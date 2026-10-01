"""Frozen 1B TPU jobs and VM-side data preparation; never allocate or train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kiwilm.colab_artifacts import file_sha256
from kiwilm.colab_kiwilm2 import build_colab_job
from kiwilm.data import (
    TOKENIZER_BUNDLE_FILE,
    TOKENIZER_BUNDLE_SCHEMA_VERSION,
    PreparedTokenData,
    metadata_fingerprint,
    prepare_smollm_corpus,
    tokenizer_bundle_fingerprint,
)
from kiwilm.tokenizer import ByteBPETokenizer

FINAL_SETTINGS = {
    "phase": "final-1b", "max_tokens": 1_000_000_000, "warmup_tokens": 20_000_000,
    "batch_size": 8, "grad_accum_steps": 4, "eval_batches": 200,
    "eval_interval": 500, "checkpoint_interval": 500, "precision": "bf16",
    "architecture": "kiwilm2", "optimizer": "muon", "muon_lr": 0.01, "seed": 42,
    "learning_rate": 3e-4, "min_learning_rate": 3e-5, "weight_decay": 0.1,
    "beta2": 0.95, "grad_clip": 1.0, "context_length": 512, "d_model": 512,
}
RECIPE = {"train_tokens": 1_000_000_000, "validation_tokens": 2_000_000,
          "tokenizer_train_documents": 100_000, "validation_documents_per_source": 10_000,
          "fineweb_probability": 0.7, "seed": 42, "vocab_size": 32_000, "min_frequency": 2}


def validate_final_job(job: dict) -> None:
    if any(job.get(key) != value for key, value in FINAL_SETTINGS.items()) or (
        job.get("use_drive") is not True or job.get("continuation_phase") is not None
        or job.get("resume_sha256") is not None
    ):
        raise ValueError("1B TPU job differs from the frozen Dense/Muon/BF16/Drive settings")
    recipe = job.get("data_recipe")
    if not isinstance(recipe, dict) or any(recipe.get(k) != v for k, v in RECIPE.items()):
        raise ValueError("1B TPU data recipe differs from the frozen controls")
    revision = recipe.get("resolved_revision")
    if not isinstance(revision, str) or len(revision) != 40 or any(
        c not in "0123456789abcdef" for c in revision
    ) or recipe.get("dataset_name") != "HuggingFaceTB/smollm-corpus":
        raise ValueError("1B TPU dataset must have a pinned immutable SmolLM revision")
    fingerprint = job.get("tokenizer_source_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(
        c not in "0123456789abcdef" for c in fingerprint
    ):
        raise ValueError("1B TPU tokenizer must have frozen source provenance")
    if not job.get("drive_backup_dir") or (job.get("drive_resume_dir") and (
        not job.get("data_fingerprint") or job["drive_resume_dir"] != job["drive_backup_dir"]
    )):
        raise ValueError("1B recovery requires the original job fingerprint and backup namespace")


def write_final_job(
    source_dir: Path, output: Path, *, tpu: str, cache_dir: str = "", backup_dir: str = "",
    resume_job: Path | None = None,
) -> None:
    """Upload only a verified tokenizer bundle; source packed data need not be present."""
    from kiwilm.tpu_setup import validate_job

    if tpu not in {"v5e1", "v6e1"}:
        raise ValueError("unsupported TPU")
    if resume_job is not None:
        job = json.loads(resume_job.read_text())
        validate_job(job, Path("/content/drive"), check_cache_symlinks=False)
        if job.get("phase") != "final-1b" or not job.get("data_fingerprint"):
            raise ValueError("resume needs the ready original 1B tpu-job.json, not a smoke job")
        if (cache_dir and cache_dir != job["drive_cache_dir"]) or (
            backup_dir and backup_dir != job["drive_backup_dir"]
        ):
            raise ValueError("1B resume cannot change the original cache or backup namespace")
        job["drive_resume_dir"] = job["drive_backup_dir"]
    else:
        metadata = json.loads((source_dir / "metadata.json").read_text())
        if metadata_fingerprint(metadata) != metadata.get("fingerprint"):
            raise ValueError("tokenizer source metadata fingerprint is invalid")
        details = metadata["tokenizer"]
        name = details["file"]
        if Path(name).name != name or "\\" in name or details.get("vocab_size") != 32_000 or (
            details.get("requested_vocab_size") != 32_000 or details.get("min_frequency") != 2
        ):
            raise ValueError("1B requires the frozen 32K BPE tokenizer")
        if file_sha256(source_dir / name) != details["sha256"]:
            raise ValueError("tokenizer source checksum mismatch")
        if ByteBPETokenizer.from_json((source_dir / name).read_text()).vocab_size != 32_000:
            raise ValueError("tokenizer source vocabulary must actually contain 32K tokens")
        tokenizer_dir = output.parent / "tokenizer"
        tokenizer_dir.mkdir(exist_ok=True)
        bundle = {"schema_version": TOKENIZER_BUNDLE_SCHEMA_VERSION,
                  "source_dataset_fingerprint": metadata["fingerprint"],
                  "tokenizer": {k: details[k] for k in (
                      "file", "sha256", "vocab_size", "requested_vocab_size",
                      "min_frequency", "special_tokens")}}
        bundle["fingerprint"] = tokenizer_bundle_fingerprint(bundle)
        for target, contents in (
            (tokenizer_dir / name, (source_dir / name).read_bytes()),
            (tokenizer_dir / TOKENIZER_BUNDLE_FILE, (json.dumps(bundle) + "\n").encode()),
        ):
            if target.exists():
                if target.read_bytes() != contents:
                    raise ValueError("existing tokenizer upload bundle differs; choose new results")
            else:
                target.write_bytes(contents)
        recipe = {**RECIPE, "dataset_name": metadata["dataset"]["name"],
                  "revision": metadata["dataset"]["requested_revision"],
                  "resolved_revision": metadata["dataset"]["resolved_revision"]}
        # Include tokenizer-source provenance: changing it changes packed metadata.
        key = (f"tpu-final-1b-seed42-{recipe['resolved_revision'][:12]}-"
               f"{details['sha256'][:12]}-{metadata['fingerprint'][:12]}")
        root = "/content/drive/MyDrive/KiwiLM2/"
        job = {**FINAL_SETTINGS, "schema_version": 1, "use_drive": True,
               "data_fingerprint": None, "tokenizer_sha256": details["sha256"],
               "tokenizer_source_fingerprint": metadata["fingerprint"], "data_recipe": recipe,
               "drive_cache_dir": cache_dir or root + "data/" + key,
               "drive_backup_dir": backup_dir or root + "checkpoints/" + f"{key}-{tpu}-muon001",
               "resume_sha256": None, "continuation_phase": None}
    validate_job(job, Path("/content/drive"), check_cache_symlinks=False)
    output.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n")


def validate_final_data(path: Path, job: dict) -> PreparedTokenData:
    data = PreparedTokenData(path, expected_fingerprint=job.get("data_fingerprint"))
    build_colab_job(path, phase="final-1b", architecture="kiwilm2")
    recipe, metadata = job["data_recipe"], data.metadata
    config = metadata["config"]
    expected = {"seed": 42, "fineweb_probability": 0.7,
                "validation_documents_per_source": 10_000, "tokenizer_train_limit": 100_000}
    if any(config.get(k) != v for k, v in expected.items()) or (
        metadata["dataset"] != {"name": recipe["dataset_name"],
                                "requested_revision": recipe["revision"],
                                "resolved_revision": recipe["resolved_revision"]}
        or metadata["splits"]["validation"]["tokens"] != 2_000_000
        or metadata["tokenizer"]["sha256"] != job["tokenizer_sha256"]
        or metadata["tokenizer"].get("reused_from") != {
            "dataset_fingerprint": job["tokenizer_source_fingerprint"],
            "tokenizer_sha256": job["tokenizer_sha256"]}
    ):
        raise ValueError("1B prepared data differs from the frozen recipe/tokenizer provenance")
    return data


def prepare_final_data(
    job: dict, *, data_dir: Path, drive_root: Path, tokenizer_dir: Path,
    require_ready: bool = False,
) -> None:
    """Resolve the fingerprint once during setup; recovery never regenerates missing data."""
    from kiwilm.colab_drive import restore_prepared_data
    from kiwilm.tpu_setup import cache_exists, validate_cache_marker, validate_job

    validate_job(job, drive_root, check_cache_symlinks=not require_ready)
    if require_ready:
        if not job.get("data_fingerprint"):
            raise ValueError("1B setup must lock the prepared fingerprint before training")
        validate_final_data(data_dir, job)
        return
    if not drive_root.is_mount():
        raise RuntimeError("mount Google Drive before preparing 1B TPU data")
    if not job.get("drive_resume_dir") and cache_exists(
        Path(job["drive_backup_dir"]) / "latest.json"
    ):
        raise ValueError("this 1B job already has Drive progress; use explicit resume mode")
    cache = Path(job["drive_cache_dir"])
    if not (data_dir / "metadata.json").is_file():
        if cache_exists(cache):
            cached = validate_final_data(cache, job)
            job["data_fingerprint"] = cached.fingerprint
            validate_cache_marker(cache, job)
            restore_prepared_data(cache, data_dir, required=True, storage_root=drive_root)
        elif job.get("drive_resume_dir") or job.get("data_fingerprint"):
            raise ValueError(
                "required 1B data cache is missing; refusing regeneration/fresh training"
            )
        else:
            bundle = json.loads((tokenizer_dir / TOKENIZER_BUNDLE_FILE).read_text())
            if bundle.get("source_dataset_fingerprint") != job["tokenizer_source_fingerprint"] or (
                bundle.get("tokenizer", {}).get("sha256") != job["tokenizer_sha256"]
            ):
                raise ValueError("uploaded tokenizer differs from the frozen 1B job")
            print("Preparing 1,000,000,000 SmolLM tokens inside this VM (no training).", flush=True)
            prepare_smollm_corpus(data_dir, **job["data_recipe"], tokenizer_from=tokenizer_dir)
    job["data_fingerprint"] = validate_final_data(data_dir, job).fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/smollm-smoke"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tpu", choices=("v5e1", "v6e1"), default="v6e1")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--backup-dir", default="")
    parser.add_argument("--resume-job", type=Path)
    args = parser.parse_args()
    write_final_job(args.source_dir, args.output, tpu=args.tpu, cache_dir=args.cache_dir,
                    backup_dir=args.backup_dir, resume_job=args.resume_job)


if __name__ == "__main__":
    main()
