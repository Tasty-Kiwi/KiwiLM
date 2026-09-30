"""Prepare exact TPU smoke inputs, without importing XLA or starting training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from kiwilm.colab_artifacts import file_sha256, reassemble_colab_artifacts, require_mapping
from kiwilm.colab_drive import CACHE_MARKER, atomic_copy_file, restore_prepared_data
from kiwilm.colab_kiwilm2 import build_colab_job
from kiwilm.data import PreparedTokenData, metadata_fingerprint
from kiwilm.tpu_smoke import validate_tied_checkpoint


def job_digest(job: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()


def validate_job(
    job: dict[str, Any], drive_root: Path, *, check_cache_symlinks: bool = True,
) -> None:
    require_mapping(job, "TPU setup job")
    if job.get("schema_version") != 1 or not isinstance(job.get("use_drive"), bool):
        raise ValueError("invalid TPU setup job")
    keys = ["data_fingerprint", "tokenizer_sha256"]
    if job.get("resume_sha256") is not None:
        keys.append("resume_sha256")
    for key in keys:
        value = job.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"invalid TPU setup {key}")
    if job["use_drive"]:
        cache_name = job.get("drive_cache_dir")
        if not isinstance(cache_name, str):
            raise ValueError("missing TPU Drive cache directory")
        cache = Path(cache_name)
        if (
            not cache.is_absolute() or ".." in cache.parts
            or not cache.is_relative_to(drive_root)
            or (check_cache_symlinks and not cache.resolve().is_relative_to(drive_root.resolve()))
        ):
            raise ValueError("TPU data cache must be inside the mounted Drive directory")
        # Never replace the mount itself or a broad Drive root.
        if len(cache.relative_to(drive_root).parts) < 4 or cache.parent.name != "data":
            raise ValueError("TPU data cache must be a specific dataset directory")


def write_setup_job(
    data_dir: Path, output: Path, *, use_drive: bool, cache_dir: str,
    resume: Path | None = None,
) -> None:
    """Validate the frozen local inputs before any allocation or upload."""
    frozen = build_colab_job(data_dir, phase="smoke", architecture="kiwilm2")
    metadata = json.loads((data_dir / "metadata.json").read_text())
    if resume is not None:
        validate_tied_checkpoint(resume)
    job = {
        "schema_version": 1, "use_drive": use_drive,
        "data_fingerprint": frozen["data_fingerprint"],
        "tokenizer_sha256": metadata["tokenizer"]["sha256"],
        "drive_cache_dir": cache_dir or (
            "/content/drive/MyDrive/KiwiLM2/data/tpu-smoke-" + frozen["data_fingerprint"]
        ),
        "resume_sha256": file_sha256(resume) if resume is not None else None,
    }
    validate_job(job, Path("/content/drive"))
    output.write_text(json.dumps(job, indent=2) + "\n")


def setup_owner(
    result_dir: Path, report_path: Path, *, session: str, tpu: str, claim: bool = False,
) -> None:
    """Claim/verify setup ownership without optimization-removable assertions."""
    job = json.loads((result_dir / "tpu-job.json").read_text())
    validate_job(job, Path("/content/drive"))
    report = require_mapping(json.loads(report_path.read_text()), "TPU setup report")
    if report.get("state") != "ready" or report.get("job_digest") != job_digest(job):
        raise ValueError("TPU setup is not ready or its job changed")
    expected = {"session": session, "tpu": tpu, "job_digest": job_digest(job)}
    owner_path = result_dir / "setup-owner.json"
    if claim:
        with owner_path.open("x") as stream:
            stream.write(json.dumps(expected) + "\n")
    elif json.loads(owner_path.read_text()) != expected:
        raise ValueError("TPU setup session ownership differs; refusing training")


def validate_cache_marker(cache: Path, job: dict[str, Any]) -> None:
    try:
        marker = json.loads((cache / CACHE_MARKER).read_text())
    except FileNotFoundError as error:
        raise ValueError("Drive cache is incomplete; not overwriting") from error
    if require_mapping(marker, "Drive cache marker").get("fingerprint") != job["data_fingerprint"]:
        raise ValueError("Drive cache marker fingerprint differs; not overwriting")
    metadata = require_mapping(json.loads((cache / "metadata.json").read_text()), "cache metadata")
    if metadata.get("fingerprint") != job["data_fingerprint"] or (
        metadata_fingerprint(metadata) != job["data_fingerprint"]
    ):
        raise ValueError("Drive cache metadata fingerprint differs; not overwriting")
    if require_mapping(metadata.get("tokenizer"), "cache tokenizer").get("sha256") != (
        job["tokenizer_sha256"]
    ):
        raise ValueError("Drive cache tokenizer differs from the frozen local dataset")


def cache_exists(cache: Path) -> bool:
    # Do not hide ENOTCONN/EIO as a cache miss on newer Python versions.
    try:
        cache.stat()
    except FileNotFoundError:
        return False
    return True


def validated_data(path: Path, job: dict[str, Any]) -> PreparedTokenData:
    data = PreparedTokenData(path, expected_fingerprint=job["data_fingerprint"])
    if data.metadata["tokenizer"]["sha256"] != job["tokenizer_sha256"]:
        raise ValueError("TPU setup tokenizer differs from the frozen local dataset")
    return data


def publish_data_cache(data_dir: Path, cache: Path, job: dict[str, Any]) -> None:
    data = validated_data(data_dir, job)
    # Reserve a NEW directory. Do not replace/delete even an incomplete old cache.
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(exist_ok=False)
    names = ["metadata.json", data.metadata["tokenizer"]["file"], *(
        split["file"] for split in data.metadata["splits"].values()
    )]
    for name in names:
        if Path(name).name != name or not name or "\\" in name:
            raise ValueError("unsafe prepared data cache filename")
        atomic_copy_file(data_dir / name, cache / name)
    validated_data(cache, job)
    (cache / CACHE_MARKER).write_text(json.dumps({"fingerprint": data.fingerprint}) + "\n")


def prepare_inputs(
    job: dict[str, Any], *, data_dir: Path, resume_dir: Path, drive_root: Path,
    require_ready: bool = False,
) -> dict[str, Any]:
    validate_job(job, drive_root, check_cache_symlinks=not require_ready)
    if job["use_drive"] and not require_ready and not drive_root.is_mount():
        raise RuntimeError("mount Google Drive before preparing TPU data")
    if require_ready and not (data_dir / "metadata.json").is_file():
        raise ValueError("TPU inputs are not ready; run setup first")
    source = "local-vm"
    if not (data_dir / "metadata.json").is_file():
        restored = False
        if job["use_drive"]:
            cache = Path(job["drive_cache_dir"])
            # A disconnected mount is an error, not an absent cache.
            if cache_exists(cache):
                validate_cache_marker(cache, job)
                restored = restore_prepared_data(
                    cache, data_dir, required=True, storage_root=drive_root,
                )
                source = "drive-cache"
        if not restored:
            manifest = data_dir / "artifact-manifest.json"
            if not manifest.is_file():
                if require_ready:
                    raise ValueError("TPU inputs are not ready; run setup first")
                return {"state": "needs-data-upload", "job_digest": job_digest(job)}
            reassemble_colab_artifacts(manifest, data_dir)
            source = "uploaded-chunks"
    validated_data(data_dir, job)
    if job["use_drive"] and not require_ready:
        if not drive_root.is_mount():
            raise RuntimeError("Google Drive disconnected during TPU setup")
        cache = Path(job["drive_cache_dir"])
        if cache_exists(cache):
            validate_cache_marker(cache, job)
        else:
            # Publish only a new specific cache; never delete an existing Drive directory.
            publish_data_cache(data_dir, cache, job)
    if job.get("resume_sha256"):
        checkpoint = resume_dir / "latest.pt"
        if not checkpoint.is_file():
            manifest = resume_dir / "artifact-manifest.json"
            if not manifest.is_file():
                if require_ready:
                    raise ValueError("resume checkpoint is not ready")
                return {
                    "state": "needs-resume-upload", "job_digest": job_digest(job),
                    "data_source": source,
                }
            reassemble_colab_artifacts(manifest, resume_dir)
        if file_sha256(checkpoint) != job["resume_sha256"]:
            raise ValueError("resume checkpoint checksum differs from the local setup job")
        validate_tied_checkpoint(checkpoint)
    return {
        "state": "ready", "job_digest": job_digest(job), "data_source": source,
        "data_fingerprint": job["data_fingerprint"], "tokenizer_sha256": job["tokenizer_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("/content/kiwilm-data-artifacts"))
    parser.add_argument("--resume-dir", type=Path, default=Path("/content/kiwilm-tpu-resume"))
    parser.add_argument("--drive-root", type=Path, default=Path("/content/drive"))
    parser.add_argument("--report", type=Path, default=Path("/content/kiwilm-tpu-setup.json"))
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    report = prepare_inputs(
        job, data_dir=args.data_dir, resume_dir=args.resume_dir, drive_root=args.drive_root,
        require_ready=args.require_ready,
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
