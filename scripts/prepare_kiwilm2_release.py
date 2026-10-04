"""Prepare and verify a local 1B release. Never trains, tags, or uploads."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import torch

from kiwilm.diagnostics import cached_generation_parity_report
from kiwilm.generation import generate
from kiwilm.inference import load_trained_model
from kiwilm.models import build_model
from kiwilm.safetensors_io import (
    export_provenance,
    export_safetensors_bundle,
    sha256_file,
    verify_safetensors_bundle,
)
from kiwilm.tokenizer import ByteBPETokenizer

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CHECKPOINT = "cdbea1da1f7ad6b9cc27f609cdbe35673e597b7741895905384a21e4060eb173"
EXPECTED_TOKENIZER = "4bcfc2d969a7a8c2285b364d709917d14e17a141e281fed9d7770db00329acf3"


def verify_package_wheel(wheel: Path, source_dir: Path) -> None:
    """Prevent an old/different package wheel from being attributed to frozen source."""
    expected = {"kiwilm/" + str(f.relative_to(source_dir)).replace(os.sep, "/"): f
                for f in source_dir.rglob("*")
                if f.is_file() and (f.suffix == ".py" or f.name == "py.typed")}
    with zipfile.ZipFile(wheel) as archive:
        names = {name for name in archive.namelist()
                 if name.startswith("kiwilm/") and not name.endswith("/")}
        if names != set(expected):
            raise ValueError("wheel package files differ from current source")
        for name, file in expected.items():
            if archive.read(name) != file.read_bytes():
                raise ValueError(f"wheel source differs: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "runs/colab/tpu-v6e1-final-1b-muon-resume1/latest.pt"))
    parser.add_argument("--checkpoint-provenance", type=Path, default=Path(
        "runs/colab/tpu-v6e1-final-1b-muon-resume1/tpu-job.json"))
    parser.add_argument("--tokenizer-from", type=Path, default=Path("data/smollm-architecture"))
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument(
        "--source-tag", help="existing tag at clean HEAD; required before publication",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validation-data", type=Path,
                        help="optional frozen prepared data for paired rounding-loss evaluation")
    parser.add_argument("--validation-batches", type=int, default=32)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path(f"artifacts/huggingface/KiwiLM-2-{args.dtype}")
    if args.validation_batches < 1:
        raise ValueError("validation batches must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing release: {args.output_dir}")
    if sha256_file(args.checkpoint) != EXPECTED_CHECKPOINT:
        raise ValueError("release checkpoint differs from the frozen, evaluated 1B checkpoint")
    if args.wheel.name != "kiwilm-0.1.0-py3-none-any.whl":
        raise ValueError("expected the built KiwiLM 0.1.0 wheel")
    verify_package_wheel(args.wheel, ROOT / "src/kiwilm")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    if args.source_tag:
        tagged = subprocess.check_output(
            ["git", "rev-parse", "--verify", f"refs/tags/{args.source_tag}^{{commit}}"],
            cwd=ROOT, text=True,
        ).strip()
        if dirty or tagged != head:
            raise ValueError("release source tag must point at clean HEAD")
    prepared = json.loads((args.tokenizer_from / "metadata.json").read_text())
    provenance = export_provenance(prepared, args.checkpoint_provenance)
    if provenance["tokenizer_sha256"] != EXPECTED_TOKENIZER:
        raise ValueError("release tokenizer differs from the frozen evaluated tokenizer")
    tokenizer_file = args.tokenizer_from / prepared["tokenizer"]["file"]
    if not tokenizer_file.resolve().is_relative_to(args.tokenizer_from.resolve()):
        raise ValueError("tokenizer artifact must be inside its prepared dataset")
    torch.set_num_threads(4)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".kiwilm-release-", dir=args.output_dir.parent))
    try:
        bundle = staging / "bundle"
        export_safetensors_bundle(
            args.checkpoint, bundle, tokenizer_path=tokenizer_file,
            expected_data_fingerprint=provenance["checkpoint_data_fingerprint"],
            expected_tokenizer_sha256=provenance["tokenizer_sha256"],
            provenance=provenance, variant="dense-tpu-v6e1-muon-0.01-1b",
            dtype=args.dtype,
        )
        metadata = json.loads((bundle / "metadata.json").read_text())
        if metadata["tokens_seen"] != 1_000_000_000 or metadata["step"] != 61036:
            raise ValueError("only the exact-1B final checkpoint is eligible")
        if metadata["model_config"]["architecture"] != "kiwilm2":
            raise ValueError("expected Dense KiwiLM 2")
        source, config = load_trained_model(
            args.checkpoint, data_fingerprint=provenance["checkpoint_data_fingerprint"],
            device=torch.device("cpu"),
        )
        portable, restored_config = load_trained_model(
            bundle, data_fingerprint=None, device=torch.device("cpu"),
        )
        if (config != restored_config
                or portable.token_embedding.weight is not portable.lm_head.weight):
            raise ValueError("export reconstruction/config/weight tying failed")
        generator = torch.Generator().manual_seed(143)
        storage_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        rounded = build_model(config).eval()
        rounded.load_state_dict({
            name: tensor.to(storage_dtype).float() if tensor.is_floating_point() else tensor
            for name, tensor in source.state_dict().items()
        }, strict=True)
        logit_checks = []
        with torch.inference_mode():
            for length in (1, 31, 512):
                values = torch.randint(config.vocab_size, (1, length), generator=generator)
                first, second = source(values), portable(values)
                reference = rounded(values)
                if not torch.equal(reference, second):
                    raise ValueError(f"rounded-reference export parity failed at length {length}")
                if not torch.isfinite(second).all():
                    raise ValueError("non-finite exported-model logits")
                delta = (first - second).abs()
                logit_checks.append({"length": length, "rounded_reference_bitwise_equal": True,
                                     "fp32_source_bitwise_equal": torch.equal(first, second),
                                     "maximum_absolute_difference_from_fp32": delta.max().item(),
                                     "mean_absolute_difference_from_fp32": delta.mean().item(),
                                     "argmax_agreement": (first.argmax(-1) == second.argmax(-1))
                                     .float().mean().item()})
        parity = cached_generation_parity_report(portable, values)
        if not parity["passed"]:
            raise ValueError("bundle cached-generation parity failed")
        tokenizer = ByteBPETokenizer.load(bundle / "tokenizer.json")
        generations = []
        for seed in (42, 46):
            for cache in ("off", "auto"):
                options = dict(max_new_tokens=32, context_length=512, temperature=0.8,
                               top_k=40, seed=seed, device=torch.device("cpu"), cache=cache)
                prompt = "Once upon a time, a fox found a box."
                a = generate(rounded, tokenizer, prompt, **options)
                b = generate(portable, tokenizer, prompt, **options)
                original = generate(source, tokenizer, prompt, **options)
                if a != b:
                    raise ValueError("source/bundle generated text differs")
                generations.append({"seed": seed, "cache": cache,
                                    "rounded_reference_text_equal": True,
                                    "fp32_source_text_equal": original == b,
                                    "text": b, "fp32_source_text": original})
        validation = {"status": "not measured; supply --validation-data"}
        if args.validation_data is not None:
            from kiwilm.data import PreparedTokenData
            from kiwilm.training import evaluate

            data = PreparedTokenData(args.validation_data)
            if data.metadata["tokenizer"]["sha256"] != EXPECTED_TOKENIZER:
                raise ValueError("rounding validation uses a different tokenizer")
            if data.metadata["splits"]["validation"]["sha256"] != (
                "afc8a779d6d941584505c17318c24e56a6ac3e1e02b906beb33dcafb87be1e1c"
            ):
                raise ValueError("rounding validation uses a different frozen split")
            results = [evaluate(
                model, data, batch_size=2, context_length=512,
                num_batches=args.validation_batches, device=torch.device("cpu"),
                generator=torch.Generator().manual_seed(143), precision="fp32",
            ) for model in (source, portable)]
            difference = results[1]["validation_loss"] - results[0]["validation_loss"]
            validation = {"status": "measured", "batches": args.validation_batches,
                          "batch_size": 2, "context_length": 512, "seed": 143,
                          "fp32_source": results[0], "exported_weights_fp32_execution": results[1],
                          "loss_change": difference, "maximum_allowed_loss_increase": 0.01}
            if difference > 0.01:
                raise ValueError(
                    "paired validation loss increases by more than 0.01 after rounding",
                )
        for origin, name in ((ROOT / "releases/kiwilm2-1b/README.md", "README.md"),
                             (ROOT / "LICENSE", "LICENSE"),
                             (ROOT / "examples/generate_from_hub.py", "generate.py"),
                             (args.wheel, args.wheel.name)):
            shutil.copyfile(origin, bundle / name)
        if args.dtype == "fp32":
            card = bundle / "README.md"
            card.write_text(card.read_text().replace(
                "Weights are exported in **BF16** Safetensors by default.",
                "This bundle was explicitly exported in **FP32** Safetensors.",
            ))
        if args.source_tag:
            card = bundle / "README.md"
            card.write_text(card.read_text().replace(
                "This card is a **local draft** until source commit/tag and Hub publication are\n"
                "completed. Follow the repository release runbook before uploading.",
                f"Release source tag: `{args.source_tag}`; commit: `{head}`. "
                "See `release.json` for verification and package checksums.",
            ))
        verify_safetensors_bundle(bundle)
        if sha256_file(args.checkpoint) != EXPECTED_CHECKPOINT:
            raise ValueError("source checkpoint changed during export")
        if args.source_tag:
            current_head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            ).strip()
            current_dirty = subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True,
            ).strip()
            current_tag = subprocess.check_output(
                ["git", "rev-parse", "--verify", f"refs/tags/{args.source_tag}^{{commit}}"],
                cwd=ROOT, text=True,
            ).strip()
            if current_head != head or current_tag != head or current_dirty:
                raise ValueError("frozen source changed during release preparation")
            verify_package_wheel(args.wheel, ROOT / "src/kiwilm")
        release = {
            "status": "local-draft-not-published", "repo_id": "Tasty-Kiwi/KiwiLM-2",
            "license": "mit", "checkpoint_sha256": EXPECTED_CHECKPOINT,
            "weights_dtype": args.dtype,
            "source_checkpoint_preserved": True, "tooling_git_head": head,
            "tooling_worktree_dirty": dirty, "source_tag": args.source_tag,
            "source_freeze_pending": args.source_tag is None, "hub_commit": None,
            "verification": {"device": "cpu", "precision": "fp32", "torch": torch.__version__,
                             "logits": logit_checks, "cached_generation": parity,
                             "generated_text_parity": generations,
                             "paired_rounding_validation": validation,
                             "live_hub_download": "pending publication"},
            "files": {f.name: {"sha256": sha256_file(f), "bytes": f.stat().st_size}
                      for f in sorted(bundle.iterdir()) if f.is_file()},
        }
        (bundle / "release.json").write_text(json.dumps(release, indent=2, allow_nan=False) + "\n")
        os.replace(bundle, args.output_dir)
        print(json.dumps({"output_dir": str(args.output_dir), **release}, indent=2))
    finally:
        shutil.rmtree(staging)


if __name__ == "__main__":
    main()
