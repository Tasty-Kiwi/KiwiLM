"""Assemble the root files around two pre-exported KiwiLM model bundles."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from kiwilm.safetensors_io import sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--direct-bundle", type=Path, required=True)
    parser.add_argument("--cpt-bundle", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def main() -> int:
    args = build_parser().parse_args()
    destination = args.release_dir
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"release directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    shutil.copytree(args.direct_bundle, destination / "direct-sft-v2")
    shutil.copytree(args.cpt_bundle, destination / "cpt-sft-v2")
    shutil.copy2(args.repo_root / "docs/huggingface-model-card.md", destination / "README.md")
    shutil.copy2(args.repo_root / "docs/model-y.svg", destination / "model-y.svg")
    shutil.copy2(args.repo_root / "LICENSE", destination / "LICENSE")
    wheels = sorted((args.repo_root / "dist").glob("kiwilm-*.whl"))
    if len(wheels) != 1:
        raise ValueError("expected exactly one built KiwiLM wheel in dist/")
    shutil.copy2(wheels[0], destination / wheels[0].name)
    evaluation_source = (
        args.repo_root
        / "examples/comparisons/sft-v2-model-y-cpt-vs-direct"
    )
    evaluation_destination = destination / "evaluation"
    evaluation_destination.mkdir()
    for name in ("evaluation.md", "report.md", "results.jsonl", "summary.json"):
        shutil.copy2(evaluation_source / name, evaluation_destination / name)

    files = {
        str(path.relative_to(destination)): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "format": "kiwilm-huggingface-release-v1",
        "repository": "Tasty-Kiwi/KiwiLM",
        "files": files,
    }
    (destination / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
