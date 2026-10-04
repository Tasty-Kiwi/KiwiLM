"""Explicit, private-only publication or draft staging of a verified KiwiLM bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

from kiwilm.safetensors_io import BUNDLE_FILES, sha256_file, verify_safetensors_bundle

RELEASE_FILES = {*BUNDLE_FILES, "README.md", "LICENSE", "generate.py",
                 "kiwilm-0.1.0-py3-none-any.whl", "release.json"}


def publish(bundle: Path, api: HfApi, *, stage_private_draft: bool = False) -> str:
    release = json.loads((bundle / "release.json").read_text())
    unfrozen = release["source_freeze_pending"] or not release.get("source_tag")
    if unfrozen and not stage_private_draft:
        raise ValueError("commit/tag and rebuild the release before publishing")
    verify_safetensors_bundle(bundle)
    if set(release["files"]) != RELEASE_FILES - {"release.json"}:
        raise ValueError("unexpected release files; training checkpoints must not be published")
    for name, details in release["files"].items():
        file = bundle / name
        if sha256_file(file) != details["sha256"] or file.stat().st_size != details["bytes"]:
            raise ValueError(f"release integrity check failed: {name}")
    repo_id = release["repo_id"]
    if repo_id != "Tasty-Kiwi/KiwiLM-2":
        raise ValueError("unexpected release repository")
    # Never accept an existing repository with unknown visibility or change it to public.
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=False)
    if not api.repo_info(repo_id=repo_id, repo_type="model").private:
        raise ValueError("refusing to upload to a non-private repository")
    result = api.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=bundle,
        allow_patterns=sorted(RELEASE_FILES),
        commit_message=("Stage private KiwiLM 2 Dense 1B draft; source freeze pending"
                        if unfrozen else "Release KiwiLM 2 Dense 1B bundle"),
    )
    if not api.repo_info(repo_id=repo_id, repo_type="model").private:
        raise ValueError("repository privacy changed during upload; inspect the repository")
    return result.oid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--publish", action="store_true", help="authorize creating/uploading privately",
    )
    parser.add_argument(
        "--stage-private-draft", action="store_true",
        help="explicitly stage an unfrozen draft privately without changing its provenance",
    )
    args = parser.parse_args()
    if not args.publish:
        print("No remote changes. Add --publish only after reviewing the frozen release.")
        return
    commit = publish(args.bundle, HfApi(), stage_private_draft=args.stage_private_draft)
    print(json.dumps({"repo_id": "Tasty-Kiwi/KiwiLM-2", "private": True, "hub_commit": commit}))


if __name__ == "__main__":
    main()
