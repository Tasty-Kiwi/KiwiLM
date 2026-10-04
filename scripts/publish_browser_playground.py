"""Explicit static-only deployment to the existing KiwiLM playground Space."""

import argparse
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

from kiwilm.safetensors_io import sha256_file

LEGACY_MANIFEST = (
    Path(__file__).resolve().parents[1] / "spaces/kiwilm-playground/legacy-manifest.json"
)


def publish(dist: Path, readme: Path, api: HfApi) -> str:
    model_dir = dist / "models/kiwilm2"
    manifest = json.loads((model_dir / "browser-manifest.json").read_text())
    if (manifest["model_repo"] != "Tasty-Kiwi/KiwiLM-2"
            or manifest["model_revision"] != "53d6abcb4d11c8a98936c19d55d2b741a99d7602"
            or manifest["verification"]["direct_cached_rollover"] != "passed"):
        raise ValueError("unexpected or unverified browser model")
    if set(manifest["files"]) != {"model.onnx", "tokenizer.json", "parity.json"}:
        raise ValueError("unexpected browser files")
    for name, details in manifest["files"].items():
        file = model_dir / name
        if sha256_file(file) != details["sha256"] or file.stat().st_size != details["bytes"]:
            raise ValueError(f"browser asset integrity failed: {name}")
    files = sorted(file for file in dist.rglob("*") if file.is_file())
    for file in files:
        name = file.relative_to(dist).as_posix()
        allowed = (name == "index.html" or name == "models/kiwilm2/browser-manifest.json"
                   or name in {f"models/kiwilm2/{name}" for name in manifest["files"]}
                   or (name.startswith("assets/") and file.suffix in {".js", ".css", ".wasm"}))
        if not allowed:
            raise ValueError(f"unexpected deployment file: {name}")
    if not (dist / "index.html").is_file() or "sdk: static" not in readme.read_text():
        raise ValueError("static application required")
    repo = "Tasty-Kiwi/KiwiLM-Playground"
    info = api.repo_info(repo, repo_type="space", files_metadata=True)
    if info.sdk != "static":
        raise ValueError("refusing to change the Space SDK or allocate compute")
    legacy = json.loads(LEGACY_MANIFEST.read_text())
    remote = {file.rfilename: file for file in info.siblings}
    for model_id, bundle in legacy["models"].items():
        for name, details in bundle["files"].items():
            path = f"models/{model_id}/{name}"
            if path not in remote:
                raise ValueError(f"preserved legacy asset missing: {path}")
            file = remote[path]
            if file.size is not None and file.size != details["bytes"]:
                raise ValueError(f"legacy asset size mismatch: {path}")
            if file.lfs is not None and file.lfs.sha256 != details["sha256"]:
                raise ValueError(f"legacy asset checksum mismatch: {path}")
    operations = [CommitOperationAdd(path_in_repo=file.relative_to(dist).as_posix(),
                                     path_or_fileobj=file) for file in files]
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme))
    # Reuse existing X/Y artifacts; preserve all models and Space history.
    result = api.create_commit(repo_id=repo, repo_type="space", operations=operations,
                               parent_commit=info.sha,
                               commit_message="Restore KiwiLM 1 choices alongside KiwiLM 2")
    return result.oid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("spaces/kiwilm-playground/dist"))
    parser.add_argument("--readme", type=Path, default=Path("spaces/kiwilm-playground/README.md"))
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.publish:
        print(json.dumps({"space": "Tasty-Kiwi/KiwiLM-Playground",
                          "commit": publish(args.dist, args.readme, HfApi())}))
    else:
        print("No remote changes. Add --publish to update the existing static playground.")
