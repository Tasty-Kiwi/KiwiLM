"""Bounded parallel CLI uploads; publish the verified manifest last."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from kiwilm.colab_artifacts import file_sha256, require_mapping


def upload_artifacts(
    directory: Path, remote_dir: str, *, session: str, colab_bin: str = "colab",
    workers: int = 3, attempts: int = 3, retry_delay: float = 1,
) -> None:
    if (
        isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 4
        or isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 5
        or not 0 <= retry_delay <= 10
    ):
        raise ValueError("uploads require 1-4 workers, 1-5 attempts and 0-10s retry delay")
    if not session:
        raise ValueError("an explicit upload session is required")
    remote = PurePosixPath(remote_dir)
    if not remote.is_absolute() or ".." in remote.parts:
        raise ValueError("remote upload directory must be absolute without traversal")
    manifest_path = directory / "artifact-manifest.json"
    manifest = require_mapping(json.loads(manifest_path.read_text()), "upload manifest")
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("parts"), list):
        raise ValueError("invalid artifact upload manifest")
    parts = manifest["parts"]
    if not parts:
        raise ValueError("upload manifest has no parts")
    seen = set()
    for raw_part in parts:
        part = require_mapping(raw_part, "upload part")
        name = part.get("name")
        if (
            not isinstance(name, str) or not name or name in {".", "..", manifest_path.name}
            or Path(name).name != name or "\\" in name or name in seen
        ):
            raise ValueError("unsafe or duplicate upload part name")
        seen.add(name)
        path = directory / name
        if path.is_symlink() or path.stat().st_size != part.get("bytes") or (
            file_sha256(path) != part.get("sha256")
        ):
            raise ValueError(f"upload part integrity mismatch: {name}")

    def upload(name: str) -> None:
        command = [colab_bin, "upload", "-s", session, str(directory / name), str(remote / name)]
        for attempt in range(1, attempts + 1):
            try:
                subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
                print(f"Uploaded {name}", flush=True)
                return
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                if attempt == attempts:
                    raise RuntimeError(
                        f"upload failed after {attempts} attempts: {name}"
                    ) from error
                print(f"Retrying {name} ({attempt + 1}/{attempts})", flush=True)
                time.sleep(min(10, retry_delay * 2 ** (attempt - 1)))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(upload, [part["name"] for part in parts]))
    # A worker must never see a committed manifest for an incomplete upload.
    upload(manifest_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("remote_dir")
    parser.add_argument("--session", required=True)
    parser.add_argument("--colab-bin", default="colab")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    upload_artifacts(
        args.directory, args.remote_dir, session=args.session, colab_bin=args.colab_bin,
        workers=args.workers, attempts=args.attempts,
    )


if __name__ == "__main__":
    main()
