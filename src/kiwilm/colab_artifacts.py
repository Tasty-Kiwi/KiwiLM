"""Verified reconstruction of chunked Colab training artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def create_colab_artifacts(
    files: Mapping[str, Path],
    output_dir: Path,
    *,
    archive_name: str = "artifacts.tar",
    chunk_size: int = 64 * 1024 * 1024,
) -> Path:
    """Create a checksummed, chunked tar archive for reliable Colab download."""

    if not files:
        raise ValueError("artifact files must not be empty")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if Path(archive_name).name != archive_name or not archive_name:
        raise ValueError("archive_name must be a safe filename")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / archive_name
    file_details: dict[str, dict[str, Any]] = {}
    with tarfile.open(archive_path, mode="w") as archive:
        for name, path in sorted(files.items()):
            if Path(name).name != name or not name:
                raise ValueError(f"unsafe artifact name: {name!r}")
            if not path.is_file():
                raise ValueError(f"artifact does not exist: {path}")
            archive.add(path, arcname=name, recursive=False)
            file_details[name] = {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    archive_details = {
        "name": archive_name,
        "bytes": archive_path.stat().st_size,
        "sha256": file_sha256(archive_path),
    }
    parts = []
    with archive_path.open("rb") as source:
        index = 0
        while chunk := source.read(chunk_size):
            name = f"{archive_name}.part-{index:04d}"
            path = output_dir / name
            path.write_bytes(chunk)
            parts.append(
                {
                    "name": name,
                    "bytes": len(chunk),
                    "sha256": hashlib.sha256(chunk).hexdigest(),
                }
            )
            index += 1
    manifest = {
        "schema_version": 1,
        "archive": archive_details,
        "files": file_details,
        "parts": parts,
    }
    manifest_path = output_dir / "artifact-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    archive_path.unlink()
    return manifest_path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def reassemble_colab_artifacts(manifest_path: Path, output_dir: Path) -> list[Path]:
    """Verify downloaded parts and safely extract the declared artifacts."""

    manifest = require_mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        "artifact manifest",
    )
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported artifact manifest schema")
    archive_details = require_mapping(manifest.get("archive"), "archive")
    file_details = require_mapping(manifest.get("files"), "files")
    parts = manifest.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("artifact manifest parts must be a non-empty list")

    output_dir.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=output_dir,
        prefix=".kiwilm-artifacts-",
        suffix=".tar",
    )
    os.close(handle)
    temporary_archive = Path(temporary_name)
    try:
        with temporary_archive.open("wb") as destination:
            for index, raw_part in enumerate(parts):
                part = require_mapping(raw_part, f"parts[{index}]")
                name = part.get("name")
                if not isinstance(name, str) or Path(name).name != name:
                    raise ValueError(f"parts[{index}] has an unsafe name")
                path = output_dir / name
                if path.stat().st_size != part.get("bytes"):
                    raise ValueError(f"artifact part size mismatch: {name}")
                if file_sha256(path) != part.get("sha256"):
                    raise ValueError(f"artifact part checksum mismatch: {name}")
                with path.open("rb") as source:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)

        if temporary_archive.stat().st_size != archive_details.get("bytes"):
            raise ValueError("artifact archive size mismatch")
        if file_sha256(temporary_archive) != archive_details.get("sha256"):
            raise ValueError("artifact archive checksum mismatch")

        expected_names = set(file_details)
        extracted: list[Path] = []
        with tarfile.open(temporary_archive, mode="r") as archive:
            members = archive.getmembers()
            member_names = {member.name for member in members}
            if member_names != expected_names:
                raise ValueError("artifact archive contents do not match manifest")
            for member in members:
                if not member.isfile() or Path(member.name).name != member.name:
                    raise ValueError(f"unsafe artifact archive member: {member.name}")
                details = require_mapping(file_details[member.name], member.name)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot extract artifact: {member.name}")
                target = output_dir / member.name
                target_handle, target_name = tempfile.mkstemp(
                    dir=output_dir,
                    prefix=f".{member.name}.",
                    suffix=".tmp",
                )
                os.close(target_handle)
                temporary_target = Path(target_name)
                try:
                    with source, temporary_target.open("wb") as destination:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
                    if temporary_target.stat().st_size != details.get("bytes"):
                        raise ValueError(f"artifact size mismatch: {member.name}")
                    if file_sha256(temporary_target) != details.get("sha256"):
                        raise ValueError(f"artifact checksum mismatch: {member.name}")
                    os.replace(temporary_target, target)
                finally:
                    temporary_target.unlink(missing_ok=True)
                extracted.append(target)
        return extracted
    finally:
        temporary_archive.unlink(missing_ok=True)
