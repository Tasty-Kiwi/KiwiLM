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
