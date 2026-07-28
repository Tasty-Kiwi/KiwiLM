"""Verified, atomic backups for checkpoints and related run artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

BACKUP_MANIFEST_NAME = "backup-manifest.json"
BACKUP_SCHEMA_VERSION = 1


def file_digest(path: str | Path) -> dict[str, int | str]:
    """Return the byte length and SHA-256 digest of a file."""

    resolved = Path(path)
    digest = sha256()
    size = 0
    with resolved.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


class VerifiedDirectoryBackup:
    """Copy named files into a directory and publish a verified manifest last."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.manifest_path = self.directory / BACKUP_MANIFEST_NAME

    def sync(
        self,
        sources: Mapping[str, str | Path],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically replace backup files and then update their manifest."""

        if not sources:
            raise ValueError("backup sources cannot be empty")
        self.directory.mkdir(parents=True, exist_ok=True)
        prior = self._read_manifest()
        files = dict(prior.get("files") or {})

        for name, source in sources.items():
            self._validate_name(name)
            source_path = Path(source)
            if not source_path.is_file():
                raise FileNotFoundError(f"backup source does not exist: {source_path}")
            destination = self.directory / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            copied = self._copy_atomically(source_path, destination)
            verified = file_digest(destination)
            if verified != copied:
                raise OSError(f"backup verification failed for {destination}")
            files[name] = copied

        merged_metadata = dict(prior.get("metadata") or {})
        merged_metadata.update(dict(metadata or {}))
        manifest = {
            "schema_version": BACKUP_SCHEMA_VERSION,
            "updated_at": datetime.now(UTC).isoformat(),
            "files": dict(sorted(files.items())),
            "metadata": merged_metadata,
        }
        self._write_manifest(manifest)
        return manifest

    def verify(self) -> dict[str, Any]:
        """Validate every file described by the published manifest."""

        manifest = self._read_manifest(required=True)
        for name, expected in manifest["files"].items():
            self._validate_name(name)
            actual = file_digest(self.directory / name)
            if actual != expected:
                raise OSError(f"backup verification failed for {name}")
        return manifest

    def _copy_atomically(self, source: Path, destination: Path) -> dict[str, int | str]:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        digest = sha256()
        size = 0
        try:
            with os.fdopen(file_descriptor, "wb") as target, source.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    target.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                target.flush()
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return {"bytes": size, "sha256": digest.hexdigest()}

    def _read_manifest(self, *, required: bool = False) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            if required:
                raise FileNotFoundError(f"backup manifest is missing: {self.manifest_path}")
            return {}
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != BACKUP_SCHEMA_VERSION:
            raise ValueError("unsupported backup manifest schema")
        if not isinstance(manifest.get("files"), dict):
            raise ValueError("backup manifest has invalid files")
        if not isinstance(manifest.get("metadata"), dict):
            raise ValueError("backup manifest has invalid metadata")
        return manifest

    def _write_manifest(self, manifest: Mapping[str, Any]) -> None:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.manifest_path.name}.",
            suffix=".tmp",
            dir=self.directory,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
            os.replace(temporary, self.manifest_path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _validate_name(name: str) -> None:
        path = PurePosixPath(name)
        if not name or path.is_absolute() or ".." in path.parts or path.name != name:
            raise ValueError(f"backup name must be a plain filename: {name!r}")
