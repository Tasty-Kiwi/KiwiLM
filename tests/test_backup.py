"""Verified directory backup coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kiwilm.backup import VerifiedDirectoryBackup, file_digest


def test_verified_backup_replaces_files_and_merges_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "drive"
    source.mkdir()
    checkpoint = source / "latest.pt"
    metrics = source / "metrics.jsonl"
    checkpoint.write_bytes(b"checkpoint-one")
    metrics.write_text('{"step": 1}\n', encoding="utf-8")

    backup = VerifiedDirectoryBackup(destination)
    first = backup.sync(
        {"latest.pt": checkpoint, "metrics.jsonl": metrics},
        metadata={"step": 1, "complete": False},
    )

    assert first["files"]["latest.pt"] == file_digest(checkpoint)
    assert (destination / "latest.pt").read_bytes() == b"checkpoint-one"
    checkpoint.write_bytes(b"checkpoint-two")
    second = backup.sync(
        {"latest.pt": checkpoint},
        metadata={"step": 2, "complete": True},
    )

    assert (destination / "latest.pt").read_bytes() == b"checkpoint-two"
    assert second["metadata"] == {"step": 2, "complete": True}
    assert "metrics.jsonl" in second["files"]
    assert backup.verify() == second


def test_backup_manifest_detects_corruption(tmp_path: Path) -> None:
    source = tmp_path / "best.pt"
    source.write_bytes(b"valid")
    destination = tmp_path / "drive"
    backup = VerifiedDirectoryBackup(destination)
    backup.sync({"best.pt": source})

    (destination / "best.pt").write_bytes(b"corrupt")

    with pytest.raises(OSError, match="verification failed"):
        backup.verify()


@pytest.mark.parametrize("name", ["", "../best.pt", "run/best.pt", "/best.pt"])
def test_backup_rejects_non_filename_names(tmp_path: Path, name: str) -> None:
    source = tmp_path / "best.pt"
    source.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="plain filename"):
        VerifiedDirectoryBackup(tmp_path / "drive").sync({name: source})


def test_backup_rejects_invalid_manifest(tmp_path: Path) -> None:
    destination = tmp_path / "drive"
    destination.mkdir()
    (destination / "backup-manifest.json").write_text(
        json.dumps({"schema_version": 999, "files": {}, "metadata": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported"):
        VerifiedDirectoryBackup(destination).verify()
