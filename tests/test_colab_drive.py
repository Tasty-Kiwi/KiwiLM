"""Tests for Colab Google Drive persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path

from kiwilm.colab_drive import (
    CACHE_MARKER,
    CheckpointBackup,
    cache_prepared_data,
    restore_checkpoint_backup,
    restore_prepared_data,
)


def test_prepared_data_cache_is_published_and_restored(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "metadata.json").write_text(
        json.dumps({"fingerprint": "a" * 64}),
        encoding="utf-8",
    )
    (prepared / "train.bin").write_bytes(b"training tokens")
    cache = tmp_path / "drive" / "data" / "smoke"

    assert cache_prepared_data(prepared, cache) is True
    assert (cache / CACHE_MARKER).is_file()
    assert cache_prepared_data(prepared, cache) is False

    restored = tmp_path / "restored"
    assert restore_prepared_data(cache, restored) is True
    assert (restored / "train.bin").read_bytes() == b"training tokens"
    assert not (restored / CACHE_MARKER).exists()


def test_checkpoint_backup_sync_and_restore(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    job = tmp_path / "job.json"
    job.write_text('{"phase":"smoke"}\n', encoding="utf-8")
    backup_dir = tmp_path / "drive" / "checkpoints" / "run-key"
    backup = CheckpointBackup(
        run_dir,
        backup_dir,
        job_path=job,
        interval_seconds=60,
    )

    assert backup.sync_if_changed() is False
    assert (backup_dir / "job.json").read_text(encoding="utf-8") == job.read_text(
        encoding="utf-8"
    )
    (run_dir / "latest.pt").write_bytes(b"checkpoint-one")
    (run_dir / "best.pt").write_bytes(b"best-one")
    (run_dir / "metrics.jsonl").write_text('{"step":1}\n', encoding="utf-8")
    assert backup.sync_if_changed() is True
    assert backup.sync_if_changed() is False
    assert (backup_dir / "latest.pt").read_bytes() == b"checkpoint-one"

    (run_dir / "latest.pt").write_bytes(b"checkpoint-two-is-larger")
    assert backup.sync_if_changed() is True
    restored_dir = tmp_path / "resumed"
    resume = restore_checkpoint_backup(backup_dir, restored_dir)
    assert resume == restored_dir / "resume.pt"
    assert resume.read_bytes() == b"checkpoint-two-is-larger"
    assert (restored_dir / "best.pt").read_bytes() == b"best-one"
    assert (restored_dir / "metrics.jsonl").is_file()
