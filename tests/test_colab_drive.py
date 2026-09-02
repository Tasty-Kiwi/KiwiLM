"""Tests for Colab Google Drive persistence helpers."""

from __future__ import annotations

import errno
import json
import shutil
from pathlib import Path

import pytest

import kiwilm.colab_drive as drive
from kiwilm.colab_drive import (
    CACHE_MARKER,
    CheckpointBackup,
    cache_prepared_data,
    restore_checkpoint_backup,
    restore_prepared_data,
)
from kiwilm.data import PreparedTokenData, prepare_from_stories


def _prepared_data(path: Path) -> dict:
    return prepare_from_stories(
        path, ["A tiny training story."], ["A tiny validation story."],
        vocab_size=300, min_frequency=1,
    )


def test_prepared_data_cache_is_published_and_restored(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    metadata = _prepared_data(prepared)
    cache = tmp_path / "drive" / "data" / "smoke"

    assert cache_prepared_data(prepared, cache) is True
    assert (cache / CACHE_MARKER).is_file()
    assert cache_prepared_data(prepared, cache) is False

    restored = tmp_path / "restored"
    assert restore_prepared_data(cache, restored) is True
    assert PreparedTokenData(restored).fingerprint == metadata["fingerprint"]
    assert not (restored / CACHE_MARKER).exists()


def test_data_restore_retries_aggregated_transport_errors_without_losing_local_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = tmp_path / "prepared"
    metadata = _prepared_data(prepared)
    cache = tmp_path / "drive" / "data"
    cache_prepared_data(prepared, cache)
    destination = tmp_path / "local"
    destination.mkdir()
    (destination / "keep.txt").write_text("previous valid data")
    real_copytree = shutil.copytree
    attempts = []

    def flaky_copy(source, target, **kwargs):
        attempts.append(target)
        assert (destination / "keep.txt").read_text() == "previous valid data"
        if len(attempts) == 1:
            Path(target).mkdir()
            (Path(target) / "partial.bin").write_bytes(b"partial")
            raise shutil.Error([
                (str(source), str(target),
                 f"[Errno {errno.ENOTCONN}] Transport endpoint is not connected"),
                (str(source), str(target), "[Errno 2] No such file or directory"),
            ])
        return real_copytree(source, target, **kwargs)

    monkeypatch.setattr(drive.shutil, "copytree", flaky_copy)
    assert restore_prepared_data(cache, destination, retry_delay=0)
    assert len(attempts) == 2
    assert PreparedTokenData(destination).fingerprint == metadata["fingerprint"]
    assert not (destination / "partial.bin").exists()
    assert not list(tmp_path.glob(".local.restore-*"))
    assert PreparedTokenData(cache).fingerprint == metadata["fingerprint"]


@pytest.mark.parametrize("corruption", ["checksum", "size", "metadata", "marker"])
def test_invalid_cache_does_not_replace_local_data(tmp_path: Path, corruption: str) -> None:
    prepared = tmp_path / "prepared"
    metadata = _prepared_data(prepared)
    cache = tmp_path / "cache"
    cache_prepared_data(prepared, cache)
    if corruption == "marker":
        (cache / CACHE_MARKER).write_text(json.dumps({"fingerprint": "b" * 64}))
    elif corruption == "metadata":
        (cache / "metadata.json").write_text("{}")
    else:
        split = cache / metadata["splits"]["train"]["file"]
        content = split.read_bytes()
        split.write_bytes(
            bytes([content[0] ^ 1]) + content[1:] if corruption == "checksum" else b"x"
        )
    destination = tmp_path / "local"
    destination.mkdir()
    (destination / "keep.txt").write_text("keep")
    with pytest.raises(ValueError):
        restore_prepared_data(cache, destination, retry_delay=0)
    assert (destination / "keep.txt").read_text() == "keep"
    assert not list(tmp_path.glob(".local.restore-*"))


@pytest.mark.parametrize("restore", [restore_prepared_data, restore_checkpoint_backup])
def test_restore_required_missing_source_never_falls_back(
    tmp_path: Path, restore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps = []
    monkeypatch.setattr(drive.time, "sleep", sleeps.append)
    source = tmp_path / "missing"
    destination = tmp_path / "local"
    assert not restore(source, destination)
    with pytest.raises(RuntimeError, match="Training has not started"):
        restore(source, destination, required=True)
    assert sleeps == [5.0, 5.0]
    assert not destination.exists()


@pytest.mark.parametrize("restore", [restore_prepared_data, restore_checkpoint_backup])
def test_unreachable_storage_does_not_look_like_missing_cache(tmp_path: Path, restore) -> None:
    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        restore(
            tmp_path / "drive" / "cache", tmp_path / "local",
            storage_root=tmp_path / "drive", attempts=2, retry_delay=0,
        )


def test_marker_transport_error_retries_even_before_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = tmp_path / "prepared"
    _prepared_data(prepared)
    cache = tmp_path / "cache"
    cache_prepared_data(prepared, cache)
    real_stat = Path.stat
    calls = 0

    def flaky_stat(path, *args, **kwargs):
        nonlocal calls
        if path == cache / CACHE_MARKER:
            calls += 1
            if calls == 1:
                raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)
    assert restore_prepared_data(cache, tmp_path / "local", retry_delay=0)
    assert calls == 2


def test_permission_error_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def denied(path):
        calls.append(path)
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(drive, "_is_file", denied)
    with pytest.raises(PermissionError):
        restore_prepared_data(tmp_path / "cache", tmp_path / "local", retry_delay=0)
    assert len(calls) == 1


def test_failed_publication_rolls_back_existing_local_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = tmp_path / "prepared"
    _prepared_data(prepared)
    cache = tmp_path / "cache"
    cache_prepared_data(prepared, cache)
    destination = tmp_path / "local"
    destination.mkdir()
    (destination / "keep.txt").write_text("keep")
    replace = drive.os.replace

    def fail_publish(src, dst):
        if Path(src).name == "data" and Path(dst) == destination:
            raise OSError(errno.ENOSPC, "disk full")
        return replace(src, dst)

    monkeypatch.setattr(drive.os, "replace", fail_publish)
    with pytest.raises(OSError, match="disk full"):
        restore_prepared_data(cache, destination, retry_delay=0)
    assert (destination / "keep.txt").read_text() == "keep"
    assert not list(tmp_path.glob(".local.*"))


@pytest.mark.parametrize("checkpoint", [False, True])
def test_source_disappearing_after_copy_begins_never_returns_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpoint: bool,
) -> None:
    source = tmp_path / "cache"
    if checkpoint:
        source.mkdir()
        (source / "latest.pt").write_bytes(b"checkpoint")
    else:
        prepared = tmp_path / "prepared"
        _prepared_data(prepared)
        cache_prepared_data(prepared, source)
    calls = 0

    def disconnected(*args, **kwargs):
        raise OSError(errno.ENOTCONN, "disconnected")

    def appears_once(path):
        nonlocal calls
        calls += 1
        return calls == 1

    monkeypatch.setattr(drive, "_is_file", appears_once)
    monkeypatch.setattr(drive.shutil, "copytree", disconnected)
    monkeypatch.setattr(drive.shutil, "copyfile", disconnected)
    restore = restore_checkpoint_backup if checkpoint else restore_prepared_data
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        restore(source, tmp_path / "local", retry_delay=0)
    assert calls == 3
    assert source.is_dir()
    assert not list(tmp_path.glob(".*restore-*"))


def test_checkpoint_retry_preserves_old_local_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "backup"
    source.mkdir()
    (source / "latest.pt").write_bytes(b"latest")
    (source / "metrics.jsonl").write_text('{"step":5000}\n')
    destination = tmp_path / "run"
    destination.mkdir()
    (destination / "resume.pt").write_bytes(b"old checkpoint")
    copyfile = shutil.copyfile
    calls = 0

    def flaky_copy(src, dst):
        nonlocal calls
        if Path(src).name == "metrics.jsonl":
            calls += 1
            assert (destination / "resume.pt").read_bytes() == b"old checkpoint"
            if calls == 1:
                raise OSError(errno.ENOTCONN, "Drive disconnected")
        return copyfile(src, dst)

    monkeypatch.setattr(drive.shutil, "copyfile", flaky_copy)
    resume = restore_checkpoint_backup(source, destination, required=True, retry_delay=0)
    assert resume.read_bytes() == b"latest"
    assert calls == 2
    assert (source / "latest.pt").read_bytes() == b"latest"
    assert not list(tmp_path.glob(".checkpoint-restore-*"))


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
