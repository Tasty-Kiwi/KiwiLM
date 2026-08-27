"""Google Drive persistence helpers for Colab training jobs."""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path

CACHE_MARKER = ".complete"
CHECKPOINT_FILES = ("latest.pt", "best.pt", "metrics.jsonl")


def atomic_copy_file(source: str | Path, destination: str | Path) -> Path:
    """Copy a file and publish it atomically within the destination directory."""

    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.parent / (
        f".{destination_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        shutil.copyfile(source_path, temporary)
        os.replace(temporary, destination_path)
    finally:
        temporary.unlink(missing_ok=True)
    return destination_path


def cache_prepared_data(data_dir: str | Path, cache_dir: str | Path) -> bool:
    """Publish prepared data to a Drive cache, with the marker written last."""

    source = Path(data_dir)
    destination = Path(cache_dir)
    marker = destination / CACHE_MARKER
    if marker.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.partial"
    try:
        shutil.copytree(source, staging)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        (staging / CACHE_MARKER).write_text(
            json.dumps({"fingerprint": metadata.get("fingerprint")}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if marker.is_file():
            return False
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
        return True
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def restore_prepared_data(cache_dir: str | Path, data_dir: str | Path) -> bool:
    """Restore a complete prepared-data cache to local Colab storage."""

    source = Path(cache_dir)
    if not (source / CACHE_MARKER).is_file():
        return False
    destination = Path(data_dir)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(CACHE_MARKER))
    return True


def restore_checkpoint_backup(backup_dir: str | Path, run_dir: str | Path) -> Path | None:
    """Copy a resumable Drive checkpoint and its sidecars to local storage."""

    source = Path(backup_dir)
    latest = source / "latest.pt"
    if not latest.is_file():
        return None
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    resume = atomic_copy_file(latest, destination / "resume.pt")
    for name in ("best.pt", "metrics.jsonl"):
        candidate = source / name
        if candidate.is_file():
            atomic_copy_file(candidate, destination / name)
    return resume


class CheckpointBackup:
    """Mirror atomically published local checkpoints to Drive in the background."""

    def __init__(
        self,
        run_dir: str | Path,
        backup_dir: str | Path,
        *,
        job_path: str | Path,
        interval_seconds: float = 30.0,
        error_callback: Callable[[Exception], None] | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.run_dir = Path(run_dir)
        self.backup_dir = Path(backup_dir)
        self.job_path = Path(job_path)
        self.interval_seconds = interval_seconds
        self.error_callback = error_callback
        self.successful_syncs = 0
        self.last_error: Exception | None = None
        self._last_signature: tuple[int, int] | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _copy_if_present(self, source: Path, name: str) -> None:
        if source.is_file():
            atomic_copy_file(source, self.backup_dir / name)

    def sync_if_changed(
        self,
        *,
        force: bool = False,
        extra_files: Iterable[tuple[str, str | Path]] = (),
    ) -> bool:
        """Copy one coherent checkpoint snapshot when ``latest.pt`` changes."""

        latest = self.run_dir / "latest.pt"
        if not latest.is_file():
            self._copy_if_present(self.job_path, "job.json")
            return False
        details = latest.stat()
        signature = (details.st_mtime_ns, details.st_size)
        if not force and signature == self._last_signature:
            return False
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._copy_if_present(self.job_path, "job.json")
        for name in CHECKPOINT_FILES:
            self._copy_if_present(self.run_dir / name, name)
        for name, path in extra_files:
            self._copy_if_present(Path(path), name)
        self._last_signature = signature
        self.successful_syncs += 1
        self.last_error = None
        return True

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self.sync_if_changed()
            except Exception as error:  # pragma: no cover - depends on Drive FUSE
                self.last_error = error
                if self.error_callback is not None:
                    self.error_callback(error)

    def start(self) -> None:
        """Start polling without blocking training."""

        if self._thread is not None:
            raise RuntimeError("checkpoint backup has already been started")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._copy_if_present(self.job_path, "job.json")
        self._thread = threading.Thread(
            target=self._run,
            name="kiwilm-drive-backup",
            daemon=True,
        )
        self._thread.start()

    def stop(
        self,
        *,
        final_sync: bool = True,
        extra_files: Iterable[tuple[str, str | Path]] = (),
    ) -> None:
        """Stop polling and optionally require a final synchronous backup."""

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        if final_sync:
            self.sync_if_changed(force=True, extra_files=extra_files)


__all__ = [
    "CACHE_MARKER",
    "CheckpointBackup",
    "atomic_copy_file",
    "cache_prepared_data",
    "restore_checkpoint_backup",
    "restore_prepared_data",
]
