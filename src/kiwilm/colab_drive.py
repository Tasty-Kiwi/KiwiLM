"""Google Drive persistence helpers for Colab training jobs."""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import stat
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeVar

from kiwilm.data import PreparedTokenData

CACHE_MARKER = ".complete"
CHECKPOINT_FILES = ("latest.pt", "best.pt", "metrics.jsonl")
_T = TypeVar("_T")
_RETRY_ERRNOS = {
    errno.ENOENT, errno.ENOTCONN, errno.EIO, errno.ESTALE,
    errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNABORTED,
    errno.EHOSTUNREACH, errno.ENETUNREACH, errno.EAGAIN,
}


def _retry_restore(
    operation: Callable[[], _T], *, description: str, attempts: int, retry_delay: float
) -> _T:
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("restore attempts must be a positive integer")
    if not 0 <= retry_delay <= 60:
        raise ValueError("restore retry_delay must be between 0 and 60 seconds")
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except OSError as error:
            codes = [error.errno]
            # copytree aggregates multiple OSErrors into string triples.
            if isinstance(error, shutil.Error) and isinstance(error.args[0], list):
                codes = [
                    int(match[1]) if (match := re.search(r"\[Errno (\d+)\]", row[2])) else None
                    for row in error.args[0]
                ]
            if not codes or any(code not in _RETRY_ERRNOS for code in codes):
                raise
            if attempt == attempts:
                raise RuntimeError(
                    f"{description} failed after {attempts} attempts. Training has not started; "
                    "no Drive backup was modified. Check/remount Google Drive and retry "
                    "with the same job settings."
                ) from error
            print(
                f"{description}: {error}; retry {attempt + 1}/{attempts} "
                f"in {retry_delay:g}s",
                flush=True,
            )
            time.sleep(retry_delay)
    raise AssertionError("unreachable")


def _is_file(path: Path) -> bool:
    """Unlike Path.is_file on newer Python, do not hide transport errors."""
    try:
        return stat.S_ISREG(path.stat().st_mode)
    except FileNotFoundError:
        return False


def _check_storage_root(storage_root: str | Path | None) -> None:
    if storage_root is not None:
        # A disappeared mount must not be treated as a new, empty cache.
        with os.scandir(storage_root) as entries:
            next(entries, None)


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


def restore_prepared_data(
    cache_dir: str | Path,
    data_dir: str | Path,
    *,
    required: bool = False,
    storage_root: str | Path | None = None,
    attempts: int = 3,
    retry_delay: float = 5.0,
) -> bool:
    """Stage, checksum-validate, then publish a cache; never modify Drive."""

    source = Path(cache_dir)
    destination = Path(data_dir)
    observed_cache = False

    def restore() -> bool:
        nonlocal observed_cache
        _check_storage_root(storage_root)
        marker = source / CACHE_MARKER
        if not _is_file(marker):
            if required or observed_cache:
                raise FileNotFoundError(
                    errno.ENOENT, "Required prepared-data cache is unavailable", str(marker)
                )
            return False
        observed_cache = True
        fingerprint = json.loads(marker.read_text(encoding="utf-8")).get("fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("prepared-data cache marker has no valid fingerprint")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            prefix=f".{destination.name}.restore-", dir=destination.parent
        ) as tmp:
            staging = Path(tmp) / "data"
            shutil.copytree(source, staging, ignore=shutil.ignore_patterns(CACHE_MARKER))
            # Checks metadata, tokenizer, split sizes and SHA-256 hashes locally.
            PreparedTokenData(staging, expected_fingerprint=fingerprint)
            # Keep the old directory outside the temporary workspace so even a
            # failed rollback cannot delete it during TemporaryDirectory cleanup.
            previous = destination.parent / f".{destination.name}.previous-{uuid.uuid4().hex}"
            if destination.exists():
                os.replace(destination, previous)
            try:
                os.replace(staging, destination)
            except OSError:
                if previous.exists():
                    try:
                        os.replace(previous, destination)
                    except OSError as error:
                        raise RuntimeError(
                            f"Local data publication failed; recover previous data from {previous}"
                        ) from error
                raise
            if previous.exists():
                shutil.rmtree(previous)
        return True

    return _retry_restore(
        restore, description=f"Drive data restore from {source}",
        attempts=attempts, retry_delay=retry_delay,
    )


def restore_checkpoint_backup(
    backup_dir: str | Path,
    run_dir: str | Path,
    *,
    required: bool = False,
    storage_root: str | Path | None = None,
    attempts: int = 3,
    retry_delay: float = 5.0,
) -> Path | None:
    """Copy a resumable Drive checkpoint and its sidecars to local storage."""

    source = Path(backup_dir)
    latest = source / "latest.pt"
    destination = Path(run_dir)
    observed_checkpoint = False

    def restore() -> Path | None:
        nonlocal observed_checkpoint
        _check_storage_root(storage_root)
        if not _is_file(latest):
            if required or observed_checkpoint:
                raise FileNotFoundError(
                    errno.ENOENT, "Required resume checkpoint is unavailable", str(latest)
                )
            return None
        observed_checkpoint = True
        destination.mkdir(parents=True, exist_ok=True)
        # Stage every available file before publishing resume.pt last.
        with TemporaryDirectory(prefix=".checkpoint-restore-", dir=destination.parent) as tmp:
            staging = Path(tmp)
            shutil.copyfile(latest, staging / "resume.pt")
            for name in ("best.pt", "metrics.jsonl"):
                candidate = source / name
                if _is_file(candidate):
                    shutil.copyfile(candidate, staging / name)
            for name in ("best.pt", "metrics.jsonl", "resume.pt"):
                staged = staging / name
                if staged.is_file():
                    os.replace(staged, destination / name)
        return destination / "resume.pt"

    return _retry_restore(
        restore, description=f"Drive checkpoint restore from {latest}",
        attempts=attempts, retry_delay=retry_delay,
    )


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
