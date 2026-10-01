"""Verified two-generation Drive persistence for single-device TPU checkpoints.

Training waits for publication. A manifest pointer is committed only after all
copied bytes have been read back and checked; a failed copy keeps the old pointer.
This protects against VM loss, not against Drive outages or cloud data loss.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypeVar

import torch

from kiwilm.colab_artifacts import file_sha256
from kiwilm.colab_drive import atomic_copy_file

_T = TypeVar("_T")
_GENERATION = re.compile(r"step-\d{8}-[0-9a-f]{32}")
_TRANSIENT = {errno.ENOENT, errno.ENOTCONN, errno.EIO, errno.ESTALE, errno.ETIMEDOUT,
              errno.ECONNRESET, errno.EAGAIN, errno.ENETUNREACH, errno.EHOSTUNREACH}


def contract_digest(contract: dict[str, Any], fingerprint: str, model_config: dict) -> str:
    values = {"contract": contract, "data_fingerprint": fingerprint, "model_config": model_config}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def _retry(operation: Callable[[], _T], *, attempts: int, delay: float) -> _T:
    if attempts < 1 or not 0 <= delay <= 60:
        raise ValueError("invalid checkpoint retry policy")
    for attempt in range(attempts):
        try:
            return operation()
        except OSError as error:
            if error.errno not in _TRANSIENT or attempt + 1 == attempts:
                raise RuntimeError(
                    "Drive checkpoint operation failed; training must stop. The previous "
                    "committed Drive checkpoint and the local checkpoint remain recoverable."
                ) from error
            print(json.dumps({"event": "drive_checkpoint_retry", "attempt": attempt + 1,
                              "error": str(error)}), flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _check_mount(root: Path | None, backup: Path) -> None:
    if root is not None and not root.is_mount():
        raise OSError(errno.ENOTCONN, "Google Drive mount is unavailable", str(root))
    if root is not None and (
        not backup.is_absolute() or ".." in backup.parts
        or not backup.is_relative_to(root) or not backup.resolve().is_relative_to(root.resolve())
        or len(backup.relative_to(root).parts) < 4 or backup.parent.name != "checkpoints"
    ):
        raise ValueError("Drive backup must be a specific checkpoint namespace inside the mount")


def _atomic_json(path: Path, values: dict) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(values, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _pointer(root: Path, name: str) -> dict | None:
    path = root / name
    try:
        values = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    if not isinstance(values, dict) or values.get("schema_version") != 1 or not (
        isinstance(values.get("generation"), str)
        and _GENERATION.fullmatch(values["generation"])
        and isinstance(values.get("manifest_sha256"), str)
        and isinstance(values.get("step"), int) and values["step"] >= 0
    ):
        raise ValueError("invalid Drive checkpoint pointer")
    return values


def _manifest(root: Path, pointer: dict, identity: str) -> dict:
    generation = root / pointer["generation"]
    if generation.is_symlink():
        raise ValueError("Drive checkpoint generation cannot be a symlink")
    path = generation / "manifest.json"
    if file_sha256(path) != pointer["manifest_sha256"]:
        raise ValueError("Drive checkpoint manifest checksum mismatch")
    manifest = json.loads(path.read_text())
    if manifest.get("contract_digest") != identity or manifest.get("step") != pointer["step"]:
        raise ValueError("Drive checkpoint provenance differs from the requested run")
    if not isinstance(manifest.get("files"), dict) or "latest.pt" not in manifest["files"]:
        raise ValueError("Drive checkpoint has no committed latest.pt")
    for name in manifest["files"]:
        if name not in {"latest.pt", "metrics.jsonl", "job.json"}:
            raise ValueError("unsafe Drive checkpoint filename")
    return manifest


class DriveCheckpointStore:
    """Synchronous, fail-closed backup; keep latest and previous committed steps."""

    def __init__(
        self, root: Path, *, identity: str, storage_root: Path | None = None,
        attempts: int = 3, retry_delay: float = 5,
    ) -> None:
        self.root, self.identity, self.storage_root = root, identity, storage_root
        self.attempts, self.retry_delay = attempts, retry_delay

    def check(self) -> None:
        """Check the mount, locked namespace, and write access before training."""
        def operation() -> None:
            _check_mount(self.storage_root, self.root)
            self.root.mkdir(parents=True, exist_ok=True)
            lock = self.root / "identity.json"
            try:
                with lock.open("x") as stream:
                    json.dump({"contract_digest": self.identity}, stream)
            except FileExistsError as error:
                if json.loads(lock.read_text()).get("contract_digest") != self.identity:
                    raise ValueError(
                        "Drive checkpoint namespace belongs to a different run"
                    ) from error
            probe = self.root / f".write-check-{uuid.uuid4().hex}"
            try:
                probe.write_bytes(b"kiwilm-checkpoint-write-check")
                if probe.read_bytes() != b"kiwilm-checkpoint-write-check":
                    raise ValueError("Drive checkpoint write/read check failed")
            finally:
                probe.unlink(missing_ok=True)
        _retry(operation, attempts=self.attempts, delay=self.retry_delay)

    def publish(self, run: Path, *, step: int, tokens: int, job: Path | None = None) -> dict:
        def operation() -> dict:
            self.check()
            old_latest = _pointer(self.root, "latest.json")
            old_previous = _pointer(self.root, "previous.json")
            if old_latest is not None and step < old_latest["step"]:
                raise ValueError("refusing to overwrite newer Drive progress with an older step")
            generation = f"step-{step:08d}-{uuid.uuid4().hex}"
            directory = self.root / generation
            directory.mkdir()
            files = {}
            candidates = {"latest.pt": run / "latest.pt", "metrics.jsonl": run / "metrics.jsonl"}
            if job is not None:
                candidates["job.json"] = job
            for name, source in candidates.items():
                if name == "latest.pt" or source.is_file():
                    expected = {"bytes": source.stat().st_size, "sha256": file_sha256(source)}
                    target = atomic_copy_file(source, directory / name)
                    if target.stat().st_size != expected["bytes"] or (
                        file_sha256(target) != expected["sha256"]
                    ):
                        raise ValueError("Drive checkpoint read-back checksum mismatch")
                    files[name] = expected
            manifest = {"schema_version": 1, "step": step, "tokens_seen": tokens,
                        "contract_digest": self.identity, "files": files}
            _atomic_json(directory / "manifest.json", manifest)
            pointer = {"schema_version": 1, "generation": generation, "step": step,
                       "manifest_sha256": file_sha256(directory / "manifest.json")}
            _manifest(self.root, pointer, self.identity)
            # Re-saving a boundary must not evict the previous distinct step.
            previous = old_latest if old_latest and old_latest["step"] < step else old_previous
            if previous:
                _atomic_json(self.root / "previous.json", previous)
            _atomic_json(self.root / "latest.json", pointer)  # Commit last.
            if _pointer(self.root, "latest.json") != pointer:
                raise ValueError("Drive checkpoint pointer read-back failed")
            # Prune only known superseded committed generations, never a broad glob.
            retained = {generation, previous["generation"] if previous else None}
            for old in (old_latest, old_previous):
                if old and old["generation"] not in retained:
                    obsolete = self.root / old["generation"]
                    try:
                        _manifest(self.root, old, self.identity)
                        shutil.rmtree(obsolete)
                    except (OSError, ValueError) as error:
                        print(json.dumps({"event": "drive_checkpoint_prune_warning",
                                          "error": str(error)}), flush=True)
            return {"event": "drive_checkpoint_committed", "step": step, "tokens_seen": tokens,
                    "backup_dir": str(self.root), "generation": generation,
                    "checkpoint_sha256": files["latest.pt"]["sha256"], "verified": True}
        return _retry(operation, attempts=self.attempts, delay=self.retry_delay)

    def validate_progress(self, completed: int) -> None:
        """Do not train behind an already committed recovery point."""
        def operation() -> None:
            _check_mount(self.storage_root, self.root)
            latest = _pointer(self.root, "latest.json")
            if latest:
                _manifest(self.root, latest, self.identity)
                if latest["step"] > completed:
                    raise ValueError(
                        "Drive has newer progress; require Drive resume or a new namespace"
                    )
        _retry(operation, attempts=self.attempts, delay=self.retry_delay)

    def restore(self, destination: Path, *, expected_step: int | None = None) -> dict:
        """Restore a verified committed generation, never silently start fresh."""
        def operation() -> dict:
            _check_mount(self.storage_root, self.root)
            pointers = [_pointer(self.root, name) for name in ("latest.json", "previous.json")]
            errors = []
            for pointer in pointers:
                if pointer is None or (
                    expected_step is not None and pointer["step"] != expected_step
                ):
                    continue
                try:
                    manifest = _manifest(self.root, pointer, self.identity)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with TemporaryDirectory(prefix=".tpu-resume-", dir=destination.parent) as tmp:
                        staged = Path(tmp)
                        for name, details in manifest["files"].items():
                            source = self.root / pointer["generation"] / name
                            if source.is_symlink():
                                raise ValueError("Drive checkpoint files cannot be symlinks")
                            shutil.copyfile(source, staged / name)
                            if (staged / name).stat().st_size != details["bytes"] or (
                                file_sha256(staged / name) != details["sha256"]
                            ):
                                raise ValueError("Drive checkpoint restore checksum mismatch")
                        destination.mkdir(parents=True, exist_ok=True)
                        for name in manifest["files"]:
                            if name != "latest.pt":
                                os.replace(staged / name, destination / name)
                        os.replace(staged / "latest.pt", destination / "latest.pt")
                    return {"step": pointer["step"], "generation": pointer["generation"],
                            "backup_dir": str(self.root), "contract_digest": self.identity,
                            "checkpoint_sha256": manifest["files"]["latest.pt"]["sha256"]}
                except ValueError as error:
                    errors.append(str(error))
            raise ValueError(f"No valid committed Drive checkpoint for requested step: {errors}")
        return _retry(operation, attempts=self.attempts, delay=self.retry_delay)


def compare_continuation(reference: Path, resumed: Path) -> dict:
    """Exact state comparison; timing, diagnostics, and VM IDs are not training state."""
    left = torch.load(reference, map_location="cpu", weights_only=True)
    right = torch.load(resumed, map_location="cpu", weights_only=True)
    mismatches = []

    def compare(a: Any, b: Any, path: str) -> None:
        if isinstance(a, torch.Tensor):
            equal = isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
        elif isinstance(a, dict):
            equal = isinstance(b, dict) and a.keys() == b.keys()
            if equal:
                for key in a:
                    compare(a[key], b[key], f"{path}.{key}")
                return
        elif isinstance(a, (list, tuple)):
            equal = isinstance(b, (list, tuple)) and len(a) == len(b)
            if equal:
                for index, (x, y) in enumerate(zip(a, b, strict=True)):
                    compare(x, y, f"{path}[{index}]")
                return
        else:
            equal = a == b
        if not equal:
            mismatches.append(path)

    for key in ("step", "data_fingerprint", "model_config", "train_config", "model_state_dict",
                "optimizer_state_dict", "batcher_state", "rng_state"):
        compare(left.get(key), right.get(key), key)
    for key in ("tokens_seen", "smoke_contract", "scaler_state", "xla_rng_state"):
        compare(left["training_state"].get(key), right["training_state"].get(key), key)
    source_vm = left["training_state"].get("vm_id")
    resumed_vm = right["training_state"].get("vm_id")
    distinct_vms = bool(source_vm and resumed_vm and source_vm != resumed_vm)
    return {"passed": not mismatches and distinct_vms, "state_equal": not mismatches,
            "distinct_vm_ids": distinct_vms, "step": right["step"],
            "tokens_seen": right["training_state"]["tokens_seen"],
            "mismatch_count": len(mismatches), "mismatches": mismatches[:20]}
