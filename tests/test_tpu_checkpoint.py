"""Local Drive failure and fresh-process recovery proofs; never allocate a TPU."""

from __future__ import annotations

import errno
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

import kiwilm.tpu_checkpoint as checkpoints
from kiwilm.config import KiwiLM2Config
from kiwilm.data import PreparedTokenData, prepare_from_stories
from kiwilm.tpu_checkpoint import DriveCheckpointStore, compare_continuation, contract_digest
from kiwilm.tpu_smoke import Runtime, probe, probe_contract, probe_settings


@pytest.fixture
def store(tmp_path: Path) -> tuple[DriveCheckpointStore, Path]:
    run = tmp_path / "run"
    run.mkdir()
    (run / "latest.pt").write_bytes(b"checkpoint-20")
    (run / "metrics.jsonl").write_text('{"step":20}\n')
    return DriveCheckpointStore(tmp_path / "drive" / "test", identity="a" * 64,
                                attempts=3, retry_delay=0), run


def test_publication_restore_and_two_distinct_generations(store, tmp_path: Path) -> None:
    backup, run = store
    first = backup.publish(run, step=20, tokens=320)
    assert first["verified"]
    (run / "latest.pt").write_bytes(b"checkpoint-40")
    backup.publish(run, step=40, tokens=640)
    backup.publish(run, step=40, tokens=640)  # Final save retains step 20, not another step 40.
    assert len(list(backup.root.glob("step-*"))) == 2
    previous = backup.restore(tmp_path / "resume", expected_step=20)
    assert previous["step"] == 20
    assert (tmp_path / "resume" / "latest.pt").read_bytes() == b"checkpoint-20"
    latest = backup.restore(tmp_path / "latest")
    assert latest["step"] == 40
    assert (tmp_path / "latest" / "latest.pt").read_bytes() == b"checkpoint-40"


def test_failed_copy_preserves_last_committed_checkpoint(
    store, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    backup, run = store
    backup.publish(run, step=20, tokens=320)
    pointer = (backup.root / "latest.json").read_bytes()
    (run / "latest.pt").write_bytes(b"checkpoint-40")
    failure = Mock(side_effect=OSError(errno.ENOTCONN, "disconnected"))
    monkeypatch.setattr(checkpoints, "atomic_copy_file", failure)
    with pytest.raises(RuntimeError, match="training must stop"):
        backup.publish(run, step=40, tokens=640)
    assert failure.call_count == 3
    assert (backup.root / "latest.json").read_bytes() == pointer
    backup.restore(tmp_path / "recovered")
    assert (tmp_path / "recovered" / "latest.pt").read_bytes() == b"checkpoint-20"


def test_copy_corruption_never_commits_pointer(store, monkeypatch: pytest.MonkeyPatch) -> None:
    backup, run = store
    original = checkpoints.atomic_copy_file

    def corrupt(source, target):
        path = original(source, target)
        path.write_bytes(b"corrupt")
        return path

    monkeypatch.setattr(checkpoints, "atomic_copy_file", corrupt)
    with pytest.raises(ValueError, match="read-back checksum"):
        backup.publish(run, step=20, tokens=320)
    assert not (backup.root / "latest.json").exists()


def test_restore_falls_back_to_verified_previous_generation(store, tmp_path: Path) -> None:
    backup, run = store
    backup.publish(run, step=20, tokens=320)
    current = backup.publish(run, step=40, tokens=640)
    (backup.root / current["generation"] / "latest.pt").write_bytes(b"corrupted latest")
    restored = backup.restore(tmp_path / "restored")
    assert restored["step"] == 20
    with pytest.raises(ValueError, match="No valid committed"):
        backup.restore(tmp_path / "must-be-40", expected_step=40)


def test_required_restore_never_starts_fresh(store, tmp_path: Path) -> None:
    backup, _ = store
    with pytest.raises(ValueError, match="No valid committed"):
        backup.restore(tmp_path / "restore")
    assert not (tmp_path / "restore").exists()


def test_namespace_and_older_progress_are_locked(store) -> None:
    backup, run = store
    backup.publish(run, step=20, tokens=320)
    with pytest.raises(ValueError, match="newer progress"):
        backup.validate_progress(0)
    with pytest.raises(ValueError, match="older step"):
        backup.publish(run, step=10, tokens=160)
    foreign = DriveCheckpointStore(backup.root, identity="b" * 64)
    with pytest.raises(ValueError, match="different run"):
        foreign.check()


def test_dead_mount_cannot_be_treated_as_a_new_directory(
    store, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backup, run = store
    backup.storage_root = backup.root.parent
    monkeypatch.setattr(Path, "is_mount", lambda _: False)
    with pytest.raises(RuntimeError, match="training must stop"):
        backup.publish(run, step=20, tokens=320)
    assert not backup.root.exists()


def test_mount_root_cannot_be_a_checkpoint_target(
    store, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backup, _ = store
    backup.storage_root = backup.root
    monkeypatch.setattr(Path, "is_mount", lambda _: True)
    with pytest.raises(ValueError, match="specific checkpoint namespace"):
        backup.check()
    assert not backup.root.exists()


def test_pointer_traversal_rejected_without_touching_outside(store, tmp_path: Path) -> None:
    backup, _ = store
    backup.check()
    outside = tmp_path / "keep"
    outside.write_text("important")
    (backup.root / "latest.json").write_text(json.dumps({
        "schema_version": 1, "generation": "../../keep", "step": 20,
        "manifest_sha256": "a" * 64,
    }))
    with pytest.raises(ValueError, match="invalid Drive checkpoint pointer"):
        backup.restore(tmp_path / "local")
    assert outside.read_text() == "important"


def test_fresh_process_continuation_matches_all_training_state(tmp_path: Path) -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        data_dir = tmp_path / "data"
        prepare_from_stories(data_dir, ["A tiny training story. " * 8],
                             ["A validation story. " * 8], vocab_size=300, min_frequency=1)
        data = PreparedTokenData(data_dir)
        config = KiwiLM2Config(
            vocab_size=data.tokenizer.vocab_size, d_model=8, context_length=8,
            num_query_heads=2, num_kv_heads=1, swiglu_dim=12,
            bigram_buckets=17, trigram_buckets=19, conv_kernel_sizes=(3, 5, 3, 5, 3, 5),
        )
        options = {"batch_size": 1, "accumulation": 2, "eval_batches": 1,
                   "eval_interval": 2, "checkpoint_interval": 2}
        settings = probe_settings(config, precision="fp32", **options)
        identity = contract_digest(probe_contract(settings, "cpu"), data.fingerprint,
                                   config.to_dict())
        store = DriveCheckpointStore(tmp_path / "drive", identity=identity, retry_delay=0)
        probe(data, tmp_path / "reference", config=config, runtime=Runtime("cpu", "fp32"),
              steps=4, warmup_steps=1, drive_store=store, vm_id="first-process", **options)
        store.restore(tmp_path / "resume-input", expected_step=2)
        # A genuinely separate interpreter: no live model/optimizer/global RNG survives.
        code = """
import json, sys, torch
from pathlib import Path
from kiwilm.config import ModelConfig
from kiwilm.data import PreparedTokenData
from kiwilm.tpu_smoke import probe, Runtime
root = Path(sys.argv[1])
torch.set_num_threads(1)
saved = torch.load(root/'resume-input/latest.pt', weights_only=True)
config = ModelConfig.from_dict(saved['model_config'])
probe(PreparedTokenData(root/'data'), root/'resumed', config=config,
      runtime=Runtime('cpu','fp32'), steps=2, warmup_steps=1, batch_size=1,
      accumulation=2, eval_batches=1, eval_interval=2, checkpoint_interval=2,
      resume=root/'resume-input/latest.pt', vm_id='second-process', require_new_vm=True)
"""
        result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True,
                                text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        comparison = compare_continuation(tmp_path / "reference/latest.pt",
                                          tmp_path / "resumed/latest.pt")
        assert comparison["passed"] and comparison["distinct_vm_ids"]
        payload = torch.load(tmp_path / "resumed/latest.pt", weights_only=True)
        payload["optimizer_state_dict"]["state"][0]["momentum_buffer"].add_(1)
        torch.save(payload, tmp_path / "changed.pt")
        assert not compare_continuation(tmp_path / "reference/latest.pt",
                                        tmp_path / "changed.pt")["passed"]
        with pytest.raises(ValueError, match="different freshly allocated VM"):
            probe(data, tmp_path / "same-vm", config=config, runtime=Runtime("cpu", "fp32"),
                  steps=2, warmup_steps=1, resume=tmp_path / "resume-input/latest.pt",
                  vm_id="first-process", require_new_vm=True, **options)
    finally:
        torch.set_num_threads(previous_threads)


def test_backup_failure_stops_before_the_next_optimizer_update(tmp_path: Path) -> None:
    prepare_from_stories(tmp_path / "data", ["A training story. " * 8],
                         ["A validation story. " * 8], vocab_size=300, min_frequency=1)
    data = PreparedTokenData(tmp_path / "data")
    config = KiwiLM2Config(
        vocab_size=data.tokenizer.vocab_size, d_model=8, context_length=8,
        num_query_heads=2, num_kv_heads=1, swiglu_dim=12,
        bigram_buckets=17, trigram_buckets=19, conv_kernel_sizes=(3, 5, 3, 5, 3, 5),
    )
    options = dict(batch_size=1, accumulation=2, eval_batches=1,
                   eval_interval=2, checkpoint_interval=2)
    settings = probe_settings(config, precision="fp32", **options)
    backup = Mock(spec=DriveCheckpointStore)
    backup.identity = contract_digest(probe_contract(settings, "cpu"), data.fingerprint,
                                     config.to_dict())
    backup.publish.side_effect = RuntimeError("Drive failed; training must stop")
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="training must stop"):
        probe(data, output, config=config, runtime=Runtime("cpu", "fp32"),
              steps=4, warmup_steps=1, drive_store=backup, **options)
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows if row["event"] == "train"] == [1, 2]
    assert torch.load(output / "latest.pt", weights_only=True)["step"] == 2
