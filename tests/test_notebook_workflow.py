"""M2 tests use only tiny CPU fixtures and local fake Drive storage."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from kiwilm.data import PreparedTokenData
from kiwilm.notebook_setup import environment_plan, run_notebook_action
from kiwilm.notebook_workflow import (
    NotebookConfig,
    _validate_payload,
    contract_identity,
    evaluate_probe,
    experiment_contract,
    preflight,
    prepare_probe_data,
    train_probe,
)
from kiwilm.safetensors_io import load_safetensors_model
from kiwilm.tpu_checkpoint import DriveCheckpointStore


@pytest.fixture
def probe(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    data = tmp_path / "data"
    prepare_probe_data(data)
    yield data, NotebookConfig(max_tokens=97, warmup_tokens=32, grad_accum_steps=2)
    torch.set_num_threads(previous)


def _train(config, data, root, *, mode="fresh", stop=None):
    return train_probe(
        config,
        data_dir=data,
        run_dir=root / "run",
        backup_root=root / "backup",
        mode=mode,
        start_training=True,
        stop_after_step=stop,
    )


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize(
    "field,value",
    [
        ("device", "auto"),
        ("precision", "bf16"),
        ("engine", "kiwilm3"),
        ("max_tokens", 1_000_001),
        ("warmup_tokens", 257),
        ("checkpoint_interval", 0),
        ("context_length", 65),
        ("batch_size", True),
        ("noise_seed", -1),
        ("seed", 2**32),
        ("grad_accum_steps", 33),
        ("lr", float("nan")),
        ("experiment", "../unsafe"),
    ],
)
def test_config_rejects_unsafe_values(field, value):
    with pytest.raises(ValueError):
        NotebookConfig(**{field: value})


def test_safe_default_no_training_or_backup_writes(probe, tmp_path):
    data, config = probe
    with pytest.raises(ValueError, match="disabled"):
        train_probe(
            config,
            data_dir=data,
            run_dir=tmp_path / "run",
            backup_root=tmp_path / "backup",
            mode="fresh",
        )
    assert not (tmp_path / "backup").exists()
    report = preflight(config, data)
    assert not report["training_started"]
    assert report["engine"] == config.engine
    assert not (tmp_path / "run").exists()


def test_contract_portable_paths_and_strict_controls(probe, tmp_path):
    data, config = probe
    first = experiment_contract(config, PreparedTokenData(data))
    assert contract_identity(first) == contract_identity(json.loads(json.dumps(first)))
    for changed in (
        replace(config, noise_seed=3),
        replace(config, seed=43),
        replace(config, max_tokens=98),
        replace(config, checkpoint_interval=1),
    ):
        changed_contract = experiment_contract(changed, PreparedTokenData(data))
        assert contract_identity(first) != contract_identity(changed_contract)
    first["implementation_sha256"] = "changed"
    assert contract_identity(first) != preflight(config, data)["identity"]
    assert str(tmp_path) not in json.dumps(first)
    assert NotebookConfig(**asdict(config)) == config


def test_fresh_process_continuation_exact(probe, tmp_path):
    """Separate Python processes emulate VM loss; compare all numerical recovery state."""
    data, config = probe
    config = replace(config, max_tokens=257, grad_accum_steps=1)
    worker = """
import json, sys, torch
from kiwilm.notebook_worker import execute
torch.set_num_threads(1)
execute('train', json.loads(sys.argv[1]))
"""
    common = {"config": asdict(config), "data_dir": str(data), "start_training": True}
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}

    def run(directory, backup, mode, stop=None):
        request = {
            **common,
            "run_dir": str(directory),
            "backup_root": str(backup),
            "mode": mode,
            "stop_after_step": stop,
        }
        subprocess.run(
            [sys.executable, "-c", worker, json.dumps(request)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )

    run(tmp_path / "reference", tmp_path / "ref-backup", "fresh")
    run(tmp_path / "first-vm", tmp_path / "backup", "fresh", 4)
    run(tmp_path / "second-vm", tmp_path / "backup", "resume")
    reference = torch.load(tmp_path / "reference/latest.pt", weights_only=True)
    resumed = torch.load(tmp_path / "second-vm/latest.pt", weights_only=True)
    _equal(reference, resumed)
    assert resumed["training_state"]["tokens_seen"] == 257
    assert resumed["step"] == 9
    assert resumed["scheduler_state_dict"] == {"tokens_seen": 257}
    assert set(resumed["batcher_state"]["generators"]) == {"validation", "noise"}

    def metrics(path):
        return [
            {
                key: value
                for key, value in json.loads(line).items()
                if key not in {"seconds", "tokens_per_second"}
            }
            for line in path.read_text().splitlines()
        ]

    assert metrics(tmp_path / "reference/metrics.jsonl") == metrics(
        tmp_path / "second-vm/metrics.jsonl"
    )


def test_resume_no_fresh_fallback_and_no_overwrite(probe, tmp_path):
    data, config = probe
    root = tmp_path / "missing"
    with pytest.raises(ValueError, match="No valid committed"):
        _train(config, data, root, mode="resume")
    assert not (root / "run/latest.pt").exists()
    done = _train(config, data, tmp_path / "valid", stop=1)
    namespace = Path(done["backup"])
    pointer = (namespace / "latest.json").read_bytes()
    with pytest.raises(FileExistsError, match="empty"):
        _train(config, data, tmp_path / "valid")
    with pytest.raises(FileExistsError, match="resume instead"):
        train_probe(
            config,
            data_dir=data,
            run_dir=tmp_path / "new-local",
            backup_root=tmp_path / "valid/backup",
            mode="fresh",
            start_training=True,
        )
    with pytest.raises(ValueError, match="No valid committed"):
        train_probe(
            replace(config, noise_seed=4),
            data_dir=data,
            run_dir=tmp_path / "wrong-seed",
            backup_root=tmp_path / "valid/backup",
            mode="resume",
            start_training=True,
        )
    assert pointer == (namespace / "latest.json").read_bytes()


@pytest.mark.parametrize(
    "missing",
    [
        "optimizer_state_dict",
        "scheduler_state_dict",
        "rng_state",
        "noise",
        "numpy_rng",
        "scaler",
        "owner",
    ],
)
def test_incomplete_state_rejected(probe, tmp_path, missing):
    data, config = probe
    result = _train(config, data, tmp_path / "fixture", stop=1)
    payload = torch.load(result["checkpoint"], weights_only=True)
    contract = payload["training_state"]["contract"]
    if missing == "noise":
        del payload["batcher_state"]["generators"]["noise"]
    elif missing == "owner":
        del payload["batcher_state"]["owner"]
    elif missing in {"numpy_rng", "scaler"}:
        del payload["training_state"][missing]
    else:
        del payload[missing]
    with pytest.raises(ValueError, match="missing"):
        _validate_payload(payload, contract, contract_identity(contract))


def test_generation_health_export_and_retention(probe, tmp_path):
    data, config = probe
    config = replace(config, max_tokens=192, grad_accum_steps=1, checkpoint_interval=1)
    result = _train(config, data, tmp_path / "fixture")
    namespace = Path(result["backup"])
    assert len(list(namespace.glob("step-*"))) == 2
    assert json.loads((namespace / "latest.json").read_text())["step"] == 6
    assert json.loads((namespace / "previous.json").read_text())["step"] == 5
    report = evaluate_probe(
        config, data_dir=data, checkpoint=Path(result["checkpoint"]), export_dir=tmp_path / "bundle"
    )
    assert report["weights_finite"] and report["cached_generation_parity"]["passed"]
    assert len(report["health"]["blocks"]) == 10
    assert (tmp_path / "bundle/model.safetensors").exists()
    bundle, loaded_config = load_safetensors_model(
        tmp_path / "bundle",
        data_fingerprint=PreparedTokenData(data).fingerprint,
        device=torch.device("cpu"),
    )
    assert loaded_config == config.model_config(PreparedTokenData(data).tokenizer.vocab_size)
    original = torch.load(result["checkpoint"], weights_only=True)["model_state_dict"]
    for name, value in bundle.state_dict().items():
        expected = (
            original[name].to(torch.bfloat16).float()
            if value.is_floating_point()
            else original[name]
        )
        assert torch.equal(value, expected)
    with pytest.raises(FileExistsError):
        evaluate_probe(
            config,
            data_dir=data,
            checkpoint=Path(result["checkpoint"]),
            export_dir=tmp_path / "bundle",
        )


def test_corrupt_latest_retains_previous_but_refuses_rollback(probe, tmp_path):
    data, config = probe
    result = _train(config, data, tmp_path / "fixture")
    namespace = Path(result["backup"])
    latest = json.loads((namespace / "latest.json").read_text())
    (namespace / latest["generation"] / "latest.pt").write_bytes(b"corrupt fixture")
    store = DriveCheckpointStore(namespace, identity=result["identity"])
    assert store.restore(tmp_path / "fallback")["step"] == 0
    with pytest.raises(ValueError, match="newer progress"):
        train_probe(
            config,
            data_dir=data,
            run_dir=tmp_path / "refuse-rollback",
            backup_root=tmp_path / "fixture/backup",
            mode="resume",
            start_training=True,
        )


def test_dead_drive_mount_fails_before_training(probe, tmp_path, monkeypatch):
    data, config = probe
    monkeypatch.setattr("kiwilm.tpu_checkpoint.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="Drive checkpoint operation failed"):
        train_probe(
            config,
            data_dir=data,
            run_dir=tmp_path / "run",
            backup_root=tmp_path / "not-a-mount/MyDrive/KiwiLM3",
            storage_root=tmp_path / "not-a-mount",
            mode="fresh",
            start_training=True,
        )
    assert not (tmp_path / "run/latest.pt").exists()


def test_setup_pins_and_worker_control(tmp_path):
    wheel = tmp_path / "kiwilm.whl"
    wheel.write_bytes(b"wheel fixture")
    plan = environment_plan(wheel, tmp_path / "env", "xla")
    assert plan["torch"] == plan["torch_xla"] == "2.9.0"
    assert plan["python"] == "3.12"
    with pytest.raises(ValueError):
        run_notebook_action(Path(sys.executable), "allocate-tpu", {})


def test_notebook_worker_subprocess_actions(probe, tmp_path, capsys):
    data, config = probe
    request = {
        "config": asdict(config),
        "data_dir": str(data),
        "run_dir": str(tmp_path / "worker-run"),
        "backup_root": str(tmp_path / "worker-backup"),
        "mode": "fresh",
    }
    python = Path(sys.executable)
    checked = run_notebook_action(python, "preflight", request)
    assert checked["contract"]["runtime_versions"]["cpu_threads"] == 1
    with pytest.raises(RuntimeError, match="worker failed"):
        run_notebook_action(python, "train", request)
    assert not (tmp_path / "worker-run").exists()
    trained = run_notebook_action(python, "train", {**request, "start_training": True})
    assert trained["tokens_seen"] == config.max_tokens
    report = run_notebook_action(
        python, "evaluate", {**request, "checkpoint": trained["checkpoint"]}
    )
    assert report["cached_generation_parity"]["passed"]
    assert (
        '"notebook_result"' not in capsys.readouterr().out
    )  # large reports are returned, not dumped


def test_metrics_corruption_and_inconsistent_progress_refused(probe, tmp_path):
    data, config = probe
    result = _train(config, data, tmp_path / "fixture", stop=1)
    run = Path(result["checkpoint"]).parent
    checkpoint = torch.load(run / "latest.pt", weights_only=True)
    contract = checkpoint["training_state"]["contract"]
    checkpoint["scheduler_state_dict"]["tokens_seen"] = 0
    with pytest.raises(ValueError, match="inconsistent"):
        _validate_payload(checkpoint, contract, result["identity"])
    # Publish integrity-valid files with a logically wrong log to test adapter-level refusal.
    (run / "metrics.jsonl").write_text('{"event":"train","step":2}\n')
    store = DriveCheckpointStore(Path(result["backup"]), identity=result["identity"])
    store.publish(run, step=1, tokens=64)
    with pytest.raises(ValueError, match="metrics do not match"):
        train_probe(
            config,
            data_dir=data,
            run_dir=tmp_path / "bad-log",
            backup_root=tmp_path / "fixture/backup",
            mode="resume",
            start_training=True,
        )


@pytest.mark.parametrize("device,precision", [("cuda", "fp16"), ("xla", "bf16")])
def test_missing_accelerator_rng_refused_without_hardware(probe, tmp_path, device, precision):
    data, config = probe
    result = _train(config, data, tmp_path / "fixture", stop=1)
    payload = torch.load(result["checkpoint"], weights_only=True)
    contract = payload["training_state"]["contract"]
    # Simulate an integrity-valid accelerator payload with its backend state omitted.
    contract["config"]["device"] = device
    contract["config"]["precision"] = precision
    contract["train_config"]["precision"] = precision
    payload["train_config"]["precision"] = precision
    payload["training_state"]["identity"] = contract_identity(contract)
    payload["rng_state"].pop("cuda", None)
    with pytest.raises(ValueError, match="backend/NumPy/scaler"):
        _validate_payload(payload, contract, contract_identity(contract))
