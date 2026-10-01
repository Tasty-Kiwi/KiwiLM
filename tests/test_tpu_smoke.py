"""CPU proofs for the experimental TPU path; real TPU results are separate."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from kiwilm.checkpoint import save_checkpoint
from kiwilm.colab_artifacts import create_colab_artifacts
from kiwilm.config import KiwiLM2Config
from kiwilm.data import PreparedTokenData, prepare_from_stories
from kiwilm.models import KiwiLM2LM
from kiwilm.optim import MuonWithAuxAdamW, split_muon_parameters
from kiwilm.tpu_setup import job_digest
from kiwilm.tpu_smoke import (
    Runtime,
    TensorMuon,
    portable_loss_report,
    prepare_tied_model,
    probe,
    validate_tied_checkpoint,
)
from kiwilm.training import TrainConfig, _validate_resume_settings


def tiny_config(vocab_size: int = 300) -> KiwiLM2Config:
    return KiwiLM2Config(
        vocab_size=vocab_size, d_model=8, context_length=8, num_query_heads=2, num_kv_heads=1,
        swiglu_dim=12, bigram_buckets=17, trigram_buckets=19,
        conv_kernel_sizes=(3, 5, 3, 5, 3, 5),
    )


def test_tensor_muon_matches_reference_updates() -> None:
    torch.manual_seed(42)
    reference = KiwiLM2LM(tiny_config())
    candidate = copy.deepcopy(reference)
    optimizers = []
    for model, kind in [(reference, MuonWithAuxAdamW), (candidate, TensorMuon)]:
        muon, auxiliary = split_muon_parameters(model)
        optimizers.append(kind(
            muon, auxiliary, muon_lr=0.01, adamw_lr=3e-4, weight_decay=0.1, beta2=0.95,
        ))
    for index in range(5):
        lr = 3e-4 * (index + 1) / 5
        for group in optimizers[0].param_groups:
            group["lr"] = lr * group["lr_multiplier"]
        optimizers[1].set_learning_rate(lr)
        for left, right in zip(reference.parameters(), candidate.parameters(), strict=True):
            left.grad = torch.randn_like(left)
            right.grad = left.grad.clone()
        for optimizer in optimizers:
            optimizer.step()
        for left, right in zip(reference.parameters(), candidate.parameters(), strict=True):
            torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-7)


def test_device_transfer_retied_before_optimizer(monkeypatch: pytest.MonkeyPatch) -> None:
    original_to = KiwiLM2LM.to

    def untie_on_transfer(model, *args, **kwargs):
        original_to(model, *args, **kwargs)
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        return model

    expected = sum(parameter.numel() for parameter in KiwiLM2LM(tiny_config()).parameters())
    monkeypatch.setattr(KiwiLM2LM, "to", untie_on_transfer)
    model = prepare_tied_model(tiny_config(), torch.device("cpu"))
    assert model.lm_head.weight is model.token_embedding.weight
    assert sum(parameter.numel() for parameter in model.parameters()) == expected
    muon, auxiliary = split_muon_parameters(model)
    assert all(parameter is not model.lm_head.weight for parameter in muon)
    assert any(parameter is model.lm_head.weight for parameter in auxiliary)


def test_unequal_tied_checkpoint_rejected(tmp_path: Path) -> None:
    model = KiwiLM2LM(tiny_config())
    state = model.state_dict()
    state["lm_head.weight"] = state["lm_head.weight"].clone() + 1
    path = tmp_path / "untied.pt"
    torch.save({"model_config": model.config.to_dict(), "model_state_dict": state}, path)
    with pytest.raises(ValueError, match="unequal tied"):
        validate_tied_checkpoint(path)


def test_portability_gate_checks_logits_not_just_loss(tmp_path: Path) -> None:
    prepare_from_stories(
        tmp_path / "data", ["A training story. " * 8], ["A validation story. " * 8],
        vocab_size=300, min_frequency=1,
    )
    data = PreparedTokenData(tmp_path / "data")
    model = prepare_tied_model(tiny_config(data.tokenizer.vocab_size), torch.device("cpu"))
    path = tmp_path / "latest.pt"
    save_checkpoint(path, model=model, step=0, data_fingerprint=data.fingerprint)
    options = dict(batch_size=1, batches=1)
    runtime = Runtime("cpu", "fp32")
    assert portable_loss_report(model, data, path, runtime, **options)["passed"]
    original_forward = model.lm_head.forward
    # A uniform offset leaves cross-entropy unchanged but must fail logit parity.
    model.lm_head.forward = lambda values: original_forward(values) + 5
    with pytest.raises(RuntimeError, match="portability failed"):
        portable_loss_report(model, data, path, runtime, **options)


def test_launcher_rejects_unsupported_tpu_before_allocation() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/run_colab_kiwilm2_tpu_smoke.sh"
    result = subprocess.run(
        ["bash", str(script)], env={**os.environ, "COLAB_TPU": "unsupported"},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert "COLAB_TPU must be v5e1 or v6e1" in result.stderr


def test_probe_checkpoint_resume_and_timing(tmp_path: Path) -> None:
    prepare_from_stories(
        tmp_path / "data", ["A tiny training story. " * 8], ["A validation story. " * 8],
        vocab_size=300, min_frequency=1,
    )
    data = PreparedTokenData(tmp_path / "data")
    config = tiny_config(data.tokenizer.vocab_size)
    options = dict(config=config, runtime=Runtime("cpu", "fp32"), batch_size=1,
                   accumulation=2, eval_batches=1, warmup_steps=1)
    first = probe(data, tmp_path / "split", steps=2, **options)
    assert first["tokens_seen"] == 32
    assert first["steady_tokens_per_second"] > 0
    probe(data, tmp_path / "split", steps=2, resume=tmp_path / "split" / "latest.pt", **options)
    probe(data, tmp_path / "whole", steps=4, **options)
    split = torch.load(tmp_path / "split" / "latest.pt", weights_only=True)
    whole = torch.load(tmp_path / "whole" / "latest.pt", weights_only=True)
    assert split["step"] == whole["step"] == 4
    with pytest.raises(ValueError, match="not the production trainer"):
        _validate_resume_settings(
            tmp_path / "split" / "latest.pt", TrainConfig(**split["train_config"])
        )
    for key in whole["model_state_dict"]:
        torch.testing.assert_close(split["model_state_dict"][key], whole["model_state_dict"][key])
    with pytest.raises(ValueError, match="already has a checkpoint"):
        probe(data, tmp_path / "split", steps=2, **options)
    with pytest.raises(ValueError, match="matching hardware smoke"):
        probe(data, tmp_path / "other", steps=2, resume=tmp_path / "split" / "latest.pt",
              **{**options, "eval_batches": 2})


def test_tpu_rejects_fp16_without_importing_xla() -> None:
    with pytest.raises(ValueError, match="requires bf16"):
        Runtime("xla", "fp16")


def test_xla_requires_one_real_tpu_and_synchronizes(monkeypatch: pytest.MonkeyPatch) -> None:
    names = ["torch_xla", "torch_xla.core", "torch_xla.core.xla_model",
             "torch_xla.debug", "torch_xla.debug.metrics", "torch_xla.runtime"]
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, attribute = name.rsplit(".", 1)
            setattr(modules[parent], attribute, module)
    calls = []
    modules["torch_xla"].device = lambda: torch.device("xla")
    modules["torch_xla"].sync = lambda **kwargs: calls.append(kwargs)
    xr = modules["torch_xla.runtime"]
    xr.device_type = lambda: "TPU"
    xr.global_runtime_device_count = lambda: 1
    runtime = Runtime("xla", "bf16")
    runtime.sync()
    assert calls == [{"wait": True}]
    xr.global_runtime_device_count = lambda: 2
    with pytest.raises(RuntimeError, match="exactly one"):
        Runtime("xla", "bf16")
    xr.device_type = lambda: "CPU"
    with pytest.raises(RuntimeError, match="real TPU"):
        Runtime("xla", "bf16")


@pytest.mark.parametrize("resume", [False, True])
def test_bootstrap_uses_standalone_python_and_bounded_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: bool,
) -> None:
    path = Path(__file__).resolve().parents[1] / "scripts" / "colab_kiwilm2_tpu_smoke.py"
    spec = importlib.util.spec_from_file_location("tpu_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CONTENT", tmp_path)
    monkeypatch.setattr(module, "ENV", tmp_path / "env")
    monkeypatch.setattr(module, "PYTHON", tmp_path / "env" / "bin" / "python")
    (tmp_path / "kiwilm-0.1.0-py3-none-any.whl").write_bytes(b"test wheel")
    (tmp_path / "kiwilm-data-artifacts").mkdir()
    (tmp_path / "kiwilm-data-artifacts" / "artifact-manifest.json").write_text("{}")
    job = {"schema_version": 1, "use_drive": False, "data_fingerprint": "a" * 64,
           "tokenizer_sha256": "b" * 64}
    (tmp_path / "kiwilm-tpu-job.json").write_text(json.dumps(job))
    (tmp_path / "kiwilm-tpu-setup.json").write_text(json.dumps({
        "state": "ready", "job_digest": job_digest(job),
    }))
    monkeypatch.setenv("KIWILM2_TPU_ACTION", "train")
    if resume:
        (tmp_path / "kiwilm-tpu-resume").mkdir()
        (tmp_path / "kiwilm-tpu-resume" / "artifact-manifest.json").write_text("{}")
        (tmp_path / "kiwilm-tpu-resume" / "latest.pt").write_bytes(b"mock resume")
    run = Mock(return_value=subprocess.CompletedProcess([], 0, stdout="", stderr=""))
    process = Mock(returncode=0)
    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module.subprocess, "Popen", Mock(return_value=process))
    module.main()
    commands = [call.args[0] for call in run.call_args_list]
    assert commands[1][-4:] == ["venv", "--python", "3.12", str(tmp_path / "env")]
    assert "torch==2.9.0" in commands[2]
    assert "torch_xla[tpu]==2.9.0" in commands[3]
    assert all("ensurepip" not in command for command in commands)
    for command in commands:
        if "-c" in command:
            compile(command[command.index("-c") + 1], "<bootstrap-command>", "exec")
    child_env = module.subprocess.Popen.call_args.kwargs["env"]
    assert child_env["LD_LIBRARY_PATH"].startswith(str(tmp_path / "env" / "lib"))
    process.wait.assert_called_once_with(timeout=1)
    process.kill.assert_not_called()
    worker = module.subprocess.Popen.call_args.args[0]
    assert "--require-ready" in commands[-2]
    assert "--steps" not in worker
    assert worker[worker.index("--eval-batches") + 1] == "50"
    assert "--artifact-dir" in worker
    if resume:
        assert worker[worker.index("--resume") + 1] == str(
            tmp_path / "kiwilm-tpu-resume" / "latest.pt"
        )
    else:
        assert "--resume" not in worker


@pytest.mark.parametrize("action", [None, "preflight", "prepare"])
def test_bootstrap_setup_cannot_launch_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str | None,
) -> None:
    path = Path(__file__).resolve().parents[1] / "scripts" / "colab_kiwilm2_tpu_smoke.py"
    spec = importlib.util.spec_from_file_location("setup_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CONTENT", tmp_path)
    monkeypatch.setattr(module, "PYTHON", tmp_path / "python")
    (tmp_path / "python").touch()
    (tmp_path / "kiwilm-0.1.0-py3-none-any.whl").touch()
    (tmp_path / "kiwilm-tpu-preflight.json").write_text("{}")
    (tmp_path / "kiwilm-tpu-job.json").write_text("{}")
    if action is None:
        monkeypatch.delenv("KIWILM2_TPU_ACTION", raising=False)
    else:
        monkeypatch.setenv("KIWILM2_TPU_ACTION", action)
    commands = Mock()
    popen = Mock(side_effect=AssertionError("setup must never start a training process"))
    monkeypatch.setattr(module, "run", commands)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    module.main()
    popen.assert_not_called()
    assert commands.call_count == (1 if action == "prepare" else 0)
    if action == "prepare":
        assert "kiwilm.tpu_setup" in commands.call_args.args[0]
        assert "--require-ready" not in commands.call_args.args[0]


def test_bootstrap_changed_job_refuses_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path(__file__).resolve().parents[1] / "scripts" / "colab_kiwilm2_tpu_smoke.py"
    spec = importlib.util.spec_from_file_location("locked_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CONTENT", tmp_path)
    monkeypatch.setattr(module, "PYTHON", tmp_path / "python")
    monkeypatch.setenv("KIWILM2_TPU_ACTION", "train")
    (tmp_path / "python").touch()
    (tmp_path / "kiwilm-0.1.0-py3-none-any.whl").touch()
    (tmp_path / "kiwilm-tpu-preflight.json").write_text("{}")
    (tmp_path / "kiwilm-tpu-job.json").write_text('{"changed":true}')
    (tmp_path / "kiwilm-tpu-setup.json").write_text(json.dumps({
        "state": "ready", "job_digest": "stale",
    }))
    popen = Mock()
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="job changed"):
        module.main()
    popen.assert_not_called()


def test_bootstrap_reassembles_actual_chunks(tmp_path: Path) -> None:
    path = Path(__file__).resolve().parents[1] / "scripts" / "colab_kiwilm2_tpu_smoke.py"
    spec = importlib.util.spec_from_file_location("tpu_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "metadata.json"
    source.write_text('{"test":true}\n')
    data_dir = tmp_path / "artifacts"
    create_colab_artifacts({"metadata.json": source}, data_dir, chunk_size=16)
    module.run(module.data_restore_command(Path(sys.executable), data_dir))
    assert (data_dir / "metadata.json").read_bytes() == source.read_bytes()


def test_full_smoke_exact_tokens_and_periodic_resume(tmp_path: Path) -> None:
    prepare_from_stories(
        tmp_path / "data", ["A tiny training story. " * 8], ["A validation story. " * 8],
        vocab_size=300, min_frequency=1,
    )
    data = PreparedTokenData(tmp_path / "data")
    config = tiny_config(data.tokenizer.vocab_size)
    options = dict(config=config, runtime=Runtime("cpu", "fp32"), batch_size=1,
                   accumulation=2, eval_batches=1, warmup_steps=1,
                   max_tokens=40, checkpoint_interval=1, eval_interval=1,
                   verify_checkpoint_reload=True)
    split_dir = tmp_path / "split"
    probe(data, split_dir, steps=2, **options)
    # Simulate metrics recorded after the last saved optimizer boundary.
    with (split_dir / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": "train", "step": 99}) + "\n")
    result = probe(data, split_dir, steps=4,
                   resume=split_dir / "latest.pt", final_diagnostics=True, **options)
    probe(data, tmp_path / "whole", steps=4, **options)
    assert result["status"] == "smoke-complete"
    assert result["tokens_seen"] == 40
    assert result["step"] == 3
    assert result["session_tokens"] == 8
    assert result["steady_tokens"] == 0  # Partial last batch excluded from throughput.
    assert result["diagnostic_device"] == "cpu"
    assert result["health"]["batch_count"] == 50
    assert result["cached_generation_parity"]["passed"]
    assert result["checkpoint_reload"]["passed"]
    assert result["checkpoint_reload"]["logits_equal"]
    assert result["weight_tying"]["identity_preserved"]
    assert not result["weight_tying"]["head_in_muon"]
    assert result["first_checkpoint_portability"]["passed"]
    assert result["final_checkpoint_portability"]["passed"]
    assert result["final_checkpoint_portability"]["absolute_loss_difference"] == 0
    rows = [json.loads(line) for line in (split_dir / "metrics.jsonl").read_text().splitlines()]
    assert not any(row["step"] == 99 for row in rows)
    assert [r["tokens_seen"] for r in rows if r["event"] == "train"] == [16, 32, 40]
    split = torch.load(split_dir / "latest.pt", weights_only=True)
    whole = torch.load(tmp_path / "whole" / "latest.pt", weights_only=True)
    for key in whole["model_state_dict"]:
        torch.testing.assert_close(split["model_state_dict"][key], whole["model_state_dict"][key])
