"""CPU proofs for the experimental TPU path; real TPU results are separate."""

from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from kiwilm.colab_artifacts import create_colab_artifacts
from kiwilm.config import KiwiLM2Config
from kiwilm.data import PreparedTokenData, prepare_from_stories
from kiwilm.models import KiwiLM2LM
from kiwilm.optim import MuonWithAuxAdamW, split_muon_parameters
from kiwilm.tpu_smoke import Runtime, TensorMuon, probe
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


def test_bootstrap_uses_standalone_python_and_bounded_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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
    process.wait.assert_called_once_with(timeout=900)


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
