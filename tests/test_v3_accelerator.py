"""V3 single-device math and fake-Drive tests; never rent a TPU or train a corpus."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kiwilm.data import PreparedTokenData
from kiwilm.tpu_smoke import Runtime
from kiwilm.v3.accelerator import (
    AcceleratorTrainConfig,
    AcceleratorTrainer,
    candidate_config,
    dense_reconstruction,
    learning_rate,
)
from kiwilm.v3.accelerator_checkpoint import FORMAT, V3CheckpointStore, restore_state, save_state
from kiwilm.v3.accelerator_worker import execute
from kiwilm.v3.accelerator_workflow import construct, namespace, preflight, prepare_demo, train
from kiwilm.v3.experiment_runner import checksum
from kiwilm.v3.masking import corrupt_tokens
from kiwilm.v3.objectives import masked_reconstruction_loss
from kiwilm.v3.tokenizer import MaskBPETokenizer
from kiwilm.v3.trainer import DenoisingTrainConfig, DenoisingTrainer

ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def fixed_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


@pytest.fixture
def prepared(tmp_path):
    data, tokenizer = tmp_path / "data", tmp_path / "tokenizer.json"
    prepare_demo(data, tokenizer)
    return data, tokenizer


def settings(**overrides):
    return replace(
        AcceleratorTrainConfig(
            device="cpu",
            precision="fp32",
            max_tokens=129,
            warmup_tokens=32,
            checkpoint_interval=1,
            eval_interval=2,
            eval_batches=1,
        ),
        **overrides,
    )


def arguments(prepared):
    data, tokenizer = prepared
    return dict(
        data_dir=data,
        tokenizer_path=tokenizer,
        candidate="hybrid-12",
        qualification=True,
        run_name="recovery-test",
    )


def make(prepared, **overrides):
    return construct(settings(**overrides), **arguments(prepared))


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and torch.equal(left, right)
        )
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for x, y in zip(left, right, strict=True):
            equal(x, y)
    else:
        assert left == right


@pytest.mark.parametrize(
    "options",
    [
        dict(device="auto"),
        dict(precision="fp16"),
        dict(max_tokens=0),
        dict(grad_accum_steps=True),
        dict(warmup_tokens=4096),
        dict(min_learning_rate=2),
        dict(seed=2**32),
        dict(learning_rate=float("inf")),
    ],
)
def test_config_refusals(options):
    with pytest.raises(ValueError):
        AcceleratorTrainConfig(**options)


def test_collapse_monitor_is_deterministic_read_only_and_matches_cpu_audit(prepared):
    from kiwilm.v3.diagnostics import gradient_probe
    from kiwilm.v3.training_diagnostics import CollapseMonitor

    trainer, old_job = make(prepared)
    assert "collapse_diagnostics" not in old_job
    monitor = CollapseMonitor(trainer)
    gradients = {name: torch.ones_like(p) for name, p in trainer.model.named_parameters()}
    for name, parameter in trainer.model.named_parameters():
        parameter.grad = gradients[name].clone()
    before = {name: p.clone() for name, p in trainer.model.named_parameters()}
    rng = torch.get_rng_state().clone()
    data_rng = trainer.data_generator.get_state().clone()
    noise_rng = trainer.noise_generator.get_state().clone()
    report = monitor.evaluate()
    assert report == monitor.evaluate()
    assert report["step"] == 0 and report["finite_block_statistics"]
    assert len(report["context_probes"]) == 8 and len(report["blocks"]) == 12
    assert trainer.model.training and trainer.optimizer_steps == trainer.tokens_seen == 0
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(data_rng, trainer.data_generator.get_state())
    assert torch.equal(noise_rng, trainer.noise_generator.get_state())
    for name, parameter in trainer.model.named_parameters():
        assert torch.equal(before[name], parameter)
        assert torch.equal(gradients[name], parameter.grad)
    audit = gradient_probe(trainer.model, monitor.windows[0], trainer.masking)
    for measured, reference in zip(report["blocks"], audit["blocks"], strict=True):
        for key in ("input_rms", "mixer_update_rms", "post_mixer_rms", "output_rms"):
            assert measured[key] == pytest.approx(reference[key])
    for row in report["context_probes"]:
        if row["masked_tokens"]:
            assert row["loss_minus_unigram"] == row["loss"] - row["unigram_loss"]
        if row["noise_level"] == 1:
            assert row["changed_visible_tokens"] == row["visible_content_tokens"] == 0
            assert row["argmax_change_fraction"] == row["maximum_probability_change"] == 0


def test_collapse_monitor_removes_hooks_on_failure(prepared, monkeypatch):
    from kiwilm.v3 import training_diagnostics as diagnostics

    trainer, _ = make(prepared)
    monitor = diagnostics.CollapseMonitor(trainer)

    def fail(*args, **kwargs):
        raise RuntimeError("diagnostic failure")

    monkeypatch.setattr(diagnostics, "context_probe", fail)
    with pytest.raises(RuntimeError, match="diagnostic failure"):
        monitor.evaluate()
    assert trainer.model.training
    for module in trainer.model.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks


def test_diagnostic_job_is_locked_separately_and_worker_reconstructs(prepared):
    from kiwilm.v3.training_diagnostics import POLICY

    trainer, original = make(prepared)
    _, diagnostic = construct(settings(), **arguments(prepared), collapse_diagnostics=True)
    assert original["contract"] == diagnostic["contract"] == trainer.contract
    assert diagnostic["collapse_diagnostics"] == POLICY
    assert namespace(Path("drive"), original) != namespace(Path("drive"), diagnostic)
    request = {
        "config": settings().to_dict(),
        "collapse_diagnostics": True,
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in arguments(prepared).items()
        },
    }
    assert execute("preflight", request)["job"]["collapse_diagnostics"] == POLICY
    with pytest.raises(ValueError, match="boolean"):
        construct(settings(), **arguments(prepared), collapse_diagnostics="yes")


def test_diagnostic_resume_matches_uninterrupted_training(prepared, tmp_path):
    config = settings(eval_interval=1)
    opts = dict(arguments(prepared), collapse_diagnostics=True)

    def launch(name, backup, mode, **extra):
        return train(
            config,
            **opts,
            run_dir=tmp_path / name,
            backup_root=tmp_path / backup,
            mode=mode,
            start_training=True,
            retry_delay=0,
            **extra,
        )

    launch("reference", "reference-drive", "fresh")
    launch("paused", "drive", "fresh", stop_after_step=1)
    launch("resumed", "drive", "resume")
    left, right = [
        torch.load(tmp_path / name / "latest.pt", weights_only=True)
        for name in ("reference", "resumed")
    ]
    for key in ("model", "optimizer", "generators", "rng", "schedule", "tokens_seen", "step"):
        equal(left[key], right[key])
    rows = [
        json.loads(line) for line in (tmp_path / "resumed/metrics.jsonl").read_text().splitlines()
    ]
    probes = [row for row in rows if row["event"] == "v3_collapse_diagnostics"]
    assert [row["step"] for row in probes] == [0, 1, 2, 3]
    assert rows[0]["event"] == "v3_validation" and rows[1] == probes[0]
    assert all(row["finite_block_statistics"] for row in probes)


def test_schedule_endpoints_and_real_candidate_depths():
    config = settings()
    assert learning_rate(config, 0) == 0
    assert learning_rate(config, config.warmup_tokens) == config.learning_rate
    assert learning_rate(config, config.max_tokens) == config.min_learning_rate
    assert learning_rate(config, 70) > learning_rate(config, 100)
    with pytest.raises(ValueError):
        learning_rate(config, 130)
    for name in ("hybrid-12", "attention-12", "hybrid-16", "attention-16"):
        model = candidate_config(name, 32001, qualification=False)
        assert model.d_model == 512 and model.swiglu_dim == 2048
        assert model.num_blocks == int(name.split("-")[1])
    with pytest.raises(ValueError):
        candidate_config("hadamard", 32001, qualification=True)


def test_dense_loss_matches_m4_value_gradients_and_empty_mask():
    clean = torch.tensor([[2, 4, 5, 3], [2, 7, 8, 3]])
    selected = torch.tensor([[False, True, False, False], [False, True, True, False]])
    logits = torch.randn(2, 4, 11, requires_grad=True)
    native = masked_reconstruction_loss(logits, clean, selected, mask_id=10)
    total, correct = dense_reconstruction(logits, clean, selected, 10, accuracy=True)
    assert torch.equal(total, native.loss_sum) and torch.equal(correct, native.correct)
    left = torch.autograd.grad(native.loss, logits, retain_graph=True)[0]
    right = torch.autograd.grad(total / selected.sum(), logits)[0]
    assert torch.equal(left, right)
    total, _ = dense_reconstruction(logits, clean, torch.zeros_like(selected), 10)
    assert total.item() == 0


def test_single_microbatch_equivalent_to_m4_initialization_and_update(prepared):
    data_path, token_path = prepared
    data, tokenizer = PreparedTokenData(data_path), MaskBPETokenizer.load(token_path)
    new, _ = make(
        prepared, grad_accum_steps=1, learning_rate=0.001, min_learning_rate=0.001, warmup_tokens=0
    )
    # Identical model/config/masks; no dropout to isolate objective/optimizer math.
    model_config = replace(new.model.config, dropout=0.0)
    new = AcceleratorTrainer(model_config, new.config, tokenizer, data)
    torch.manual_seed(42)
    from kiwilm.v3 import build_encoder

    old = DenoisingTrainer(
        build_encoder(model_config),
        tokenizer,
        data,
        DenoisingTrainConfig(max_steps=4, context_length=16, weight_decay=0.01),
    )
    left, right = new.train_step(), old.train_step()
    assert left["loss"] == pytest.approx(right["loss"], abs=1e-6)
    assert left["masked_tokens"] == right["masked_tokens"]
    for x, y in zip(new.model.parameters(), old.model.parameters(), strict=True):
        torch.testing.assert_close(x, y, atol=1e-8, rtol=1e-6)


def test_accumulation_normalizes_total_masks_not_average_means(prepared):
    trainer, _ = make(prepared, learning_rate=0.001, min_learning_rate=0.001, warmup_tokens=0)
    # Capture the actual CPU-corrupted microbatches for an independent sum/denominator calculation.
    reference = copy.deepcopy(trainer.model).eval()
    trainer.model.config = replace(trainer.model.config, dropout=0.0)
    for model in (trainer.model, reference):
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        for block in model.blocks:
            if hasattr(block.mixer, "dropout") and isinstance(block.mixer.dropout, float):
                block.mixer.dropout = 0.0
    data_rng = torch.Generator().manual_seed(trainer.config.data_seed)
    noise_rng = torch.Generator().manual_seed(trainer.config.noise_seed)
    losses, masks = [], 0
    for _ in range(2):
        clean, _ = trainer.data.get_batch(
            "train", batch_size=2, context_length=16, generator=data_rng
        )
        corrupt = corrupt_tokens(clean, trainer.masking, generator=noise_rng)
        logits = reference(
            corrupt.input_ids,
            attention_mask=corrupt.attention_mask,
            noise_level=corrupt.noise_level,
        )
        losses.append(
            masked_reconstruction_loss(
                logits, clean, corrupt.masked_positions, mask_id=trainer.tokenizer.mask_id
            ).loss_sum
        )
        masks += corrupt.masked_positions.sum().item()
    expected = sum(losses) / masks
    assert trainer.train_step()["loss"] == pytest.approx(expected.item(), abs=1e-6)


def test_exact_partial_budget_validation_rng_and_empty_masks(prepared):
    trainer, _ = make(prepared)
    before = [
        torch.get_rng_state(),
        trainer.data_generator.get_state(),
        trainer.noise_generator.get_state(),
    ]
    first, second = trainer.evaluate(), trainer.evaluate()
    assert first == second
    for a, b in zip(
        before,
        [
            torch.get_rng_state(),
            trainer.data_generator.get_state(),
            trainer.noise_generator.get_state(),
        ],
        strict=True,
    ):
        assert torch.equal(a, b)
    rows = []
    while trainer.tokens_seen < trainer.config.max_tokens:
        rows.append(trainer.train_step())
    assert trainer.tokens_seen == 129 and rows[-1]["input_tokens"] == 1
    assert rows[-1]["learning_rate"] == trainer.config.min_learning_rate
    assert sum(row["input_tokens"] for row in rows) == 129
    with pytest.raises(ValueError, match="exhausted"):
        trainer.train_step()
    fresh, _ = make(prepared)
    # Entirely protected content gives an empty mask; no optimizer/decay update.
    fresh.masking = replace(
        fresh.masking, protected_token_ids=tuple(range(fresh.tokenizer.mask_id))
    )
    weights = [p.clone() for p in fresh.model.parameters()]
    row = fresh.train_step()
    assert row["skipped_empty_mask"] and fresh.optimizer_steps == 0 and not fresh.optimizer.state
    assert all(torch.equal(p, w) for p, w in zip(fresh.model.parameters(), weights, strict=True))


def test_fail_nonfinite_before_update_and_no_checkpoint_failed_boundary(
    prepared, tmp_path, monkeypatch
):
    trainer, job = make(prepared)
    original = trainer.model.forward
    monkeypatch.setattr(trainer.model, "forward", lambda *a, **k: original(*a, **k) * float("nan"))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        trainer.train_step()
    (tmp_path / "metrics").touch()
    with pytest.raises(RuntimeError, match="incomplete"):
        save_state(trainer, tmp_path / "bad.pt", job=job, metrics=tmp_path / "metrics")


def test_fresh_safety_no_writes_and_namespace_isolation(prepared, tmp_path):
    kwargs = arguments(prepared)
    with pytest.raises(ValueError, match="disabled"):
        train(
            settings(),
            **kwargs,
            run_dir=tmp_path / "run",
            backup_root=tmp_path / "drive",
            mode="fresh",
        )
    assert not (tmp_path / "run").exists() and not (tmp_path / "drive").exists()
    preview = preflight(settings(), **kwargs)
    assert not preview["training_started"] and not preview["live_continuation_qualified"]
    _trainer, job = make(prepared)
    other = {**job, "candidate": "attention-12"}
    assert namespace(tmp_path, job) != namespace(tmp_path, other)
    assert str(tmp_path) not in json.dumps(job)
    with pytest.raises(ValueError, match="mounted Drive"):
        train(
            settings(device="xla", precision="bf16"),
            **kwargs,
            run_dir=tmp_path / "run",
            backup_root=tmp_path / "drive",
            mode="fresh",
            start_training=True,
        )
    with pytest.raises(ValueError, match="explicit fresh"):
        train(
            settings(),
            **kwargs,
            run_dir=tmp_path / "run",
            backup_root=tmp_path / "drive",
            mode="auto",
            start_training=True,
        )


def test_native_state_roundtrip_refusals_and_job_portability(prepared, tmp_path):
    trainer, job = make(prepared)
    row = trainer.train_step()
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(json.dumps(row) + "\n")
    path = save_state(trainer, tmp_path / "latest.pt", job=job, metrics=metrics, vm_id="vm-one")
    restored, same = make(prepared)
    receipt = restore_state(
        restored,
        path,
        job=same,
        expected_sha256=checksum(path),
        metrics=metrics,
        vm_id="vm-two",
        require_new_vm=True,
    )
    assert receipt["distinct_vm_ids"] and restored.step == 1 and restored.tokens_seen == 64
    assert restored.model.token_embedding.weight is restored.model.reconstruction_head.weight
    with pytest.raises(ValueError, match="distinct"):
        restore_state(
            restored,
            path,
            job=same,
            expected_sha256=checksum(path),
            metrics=metrics,
            vm_id="vm-one",
            require_new_vm=True,
        )
    with pytest.raises(ValueError, match="checksum"):
        restore_state(restored, path, job=same, expected_sha256="0" * 64, metrics=metrics)
    broken = torch.load(path, weights_only=True)
    broken["rng"].pop("numpy")
    torch.save(broken, tmp_path / "broken.pt")
    with pytest.raises(ValueError, match="RNG"):
        restore_state(
            restored,
            tmp_path / "broken.pt",
            job=same,
            expected_sha256=checksum(tmp_path / "broken.pt"),
            metrics=metrics,
        )
    with metrics.open("a") as stream:
        stream.write('{"event":"v3_train","step":2}\n')
    with pytest.raises(ValueError, match="metrics differ"):
        restore_state(restored, path, job=same, expected_sha256=checksum(path), metrics=metrics)


def test_real_fresh_process_resume_matches_continuous(prepared, tmp_path):
    data, token = prepared
    common = {
        "config": settings().to_dict(),
        "data_dir": str(data),
        "tokenizer_path": str(token),
        "candidate": "hybrid-12",
        "qualification": True,
        "run_name": "fresh-process",
        "start_training": True,
    }

    def launch(root, drive, mode, stop=None):
        request = {
            **common,
            "run_dir": str(root),
            "backup_root": str(drive),
            "mode": mode,
            "stop_after_step": stop,
        }
        path = tmp_path / f"request-{root.name}.json"
        path.write_text(json.dumps(request))
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        result = subprocess.run(
            [sys.executable, "-m", "kiwilm.v3.accelerator_worker", "train", str(path)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    launch(tmp_path / "reference", tmp_path / "drive-ref", "fresh")
    launch(tmp_path / "interrupted", tmp_path / "drive", "fresh", 1)
    launch(tmp_path / "replacement", tmp_path / "drive", "resume")
    left, right = [
        torch.load(tmp_path / name / "latest.pt", weights_only=True)
        for name in ("reference", "replacement")
    ]
    for key in (
        "model",
        "optimizer",
        "generators",
        "rng",
        "schedule",
        "step",
        "optimizer_steps",
        "tokens_seen",
        "masked_tokens_seen",
    ):
        equal(left[key], right[key])
    assert right["tokens_seen"] == 129 and right["format"] == FORMAT
    roots = list((tmp_path / "drive/checkpoints").iterdir())
    assert len(roots) == 1
    generations = list(roots[0].glob("step-*"))
    assert len(generations) == 2
    assert sorted(json.loads((p / "manifest.json").read_text())["step"] for p in generations) == [
        2,
        3,
    ]


def test_drive_publish_failure_retains_local_latest_and_old_pointer(
    prepared, tmp_path, monkeypatch
):
    config = settings()
    opts = arguments(prepared)
    paused = train(
        config,
        **opts,
        run_dir=tmp_path / "first",
        backup_root=tmp_path / "drive",
        mode="fresh",
        start_training=True,
        stop_after_step=1,
        retry_delay=0,
    )
    root = Path(paused["backup_dir"])
    old_pointer = (root / "latest.json").read_bytes()
    trainer, job = make(prepared)
    store = V3CheckpointStore(root, job=job, retry_delay=0)
    store.restore(trainer, tmp_path / "restored")
    row = trainer.train_step()
    with (tmp_path / "restored/metrics.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    monkeypatch.setattr(
        store.transport,
        "publish",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Drive offline")),
    )
    with pytest.raises(RuntimeError, match="offline"):
        store.publish(trainer, tmp_path / "restored")
    assert (root / "latest.json").read_bytes() == old_pointer
    assert torch.load(tmp_path / "restored/latest.pt", weights_only=True)["step"] == 2
    with pytest.raises(FileExistsError):
        train(
            config,
            **opts,
            run_dir=tmp_path / "new",
            backup_root=tmp_path / "drive",
            mode="fresh",
            start_training=True,
        )
    with pytest.raises(FileExistsError):
        train(
            config,
            **opts,
            run_dir=tmp_path / "first",
            backup_root=tmp_path / "drive",
            mode="resume",
            start_training=True,
        )


def test_worker_preparation_and_mount_refusal(prepared, tmp_path):
    with pytest.raises(FileExistsError):
        execute("prepare-demo", {"data_dir": str(prepared[0]), "tokenizer_path": str(prepared[1])})
    with pytest.raises(RuntimeError, match="mount unavailable"):
        execute(
            "restore-data",
            {
                "data_dir": str(tmp_path / "new"),
                "data_cache": str(tmp_path / "cache"),
                "storage_root": str(tmp_path),
            },
        )
    with pytest.raises(ValueError, match="unknown"):
        execute(
            "wrong",
            {
                "config": settings().to_dict(),
                **{
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in arguments(prepared).items()
                },
            },
        )


def test_backend_restrictions_no_fallback(monkeypatch):
    import kiwilm.tpu_smoke as runtime_module

    with pytest.raises(ValueError, match="bf16"):
        Runtime("xla", "fp32")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        runtime_module.Runtime("cuda", "bf16")


def test_cuda_bf16_requires_native_support_before_model_transfer(prepared, monkeypatch):
    from kiwilm.v3 import accelerator

    monkeypatch.setattr(
        accelerator,
        "Runtime",
        lambda *a: SimpleNamespace(device=torch.device("cuda"), precision="bf16"),
    )
    calls = []
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **kw: calls.append(kw) or False)
    with pytest.raises(ValueError, match="BF16 unsupported"):
        make(prepared, device="cuda", precision="bf16")
    assert calls == [{"including_emulation": False}]


@pytest.mark.parametrize("damage", ["moments", "lr", "tie", "budget"])
def test_corrupt_native_state_refused_before_model_mutation(prepared, tmp_path, damage):
    trainer, job = make(prepared)
    row = trainer.train_step()
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(json.dumps(row) + "\n")
    path = save_state(trainer, tmp_path / "state.pt", job=job, metrics=metrics)
    payload = torch.load(path, weights_only=True)
    if damage == "moments":
        payload["optimizer"]["state"].clear()
    elif damage == "lr":
        payload["optimizer"]["param_groups"][0]["lr"] = torch.tensor(0.7)
    elif damage == "tie":
        payload["model"]["reconstruction_head.weight"] = (
            payload["model"]["token_embedding.weight"].clone() + 1
        )
    else:
        payload["tokens_seen"] = trainer.config.max_tokens + 1
    torch.save(payload, path)
    target, _ = make(prepared)
    before = [p.clone() for p in target.model.parameters()]
    with pytest.raises(ValueError):
        restore_state(target, path, job=job, expected_sha256=checksum(path), metrics=metrics)
    assert target.step == 0
    assert all(torch.equal(a, b) for a, b in zip(before, target.model.parameters(), strict=True))


def test_mock_backend_rng_capture_restore_hooks(prepared, tmp_path, monkeypatch):
    # API-hook proof on CPU only: not a simulated TPU speed/equivalence result.
    import kiwilm.v3.accelerator as module

    class BackendRNG:
        seed = 0

        def get_rng_state(self, device):
            assert device.type == "cpu"
            return self.seed

        def set_rng_state(self, seed, device):
            assert device.type == "cpu"
            self.seed = seed

        def xla_device_kind(self, device):
            return "mock-api-only"

    monkeypatch.setattr(module, "version", lambda name: str(torch.__version__).split("+")[0])
    data, token = prepared
    runtime = Runtime("cpu", "fp32")
    runtime.xm = BackendRNG()
    tokenizer = MaskBPETokenizer.load(token)
    trainer = AcceleratorTrainer(
        candidate_config("hybrid-12", tokenizer.vocab_size, qualification=True),
        settings(),
        tokenizer,
        PreparedTokenData(data),
        runtime=runtime,
    )
    assert runtime.xm.seed == 42
    job = {"contract": trainer.contract}
    metrics = tmp_path / "metrics.jsonl"
    metrics.touch()
    runtime.xm.seed = 987654
    path = save_state(trainer, tmp_path / "state.pt", job=job, metrics=metrics)
    runtime.xm.seed = 0
    restore_state(trainer, path, job=job, expected_sha256=checksum(path), metrics=metrics)
    assert runtime.xm.seed == 987654


def test_tpu_notebook_safe_generated_default(tmp_path, monkeypatch):
    import nbformat

    spec = importlib.util.spec_from_file_location(
        "tpu_builder", ROOT / "scripts/build_kiwilm3_tpu_notebook.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    saved = nbformat.read(ROOT / "notebooks/kiwilm3-tpu.ipynb", as_version=4)
    nbformat.validate(saved)
    assert [c.source for c in saved.cells] == [c.source for c in builder.notebook().cells]
    monkeypatch.chdir(tmp_path)
    state = {}
    for cell in saved.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, "v3-tpu-notebook", "exec"), state)
    for name in (
        "MOUNT_DRIVE",
        "SETUP_ENVIRONMENT",
        "PREPARE_DEMO",
        "RESTORE_DATA",
        "RUN_PREFLIGHT",
        "START_TRAINING",
    ):
        assert state[name] is False
    assert not list(tmp_path.iterdir())
