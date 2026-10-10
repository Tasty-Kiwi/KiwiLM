"""Collapse regression tests: synthetic local training, never rented hardware."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kiwilm.colab_artifacts import file_sha256
from kiwilm.v3 import KiwiLM3Config, build_encoder
from kiwilm.v3.accelerator import AcceleratorTrainConfig
from kiwilm.v3.accelerator_checkpoint import save_state
from kiwilm.v3.accelerator_workflow import construct, prepare_demo
from kiwilm.v3.collapse_isolation import interventions, isolate_checkpoint
from kiwilm.v3.diagnostics import (
    _token_cosine,
    audit_checkpoint,
    collapse_flags,
    context_probe,
    gradient_probe,
    overfit_probe,
)
from kiwilm.v3.experiments import canonical_digest
from kiwilm.v3.masking import MaskingConfig

ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def fixed_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny():
    torch.manual_seed(42)
    model = build_encoder(
        KiwiLM3Config(
            vocab_size=33,
            d_model=16,
            num_blocks=3,
            context_length=16,
            num_heads=2,
            swiglu_dim=48,
            noise_embedding_dim=16,
        )
    )
    clean = torch.tensor([[2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 3]])
    return model, clean, MaskingConfig(vocab_size=33, mask_id=32)


def test_native_trainer_overfits_both_context_directions_without_label_leakage():
    torch.manual_seed(739)
    state = torch.get_rng_state().clone()
    result = overfit_probe(steps=300)
    assert torch.equal(state, torch.get_rng_state())
    assert result["passed"]
    assert result["trajectory"][-1]["predictions"] == [4, 5, 4, 5]
    assert result["swapped_cues"]["predictions"] == [5, 4, 5, 4]
    assert result["erased_cues"]["accuracy"] == 0.5
    assert result["trajectory"][-1]["loss"] < result["trajectory"][0]["loss"] / 20
    assert result["tokens_seen"] == 300 * 64
    assert result["optimizer_steps"] + result["empty_mask_skips"] == 300
    assert result["training"]["device"] == "cpu"


def test_overfit_deterministic_and_short_failure_is_not_a_pass():
    first = overfit_probe(steps=5)
    assert first == overfit_probe(steps=5)
    assert not first["passed"]


@pytest.mark.parametrize(
    "options",
    [
        {"steps": 0},
        {"steps": 2001},
        {"learning_rate": float("nan")},
        {"width": 64},
        {"width": 512, "steps": 301},
    ],
)
def test_overfit_bounds(options):
    with pytest.raises(ValueError):
        overfit_probe(**options)


def test_visible_context_changes_only_context_and_preserves_model_state(monkeypatch):
    model, clean, masking = tiny()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    calls = []
    original = model.forward

    def spy(ids, *, attention_mask, noise_level):
        calls.append((ids.clone(), attention_mask.clone(), noise_level.clone()))
        return original(ids, attention_mask=attention_mask, noise_level=noise_level)

    monkeypatch.setattr(model, "forward", spy)
    result = context_probe(model, clean, masking)
    assert len(calls) == 2 and result["changed_visible_tokens"] > 0
    for _, valid, noise in calls:
        assert torch.equal(valid, calls[0][1]) and torch.equal(noise, calls[0][2])
    selected = calls[0][0] == masking.mask_id
    assert torch.equal(calls[0][0][selected], calls[1][0][selected])
    assert torch.equal(calls[0][0][:, [0, -1]], calls[1][0][:, [0, -1]])
    assert result["masked_tokens"] == sum(result["argmax_counts"].values())
    assert model.training and torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_full_mask_is_context_negative_control():
    model, clean, masking = tiny()
    result = context_probe(model, clean, masking, noise_level=1.0)
    assert result["changed_visible_tokens"] == 0
    assert result["maximum_probability_change"] == result["argmax_change_fraction"] == 0
    assert result["reversed_context_loss_minus_original"] == 0


def test_context_probe_enters_runtime_autocast_for_both_forwards(monkeypatch):
    model, clean, masking = tiny()
    reference = context_probe(model, clean, masking)
    active, entries = False, 0

    @contextmanager
    def autocast():
        nonlocal active, entries
        active = True
        entries += 1
        try:
            yield
        finally:
            active = False

    original = model.forward

    def spy(ids, **kwargs):
        assert active and not model.training
        return original(ids, **kwargs)

    monkeypatch.setattr(model, "forward", spy)
    runtime = SimpleNamespace(device=torch.device("cpu"), autocast=autocast)
    assert context_probe(model, clean, masking, runtime=runtime) == reference
    assert entries == 2 and not active and model.training


def test_gradient_flow_does_not_mutate_weights_buffers_rng_or_mode():
    model, clean, masking = tiny()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    result = gradient_probe(model, clean, masking)
    assert result["all_finite"] and result["all_parameter_gradients_nonzero"]
    assert all(
        row["mixer_gradient_norm"] > 0 and row["mlp_gradient_norm"] > 0 for row in result["blocks"]
    )
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())
    assert model.training and torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    assert all(not block._forward_hooks and not block._forward_pre_hooks for block in model.blocks)


def test_hook_cleanup_on_failed_forward(monkeypatch):
    model, clean, masking = tiny()

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        gradient_probe(model, clean, masking)
    assert model.training
    for block in model.blocks:
        assert not block._forward_hooks and not block._forward_pre_hooks
        assert not block.mlp._forward_hooks and not block.mlp_norm._forward_pre_hooks
        assert not block.mixer._forward_hooks


def test_mean_pairwise_cosine():
    valid = torch.ones((1, 3), dtype=torch.bool)
    assert _token_cosine(torch.ones(1, 3, 4), valid) == pytest.approx(1)
    assert _token_cosine(torch.eye(3)[None], valid) == pytest.approx(0)
    assert _token_cosine(torch.ones(1, 1, 4), valid[:, :1]) is None


def test_collapse_flags_require_changed_context_and_track_mixer_not_only_mlp():
    gradient = {
        "blocks": [{"block": 0, "mixer_contribution": 117, "output_mean_token_cosine": 0.99999}]
    }
    assert not collapse_flags(gradient, [dict(changed_visible_tokens=0)])[
        "no_argmax_response_to_changed_context"
    ]
    flags = collapse_flags(
        gradient,
        [{"changed_visible_tokens": 50, "argmax_counts": {"15": 60}, "argmax_change_fraction": 0}],
    )
    assert flags["mixer_update_over_10x_input_blocks"] == [0]
    assert flags["mean_output_token_cosine_over_0_9999_blocks"] == [0]
    assert flags["same_single_argmax_across_partial_context_probes"]
    assert flags["no_argmax_response_to_changed_context"]


@pytest.fixture
def generation(tmp_path):
    data, tokenizer = tmp_path / "data", tmp_path / "tokenizer.json"
    prepare_demo(data, tokenizer)
    trainer, job = construct(
        AcceleratorTrainConfig(device="cpu", precision="fp32", max_tokens=256, warmup_tokens=32),
        data_dir=data,
        tokenizer_path=tokenizer,
        candidate="hybrid-12",
        qualification=True,
        run_name="collapse-test",
    )
    run = tmp_path / "generation"
    run.mkdir()
    (run / "metrics.jsonl").write_text(json.dumps(trainer.train_step()) + "\n")
    (run / "job.json").write_text(
        json.dumps(
            {
                "job": job,
                "identity": canonical_digest(job),
                "tokenizer_json": trainer.tokenizer.to_json(),
            }
        )
    )
    save_state(trainer, run / "latest.pt", job=job, metrics=run / "metrics.jsonl")
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "contract_digest": canonical_digest(job),
                "step": trainer.step,
                "tokens_seen": trainer.tokens_seen,
                "files": {
                    name: {"bytes": (run / name).stat().st_size, "sha256": file_sha256(run / name)}
                    for name in ("latest.pt", "job.json", "metrics.jsonl")
                },
            }
        )
    )
    return run, data


def test_native_checkpoint_readonly_audit_and_repeatability(generation):
    run, data = generation
    before = {path.name: file_sha256(path) for path in run.iterdir()}
    first = audit_checkpoint(run, data, batches=1)
    assert first == audit_checkpoint(run, data, batches=1)
    assert first["gradient_probe"]["all_finite"]
    assert len(first["context_probes"]) == 4
    assert first["step"] == 1
    assert before == {path.name: file_sha256(path) for path in run.iterdir()}


def test_real_tokenizer_option_never_samples_its_corpus(generation, monkeypatch):
    from kiwilm.data import PreparedTokenData

    _, data = generation

    def fail(*args, **kwargs):
        pytest.fail("overfit fixture must not sample a real prepared corpus")

    monkeypatch.setattr(PreparedTokenData, "get_batch", fail)
    result = overfit_probe(steps=5, tokenizer_data_dir=data)
    assert result["config"]["vocab_size"] == 294
    assert result["tokens_seen"] == 5 * 64


@pytest.mark.parametrize("name", ["latest.pt", "job.json", "metrics.jsonl"])
def test_refuse_corrupt_generation_before_loading(generation, name, monkeypatch):
    run, data = generation
    with (run / name).open("ab") as stream:
        stream.write(b"corruption")

    def fail(*args, **kwargs):
        pytest.fail("must verify checksums before deserializing weights")

    monkeypatch.setattr(torch, "load", fail)
    with pytest.raises(ValueError, match="checksum mismatch"):
        audit_checkpoint(run, data)


def test_refuse_semantically_mismatched_manifest(generation):
    run, data = generation
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["step"] += 1
    (run / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance mismatch"):
        audit_checkpoint(run, data)


def test_isolation_hooks_are_differentiable_temporary_and_remove_on_error():
    model, clean, masking = tiny()
    initial = {name: p.clone() for name, p in model.named_parameters()}
    with (
        pytest.raises(RuntimeError, match="injected"),
        interventions(model, conditioning_scale=0.0, mixer_scale=0.25),
    ):
        row = gradient_probe(model, clean, masking)
        assert row["conditioning_rms"] == 0
        assert row["parameter_gradient_norms"]["noise_conditioning.output.weight"] == 0
        assert row["blocks"][0]["mixer_gradient_norm"] > 0
        raise RuntimeError("injected")
    assert model.training
    for name, p in model.named_parameters():
        assert torch.equal(p, initial[name]) and p.grad is None
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    with (
        pytest.raises(ValueError, match="select conditioning"),
        interventions(model, conditioning_scale=0.5),
    ):
        pytest.fail("invalid intervention accepted")


def test_checkpoint_counterfactuals_are_readonly_repeatable(generation):
    run, data = generation
    hashes = {p.name: file_sha256(p) for p in run.iterdir()}
    rng = torch.get_rng_state().clone()
    first = isolate_checkpoint(run, data, replay_steps=0)
    assert first == isolate_checkpoint(run, data, replay_steps=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert len(first["counterfactuals"]) == 4 and first["fresh_replays"] == []
    assert hashes == {p.name: file_sha256(p) for p in run.iterdir()}
    for case in first["counterfactuals"]:
        assert case["gradient"]["all_finite"]
        if "conditioning-off" in case["case"]:
            assert case["gradient"]["conditioning_rms"] == 0


def test_collected_bundle_requires_completion_provenance(generation):
    run, data = generation
    document = json.loads((run / "job.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    spec = {"request": {"config": document["job"]["contract"]["training"]}}
    summary = {
        "job": document["job"],
        "identity": document["identity"],
        "step": manifest["step"],
        "tokens_seen": manifest["tokens_seen"],
        "last_commit": {"checkpoint_sha256": manifest["files"]["latest.pt"]["sha256"]},
    }
    for name, contents in (("summary.json", summary), ("spec.json", spec)):
        (run / name).write_text(json.dumps(contents))
        manifest["files"][name] = {
            "bytes": (run / name).stat().st_size,
            "sha256": file_sha256(run / name),
        }
    (run / "manifest.json").unlink()
    (run / "artifact-manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": manifest["files"]})
    )
    marker = run / "download-complete.json"
    marker.write_text(json.dumps({"spec_sha256": canonical_digest(spec)}))
    assert audit_checkpoint(run, data, batches=1)["step"] == 1
    marker.write_text(json.dumps({"spec_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="completion/provenance"):
        isolate_checkpoint(run, data, replay_steps=0)


def test_fresh_factorial_replay_is_bounded_and_deterministic(generation):
    run, data = generation
    config = AcceleratorTrainConfig(
        device="cpu",
        precision="fp32",
        max_tokens=5_000_000,
        warmup_tokens=1_000_000,
        batch_size=8,
        grad_accum_steps=4,
        learning_rate=0.0003,
        min_learning_rate=0.00003,
    )
    document = json.loads((run / "job.json").read_text())
    trainer, job = construct(
        config,
        data_dir=data,
        tokenizer_path=data.parent / "tokenizer.json",
        candidate="hybrid-12",
        qualification=True,
        run_name="factorial-test",
    )
    document.update(job=job, identity=canonical_digest(job))
    (run / "job.json").write_text(json.dumps(document))
    metrics = run / "metrics.jsonl"
    metrics.write_text("")
    save_state(trainer, run / "latest.pt", job=job, metrics=metrics)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "contract_digest": canonical_digest(job),
                "step": 0,
                "tokens_seen": 0,
                "files": {
                    name: {"bytes": (run / name).stat().st_size, "sha256": file_sha256(run / name)}
                    for name in ("latest.pt", "job.json", "metrics.jsonl")
                },
            }
        )
    )
    first = isolate_checkpoint(run, data, replay_steps=2)
    assert first == isolate_checkpoint(run, data, replay_steps=2)
    assert len(first["fresh_replays"]) == 4
    assert all(row["tokens_seen"] == 64 for row in first["fresh_replays"])
    for case in first["fresh_replays"]:
        assert case["steps"] == 2
        assert [r["step"] for r in case["trajectory"]] == [0, 2]
        assert case["training"][0]["learning_rate"] == pytest.approx(
            config.learning_rate * 512 / config.warmup_tokens
        )
    paired_fields = ("step", "input_tokens", "masked_tokens", "eligible_tokens", "learning_rate")
    paired = [
        [{key: row[key] for key in paired_fields} for row in case["training"]]
        for case in first["fresh_replays"]
    ]
    assert all(rows == paired[0] for rows in paired)
    with pytest.raises(ValueError, match="0-64"):
        isolate_checkpoint(run, data, replay_steps=65)


def test_isolation_cli_refuses_overwrite_and_unbounded_replay(tmp_path):
    command = [
        sys.executable,
        str(ROOT / "scripts" / "isolate_kiwilm3_collapse.py"),
        "--run-dir",
        str(tmp_path / "missing"),
        "--data-dir",
        str(tmp_path / "missing-data"),
    ]
    target = tmp_path / "existing.json"
    target.write_text("keep me")
    result = subprocess.run([*command, "--output", str(target)], capture_output=True, text=True)
    assert result.returncode == 2 and "refusing to overwrite" in result.stderr
    assert target.read_text() == "keep me"
    result = subprocess.run(
        [*command, "--output", str(tmp_path / "new.json"), "--replay-steps", "65"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "0-64 replay steps" in result.stderr
    assert not (tmp_path / "new.json").exists()


def test_cli_help_and_no_implicit_checkpoint_overwrite(tmp_path):
    command = [sys.executable, str(ROOT / "scripts" / "diagnose_kiwilm3.py")]
    result = subprocess.run([*command, "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "Local-only" in result.stdout
    target = tmp_path / "existing.json"
    target.write_text("keep me")
    result = subprocess.run([*command, "--output", str(target)], capture_output=True, text=True)
    assert result.returncode == 2 and "refusing to overwrite" in result.stderr
    assert target.read_text() == "keep me"


def test_cli_explicit_seed_and_exclusive_output(tmp_path):
    target = tmp_path / "probe.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "diagnose_kiwilm3.py"),
            "--overfit-steps",
            "1",
            "--learning-rates",
            "0.001",
            "--seed",
            "43",
            "--output",
            str(target),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    probe = json.loads(target.read_text())["overfit_probes"][0]
    assert probe["training"]["seed"] == 43
    assert "Completed:" in result.stderr
