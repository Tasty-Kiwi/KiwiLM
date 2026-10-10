"""Paired mask-policy controls and native trainer parity; no rented hardware."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import kiwilm.v3.accelerator as engine
from kiwilm.colab_artifacts import file_sha256
from kiwilm.data import PreparedTokenData
from kiwilm.v3 import KiwiLM3Config, build_encoder
from kiwilm.v3.accelerator_checkpoint import save_state
from kiwilm.v3.accelerator_workflow import prepare_demo
from kiwilm.v3.experiments import canonical_digest
from kiwilm.v3.masking import MaskingConfig, corrupt_tokens
from kiwilm.v3.noise_diagnostic import (
    aggregate_probes,
    evaluate_policy_model,
    noise_diagnostic,
    paired_mask_policy,
    run_policy_pair,
)
from kiwilm.v3.tokenizer import MaskBPETokenizer


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(context=16):
    return KiwiLM3Config(
        vocab_size=33,
        context_length=context,
        d_model=16,
        num_blocks=3,
        num_heads=2,
        swiglu_dim=48,
        noise_embedding_dim=16,
        dropout=0.1,
    )


def test_paired_selection_uses_same_uniforms_and_rng_preserves_native_variable_policy():
    clean = torch.tensor([[0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 3]]).repeat(2, 1)
    masking = MaskingConfig(vocab_size=33, mask_id=32)
    rng = torch.get_rng_state().clone()
    native_rng = torch.Generator().manual_seed(31)
    native = corrupt_tokens(clean, masking, generator=native_rng)
    results = []
    for policy in ("variable", "fixed-015"):
        records = []
        generator = torch.Generator().manual_seed(31)
        with paired_mask_policy(policy, records):
            result = engine.corrupt_tokens(clean, masking, generator=generator)
        assert torch.equal(generator.get_state(), native_rng.get_state())
        assert not result.masked_positions[:, [0, 1, -1]].any()
        assert torch.equal(result.eligible_positions, native.eligible_positions)
        assert records[0]["selected_per_window"] == result.masked_positions.sum(-1).tolist()
        results.append(result)
    assert torch.equal(results[0].input_ids, native.input_ids)
    assert torch.equal(results[0].noise_level, native.noise_level)
    uniform_rng = torch.Generator().manual_seed(31)
    torch.rand(2, generator=uniform_rng)
    torch.rand(2, generator=uniform_rng)
    uniforms = torch.rand(clean.shape, generator=uniform_rng)
    assert torch.equal(results[1].masked_positions, (uniforms < 0.15) & native.eligible_positions)
    assert torch.equal(results[1].noise_level, torch.full((2,), 0.15))
    assert torch.equal(rng, torch.get_rng_state()) and engine.corrupt_tokens is corrupt_tokens


def test_policy_cleanup_nested_rejection_and_forced_noise_guard():
    with pytest.raises(RuntimeError, match="injected"), paired_mask_policy("fixed-015", []):
        with pytest.raises(RuntimeError, match="exclusive"), paired_mask_policy("variable", []):
            pytest.fail("nested policy accepted")
        raise RuntimeError("injected")
    assert engine.corrupt_tokens is corrupt_tokens
    with paired_mask_policy("fixed-015", []), pytest.raises(ValueError, match="native sampled"):
        engine.corrupt_tokens(
            torch.ones(1, 16, dtype=torch.long),
            MaskingConfig(vocab_size=33, mask_id=32),
            generator=torch.Generator(),
            noise_level=0.5,
        )
    with pytest.raises(ValueError, match="unknown"), paired_mask_policy("bad", []):
        pytest.fail("invalid policy accepted")
    assert engine.corrupt_tokens is corrupt_tokens


def test_aggregation_is_target_weighted_and_full_noise_remains_separate():
    def row(count, loss, level=0.15):
        return {
            "noise_level": level,
            "masked_tokens": count,
            "loss": loss,
            "unigram_loss": 2.0,
            "masked_accuracy": 0.5,
            "unigram_accuracy": 0.25,
            "reversed_context_loss_minus_original": 0.1,
            "argmax_change_fraction": 0.2,
            "target_margin_mean": -1.0,
            "maximum_probability_change": 0.01,
            "visible_content_tokens": 100,
            "changed_visible_tokens": 80,
            "argmax_counts": {"15": count},
            "finite_logits_and_hidden": True,
        }

    results = aggregate_probes(
        [row(10, 1.0), row(90, 3.0), row(100, 9.0, 1.0), {"noise_level": 0.5, "masked_tokens": 0}]
    )
    assert results[0]["loss"] == 2.8 and results[0]["masked_tokens"] == 100
    assert results[0]["loss_minus_unigram"] == pytest.approx(0.8)
    assert results[0]["argmax_counts"] == {"15": 100}
    assert results[1]["masked_tokens"] == 0 and "loss" not in results[1]
    assert results[3]["loss"] == 9.0


def test_aligned_evaluation_preserves_rng_weights_gradients_and_mode(monkeypatch):
    model = build_encoder(config())
    windows = [torch.tensor([[2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 3]])] * 4
    masking = MaskingConfig(vocab_size=33, mask_id=32)
    log_prob = torch.full((32,), -3.0)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    result = evaluate_policy_model(model, windows, masking, log_prob)
    assert model.training and torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())
    assert result["gradient"]["all_finite"]
    assert len(result["rows"]) == 16 and result["by_noise"][0]["masked_tokens"] > 0
    assert result["by_noise"][-1]["changed_visible_tokens"] == 0
    assert result["by_noise"][-1]["maximum_probability_change"] == 0
    for row in result["rows"]:
        if row["masked_tokens"]:
            assert 0 <= row["pre_norm_all_tokens"]["centered_energy_fraction"] <= 1

    def fail(*args, **kwargs):
        raise RuntimeError("injected")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        evaluate_policy_model(model, windows, masking, log_prob)
    assert model.training
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


@pytest.fixture
def prepared(tmp_path):
    data_dir, tokenizer_path = tmp_path / "data", tmp_path / "tokenizer.json"
    prepare_demo(data_dir, tokenizer_path)
    return PreparedTokenData(data_dir), MaskBPETokenizer.load(tokenizer_path)


def test_native_train_step_parity_determinism_and_identical_pairing(prepared):
    data, tokenizer = prepared
    model_config = KiwiLM3Config(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        context_length=16,
        num_heads=2,
        num_blocks=3,
        swiglu_dim=48,
        noise_embedding_dim=16,
        dropout=0.1,
    )
    training = engine.AcceleratorTrainConfig(
        device="cpu",
        precision="fp32",
        max_tokens=4096,
        warmup_tokens=512,
        batch_size=2,
        grad_accum_steps=1,
    )
    masking = MaskingConfig(vocab_size=tokenizer.vocab_size, mask_id=tokenizer.mask_id)
    rng = torch.get_rng_state().clone()
    result = run_policy_pair(model_config, training, tokenizer, data, masking, steps=2)
    assert torch.equal(rng, torch.get_rng_state())
    assert result == run_policy_pair(model_config, training, tokenizer, data, masking, steps=2)
    assert result[0]["trajectory"][0] == result[1]["trajectory"][0]
    direct = engine.AcceleratorTrainer(model_config, training, tokenizer, data, masking=masking)
    for row in result[0]["training_rows"]:
        native = direct.train_step()
        assert native == {key: row[key] for key in native}
    assert all(c["input_tokens"] == 64 for c in result)
    for a, b in zip(result[0]["training_rows"], result[1]["training_rows"], strict=True):
        assert a["pairing"]["clean_sha256"] == b["pairing"]["clean_sha256"]
        assert a["pairing"]["noise_rng_end_sha256"] == b["pairing"]["noise_rng_end_sha256"]
        assert a["torch_rng_end_sha256"] == b["torch_rng_end_sha256"]
    assert engine.corrupt_tokens is corrupt_tokens


def test_verified_checkpoint_workflow_is_fresh_and_readonly(prepared, tmp_path):
    data, tokenizer = prepared
    training = engine.AcceleratorTrainConfig(
        device="cpu",
        precision="fp32",
        max_tokens=5_000_000,
        warmup_tokens=1_000_000,
        batch_size=8,
        grad_accum_steps=4,
    )
    model_config = KiwiLM3Config(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        context_length=512,
        num_heads=2,
        num_blocks=3,
        swiglu_dim=48,
        noise_embedding_dim=16,
        dropout=0.1,
    )
    trainer = engine.AcceleratorTrainer(model_config, training, tokenizer, data)
    job = {"contract": trainer.contract, "run_name": "noise-test"}
    run = tmp_path / "generation"
    run.mkdir()
    metrics = run / "metrics.jsonl"
    metrics.write_text("")
    (run / "job.json").write_text(
        json.dumps(
            {"job": job, "identity": canonical_digest(job), "tokenizer_json": tokenizer.to_json()}
        )
    )
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
    hashes = {p.name: file_sha256(p) for p in run.iterdir()}
    rng = torch.get_rng_state().clone()
    result = noise_diagnostic(run, data.data_dir, steps=1)
    assert result["source"]["checkpoint_sha256"] == hashes["latest.pt"]
    assert result["canonical_fix"] is None and not result["cloud_allocated"]
    assert not result["checkpoint_modified"]
    assert all(c["steps"] == 1 and c["input_tokens"] == 1024 for c in result["cases"])
    assert torch.equal(rng, torch.get_rng_state())
    assert hashes == {p.name: file_sha256(p) for p in run.iterdir()}


@pytest.mark.parametrize("steps", [0, 129, True])
def test_bounds_before_checkpoint_access(steps):
    with pytest.raises(ValueError, match="1-128"):
        noise_diagnostic(Path("missing"), Path("missing"), steps=steps)


def test_cli_refuses_overwrite_and_unbounded_replay(tmp_path):
    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "scripts/diagnose_kiwilm3_noise.py"),
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
    new = tmp_path / "new.json"
    result = subprocess.run(
        [*command, "--output", str(new), "--steps", "129"], capture_output=True, text=True
    )
    assert result.returncode == 2 and "1-128" in result.stderr and not new.exists()
