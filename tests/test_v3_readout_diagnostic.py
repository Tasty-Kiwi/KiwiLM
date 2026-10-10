"""Local readout test integrity; tiny synthetic fixtures, never cloud training."""

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from kiwilm.colab_artifacts import file_sha256
from kiwilm.v3 import KiwiLM3Config, build_encoder
from kiwilm.v3.accelerator import AcceleratorTrainConfig
from kiwilm.v3.accelerator_checkpoint import save_state
from kiwilm.v3.accelerator_workflow import construct, prepare_demo
from kiwilm.v3.experiments import canonical_digest
from kiwilm.v3.readout_diagnostic import (
    CASES,
    FixedCorpusTask,
    centered_energy,
    corpus_task,
    readout_diagnostic,
    run_case,
    score_task,
    task_loss,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def task():
    clean = torch.full((4, 16), 8, dtype=torch.long)
    clean[:, 8] = torch.tensor([4, 5, 4, 5])
    clean[:2, 1] = torch.tensor([6, 7])
    clean[2:, 14] = torch.tensor([6, 7])
    return FixedCorpusTask(clean, (4, 5), (0, 16, 32, 48), 8)


def config():
    return KiwiLM3Config(
        vocab_size=33,
        context_length=16,
        d_model=16,
        num_blocks=3,
        num_heads=2,
        swiglu_dim=48,
        noise_embedding_dim=16,
        dropout=0.1,
    )


def test_fixed_inputs_no_leakage_balance_swap_and_erased_negative_control():
    t = task()
    original, targets, selected = t.inputs(32)
    swapped, swapped_targets, swapped_selected = t.inputs(32, mode="swap")
    erased, erased_targets, erased_selected = t.inputs(32, mode="erase")
    assert selected.sum() == 4
    assert original[selected].tolist() == [32] * 4
    assert targets[selected].tolist() == [4, 5, 4, 5]
    assert not torch.isin(original, torch.tensor([4, 5])).any()
    assert swapped_targets[selected].tolist() == [5, 4, 5, 4]
    assert torch.equal(original[[1, 0, 3, 2]], swapped)
    assert torch.equal(selected, swapped_selected) and torch.equal(selected, erased_selected)
    assert torch.equal(erased_targets, targets)
    assert torch.unique(erased, dim=0).shape[0] == 1
    with pytest.raises(ValueError, match="target IDs cannot"):
        bad = t.clean.clone()
        bad[0, 0] = 4
        FixedCorpusTask(bad, (4, 5), t.starts, 8)


def test_corpus_search_is_deterministic_nonoverlapping_and_excludes_visible_labels():
    t = task()
    data = SimpleNamespace(tokens=lambda split: t.clean.numpy().flatten())
    tokenizer = SimpleNamespace(
        vocab_size=33,
        mask_id=32,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        decode=lambda ids: {4: ",", 5: " the"}.get(ids[0], "other"),
    )
    found = corpus_task(data, tokenizer, context_length=16, prefix_tokens=64)
    assert found.to_dict() == t.to_dict()
    assert (
        found.to_dict()
        == corpus_task(data, tokenizer, context_length=16, prefix_tokens=64).to_dict()
    )
    data.tokens = lambda split: torch.full((64,), 8).numpy()
    with pytest.raises(ValueError, match="not enough"):
        corpus_task(data, tokenizer, context_length=16, prefix_tokens=64)


def test_full_vocabulary_loss_and_gradients_match_selected_reference():
    torch.manual_seed(9)
    t = task()
    _, targets, selected = t.inputs(32)
    logits = torch.randn(4, 16, 33, requires_grad=True)
    dense = task_loss(logits, targets, selected, 32, t.target_ids)
    reference = F.cross_entropy(logits[selected][:, :32], targets[selected])
    assert torch.allclose(dense, reference)
    actual = torch.autograd.grad(dense, logits, retain_graph=True)[0]
    expected = torch.autograd.grad(reference, logits)[0]
    assert torch.allclose(actual, expected)
    assert torch.count_nonzero(actual[~selected]) == 0
    assert torch.count_nonzero(actual[..., 32]) == 0
    restricted = task_loss(logits, targets, selected, 32, t.target_ids, two_target=True)
    assert torch.allclose(
        restricted, F.cross_entropy(logits[selected][:, [4, 5]], torch.tensor([0, 1, 0, 1]))
    )
    bad = targets.clone()
    bad[0, 8] = 6
    with pytest.raises(ValueError, match="unrelated target"):
        task_loss(logits, bad, selected, 32, t.target_ids, two_target=True)


def test_centered_energy_constant_and_varying_sets():
    assert centered_energy(torch.ones(4, 8))["centered_energy_fraction"] == 0
    values = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    assert centered_energy(values)["centered_energy_fraction"] == 1
    assert centered_energy(torch.zeros(4, 8))["rms"] == 0


def test_score_preserves_weights_rng_gradients_mode_and_cleans_hooks(monkeypatch):
    model = build_encoder(config())
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    result = score_task(model, task(), 32)
    assert model.training and torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    assert result["finite_logits_and_hidden"]
    assert result["balanced_unigram_ce"] == math.log(2)
    erased = score_task(model, task(), 32, mode="erase")
    assert len(set(erased["predictions"])) == 1 and erased["two_target_accuracy"] == 0.5

    def fail(*args, **kwargs):
        raise RuntimeError("injected")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        score_task(model, task(), 32)
    assert not model.final_norm._forward_pre_hooks and model.training


def test_factorial_initial_logits_rng_and_lr_are_paired_and_deterministic():
    rng = torch.get_rng_state().clone()
    rows = [
        run_case(config(), task(), 32, name=name, untied=untied, two_target=two, steps=2, seed=42)
        for name, untied, two in CASES
    ]
    assert torch.equal(rng, torch.get_rng_state())
    assert all(row["trajectory"][0] == rows[0]["trajectory"][0] for row in rows)
    assert rows[1]["trainable_parameters"] - rows[0]["trainable_parameters"] == 33 * 16
    assert rows[2]["trainable_parameters"] == rows[0]["trainable_parameters"]
    assert all(row["input_tokens"] == 128 and row["masked_targets_seen"] == 8 for row in rows)
    assert len({tuple(x["learning_rate"] for x in row["training_rows"]) for row in rows}) == 1
    again = run_case(
        config(), task(), 32, name="tied-full", untied=False, two_target=False, steps=2, seed=42
    )
    assert again == rows[0]


def test_verified_checkpoint_workflow_never_resumes_or_changes_original(tmp_path, monkeypatch):
    data, tokenizer = tmp_path / "data", tmp_path / "tokenizer.json"
    prepare_demo(data, tokenizer)
    trainer, job = construct(
        AcceleratorTrainConfig(device="cpu", precision="fp32", max_tokens=256, warmup_tokens=32),
        data_dir=data,
        tokenizer_path=tokenizer,
        candidate="hybrid-12",
        qualification=True,
        run_name="readout-test",
    )
    run = tmp_path / "generation"
    run.mkdir()
    metrics = run / "metrics.jsonl"
    metrics.write_text("")
    (run / "job.json").write_text(
        json.dumps(
            {
                "job": job,
                "identity": canonical_digest(job),
                "tokenizer_json": trainer.tokenizer.to_json(),
            }
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
    monkeypatch.setattr("kiwilm.v3.readout_diagnostic.corpus_task", lambda *args: task())
    result = readout_diagnostic(run, data, steps=1)
    assert result["source"]["checkpoint_sha256"] == hashes["latest.pt"]
    assert result["source"]["job_identity"] == canonical_digest(job)
    assert result["frozen_checkpoint_score"]["finite_logits_and_hidden"]
    assert not result["cloud_allocated"] and not result["checkpoint_modified"]
    assert result["canonical_fix"] is None and len(result["cases"]) == 4
    assert all(case["steps"] == 1 for case in result["cases"])
    assert hashes == {p.name: file_sha256(p) for p in run.iterdir()}
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("steps", [0, 201, True])
def test_bounds_before_checkpoint_access(steps):
    with pytest.raises(ValueError, match="1-200"):
        readout_diagnostic(Path("missing"), Path("missing"), steps=steps)


def test_cli_no_overwrite_or_unbounded_training(tmp_path):
    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "scripts/diagnose_kiwilm3_readout.py"),
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
    output = tmp_path / "new.json"
    result = subprocess.run(
        [*command, "--output", str(output), "--steps", "201"], capture_output=True, text=True
    )
    assert result.returncode == 2 and "1-200" in result.stderr and not output.exists()
