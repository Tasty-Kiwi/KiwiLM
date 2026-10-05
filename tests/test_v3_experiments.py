"""M5 fixed-slot sampler and B/C execution: synthetic CPU evidence, not model quality."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from kiwilm.data import PreparedTokenData, prepare_from_stories
from kiwilm.v3 import KiwiLM3Config, build_encoder
from kiwilm.v3.evaluation import cloze_scores, distributions, evaluate_transfer, health_audit
from kiwilm.v3.experiment_runner import checksum, preflight, run_candidate, write_comparisons
from kiwilm.v3.experiments import (
    CANDIDATES,
    COMPARISONS,
    ExperimentControls,
    build_suite,
    validate_comparison,
    validate_suite,
)
from kiwilm.v3.sampling import SamplingConfig, generate_slots, sample_masked
from kiwilm.v3.tokenizer import MaskBPETokenizer
from kiwilm.v3.trainer import DenoisingTrainConfig, DenoisingTrainer
from kiwilm.v3.weights import load_encoder_weights, save_encoder_weights

ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def fixed_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def data(tmp_path):
    prepare_from_stories(
        tmp_path / "data",
        ["The cat saw the red ball. " * 40] * 3,
        ["The cat saw the red ball. " * 40],
        vocab_size=300,
        min_frequency=1,
        show_progress=False,
    )
    return PreparedTokenData(tmp_path / "data")


@pytest.fixture
def tokenizer(data):
    return MaskBPETokenizer.from_base(data.tokenizer)


def tiny_model(tokenizer, *, depth=12, attention=False):
    torch.manual_seed(42)
    return build_encoder(
        KiwiLM3Config(
            vocab_size=tokenizer.vocab_size,
            d_model=16,
            num_heads=2,
            swiglu_dim=48,
            noise_embedding_dim=16,
            num_blocks=depth,
            context_length=16,
            dropout=0.1,
            mixer_schedule=("attention",) * depth if attention else None,
        )
    )


def tiny_suite(data, tokenizer, **overrides):
    controls = ExperimentControls(
        max_steps=4,
        batch_size=2,
        context_length=16,
        validation_batches=2,
        checkpoint_interval=2,
        generation_slots=4,
        generation_seeds=(42, 43),
        sampler_steps=3,
        **overrides,
    )
    return build_suite(
        data_fingerprint=data.fingerprint,
        tokenizer_sha256=tokenizer.fingerprint,
        vocab_size=tokenizer.vocab_size,
        controls=controls,
        qualification=True,
    )


@pytest.mark.parametrize("depth,attention", [(12, False), (12, True), (16, False), (16, True)])
@pytest.mark.parametrize("temperature", [0.0, 0.8, 1e-300])
def test_real_backbone_sampling_preserves_context_and_rng(tokenizer, depth, attention, temperature):
    model = tiny_model(tokenizer, depth=depth, attention=attention)
    ids = torch.tensor(
        [
            [2, 4, tokenizer.mask_id, tokenizer.mask_id, 5, 3, 0],
            [2, tokenizer.mask_id, 7, 8, 9, 3, 0],
        ]
    )
    original = ids.clone()
    rng = torch.get_rng_state()
    config = SamplingConfig(16, temperature, 2, 42)
    first = sample_masked(model, tokenizer, ids, config=config)
    second = sample_masked(model, tokenizer, ids, config=config)
    assert torch.equal(first.token_ids, second.token_ids)
    assert torch.equal(ids, original) and model.training
    assert torch.equal(torch.get_rng_state(), rng)
    visible = ids != tokenizer.mask_id
    assert torch.equal(first.token_ids[visible], ids[visible])
    assert first.trace[-1]["remaining"] == [0, 0]
    for row in range(2):
        counts = [entry["remaining"][row] for entry in first.trace]
        assert counts == sorted(counts, reverse=True)
    assert (first.token_ids[~visible] >= 4).all()
    assert (first.token_ids[~visible] < tokenizer.mask_id).all()


def test_confidence_ranked_reveal_and_exact_top_k(tokenizer, monkeypatch):
    model = tiny_model(tokenizer)
    calls = []

    def logits(ids, *, attention_mask, noise_level):
        calls.append((ids.clone(), noise_level.clone()))
        output = torch.zeros(*ids.shape, tokenizer.vocab_size)
        # Position 2 is more confident; it must be frozen in the second pass.
        output[:, 1, 4] = 2
        output[:, 2, 5] = 12
        return output

    monkeypatch.setattr(model, "forward", logits)
    ids = torch.tensor([[2, tokenizer.mask_id, tokenizer.mask_id, 3]])
    sampled = sample_masked(model, tokenizer, ids, config=SamplingConfig(2, 0.8, 1, 42))
    assert sampled.token_ids.tolist() == [[2, 4, 5, 3]]
    assert calls[1][0].tolist() == [[2, tokenizer.mask_id, 5, 3]]
    assert [level.item() for _, level in calls] == [1.0, 0.5]
    assert sampled.trace[0]["remaining"] == [1]


def test_sampler_padding_invalid_slots_failure_restore_and_no_holes(tokenizer, monkeypatch):
    model = tiny_model(tokenizer)
    ids = torch.tensor([[2, tokenizer.mask_id, 3]])
    with pytest.raises(ValueError, match="MASK slots"):
        sample_masked(model, tokenizer, ids, attention_mask=torch.tensor([[True, False, True]]))
    with pytest.raises(ValueError, match="boolean"):
        sample_masked(model, tokenizer, ids, attention_mask=torch.ones_like(ids))
    monkeypatch.setattr(
        model, "forward", lambda *a, **k: torch.full((1, 3, tokenizer.vocab_size), float("nan"))
    )
    with pytest.raises(FloatingPointError, match="nonfinite"):
        sample_masked(model, tokenizer, ids)
    assert model.training
    clean = torch.tensor([[2, 4, 3, 0]])
    result = sample_masked(model, tokenizer, clean)
    assert result.trace == () and torch.equal(clean, result.token_ids)


@pytest.mark.parametrize(
    "options",
    [
        dict(steps=0),
        dict(steps=True),
        dict(seed=-1),
        dict(seed=2**32),
        dict(top_k=-1),
        dict(temperature=-1),
        dict(temperature=float("nan")),
        dict(temperature=True),
    ],
)
def test_sampling_config_invalid(options):
    with pytest.raises(ValueError):
        SamplingConfig(**options)


def test_slots_roundtrip_parity_and_context_bounds(tokenizer, tmp_path):
    model = tiny_model(tokenizer)
    kwargs = dict(prompt="The ", suffix=" cat.", output_slots=4, config=SamplingConfig(4, 0, 0))
    first = generate_slots(model, tokenizer, **kwargs)
    assert first["prompt"] == "The " and first["suffix"] == " cat."
    for suffix in (".pt", ".safetensors"):
        path = save_encoder_weights(
            model, tmp_path / f"model{suffix}", dtype="fp32", tokenizer_sha256=tokenizer.fingerprint
        )
        loaded, _ = load_encoder_weights(path, expected_tokenizer_sha256=tokenizer.fingerprint)
        assert first["token_ids"] == generate_slots(loaded, tokenizer, **kwargs)["token_ids"]
    with pytest.raises(ValueError, match="no silent truncation"):
        generate_slots(model, tokenizer, "cat " * 30, output_slots=4)
    with pytest.raises(ValueError, match="positive"):
        generate_slots(model, tokenizer, "", output_slots=0)


def test_matrix_dense_swiglu_profiles_and_frozen_controls(data, tokenizer):
    suite = tiny_suite(data, tokenizer)
    assert set(suite["candidates"]) == set(CANDIDATES)
    assert suite["dropped_experiments"] == ["A", "D"]
    assert not suite["training_started"] and suite["train_from_scratch"]
    assert set(suite["comparisons"]) == set(COMPARISONS)
    profiles = suite["candidates"]
    for name in CANDIDATES:
        config = KiwiLM3Config.from_dict(profiles[name]["model"])
        assert config.num_blocks == int(name.split("-")[1])
        assert all(row["mlp"] == "swiglu" for row in profiles[name]["profile"]["blocks"])
        assert profiles[name]["profile"]["attention_blocks"] == (
            config.num_blocks
            if name.startswith("attention")
            else (4 if config.num_blocks == 12 else 6)
        )
    # Tiny width16 keeps the XXL kernels: its cost ordering is not representative
    # of the full-width architectures. Test that ordering on full profiles below.
    assert (
        profiles["hybrid-16"]["profile"]["parameters"]
        > profiles["hybrid-12"]["profile"]["parameters"]
    )
    assert validate_suite(json.loads(json.dumps(suite))).max_steps == 4
    for mutate in (
        lambda s: s["controls"].update(data_seed=999),
        lambda s: s["candidates"]["hybrid-12"]["profile"].update(parameters=1),
        lambda s: s.update(objective="other"),
    ):
        modified = copy.deepcopy(suite)
        mutate(modified)
        with pytest.raises(ValueError, match="canonical"):
            validate_suite(modified)


@pytest.mark.parametrize(
    "options",
    [
        dict(dropout=True),
        dict(dropout=float("nan")),
        dict(initialization_seed=-1),
        dict(generation_seeds=(42, 42)),
        dict(validation_batches=101),
    ],
)
def test_experiment_controls_invalid(options):
    with pytest.raises(ValueError):
        ExperimentControls(**options)


def test_preflight_no_weights_or_writes_and_provenance(data, tokenizer, tmp_path, monkeypatch):
    suite = tiny_suite(data, tokenizer)
    monkeypatch.setattr(
        "kiwilm.v3.experiment_runner.build_encoder",
        lambda *a: pytest.fail("must not allocate weights"),
    )
    root = tmp_path / "run"
    assert not run_candidate(suite, "hybrid-12", data, tokenizer, root)["training_started"]
    assert not root.exists()
    with pytest.raises(ValueError, match="explicit boolean"):
        run_candidate(suite, "hybrid-12", data, tokenizer, root, start_training="false")
    mismatch = copy.deepcopy(suite)
    mismatch = build_suite(
        data_fingerprint="0" * 64,
        tokenizer_sha256=tokenizer.fingerprint,
        vocab_size=tokenizer.vocab_size,
        qualification=True,
    )
    with pytest.raises(ValueError, match="provenance"):
        preflight(mismatch, "hybrid-12", data, tokenizer)
    with pytest.raises(ValueError, match="four B/C"):
        preflight(suite, "hadamard", data, tokenizer)


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_completed_candidate_and_resume_exact(data, tokenizer, tmp_path, candidate):
    suite = tiny_suite(data, tokenizer)
    continuous = run_candidate(
        suite, candidate, data, tokenizer, tmp_path / "continuous", start_training=True
    )
    root = tmp_path / "continued"
    paused = run_candidate(
        suite, candidate, data, tokenizer, root, start_training=True, stop_after_step=2
    )
    assert paused["status"] == "paused" and paused["validation"] is None
    receipt = root / "step-000002.json"
    resumed = run_candidate(
        suite,
        candidate,
        data,
        tokenizer,
        root,
        start_training=True,
        resume_receipt=receipt,
        expected_checkpoint_sha256=paused["checkpoint_sha256"],
    )
    assert resumed["step"] == 4 and resumed["status"] == "complete"
    assert all(resumed["inference_parity"].values())
    assert resumed["health"]["all_finite"] and resumed["health"]["all_mlp_gradients_nonzero"]
    assert len(resumed["health"]["blocks"]) == int(candidate.split("-")[1])
    assert len(resumed["generation"]["rows"]) == 6
    assert resumed["validation"] == continuous["validation"]
    for key in ("data_rng_sha256", "noise_rng_sha256", "tokens_seen", "health"):
        assert resumed[key] == continuous[key]
    left, right = (
        torch.load(report["checkpoint"], weights_only=True)["model"]
        for report in (continuous, resumed)
    )
    assert left.keys() == right.keys()
    assert all(torch.equal(left[key], right[key]) for key in left)
    assert not (root / ".training.lock").exists()
    assert (root / "step-000002.pt").exists()
    with pytest.raises(FileExistsError):
        run_candidate(suite, candidate, data, tokenizer, root, start_training=True)


def test_resume_refuses_changed_controls_tamper_tail_or_implicit_fresh(data, tokenizer, tmp_path):
    suite = tiny_suite(data, tokenizer)
    root = tmp_path / "run"
    report = run_candidate(
        suite, "hybrid-12", data, tokenizer, root, start_training=True, stop_after_step=2
    )
    kwargs = dict(
        start_training=True,
        resume_receipt=root / "step-000002.json",
        expected_checkpoint_sha256=report["checkpoint_sha256"],
    )
    with pytest.raises(ValueError, match="checksum"):
        run_candidate(
            suite,
            "hybrid-12",
            data,
            tokenizer,
            root,
            **{**kwargs, "expected_checkpoint_sha256": "0" * 64},
        )
    with pytest.raises(ValueError, match="receipt"):
        run_candidate(suite, "attention-12", data, tokenizer, root, **kwargs)
    with pytest.raises(ValueError, match="both"):
        run_candidate(
            suite,
            "hybrid-12",
            data,
            tokenizer,
            root,
            start_training=True,
            resume_receipt=kwargs["resume_receipt"],
        )
    with (root / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps({"step": 3}) + "\n")
    with pytest.raises(ValueError, match="no implicit truncation"):
        run_candidate(suite, "hybrid-12", data, tokenizer, root, **kwargs)


def test_health_and_cloze_leave_training_rng_and_gradients_unchanged(data, tokenizer):
    model = tiny_model(tokenizer)
    trainer = DenoisingTrainer(model, tokenizer, data, DenoisingTrainConfig(max_steps=4))
    trainer.train_step()
    grad = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    rngs = [
        torch.get_rng_state(),
        trainer.data_generator.get_state(),
        trainer.noise_generator.get_state(),
    ]
    first, second = health_audit(trainer, batches=2), health_audit(trainer, batches=2)
    assert first == second and first["all_finite"]
    for original, actual in zip(grad, model.parameters(), strict=True):
        assert (original is None and actual.grad is None) or torch.equal(original, actual.grad)
    for original, actual in zip(
        rngs,
        [
            torch.get_rng_state(),
            trainer.data_generator.get_state(),
            trainer.noise_generator.get_state(),
        ],
        strict=True,
    ):
        assert torch.equal(original, actual)
    score = cloze_scores(model, tokenizer, prefix="", suffix=" cat.", choices=["a", "b"])
    assert len(score["scores"]) == 2 and score["context_limit"] == 16 and model.training
    with pytest.raises(ValueError, match="equal-token"):
        cloze_scores(model, tokenizer, prefix="", suffix="", choices=["a", "a long choice"])
    with pytest.raises(ValueError, match="distinct"):
        cloze_scores(model, tokenizer, prefix="", suffix="", choices=["a", "a"])
    with pytest.raises(ValueError, match="no truncation"):
        cloze_scores(model, tokenizer, prefix="cat " * 30, suffix="", choices=["a", "b"])
    assert distributions([1.0, 2.0, 3.0])["median"] == 2.0


def test_four_way_comparison_refuses_bad_provenance_and_writes_portable(
    data, tokenizer, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    suite = tiny_suite(data, tokenizer)
    reports = {
        candidate: run_candidate(
            suite, candidate, data, tokenizer, Path("runs") / candidate, start_training=True
        )
        for candidate in CANDIDATES
    }
    result = write_comparisons(suite, reports, "comparison")
    assert result["canonical_winner"] is None and len(result["comparisons"]) == 4
    assert "/Users/" not in (tmp_path / "comparison/results.json").read_text()
    for key, value in [
        ("step", 3),
        ("data_rng_sha256", "other"),
        ("noise_rng_sha256", "other"),
        ("runtime", {"device": "tpu"}),
        ("checkpoint_sha256", "bad"),
    ]:
        broken = copy.deepcopy(reports)
        broken["hybrid-12"][key] = value
        with pytest.raises(ValueError):
            validate_comparison(broken, suite, "B-12")
    broken = copy.deepcopy(reports)
    for report in broken.values():
        report["validation"].pop("evaluation_contract")
    with pytest.raises(ValueError, match="not aligned"):
        validate_comparison(broken, suite, "B-12")
    with pytest.raises(ValueError, match="only B/C"):
        validate_comparison(reports, suite, "A")
    with pytest.raises(FileExistsError):
        write_comparisons(suite, reports, "comparison")


def test_cli_plan_preflight_opt_in_and_data_free_generation(data, tokenizer, tmp_path):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    suite_path = tmp_path / "suite.json"
    base = [sys.executable, str(ROOT / "scripts/run_kiwilm3_experiments.py")]
    inputs = [
        "--suite",
        str(suite_path),
        "--data-dir",
        str(data.data_dir),
        "--tokenizer",
        str(tokenizer_path),
    ]
    planned = subprocess.run(
        [*base, "plan", "--qualification", *inputs], check=True, capture_output=True, text=True
    )
    assert not json.loads(planned.stdout)["training_started"]
    checked = subprocess.run(
        [*base, "preflight", "--candidate", "hybrid-12", *inputs],
        check=True,
        capture_output=True,
        text=True,
    )
    assert not json.loads(checked.stdout)["training_started"]
    failed = subprocess.run(
        [*base, "train", "--candidate", "hybrid-12", *inputs], capture_output=True, text=True
    )
    assert failed.returncode != 0 and "--start-training" in failed.stderr
    assert not (tmp_path / "runs").exists()
    weights = save_encoder_weights(
        tiny_model(tokenizer), tmp_path / "weights.pt", tokenizer_sha256=tokenizer.fingerprint
    )
    generated = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/generate_kiwilm3.py"),
            "--checkpoint",
            str(weights),
            "--checkpoint-sha256",
            checksum(weights),
            "--tokenizer",
            str(tokenizer_path),
            "--output-slots",
            "4",
            "--steps",
            "4",
            "--temperature",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(generated.stdout)["trace"][-1]["remaining"] == [0]


def test_full_width_recipe_is_dense_and_explicitly_unqualified():
    suite = build_suite(data_fingerprint="a" * 64, tokenizer_sha256="b" * 64, vocab_size=32001)
    assert suite["budget"]["input_positions"] == 1_024_000
    assert not suite["runtime"]["accelerator_qualified"]
    assert suite["candidates"]["hybrid-12"]["model"]["d_model"] == 512
    assert suite["candidates"]["attention-16"]["model"]["swiglu_dim"] == 2048
    assert (
        suite["candidates"]["hybrid-12"]["profile"]["parameters"]
        < suite["candidates"]["attention-12"]["profile"]["parameters"]
    )


def test_transfer_uses_only_compatible_ids_and_preserves_model_rng(data, tokenizer, tmp_path):
    trainer = DenoisingTrainer(
        tiny_model(tokenizer), tokenizer, data, DenoisingTrainConfig(max_steps=4)
    )
    rng = torch.get_rng_state()
    first = evaluate_transfer(trainer, data, batches=2)
    assert first["data_fingerprint"] == data.fingerprint
    assert first["autoregressive_perplexity"] is None and len(first["rows"]) == 4
    assert torch.equal(torch.get_rng_state(), rng) and trainer.model.training
    prepare_from_stories(
        tmp_path / "different",
        ["different unseen merges " * 40],
        ["different unseen merges " * 40],
        vocab_size=300,
        min_frequency=1,
        show_progress=False,
    )
    with pytest.raises(ValueError, match="ID-preserving"):
        evaluate_transfer(trainer, PreparedTokenData(tmp_path / "different"), batches=2)


def test_experiment_notebook_generated_and_disabled(tmp_path, monkeypatch):
    import nbformat

    spec = importlib.util.spec_from_file_location(
        "m5_builder", ROOT / "scripts/build_kiwilm3_experiment_notebook.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    saved = nbformat.read(ROOT / "notebooks/kiwilm3-experiments.ipynb", as_version=4)
    nbformat.validate(saved)
    assert [cell.source for cell in saved.cells] == [
        cell.source for cell in builder.notebook().cells
    ]
    monkeypatch.chdir(tmp_path)
    state = {}
    for cell in saved.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, "m5-notebook", "exec"), state)
    assert not state["START_TRAINING"] and not state["INSTALL_PACKAGE"]
    assert state["suite"] is None and not list(tmp_path.iterdir())
