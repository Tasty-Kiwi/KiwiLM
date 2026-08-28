"""Structural and causal coverage for KiwiLM 2."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from kiwilm.checkpoint import save_checkpoint
from kiwilm.colab_kiwilm2 import build_colab_job, checkpoint_backup_key
from kiwilm.compile_benchmark import benchmark_slim_runtime, select_slim_runtime
from kiwilm.config import KiwiLM2Config, KiwiLM2SlimConfig, ModelConfig
from kiwilm.data import PreparedTokenData, prepare_smollm_corpus
from kiwilm.diagnostics import model_health_report
from kiwilm.inference import load_trained_model
from kiwilm.model_profile import profile_kiwilm2
from kiwilm.models import (
    GatedHadamardMLP,
    HadamardMLP,
    KiwiLM2GQA,
    KiwiLM2LM,
    XXLCausalGatedConv,
    build_model,
    fast_walsh_hadamard,
)
from kiwilm.optim import MuonWithAuxAdamW, split_muon_parameters


def small_config(*, slim: bool = False, **overrides: object) -> KiwiLM2Config:
    values: dict[str, object] = {
        "vocab_size": 71,
        "context_length": 8,
        "d_model": 8,
        "dropout": 0.0,
        "num_query_heads": 2,
        "num_kv_heads": 1,
        "swiglu_dim": 12,
        "bigram_buckets": 17,
        "trigram_buckets": 19,
        "conv_kernel_sizes": (3, 5, 3, 5, 3, 5),
    }
    values.update(overrides)
    config_type = KiwiLM2SlimConfig if slim else KiwiLM2Config
    return config_type(**values)


@pytest.mark.parametrize("slim", [False, True])
def test_fixed_schedule_tied_embeddings_causality_and_backward(slim: bool) -> None:
    torch.manual_seed(101)
    config = small_config(slim=slim)
    model = build_model(config).eval()
    assert isinstance(model, KiwiLM2LM)
    assert model.lm_head.weight is model.token_embedding.weight
    assert len(model.blocks) == 10
    assert [
        "gqa" if isinstance(block.mixer, KiwiLM2GQA) else "conv" for block in model.blocks
    ] == list(config.mixer_schedule)
    assert [
        block.mixer.kernel_size
        for block in model.blocks
        if isinstance(block.mixer, XXLCausalGatedConv)
    ] == list(config.conv_kernel_sizes)
    assert all(isinstance(block.mlp, GatedHadamardMLP) == slim for block in model.blocks)

    original = torch.randint(config.vocab_size, (2, config.context_length))
    changed = original.clone()
    changed[:, 5:] = torch.randint(config.vocab_size, (2, 3))
    logits = model(original)
    changed_logits = model(changed)
    torch.testing.assert_close(logits[:, :5], changed_logits[:, :5])
    logits.float().mean().backward()
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


@pytest.mark.parametrize("slim", [False, True])
def test_cached_generation_matches_full_forward_and_rollover(slim: bool) -> None:
    torch.manual_seed(103)
    config = small_config(slim=slim)
    model = KiwiLM2LM(config).eval()
    tokens = torch.randint(config.vocab_size, (2, 3))
    logits, cache = model.prefill(tokens)
    torch.testing.assert_close(logits, model(tokens), rtol=1e-5, atol=1e-6)
    for _ in range(8):
        next_token = torch.randint(config.vocab_size, (2, 1))
        tokens = torch.cat((tokens, next_token), dim=1)
        cached, cache = model.decode_step(next_token, cache)
        expected = model(tokens[:, -config.context_length :])[:, -1:]
        torch.testing.assert_close(cached[:, -1:], expected, rtol=2e-5, atol=2e-6)


def test_hadamard_is_orthonormal_and_differentiable() -> None:
    values = torch.randn(3, 8, requires_grad=True)
    transformed = fast_walsh_hadamard(values)
    torch.testing.assert_close(transformed.square().sum(dim=-1), values.square().sum(dim=-1))
    transformed.sum().backward()
    assert values.grad is not None and bool(torch.isfinite(values.grad).all())


def test_gated_hadamard_matches_reference_and_uses_depth_scaled_residual() -> None:
    torch.manual_seed(107)
    module = GatedHadamardMLP(8, dropout=0.0, residual_scale=1 / math.sqrt(20))
    values = torch.randn(2, 3, 8, requires_grad=True)
    gate = fast_walsh_hadamard(values * module.gate_scale + module.gate_bias)
    value = fast_walsh_hadamard(values * module.value_scale + module.value_bias)
    expected = fast_walsh_hadamard(
        (torch.nn.functional.silu(gate) * value) * module.output_scale
        + module.output_bias
    ) * module.residual_scale
    torch.testing.assert_close(module(values), expected)
    assert float(module.residual_scale.detach()) == pytest.approx(1 / math.sqrt(20))
    for scale in (module.gate_scale, module.value_scale, module.output_scale):
        assert set(scale.detach().tolist()) <= {-1.0, 1.0}
    for bias in (module.gate_bias, module.value_bias, module.output_bias):
        assert torch.count_nonzero(bias) == 0
    module(values).sum().backward()
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in module.parameters()
    )


def test_config_round_trip_and_validation() -> None:
    for config in (small_config(), small_config(slim=True)):
        serialized = config.to_dict()
        assert json.loads(json.dumps(serialized)) == serialized
        assert ModelConfig.from_dict(serialized) == config
    with pytest.raises(ValueError, match="frozen"):
        small_config(mixer_schedule=("gqa",))
    with pytest.raises(ValueError, match="power of two"):
        small_config(slim=True, d_model=12, num_query_heads=3)
    with pytest.raises(ValueError, match="hadamard_variant"):
        small_config(slim=True, hadamard_variant="unknown")


def test_pre_v2_slim_config_loads_the_minimal_negative_baseline() -> None:
    serialized = small_config(slim=True).to_dict()
    serialized.pop("hadamard_variant")
    loaded = ModelConfig.from_dict(serialized)
    assert isinstance(loaded, KiwiLM2SlimConfig)
    assert loaded.hadamard_variant == "minimal_v1"
    model = KiwiLM2LM(loaded)
    assert all(isinstance(block.mlp, HadamardMLP) for block in model.blocks)


def test_pre_v2_checkpoint_reconstructs_with_original_state_shape(tmp_path: Path) -> None:
    config = small_config(slim=True, hadamard_variant="minimal_v1")
    source = KiwiLM2LM(config)
    checkpoint = save_checkpoint(tmp_path / "minimal-v1.pt", model=source, step=3)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["model_config"].pop("hadamard_variant")
    torch.save(payload, checkpoint)

    loaded, loaded_config = load_trained_model(
        checkpoint,
        data_fingerprint=None,
        device=torch.device("cpu"),
    )

    assert isinstance(loaded_config, KiwiLM2SlimConfig)
    assert loaded_config.hadamard_variant == "minimal_v1"
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], tensor)


def test_compile_benchmark_checks_parity_and_runtime_selection() -> None:
    result = benchmark_slim_runtime(
        small_config(),
        small_config(slim=True),
        device=torch.device("cpu"),
        batch_size=1,
        precision="fp32",
        warmup_iterations=1,
        measured_iterations=1,
        compile_backend="eager",
    )
    assert result["compiled_parity"] is True
    assert result["selected_runtime"] in {"eager", "compiled"}
    assert result["promotion_throughput_ratio"] > 0

    synthetic = {
        "dense_eager": {"median_tokens_per_second": 100.0},
        "slim_eager": {"median_tokens_per_second": 90.0},
        "slim_compiled": {"median_tokens_per_second": 110.0},
        "compiled_parity": True,
    }
    assert select_slim_runtime(synthetic)[0] == "compiled"
    synthetic["slim_compiled"]["median_tokens_per_second"] = 95.0
    assert select_slim_runtime(synthetic)[0] == "eager"


def test_profile_separates_memory_dense_and_cache_costs() -> None:
    dense = KiwiLM2LM(small_config())
    slim = KiwiLM2LM(small_config(slim=True))
    minimal = KiwiLM2LM(small_config(slim=True, hadamard_variant="minimal_v1"))
    dense_profile = profile_kiwilm2(dense, sequence_length=8)
    slim_profile = profile_kiwilm2(slim, sequence_length=8)
    minimal_profile = profile_kiwilm2(minimal, sequence_length=8)
    assert dense_profile["parameters"]["ngram"] == (17 + 19) * 8
    assert dense_profile["parameters"]["token_embedding"] == 71 * 8
    assert (
        dense_profile["parameters"]["dense_non_embedding"]
        > slim_profile["parameters"]["dense_non_embedding"]
    )
    assert dense_profile["kv_cache"]["elements"] == 2 * 4 * 1 * 8 * 4
    assert dense_profile["estimated_flops_per_token"]["total"] > 0
    assert slim_profile["hadamard_variant"] == "gated_v2"
    assert minimal_profile["hadamard_variant"] == "minimal_v1"
    assert (
        slim_profile["parameters"]["total"] - minimal_profile["parameters"]["total"]
        == 10 * (2 * slim.config.d_model + 1)
    )
    assert (
        slim_profile["estimated_flops_per_token"]["mlp"]
        > minimal_profile["estimated_flops_per_token"]["mlp"]
    )


def test_health_report_covers_every_block_and_ngram_table() -> None:
    model = KiwiLM2LM(small_config(slim=True))
    inputs = torch.randint(71, (2, 8))
    report = model_health_report(model, inputs, torch.roll(inputs, -1, dims=1))
    assert report["logits_finite"] is True
    assert len(report["blocks"]) == 10
    assert all(block["mixer_gradient_norm"] > 0 for block in report["blocks"])
    assert all(block["mlp_gradient_norm"] > 0 for block in report["blocks"])
    assert all(block["mlp_residual_scale"] is not None for block in report["blocks"])
    assert all(block["post_mlp_residual_rms"] > 0 for block in report["blocks"])
    assert "health_passed" in report
    assert report["ngram"]["bigram_gradient_norm"] > 0
    assert report["ngram"]["trigram_gradient_norm"] > 0


def test_muon_split_uses_linear_matrices_but_not_tables_or_depthwise() -> None:
    model = KiwiLM2LM(small_config())
    muon, adamw = split_muon_parameters(model)
    muon_ids = {id(parameter) for parameter in muon}
    adamw_ids = {id(parameter) for parameter in adamw}
    assert id(model.token_embedding.weight) in adamw_ids
    assert id(model.ngram_embedding.bigram.weight) in adamw_ids
    assert all(parameter.ndim == 2 for parameter in muon)
    assert any(
        id(block.mixer.query.weight) in muon_ids
        for block in model.blocks
        if isinstance(block.mixer, KiwiLM2GQA)
    )
    conv = next(
        block.mixer for block in model.blocks if isinstance(block.mixer, XXLCausalGatedConv)
    )
    assert id(conv.depthwise.weight) in adamw_ids

    optimizer = MuonWithAuxAdamW(
        muon,
        adamw,
        muon_lr=0.02,
        adamw_lr=3e-4,
        weight_decay=0.1,
        beta2=0.95,
    )
    loss = model(torch.randint(71, (1, 4))).float().mean()
    loss.backward()
    optimizer.step()
    saved_state = optimizer.state_dict()
    replacement = MuonWithAuxAdamW(
        muon,
        adamw,
        muon_lr=0.02,
        adamw_lr=3e-4,
        weight_decay=0.1,
        beta2=0.95,
    )
    replacement.load_state_dict(saved_state)
    assert len(replacement.state) == len(optimizer.state)
    assert all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


def test_smollm_preparation_is_exact_disjoint_and_reproducible(tmp_path: Path) -> None:
    rows = {
        "fineweb-edu-dedup": [
            {
                "text": (
                    "FineWeb contaminated [BOS] document."
                    if index == 2
                    else f"FineWeb educational document number {index}."
                )
            }
            for index in range(40)
        ],
        "cosmopedia-v2": [
            {
                "text": (
                    "Cosmopedia contaminated [BOS] section."
                    if index == 2
                    else f"Cosmopedia synthetic textbook section {index}."
                )
            }
            for index in range(40)
        ],
    }

    def loader(_name: str, config: str, **_kwargs: object):
        return rows[config]

    output = tmp_path / "smollm"
    metadata = prepare_smollm_corpus(
        output,
        resolved_revision="a" * 40,
        train_tokens=120,
        validation_tokens=40,
        tokenizer_train_documents=8,
        validation_documents_per_source=2,
        vocab_size=300,
        show_progress=False,
        load_dataset_fn=loader,
    )
    assert metadata["splits"]["train"]["tokens"] == 120
    assert metadata["splits"]["validation"]["tokens"] == 40
    assert metadata["splits"]["train"]["skipped_reserved_token_stories"] >= 1
    assert metadata["config"]["python_edu_included"] is False
    prepared = PreparedTokenData(output)
    assert prepared.fingerprint == metadata["fingerprint"]
    assert math.isfinite(float(prepared.tokens("train").mean()))

def test_colab_job_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePreparedData:
        fingerprint = "b" * 64

        def __init__(self, _data_dir: object) -> None:
            self.metadata = {
                "config": {
                    "python_edu_included": False,
                    "source_configs": ["fineweb-edu-dedup", "cosmopedia-v2"],
                },
                "tokenizer": {"vocab_size": 32_000, "sha256": "c" * 64},
                "splits": {"train": {"tokens": 120}},
            }

    monkeypatch.setattr(
        "kiwilm.colab_kiwilm2.PreparedTokenData",
        FakePreparedData,
    )
    job = build_colab_job(
        "unused",
        phase="smoke",
        architecture="kiwilm2",
        max_tokens=120,
        batch_size=1,
        grad_accum_steps=1,
    )
    assert job["data_fingerprint"] == "b" * 64
    assert job["max_tokens"] == 120
    assert job["max_steps"] == 101
    with pytest.raises(ValueError, match="Muon"):
        build_colab_job(
            "unused",
            phase="smoke",
            architecture="kiwilm2_slim",
            optimizer="muon",
            max_tokens=120,
        )


def test_colab_job_can_prepare_data_in_vm_and_has_stable_backup_key() -> None:
    job = build_colab_job(
        None,
        phase="smoke",
        architecture="kiwilm2",
        max_tokens=120,
        batch_size=1,
        grad_accum_steps=1,
    )
    assert job["prepare_data_in_vm"] is True
    assert job["data_fingerprint"] is None
    assert job["prepared_train_tokens"] == 120
    assert job["drive_backups"] is True
    assert job["drive_root"] == "/content/drive/MyDrive/KiwiLM2"
    assert job["data_cache_key"] == "smoke-120-seed42"
    assert job["eval_batches"] == 50
    key = checkpoint_backup_key(job)
    assert key.startswith("smoke-kiwilm2-adamw-")
    legacy_job = {
        name: value for name, value in job.items() if name != "eval_batches"
    }
    assert key == checkpoint_backup_key(legacy_job)
    changed = {**job, "batch_size": 2}
    assert checkpoint_backup_key(changed) != key

    slim = build_colab_job(
        None,
        phase="smoke",
        architecture="kiwilm2_slim",
        max_tokens=120,
        batch_size=1,
        grad_accum_steps=1,
    )
    assert slim["hadamard_variant"] == "gated_v2"
    assert slim["compile_policy"] == "auto"
    assert slim["schema_version"] == 2
    assert "gated_v2" in checkpoint_backup_key(slim)
    assert checkpoint_backup_key({**slim, "compile_policy": "eager"}) == checkpoint_backup_key(
        slim
    )
    legacy_shaped = {**slim, "hadamard_variant": "minimal_v1"}
    assert checkpoint_backup_key(legacy_shaped) != checkpoint_backup_key(slim)
    with pytest.raises(ValueError, match="compile_policy"):
        build_colab_job(
            None,
            phase="smoke",
            architecture="kiwilm2_slim",
            compile_policy="sometimes",
        )

    architecture = build_colab_job(
        None,
        phase="architecture",
        architecture="kiwilm2",
        batch_size=8,
        grad_accum_steps=4,
    )
    assert architecture["max_tokens"] == 250_000_000
    assert architecture["max_steps"] == 15_359
    assert architecture["warmup_tokens"] == 5_000_000
    assert architecture["eval_batches"] == 200
    assert checkpoint_backup_key(architecture) != checkpoint_backup_key(
        {**architecture, "eval_batches": 50}
    )
