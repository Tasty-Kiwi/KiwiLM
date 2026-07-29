"""Coverage for Model Z-P and the matched X/Y/Z smoke benchmark."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from kiwilm.checkpoint import save_checkpoint
from kiwilm.config import ModelConfig, ModelZParallelConfig
from kiwilm.data import PreparedTokenData, prepare_from_stories
from kiwilm.generation import generate_token_stream, generate_tokens
from kiwilm.inference import load_trained_model
from kiwilm.models import (
    PARALLEL_BRANCH_SCALE,
    ModelZParallelBlock,
    ModelZParallelCache,
    ModelZParallelLM,
    build_model,
)


def small_config(**overrides: object) -> ModelZParallelConfig:
    values: dict[str, object] = {
        "vocab_size": 67,
        "context_length": 12,
        "d_model": 16,
        "dropout": 0.0,
        "kernel_size": 3,
        "cnn_dilations": (1, 2),
        "num_heads": 2,
        "swiglu_dim": 48,
    }
    values.update(overrides)
    return ModelZParallelConfig(**values)


def test_default_shape_parameter_count_weight_tying_and_structure() -> None:
    config = ModelZParallelConfig()
    model = build_model(config)
    logits = model(torch.randint(config.vocab_size, (1, 8)))

    assert logits.shape == (1, 8, config.vocab_size)
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_387_008
    assert isinstance(model, ModelZParallelLM)
    assert model.lm_head.weight is model.token_embedding.weight
    assert len(model.blocks) == 2
    assert all(isinstance(block, ModelZParallelBlock) for block in model.blocks)
    assert [block.cnn.conv.conv.dilation[0] for block in model.blocks] == [1, 2]
    assert [block.feedforward.gate_projection.out_features for block in model.blocks] == [
        1280,
        1280,
    ]
    assert math.isclose(PARALLEL_BRANCH_SCALE, 1 / math.sqrt(2))


def test_parallel_merge_adds_one_residual_and_uses_identical_inputs() -> None:
    block = ModelZParallelBlock(
        4,
        kernel_size=3,
        dilation=1,
        num_heads=1,
        swiglu_dim=8,
        dropout=0.0,
        rms_norm_eps=1e-5,
    )
    seen: list[Tensor] = []

    class ConstantBranch(nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = value

        def mix(self, values: Tensor) -> Tensor:
            seen.append(values)
            return torch.full_like(values, self.value)

    block.cnn = ConstantBranch(1.0)  # type: ignore[assignment]
    block.attention = ConstantBranch(2.0)  # type: ignore[assignment]
    block.feedforward = nn.Identity()  # type: ignore[assignment]
    values = torch.randn(2, 3, 4)

    output = block(values)

    assert seen[0] is values
    assert seen[1] is values
    torch.testing.assert_close(
        output,
        values + torch.full_like(values, 3 * PARALLEL_BRANCH_SCALE),
    )


def test_model_is_strictly_causal_deterministic_and_supports_backward() -> None:
    torch.manual_seed(79)
    config = small_config(dropout=0.5)
    model = ModelZParallelLM(config).eval()
    original = torch.randint(config.vocab_size, (2, config.context_length))
    changed = original.clone()
    changed[:, 7:] = torch.randint(config.vocab_size, (2, 5))

    first = model(original)
    second = model(original)
    changed_logits = model(changed)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[:, :7], changed_logits[:, :7])
    first.mean().backward()
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_config_round_trip_and_validation() -> None:
    config = small_config(
        kernel_size=5,
        cnn_dilations=(1, 3),
        num_heads=4,
        swiglu_dim=80,
        rms_norm_eps=1e-6,
        tie_embeddings=False,
    )
    serialized = config.to_dict()

    assert json.loads(json.dumps(serialized)) == serialized
    assert ModelZParallelConfig.from_dict(serialized) == config
    assert ModelConfig.from_dict(serialized) == config
    with pytest.raises(ValueError, match="architecture"):
        small_config(architecture="model_x")
    with pytest.raises(ValueError, match="cnn_dilations"):
        small_config(cnn_dilations=(1,))
    with pytest.raises(ValueError, match="swiglu_dim"):
        small_config(swiglu_dim=0)
    with pytest.raises(ValueError, match="rms_norm_eps"):
        small_config(rms_norm_eps=0)
    with pytest.raises(ValueError, match="divisible"):
        small_config(d_model=18, num_heads=4)
    with pytest.raises(ValueError, match="even"):
        small_config(d_model=24, num_heads=8)


def test_cache_matches_forward_generation_stream_and_rollover() -> None:
    torch.manual_seed(83)
    config = small_config(context_length=8)
    model = ModelZParallelLM(config).eval()
    prompt = torch.randint(config.vocab_size, (2, 4))

    logits, cache = model.prefill(prompt)
    torch.testing.assert_close(logits, model(prompt))
    assert len(cache.cnn) == 2
    assert len(cache.attention) == 2

    tokens = prompt
    for _ in range(7):
        next_token = torch.randint(config.vocab_size, (2, 1))
        tokens = torch.cat((tokens, next_token), dim=1)
        cached_logits, cache = model.decode_step(next_token, cache)
        expected = model(tokens[:, -config.context_length :])[:, -1:, :]
        torch.testing.assert_close(cached_logits[:, -1:, :], expected)

    uncached = generate_tokens(
        model,
        prompt[:1],
        max_new_tokens=12,
        temperature=0,
        cache="off",
    )
    cached = generate_tokens(
        model,
        prompt[:1],
        max_new_tokens=12,
        temperature=0,
        cache="auto",
    )
    streamed = torch.cat(
        (
            prompt[:1],
            *generate_token_stream(
                model,
                prompt[:1],
                max_new_tokens=12,
                temperature=0,
                cache="auto",
            ),
        ),
        dim=1,
    )
    assert torch.equal(cached, uncached)
    assert torch.equal(streamed, uncached)


def test_decode_step_rejects_malformed_cache() -> None:
    model = ModelZParallelLM(small_config()).eval()
    _, cache = model.prefill(torch.randint(67, (2, 4)))

    with pytest.raises(ValueError, match="incompatible structure"):
        model.decode_step(
            torch.randint(67, (2, 1)),
            ModelZParallelCache(
                token_ids=cache.token_ids,
                cnn=cache.cnn[:-1],
                attention=cache.attention,
            ),
        )

    key, value = cache.attention[0]
    with pytest.raises(ValueError, match="attention cache"):
        model.decode_step(
            torch.randint(67, (2, 1)),
            ModelZParallelCache(
                token_ids=cache.token_ids,
                cnn=cache.cnn,
                attention=[
                    (key[:, :, :-1], value[:, :, :-1]),
                    cache.attention[1],
                ],
            ),
        )


def test_branch_diagnostics_are_finite_and_complete() -> None:
    model = ModelZParallelLM(small_config()).eval()
    diagnostics = model.branch_diagnostics(torch.randint(67, (3, 8)))

    assert len(diagnostics) == 2
    assert [row["dilation"] for row in diagnostics] == [1.0, 2.0]
    assert all(math.isfinite(value) for row in diagnostics for value in row.values())
    assert all(-1 <= row["branch_cosine_similarity"] <= 1 for row in diagnostics)


def test_checkpoint_reconstruction_on_available_devices(tmp_path: Path) -> None:
    config = small_config(context_length=8)
    checkpoint = save_checkpoint(
        tmp_path / "model-z-parallel.pt",
        model=build_model(config),
        step=1,
        model_config=config,
        data_fingerprint="z" * 64,
    )
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")

    for device_name in devices:
        device = torch.device(device_name)
        model, loaded_config = load_trained_model(
            checkpoint,
            data_fingerprint="z" * 64,
            device=device,
        )
        input_ids = torch.randint(config.vocab_size, (2, 8), device=device)
        loss = model(input_ids).float().mean()
        loss.backward()

        assert loaded_config == config
        assert torch.isfinite(loss)


def test_reduced_xyz_smoke_benchmark_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "benchmark"
    prepare_from_stories(
        data_dir,
        [
            "Once there was a small green bird.",
            "The bird found a kind friend.",
        ]
        * 6,
        ["They went home before dark."] * 4,
        vocab_size=300,
        min_frequency=1,
    )
    data = PreparedTokenData(data_dir)
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "suite_version": 1,
                "max_new_tokens": 1,
                "sampling_profiles": [
                    {
                        "id": "focused",
                        "temperature": 0,
                        "top_k": 0,
                        "seed": 42,
                    }
                ],
                "prompts": [{"id": "once", "prompt": "Once"}],
            }
        ),
        encoding="utf-8",
    )
    scripts_dir = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    module_path = scripts_dir / "run_model_xyz_smoke_benchmark.py"
    spec = importlib.util.spec_from_file_location("model_xyz_smoke", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    exit_code = module.main(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--suite",
            str(suite),
            "--device",
            "cpu",
            "--expected-data-fingerprint",
            data.fingerprint,
            "--max-steps",
            "1",
            "--batch-size",
            "2",
            "--precision",
            "fp32",
            "--context-length",
            "8",
            "--d-model",
            "8",
            "--attention-heads",
            "1",
            "--model-x-swiglu-dim",
            "12",
            "--model-y-swiglu-dim",
            "12",
            "--model-z-swiglu-dim",
            "24",
            "--warmup-steps",
            "0",
            "--eval-interval",
            "1",
            "--eval-batches",
            "1",
            "--post-eval-batches",
            "1",
            "--checkpoint-interval",
            "1",
            "--log-interval",
            "0",
            "--sample-tokens",
            "0",
            "--generation-tokens",
            "1",
            "--generation-repeats",
            "1",
            "--diagnostic-batch-size",
            "2",
        ]
    )
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert summary["models"]["model_x"]["training"]["step"] == 1
    assert summary["models"]["model_y"]["training"]["step"] == 1
    assert summary["models"]["model_z_parallel"]["training"]["step"] == 1
    assert len(summary["models"]["model_z_parallel"]["branch_diagnostics"]) == 2
    assert summary["comparison"]["generation_count"] == 3
    assert summary["settings"]["training_targets_per_model"] == 16
    assert (output_dir / "comparison" / "report.md").is_file()
    assert (output_dir / "comparison" / "results.jsonl").is_file()
