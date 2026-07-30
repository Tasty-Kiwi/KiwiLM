from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import kiwilm.training as training_module
from kiwilm.checkpoint import (
    CheckpointCompatibilityError,
    load_checkpoint,
    save_checkpoint,
)
from kiwilm.config import GatedCNNConfig
from kiwilm.generation import (
    generate,
    generate_stream,
    generate_token_stream,
    generate_tokens,
)
from kiwilm.training import (
    TrainConfig,
    choose_device,
    evaluate,
    learning_rate_at_step,
    learning_rate_at_tokens,
    train,
)


class TinyData:
    def __init__(self, vocab_size: int, fingerprint: str = "a" * 64) -> None:
        self.vocab_size = vocab_size
        self.fingerprint = fingerprint
        self.generator = torch.Generator(device="cpu").manual_seed(99)

    def get_batch(
        self,
        split: str,
        *,
        batch_size: int,
        context_length: int,
        device: str | torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert split in {"train", "validation"}
        source = generator if generator is not None else self.generator
        windows = torch.randint(
            self.vocab_size,
            (batch_size, context_length + 1),
            generator=source,
        )
        inputs, targets = windows[:, :-1], windows[:, 1:]
        return inputs.to(device), targets.to(device)

    def state_dict(self) -> dict[str, object]:
        return {
            "fingerprint": self.fingerprint,
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state["fingerprint"] != self.fingerprint:
            raise ValueError("fingerprint mismatch")
        generator_state = state["generator_state"]
        assert isinstance(generator_state, torch.Tensor)
        self.generator.set_state(generator_state)


def test_weighted_loss_preserves_target_count_and_changes_only_denominator() -> None:
    logits = torch.tensor([[[3.0, 0.0], [0.0, 1.0]]])
    targets = torch.tensor([[0, 1]])
    weights = torch.tensor([[1.0, 3.0]])

    unweighted, valid_targets, unweighted_total = (
        training_module._loss_sum_and_count(logits, targets)
    )
    weighted, weighted_targets, weight_total = training_module._loss_sum_and_count(
        logits,
        targets,
        loss_weights=weights,
    )
    per_target = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 2),
        targets.reshape(-1),
        reduction="none",
    )

    torch.testing.assert_close(unweighted, per_target.sum())
    torch.testing.assert_close(weighted, per_target[0] + 3 * per_target[1])
    assert valid_targets == weighted_targets == 2
    assert unweighted_total == 2.0
    assert weight_total == 4.0


class TinyLM(nn.Module):
    def __init__(self, config: GatedCNNConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.projection = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding(input_ids))


def tiny_config(**overrides: object) -> GatedCNNConfig:
    values: dict[str, object] = {
        "vocab_size": 8,
        "context_length": 4,
        "d_model": 4,
        "dropout": 0.0,
        "num_layers": 1,
        "kernel_size": 3,
        "dilations": (1,),
    }
    values.update(overrides)
    return GatedCNNConfig(**values)


def test_train_defaults_device_and_schedule() -> None:
    settings = TrainConfig()
    assert settings.max_steps == 2_000
    assert settings.batch_size == 32
    assert choose_device("cpu") == torch.device("cpu")
    assert learning_rate_at_step(0, settings) == pytest.approx(settings.lr / 100)
    assert learning_rate_at_step(99, settings) == pytest.approx(settings.lr)
    assert learning_rate_at_step(settings.max_steps, settings) == pytest.approx(
        settings.min_lr
    )
    short = TrainConfig(max_steps=10)
    assert learning_rate_at_step(0, short) == pytest.approx(short.lr / 10)
    assert learning_rate_at_step(9, short) == pytest.approx(short.lr)
    token_settings = TrainConfig(
        max_tokens=100, warmup_tokens=10, lr=1.0, min_lr=0.1
    )
    assert learning_rate_at_tokens(5, token_settings) == pytest.approx(0.5)
    assert learning_rate_at_tokens(10, token_settings) == pytest.approx(1.0)
    assert learning_rate_at_tokens(100, token_settings) == pytest.approx(0.1)


def test_token_budget_trims_final_batch_exactly(tmp_path: Path) -> None:
    config = tiny_config()
    result = train(
        config,
        TinyData(config.vocab_size),
        tmp_path,
        TrainConfig(
            max_steps=10,
            max_tokens=13,
            warmup_tokens=4,
            batch_size=2,
            grad_accum_steps=2,
            eval_interval=0,
            checkpoint_interval=1,
            log_interval=0,
            sample_tokens=0,
        ),
        device="cpu",
        model=TinyLM(config),
        log_fn=None,
    )
    checkpoint = torch.load(tmp_path / "latest.pt", weights_only=True)
    assert result["tokens_seen"] == 13
    assert result["stop_reason"] == "max_tokens"
    assert checkpoint["training_state"]["tokens_seen"] == 13


def test_non_cuda_mixed_precision_is_rejected(tmp_path: Path) -> None:
    config = tiny_config()
    with pytest.raises(ValueError, match="only on CUDA"):
        train(
            config,
            TinyData(config.vocab_size),
            tmp_path,
            TrainConfig(max_steps=1, precision="fp16"),
            device="cpu",
            model=TinyLM(config),
            log_fn=None,
        )


def test_token_budget_safety_cap_saves_latest_checkpoint(tmp_path: Path) -> None:
    config = tiny_config()
    with pytest.raises(RuntimeError, match="before max_tokens"):
        train(
            config,
            TinyData(config.vocab_size),
            tmp_path,
            TrainConfig(
                max_steps=1,
                max_tokens=100,
                batch_size=1,
                eval_interval=0,
                checkpoint_interval=0,
                log_interval=0,
                sample_tokens=0,
            ),
            device="cpu",
            model=TinyLM(config),
            log_fn=None,
        )
    assert (tmp_path / "latest.pt").is_file()


def test_evaluate_and_short_training_run(tmp_path) -> None:
    config = tiny_config()
    data = TinyData(config.vocab_size)
    model = TinyLM(config)
    measured = evaluate(
        model,
        data,
        batch_size=2,
        context_length=config.context_length,
        num_batches=2,
        device="cpu",
        generator=torch.Generator(device="cpu").manual_seed(5),
    )
    assert math.isfinite(measured["validation_loss"])
    assert measured["perplexity"] == pytest.approx(
        math.exp(measured["validation_loss"])
    )

    result = train(
        config,
        data,
        tmp_path,
        TrainConfig(
            max_steps=2,
            batch_size=2,
            grad_accum_steps=2,
            lr=1e-2,
            min_lr=1e-3,
            warmup_steps=1,
            eval_interval=1,
            eval_batches=1,
            checkpoint_interval=1,
            log_interval=1,
        ),
        device="cpu",
        model=model,
        log_fn=None,
    )

    assert result["step"] == 2
    assert result["best_validation_loss"] is not None
    assert (tmp_path / "latest.pt").is_file()
    assert (tmp_path / "best.pt").is_file()
    events = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert {event["event"] for event in events} == {"train", "validation"}
    assert all("step" in event for event in events)


def test_checkpoint_round_trip_compatibility_and_rng_restore(tmp_path) -> None:
    config = tiny_config()
    model = TinyLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    data = TinyData(config.vocab_size)
    batch_generator = torch.Generator(device="cpu").manual_seed(123)

    inputs, targets = data.get_batch(
        "train",
        batch_size=2,
        context_length=config.context_length,
        generator=batch_generator,
    )
    loss = torch.nn.functional.cross_entropy(
        model(inputs).reshape(-1, config.vocab_size),
        targets.reshape(-1),
    )
    loss.backward()
    optimizer.step()
    expected_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    random.seed(321)
    torch.manual_seed(321)
    checkpoint_path = save_checkpoint(
        tmp_path / "roundtrip.pt",
        model=model,
        optimizer=optimizer,
        step=7,
        model_config=config,
        train_config=TrainConfig(max_steps=10),
        data_fingerprint=data.fingerprint,
        generators={"train": batch_generator},
        batcher=data,
        metrics={"best_validation_loss": 1.25},
    )
    expected_python_random = random.random()
    expected_torch_random = torch.rand(4)
    expected_batch_random = torch.randint(
        100, (4,), generator=batch_generator
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    random.random()
    torch.rand(10)
    torch.randint(100, (10,), generator=batch_generator)

    loaded = load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        expected_model_config=config,
        expected_data_fingerprint=data.fingerprint,
        generators={"train": batch_generator},
        batcher=data,
    )
    assert loaded["step"] == 7
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, expected_parameters[name])
    assert random.random() == expected_python_random
    assert torch.equal(torch.rand(4), expected_torch_random)
    assert torch.equal(
        torch.randint(100, (4,), generator=batch_generator),
        expected_batch_random,
    )

    with pytest.raises(CheckpointCompatibilityError, match="fingerprint"):
        load_checkpoint(
            checkpoint_path,
            model=model,
            expected_model_config=config,
            expected_data_fingerprint="b" * 64,
        )
    with pytest.raises(CheckpointCompatibilityError, match="configuration"):
        load_checkpoint(
            checkpoint_path,
            model=model,
            expected_model_config=tiny_config(d_model=6),
            expected_data_fingerprint=data.fingerprint,
        )


def test_resume_locks_optimizer_and_sampling_settings(tmp_path) -> None:
    config = tiny_config()
    data = TinyData(config.vocab_size)
    first_settings = TrainConfig(
        max_steps=1,
        batch_size=2,
        warmup_steps=0,
        weight_decay=0.2,
        beta2=0.8,
        eval_interval=1,
        eval_batches=1,
        checkpoint_interval=1,
        log_interval=0,
    )
    train(
        config,
        data,
        tmp_path / "first",
        first_settings,
        device="cpu",
        model=TinyLM(config),
        log_fn=None,
    )

    with pytest.raises(
        CheckpointCompatibilityError,
        match=r"locked fields.*weight_decay",
    ):
        train(
            config,
            TinyData(config.vocab_size),
            tmp_path / "incompatible",
            TrainConfig(
                **{
                    **first_settings.to_dict(),
                    "weight_decay": 0.05,
                }
            ),
            device="cpu",
            resume_from=tmp_path / "first" / "latest.pt",
            model=TinyLM(config),
            log_fn=None,
        )

    resumed = train(
        config,
        TinyData(config.vocab_size),
        tmp_path / "resumed",
        TrainConfig(
            **{
                **first_settings.to_dict(),
                "eval_interval": 0,
                "checkpoint_interval": 0,
                "log_interval": 1,
            }
        ),
        device="cpu",
        resume_from=tmp_path / "first" / "latest.pt",
        model=TinyLM(config),
        log_fn=None,
    )
    assert resumed["step"] == 1


def test_resume_truncates_metrics_ahead_of_checkpoint(tmp_path: Path) -> None:
    config = tiny_config()
    settings = TrainConfig(
        max_steps=1,
        batch_size=2,
        warmup_steps=0,
        eval_interval=1,
        eval_batches=1,
        checkpoint_interval=1,
        log_interval=1,
    )
    output_dir = tmp_path / "run"
    train(
        config,
        TinyData(config.vocab_size),
        output_dir,
        settings,
        device="cpu",
        model=TinyLM(config),
        log_fn=None,
    )
    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"event":"train","step":2,"train_loss":999.0}\n')

    train(
        config,
        TinyData(config.vocab_size),
        output_dir,
        settings,
        device="cpu",
        resume_from=output_dir / "latest.pt",
        model=TinyLM(config),
        log_fn=None,
    )
    records = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert records
    assert all(record["step"] <= 1 for record in records)


def test_resumed_training_matches_uninterrupted_training(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tiny_config()
    base_model = TinyLM(config)
    first_model = deepcopy(base_model)
    uninterrupted_model = deepcopy(base_model)
    settings = TrainConfig(
        max_steps=2,
        batch_size=2,
        lr=1e-2,
        min_lr=1e-3,
        warmup_steps=0,
        eval_interval=1,
        eval_batches=1,
        checkpoint_interval=1,
        log_interval=0,
        seed=77,
    )
    real_save_checkpoint = training_module.save_checkpoint

    class PlannedStop(RuntimeError):
        pass

    def save_then_stop(path, **kwargs):
        saved = real_save_checkpoint(path, **kwargs)
        if Path(path).name == "latest.pt" and kwargs["step"] == 1:
            raise PlannedStop
        return saved

    monkeypatch.setattr(training_module, "save_checkpoint", save_then_stop)
    with pytest.raises(PlannedStop):
        train(
            config,
            TinyData(config.vocab_size),
            tmp_path / "partial",
            settings,
            device="cpu",
            model=first_model,
            log_fn=None,
        )
    monkeypatch.setattr(training_module, "save_checkpoint", real_save_checkpoint)

    resumed_model = TinyLM(config)
    train(
        config,
        TinyData(config.vocab_size),
        tmp_path / "partial",
        settings,
        device="cpu",
        resume_from=tmp_path / "partial" / "latest.pt",
        model=resumed_model,
        log_fn=None,
    )
    train(
        config,
        TinyData(config.vocab_size),
        tmp_path / "uninterrupted",
        settings,
        device="cpu",
        model=uninterrupted_model,
        log_fn=None,
    )

    for resumed_parameter, uninterrupted_parameter in zip(
        resumed_model.parameters(),
        uninterrupted_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(resumed_parameter, uninterrupted_parameter)


class ScriptedLM(nn.Module):
    def __init__(self, *, eos_after_token: int | None = None) -> None:
        super().__init__()
        self.config = SimpleNamespace(context_length=2)
        self.anchor = nn.Parameter(torch.zeros(()))
        self.eos_after_token = eos_after_token
        self.seen_lengths: list[int] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.seen_lengths.append(input_ids.shape[1])
        logits = torch.full(
            (*input_ids.shape, 6),
            -20.0,
            device=input_ids.device,
        )
        logits[..., 1] = 3.0
        logits[..., 2] = 2.0
        if self.eos_after_token is not None:
            logits[..., 1] = -20.0
            next_ids = torch.where(
                input_ids.eq(self.eos_after_token),
                torch.tensor(3, device=input_ids.device),
                torch.tensor(self.eos_after_token, device=input_ids.device),
            )
            logits.scatter_(-1, next_ids.unsqueeze(-1), 20.0)
        return logits


class TinyTokenizer:
    bos_id = 0
    eos_id = 3

    def encode(
        self, text: str, *, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        del text
        result = [0] if add_bos else []
        if add_eos:
            result.append(self.eos_id)
        return result

    def decode(
        self, ids: list[int], *, skip_special_tokens: bool = True
    ) -> str:
        if skip_special_tokens:
            ids = [token_id for token_id in ids if token_id not in {0, 3}]
        return " ".join(str(token_id) for token_id in ids)


def test_greedy_generation_crops_context_and_stops_at_eos() -> None:
    model = ScriptedLM(eos_after_token=2)
    output = generate_tokens(
        model,
        torch.tensor([[0, 4, 5]]),
        max_new_tokens=10,
        temperature=0,
        eos_id=3,
    )
    assert output.tolist() == [[0, 4, 5, 2, 3]]
    assert max(model.seen_lengths) <= model.config.context_length
    assert generate(
        ScriptedLM(eos_after_token=2),
        TinyTokenizer(),
        "Once",
        max_new_tokens=10,
        temperature=0,
    ) == "2"


def test_top_k_sampling_is_seeded_and_restricted() -> None:
    model = ScriptedLM()
    prompt = torch.tensor([0])
    first = generate_tokens(
        model,
        prompt,
        max_new_tokens=12,
        temperature=1.0,
        top_k=2,
        seed=17,
    )
    second = generate_tokens(
        model,
        prompt,
        max_new_tokens=12,
        temperature=1.0,
        top_k=2,
        seed=17,
    )
    assert torch.equal(first, second)
    assert set(first[0, 1:].tolist()) <= {1, 2}


def test_token_and_text_streaming_match_buffered_generation() -> None:
    prompt = torch.tensor([0])
    streamed_tokens = list(
        generate_token_stream(
            ScriptedLM(),
            prompt,
            max_new_tokens=12,
            temperature=1.0,
            top_k=2,
            seed=17,
        )
    )
    buffered_tokens = generate_tokens(
        ScriptedLM(),
        prompt,
        max_new_tokens=12,
        temperature=1.0,
        top_k=2,
        seed=17,
    )
    reconstructed = torch.cat(
        (prompt.unsqueeze(0), *streamed_tokens),
        dim=1,
    )
    torch.testing.assert_close(reconstructed, buffered_tokens)

    streamed_text = "".join(
        generate_stream(
            ScriptedLM(eos_after_token=2),
            TinyTokenizer(),
            "Once",
            max_new_tokens=10,
            temperature=0,
        )
    )
    buffered_text = generate(
        ScriptedLM(eos_after_token=2),
        TinyTokenizer(),
        "Once",
        max_new_tokens=10,
        temperature=0,
    )
    assert streamed_text == buffered_text


def test_closing_token_stream_restores_model_training_mode() -> None:
    model = ScriptedLM().train()
    stream = generate_token_stream(
        model,
        torch.tensor([0]),
        max_new_tokens=10,
        temperature=0,
    )

    next(stream)
    assert not model.training
    stream.close()
    assert model.training
