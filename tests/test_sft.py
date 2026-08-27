"""Preparation, masking, warm-start, and CLI coverage for SFT."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from kiwilm.checkpoint import save_checkpoint
from kiwilm.config import KiwiLM2Config
from kiwilm.data import prepare_from_stories
from kiwilm.models import build_model
from kiwilm.sft import (
    DEFAULT_REQUIRED_WORD_WEIGHT,
    INSTRUCT_RECORD_SEPARATOR,
    SFT_FORMAT_V2,
    SFT_V2_INSTRUCTION,
    PreparedSFTData,
    SFTBatchSampler,
    iter_raw_instruct_records,
    load_prepared_data,
    parse_tinystories_instruct_record,
    prepare_instruct_from_records,
)
from kiwilm.training import TrainConfig, evaluate, train

TEST_VOCAB_SIZE = 300

RECORD_A = """\
Features: Dialogue, MoralValue
Words: oak, gloomy, kind
Summary: Sara and Ben help each other get home.
Story:
Sara and Ben sat by the old oak tree.
"It is getting gloomy," Sara said.
Ben was kind and helped her get home.
"""

RECORD_B = """\
Words: red, ball, share
Random sentence: The red ball rolled under the chair.
Story:
Lily gave Ben a red ball.
Ben shared the ball with Lily.
Summary: Lily and Ben learn to share a red ball.
"""


def _tokenizer_source(path: Path) -> dict:
    return prepare_from_stories(
        path,
        [RECORD_A, RECORD_B] * 4,
        [RECORD_A, RECORD_B],
        vocab_size=TEST_VOCAB_SIZE,
        min_frequency=1,
    )


def _prepare_sft(path: Path, tokenizer_source: Path) -> dict:
    return prepare_instruct_from_records(
        path,
        train_records=[RECORD_A, RECORD_B] * 3,
        validation_records=[RECORD_A, RECORD_B],
        tokenizer_from=tokenizer_source,
        dataset_name="test/instruct",
        revision="abc123",
        train_limit=4,
        validation_limit=2,
    )


def _prepare_sft_v2(path: Path, tokenizer_source: Path) -> dict:
    return prepare_instruct_from_records(
        path,
        train_records=[RECORD_A, RECORD_B] * 3,
        validation_records=[RECORD_A, RECORD_B],
        tokenizer_from=tokenizer_source,
        dataset_name="test/instruct",
        revision="abc123",
        train_limit=4,
        validation_limit=2,
        sft_format=SFT_FORMAT_V2,
    )


def test_parser_canonicalizes_fields_and_removes_trailing_metadata() -> None:
    example = parse_tinystories_instruct_record(RECORD_B)

    assert example.prompt == (
        "Words: red, ball, share\n"
        "Summary: Lily and Ben learn to share a red ball.\n"
        "Random sentence: The red ball rolled under the chair.\n"
        "Story:\n"
    )
    assert example.response == (
        "Lily gave Ben a red ball.\nBen shared the ball with Lily."
    )
    assert example.required_words == ("red", "ball", "share")
    v2 = parse_tinystories_instruct_record(RECORD_B, sft_format=SFT_FORMAT_V2)
    assert v2.prompt == SFT_V2_INSTRUCTION + example.prompt
    with pytest.raises(ValueError, match="Story"):
        parse_tinystories_instruct_record("Words: one, two")
    with pytest.raises(ValueError, match="conditioning"):
        parse_tinystories_instruct_record("Story:\nA story.")


def test_raw_record_stream_handles_chunk_boundaries_and_final_record(
    tmp_path: Path,
) -> None:
    source = tmp_path / "records.txt"
    source.write_text(
        RECORD_A
        + INSTRUCT_RECORD_SEPARATOR
        + "\n"
        + RECORD_B
        + INSTRUCT_RECORD_SEPARATOR,
        encoding="utf-8",
    )

    records = list(iter_raw_instruct_records(source))

    assert len(records) == 2
    assert "Sara and Ben" in records[0]
    assert "Lily gave Ben" in records[1]


def test_raw_record_stream_replaces_malformed_utf8_without_losing_boundaries(
    tmp_path: Path,
) -> None:
    source = tmp_path / "malformed.txt"
    source.write_bytes(
        b"Words: kind\nStory:\nA kind quote: \xe2\x80\n"
        + INSTRUCT_RECORD_SEPARATOR.encode()
        + b"\nWords: red\nStory:\nA red ball.\n"
        + INSTRUCT_RECORD_SEPARATOR.encode()
    )

    records = list(iter_raw_instruct_records(source))

    assert len(records) == 2
    assert "\ufffd" in records[0]
    assert "A red ball." in records[1]


def test_preparation_reuses_tokenizer_and_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source_metadata = _tokenizer_source(source)
    first_metadata = _prepare_sft(tmp_path / "first", source)
    second_metadata = _prepare_sft(tmp_path / "second", source)

    assert first_metadata == second_metadata
    assert first_metadata["splits"]["train"]["records"] == 4
    assert first_metadata["splits"]["validation"]["records"] == 2
    assert first_metadata["splits"]["train"]["response_targets"] > 0
    assert first_metadata["config"]["source_decoding"] == "utf-8-replace"
    assert first_metadata["tokenizer"]["reused_from"] == {
        "dataset_fingerprint": source_metadata["fingerprint"],
        "tokenizer_sha256": source_metadata["tokenizer"]["sha256"],
    }
    source_tokenizer = (source / source_metadata["tokenizer"]["file"]).read_bytes()
    prepared_tokenizer = (
        tmp_path / "first" / first_metadata["tokenizer"]["file"]
    ).read_bytes()
    assert prepared_tokenizer == source_tokenizer
    assert isinstance(load_prepared_data(tmp_path / "first"), PreparedSFTData)


def test_v2_preparation_adds_constraint_masks_and_weighted_batches(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _tokenizer_source(source)
    v1_metadata = _prepare_sft(tmp_path / "v1", source)
    v2_metadata = _prepare_sft_v2(tmp_path / "v2", source)
    data = PreparedSFTData(tmp_path / "v2")
    chunks = data.chunks("train", context_length=8)
    inputs, targets, loss_weights = data.sft_batch(
        "train",
        list(range(len(chunks))),
        context_length=8,
        include_loss_weights=True,
    )

    assert v2_metadata["fingerprint"] != v1_metadata["fingerprint"]
    assert v2_metadata["config"]["sft_format"] == SFT_FORMAT_V2
    assert (
        v2_metadata["config"]["required_word_weight"]
        == DEFAULT_REQUIRED_WORD_WEIGHT
    )
    assert v2_metadata["splits"]["train"]["constraint_targets"] > 0
    # "share" does not count as an exact occurrence inside "shared".
    assert v2_metadata["splits"]["train"]["matched_required_words"] == 10
    assert data.format_prompt("Features: Dialogue\nStory:\n").startswith(
        SFT_V2_INSTRUCTION
    )
    assert data.format_prompt(SFT_V2_INSTRUCTION + "Story:\n").count(
        SFT_V2_INSTRUCTION
    ) == 1
    assert inputs.shape == targets.shape == loss_weights.shape
    supervised = targets.ne(-100)
    assert torch.all(loss_weights[~supervised] == 0)
    assert torch.all(loss_weights[supervised] >= 1)
    assert int(loss_weights.eq(DEFAULT_REQUIRED_WORD_WEIGHT).sum()) == v2_metadata[
        "splits"
    ]["train"]["constraint_targets"]

    mask_path = (
        tmp_path
        / "v2"
        / v2_metadata["splits"]["train"]["constraint_mask_file"]
    )
    mask_path.write_bytes(mask_path.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="constraint mask"):
        PreparedSFTData(tmp_path / "v2")


def test_preparation_skips_incomplete_fragments_and_limits_valid_records(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _tokenizer_source(source)
    metadata = prepare_instruct_from_records(
        tmp_path / "sft",
        train_records=["tail of a previous story", RECORD_A, RECORD_B],
        validation_records=[RECORD_A],
        tokenizer_from=source,
        train_limit=2,
        validation_limit=1,
    )

    assert metadata["splits"]["train"]["records"] == 2
    assert metadata["splits"]["train"]["skipped_records"] == 1
    assert metadata["splits"]["validation"]["skipped_records"] == 0


def test_response_chunks_cover_targets_once_and_mask_prompts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _tokenizer_source(source)
    metadata = _prepare_sft(tmp_path / "sft", source)
    data = PreparedSFTData(tmp_path / "sft")
    chunks = data.chunks("validation", context_length=8)
    inputs, targets = data.sft_batch(
        "validation",
        list(range(len(chunks))),
        context_length=8,
    )
    valid_targets = targets[targets.ne(-100)].tolist()
    expected: list[int] = []
    for record in (RECORD_A, RECORD_B):
        example = parse_tinystories_instruct_record(record)
        expected.extend(data.tokenizer.encode(example.response))
        expected.append(data.tokenizer.eos_id)

    assert valid_targets == expected
    assert len(valid_targets) == metadata["splits"]["validation"]["response_targets"]
    assert bool(targets.eq(-100).any())
    assert inputs.shape == targets.shape == (len(chunks), 8)


def test_sampling_is_seeded_and_integrity_is_validated(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _tokenizer_source(source)
    metadata = _prepare_sft(tmp_path / "sft", source)
    first = PreparedSFTData(tmp_path / "sft", seed=9)
    second = PreparedSFTData(tmp_path / "sft", seed=9)

    first_batch = first.get_batch(
        "train",
        batch_size=3,
        context_length=16,
    )
    second_batch = second.get_batch(
        "train",
        batch_size=3,
        context_length=16,
    )

    assert all(
        torch.equal(left, right)
        for left, right in zip(first_batch, second_batch, strict=True)
    )
    token_path = tmp_path / "sft" / metadata["splits"]["train"]["token_file"]
    token_path.write_bytes(token_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        PreparedSFTData(tmp_path / "sft")


def test_sft_sampler_covers_an_epoch_once_and_resumes_exactly(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _tokenizer_source(source)
    metadata = _prepare_sft(tmp_path / "sft", source)
    data = PreparedSFTData(tmp_path / "sft")
    chunks = data.chunks("train", context_length=8)
    sequential = SFTBatchSampler(
        data,
        "train",
        context_length=8,
        seed=9,
        shuffle=False,
    )
    _, targets = sequential.get_batch(batch_size=len(chunks))

    assert int(targets.ne(-100).sum()) == metadata["splits"]["train"][
        "response_targets"
    ]

    shuffled = SFTBatchSampler(data, "train", context_length=8, seed=17)
    shuffled.next_indices(3)
    state = shuffled.state_dict()
    expected = shuffled.next_indices(len(chunks) + 2)
    resumed = SFTBatchSampler(data, "train", context_length=8, seed=17)
    resumed.load_state_dict(state)

    assert torch.equal(resumed.next_indices(len(chunks) + 2), expected)


def test_sft_evaluation_uses_the_same_seeded_chunks_every_time(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _tokenizer_source(source)
    _prepare_sft(tmp_path / "sft", source)
    data = PreparedSFTData(tmp_path / "sft")
    config = KiwiLM2Config(
        vocab_size=data.tokenizer.vocab_size,
        context_length=16,
        d_model=16,
        dropout=0.0,
        num_query_heads=2,
        num_kv_heads=1,
        swiglu_dim=24,
        bigram_buckets=16,
        trigram_buckets=16,
    )
    model = build_model(config)
    first_generator = torch.Generator().manual_seed(1)
    second_generator = torch.Generator().manual_seed(999)
    torch.rand(20, generator=first_generator)

    first = evaluate(
        model,
        data,
        batch_size=2,
        context_length=16,
        num_batches=3,
        device="cpu",
        generator=first_generator,
        batch_mode="sft",
        seed=43,
    )
    second = evaluate(
        model,
        data,
        batch_size=2,
        context_length=16,
        num_batches=3,
        device="cpu",
        generator=second_generator,
        batch_mode="sft",
        seed=43,
    )

    assert first == second


def test_weight_only_sft_warm_start_and_exact_token_budget(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _tokenizer_source(source)
    _prepare_sft(tmp_path / "sft", source)
    data = PreparedSFTData(tmp_path / "sft")
    config = KiwiLM2Config(
        vocab_size=data.tokenizer.vocab_size,
        context_length=16,
        d_model=16,
        dropout=0.0,
        num_query_heads=2,
        num_kv_heads=1,
        swiglu_dim=24,
        bigram_buckets=16,
        trigram_buckets=16,
    )
    source_checkpoint = save_checkpoint(
        tmp_path / "source.pt",
        model=build_model(config),
        step=17,
        model_config=config,
        data_fingerprint="p" * 64,
    )
    result = train(
        config,
        data,
        tmp_path / "run",
        TrainConfig(
            max_steps=10,
            batch_size=2,
            grad_accum_steps=2,
            lr=1e-4,
            min_lr=1e-5,
            warmup_steps=0,
            max_tokens=13,
            warmup_tokens=2,
            batch_mode="sft",
            eval_mode="sft",
            eval_interval=1,
            eval_batches=1,
            checkpoint_interval=1,
            log_interval=0,
            sample_tokens=0,
        ),
        init_from=source_checkpoint,
        device="cpu",
        log_fn=None,
    )
    checkpoint = torch.load(tmp_path / "run" / "latest.pt", weights_only=True)

    assert result["tokens_seen"] == 13
    assert result["stop_reason"] == "max_tokens"
    assert result["initialization"]["source_step"] == 17
    assert checkpoint["training_state"]["initialization"] == result["initialization"]
    assert checkpoint["data_fingerprint"] == data.fingerprint
    assert checkpoint["train_config"]["batch_mode"] == "sft"
    assert checkpoint["step"] < 17


def test_v2_weighted_sft_trains_to_an_exact_unweighted_token_budget(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _tokenizer_source(source)
    _prepare_sft_v2(tmp_path / "sft-v2", source)
    data = PreparedSFTData(tmp_path / "sft-v2")
    config = KiwiLM2Config(
        vocab_size=data.tokenizer.vocab_size,
        context_length=16,
        d_model=16,
        dropout=0.0,
        num_query_heads=2,
        num_kv_heads=1,
        swiglu_dim=24,
        bigram_buckets=16,
        trigram_buckets=16,
    )

    result = train(
        config,
        data,
        tmp_path / "run-v2",
        TrainConfig(
            max_steps=5,
            batch_size=2,
            grad_accum_steps=2,
            lr=1e-4,
            min_lr=1e-5,
            warmup_steps=0,
            max_tokens=13,
            warmup_tokens=2,
            batch_mode="sft",
            eval_mode="sft",
            eval_interval=1,
            eval_batches=1,
            checkpoint_interval=1,
            log_interval=0,
            sample_tokens=0,
        ),
        device="cpu",
        log_fn=None,
    )

    assert result["tokens_seen"] == 13
    assert result["stop_reason"] == "max_tokens"
    assert math.isfinite(result["best_validation_loss"])


def test_preparation_rejects_existing_target_and_invalid_limits(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _tokenizer_source(source)
    _prepare_sft(tmp_path / "sft", source)

    with pytest.raises(FileExistsError, match="force=True"):
        _prepare_sft(tmp_path / "sft", source)
    with pytest.raises(ValueError, match="non-negative"):
        prepare_instruct_from_records(
            tmp_path / "negative",
            train_records=[RECORD_A],
            validation_records=[RECORD_B],
            tokenizer_from=source,
            train_limit=-1,
        )


def test_metadata_contains_no_machine_specific_tokenizer_path(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _tokenizer_source(source)
    metadata = _prepare_sft(tmp_path / "sft", source)

    assert str(source) not in json.dumps(metadata)
