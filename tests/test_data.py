from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import kiwilm.data as data_module
from kiwilm.data import (
    PreparedTokenData,
    StoryBatchSampler,
    prepare_from_stories,
    prepare_tinystories,
)
from kiwilm.tokenizer import ByteBPETokenizer

TEST_VOCAB_SIZE = 300


def _prepare_test_data(path: Path) -> dict:
    return prepare_from_stories(
        path,
        ["A tiny story.", "Another tiny story!"],
        ["A validation story."],
        vocab_size=TEST_VOCAB_SIZE,
        min_frequency=1,
    )


def test_byte_bpe_round_trip_special_tokens_and_json(tmp_path: Path) -> None:
    tokenizer = ByteBPETokenizer.train(
        ["Hello world", "Привет 👋"],
        vocab_size=TEST_VOCAB_SIZE,
        min_frequency=1,
    )

    assert (tokenizer.pad_id, tokenizer.unk_id, tokenizer.bos_id, tokenizer.eos_id) == (
        0,
        1,
        2,
        3,
    )
    text = "Привет 👋"
    plain = tokenizer.encode(text)
    bounded = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert bounded == [tokenizer.bos_id, *plain, tokenizer.eos_id]
    assert tokenizer.decode(bounded) == text

    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = ByteBPETokenizer.load(path)
    assert loaded.encode(text, add_bos=True, add_eos=True) == bounded
    assert loaded.decode(bounded) == text
    with pytest.raises(ValueError, match="reserved tokenizer control token"):
        loaded.encode("A story containing [EOS] as ordinary source text")


def test_byte_bpe_streaming_decode_preserves_split_utf8() -> None:
    tokenizer = ByteBPETokenizer.train(
        ["Hello € 😊"],
        vocab_size=260,
        min_frequency=1,
    )
    text = "Hello € 😊"
    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)

    chunks = list(tokenizer.decode_stream(token_ids))

    assert "".join(chunks) == text
    assert tokenizer.decode(token_ids) == text


def test_packing_preserves_story_boundaries_and_shifted_batches(
    tmp_path: Path,
) -> None:
    stories = ["first", "second"]
    metadata = prepare_from_stories(
        tmp_path,
        stories,
        ["validation"],
        vocab_size=TEST_VOCAB_SIZE,
        min_frequency=1,
    )
    prepared = PreparedTokenData(tmp_path, seed=7)
    expected = [
        token_id
        for story in stories
        for token_id in prepared.tokenizer.encode(
            story,
            add_bos=True,
            add_eos=True,
        )
    ]

    assert prepared.tokens("train").tolist() == expected
    assert metadata["splits"]["train"]["stories"] == 2
    assert metadata["splits"]["train"]["tokens"] == len(expected)
    first_length = len(
        prepared.tokenizer.encode(stories[0], add_bos=True, add_eos=True)
    )
    assert expected[first_length - 1] == prepared.tokenizer.eos_id
    assert expected[first_length] == prepared.tokenizer.bos_id

    context_length = len(expected) - 1
    inputs, targets = prepared.get_batch(
        "train",
        batch_size=1,
        context_length=context_length,
    )
    assert inputs.tolist() == [expected[:-1]]
    assert targets.tolist() == [expected[1:]]


def test_batches_and_rng_resume_are_deterministic(tmp_path: Path) -> None:
    _prepare_test_data(tmp_path)
    first = PreparedTokenData(tmp_path, seed=123)
    second = PreparedTokenData(tmp_path, seed=123)

    batch_a = first.get_batch("train", batch_size=4, context_length=3)
    batch_b = second.get_batch("train", batch_size=4, context_length=3)
    assert torch.equal(batch_a[0], batch_b[0])
    assert torch.equal(batch_a[1], batch_b[1])

    state = first.state_dict()
    expected = first.get_batch("train", batch_size=3, context_length=4)
    second.load_state_dict(state)
    resumed = second.get_batch("train", batch_size=3, context_length=4)
    assert torch.equal(expected[0], resumed[0])
    assert torch.equal(expected[1], resumed[1])
    assert torch.equal(expected[0][:, 1:], expected[1][:, :-1])


def test_story_batches_cover_targets_once_without_crossing_boundaries(
    tmp_path: Path,
) -> None:
    stories = ["a", "a somewhat longer story", ""]
    prepare_from_stories(
        tmp_path,
        stories,
        ["validation"],
        vocab_size=TEST_VOCAB_SIZE,
        min_frequency=1,
    )
    data = PreparedTokenData(tmp_path)
    chunks = data.story_chunks("train", context_length=3)
    sampler = StoryBatchSampler(
        data, "train", context_length=3, seed=17, shuffle=False
    )
    inputs, targets = sampler.get_batch(batch_size=len(chunks))

    expected_targets = sum(
        len(data.tokenizer.encode(story, add_bos=True, add_eos=True)) - 1
        for story in stories
    )
    assert targets.ne(-100).sum().item() == expected_targets
    assert inputs.shape == targets.shape == (len(chunks), 3)
    for row in range(len(chunks)):
        valid = targets[row].ne(-100)
        valid_count = int(valid.sum())
        assert torch.equal(
            inputs[row, 1:valid_count],
            targets[row, : valid_count - 1],
        )
        assert torch.all(inputs[row, ~valid] == data.tokenizer.pad_id)


def test_story_sampler_resume_and_corrupt_offset_cache_rebuild(
    tmp_path: Path,
) -> None:
    metadata = _prepare_test_data(tmp_path)
    data = PreparedTokenData(tmp_path)
    sampler = StoryBatchSampler(data, "train", context_length=3, seed=9)
    sampler.next_indices(1)
    state = sampler.state_dict()
    expected = sampler.next_indices(4)

    resumed = StoryBatchSampler(data, "train", context_length=3, seed=9)
    resumed.load_state_dict(state)
    assert torch.equal(resumed.next_indices(4), expected)

    split_sha = metadata["splits"]["train"]["sha256"]
    cache_path = tmp_path / f"train-story-offsets-{split_sha}.npy"
    cache_path.write_bytes(b"broken")
    reloaded = PreparedTokenData(tmp_path)
    offsets = reloaded.story_offsets("train")
    assert len(offsets) == metadata["splits"]["train"]["stories"]


def test_metadata_fingerprint_and_binary_corruption_are_rejected(
    tmp_path: Path,
) -> None:
    metadata = _prepare_test_data(tmp_path)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        PreparedTokenData(tmp_path, expected_fingerprint="0" * 64)

    metadata_path = tmp_path / "metadata.json"
    changed = json.loads(metadata_path.read_text(encoding="utf-8"))
    changed["config"]["train_limit"] = 99
    metadata_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata fingerprint mismatch"):
        PreparedTokenData(tmp_path)

    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    train_path = tmp_path / metadata["splits"]["train"]["file"]
    contents = bytearray(train_path.read_bytes())
    contents[0] ^= 1
    train_path.write_bytes(contents)
    with pytest.raises(ValueError, match="train checksum mismatch"):
        PreparedTokenData(tmp_path)


def test_tinystories_loader_is_streaming_capped_and_train_only(
    tmp_path: Path,
) -> None:
    rows = {
        "train": [{"text": "train-only-token"} for _ in range(4)],
        "validation": [{"text": "validation-only-token"} for _ in range(3)],
    }
    calls: list[dict] = []

    def fake_load_dataset(name: str, **kwargs):
        calls.append({"name": name, **kwargs})
        return iter(rows[kwargs["split"]])

    metadata = prepare_tinystories(
        tmp_path,
        revision="revision-name",
        resolved_revision="resolved-sha",
        train_limit=2,
        validation_limit=1,
        vocab_size=TEST_VOCAB_SIZE,
        min_frequency=1,
        show_progress=False,
        load_dataset_fn=fake_load_dataset,
    )

    assert metadata["dataset"] == {
        "name": "roneneldan/TinyStories",
        "requested_revision": "revision-name",
        "resolved_revision": "resolved-sha",
    }
    assert metadata["splits"]["train"]["stories"] == 2
    assert metadata["splits"]["validation"]["stories"] == 1
    assert [call["split"] for call in calls] == ["train", "train", "validation"]
    assert all(call["streaming"] is True for call in calls)
    assert all(call["revision"] == "resolved-sha" for call in calls)


def test_vocab_and_context_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="65535"):
        ByteBPETokenizer.train(["story"], vocab_size=65_536)
    _prepare_test_data(tmp_path)
    prepared = PreparedTokenData(tmp_path)
    with pytest.raises(ValueError, match="needs at least"):
        prepared.get_batch(
            "validation",
            batch_size=1,
            context_length=len(prepared.tokens("validation")),
        )


def test_preparation_requires_force_to_replace_existing_data(tmp_path: Path) -> None:
    original = _prepare_test_data(tmp_path)
    original_files = {
        tmp_path / original["tokenizer"]["file"],
        tmp_path / original["splits"]["train"]["file"],
        tmp_path / original["splits"]["validation"]["file"],
    }
    with pytest.raises(FileExistsError, match="force=True"):
        _prepare_test_data(tmp_path)

    replaced = prepare_from_stories(
        tmp_path,
        ["replacement training story"],
        ["replacement validation story"],
        vocab_size=TEST_VOCAB_SIZE,
        min_frequency=1,
        force=True,
    )
    assert replaced["fingerprint"] != original["fingerprint"]
    assert PreparedTokenData(tmp_path).fingerprint == replaced["fingerprint"]
    replacement_files = {
        tmp_path / replaced["tokenizer"]["file"],
        tmp_path / replaced["splits"]["train"]["file"],
        tmp_path / replaced["splits"]["validation"]["file"],
    }
    assert not any(path.exists() for path in original_files - replacement_files)
    assert replaced["config"]["streaming"] is False


def test_existing_target_is_rejected_before_consuming_sources_or_loading(
    tmp_path: Path,
) -> None:
    _prepare_test_data(tmp_path)
    consumed = 0

    def stories():
        nonlocal consumed
        consumed += 1
        yield "must not be consumed"

    with pytest.raises(FileExistsError, match="force=True"):
        prepare_from_stories(tmp_path, stories(), stories())
    assert consumed == 0

    loader_called = False

    def fake_load_dataset(*args, **kwargs):
        nonlocal loader_called
        loader_called = True
        return []

    with pytest.raises(FileExistsError, match="force=True"):
        prepare_tinystories(tmp_path, load_dataset_fn=fake_load_dataset)
    assert loader_called is False


def test_local_caps_do_not_consume_entire_iterables(tmp_path: Path) -> None:
    consumed = 0

    def training_stories():
        nonlocal consumed
        while True:
            consumed += 1
            yield f"story {consumed}"

    metadata = prepare_from_stories(
        tmp_path,
        training_stories(),
        ["validation story", "unused validation story"],
        train_limit=2,
        validation_limit=1,
        vocab_size=TEST_VOCAB_SIZE,
        min_frequency=1,
    )

    assert consumed == 2
    assert metadata["splits"]["train"]["stories"] == 2
    assert metadata["splits"]["validation"]["stories"] == 1


def test_failed_forced_commit_preserves_previous_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _prepare_test_data(tmp_path)
    original_replace = data_module.os.replace
    replacement_count = 0

    def fail_during_artifact_commit(source, destination) -> None:
        nonlocal replacement_count
        replacement_count += 1
        if replacement_count == 2:
            raise OSError("simulated commit failure")
        original_replace(source, destination)

    monkeypatch.setattr(data_module.os, "replace", fail_during_artifact_commit)
    with pytest.raises(OSError, match="simulated commit failure"):
        prepare_from_stories(
            tmp_path,
            ["replacement training story"],
            ["replacement validation story"],
            vocab_size=TEST_VOCAB_SIZE,
            min_frequency=1,
            force=True,
        )

    assert PreparedTokenData(tmp_path).fingerprint == original["fingerprint"]
