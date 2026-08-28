from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import kiwilm.data as data_module
from kiwilm.data import (
    DEFAULT_SIMPLESTORIES_DATASET_REVISION,
    PreparedTokenData,
    StoryBatchSampler,
    export_tokenizer_bundle,
    prepare_from_stories,
    prepare_simplestories,
    prepare_tinystories,
)
from kiwilm.tokenizer import ByteBPETokenizer, ReservedTokenError

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
    offset_ids, offsets = tokenizer.encode_with_offsets(text)
    bounded = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert offset_ids == plain
    assert len(offsets) == len(plain)
    assert all(0 <= start <= end <= len(text) for start, end in offsets)
    assert bounded == [tokenizer.bos_id, *plain, tokenizer.eos_id]
    assert tokenizer.decode(bounded) == text

    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = ByteBPETokenizer.load(path)
    assert loaded.encode(text, add_bos=True, add_eos=True) == bounded
    assert loaded.decode(bounded) == text
    with pytest.raises(ReservedTokenError, match="reserved tokenizer control token"):
        loaded.encode("A story containing [EOS] as ordinary source text")


def test_packing_skips_reserved_control_token_documents(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "filtered"
    _prepare_test_data(source_dir)
    metadata = prepare_from_stories(
        output_dir,
        ["first clean story", "malicious [BOS] boundary", "second clean story"],
        ["clean validation story"],
        vocab_size=TEST_VOCAB_SIZE,
        min_frequency=1,
        tokenizer_from=source_dir,
    )
    prepared = PreparedTokenData(output_dir)
    expected = [
        token_id
        for story in ("first clean story", "second clean story")
        for token_id in prepared.tokenizer.encode(
            story,
            add_bos=True,
            add_eos=True,
        )
    ]

    assert metadata["splits"]["train"]["stories"] == 2
    assert metadata["splits"]["train"]["skipped_reserved_token_stories"] == 1
    assert "skipped_reserved_token_stories" not in metadata["splits"]["validation"]
    assert prepared.tokens("train").tolist() == expected


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


def test_simplestories_uses_test_as_validation_and_frozen_tokenizer(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "simplestories"
    source_metadata = _prepare_test_data(source_dir)
    rows = {
        "train": [{"story": "first"}, {"story": "second"}, {"story": "unused"}],
        "test": [{"story": "validation"}, {"story": "unused"}],
    }
    calls: list[dict] = []

    def fake_load_dataset(name: str, **kwargs):
        calls.append({"name": name, **kwargs})
        return iter(rows[kwargs["split"]])

    metadata = prepare_simplestories(
        output_dir,
        tokenizer_from=source_dir,
        resolved_revision=DEFAULT_SIMPLESTORIES_DATASET_REVISION,
        train_limit=2,
        validation_limit=1,
        vocab_size=TEST_VOCAB_SIZE,
        min_frequency=1,
        show_progress=False,
        load_dataset_fn=fake_load_dataset,
    )

    source_tokenizer = source_dir / source_metadata["tokenizer"]["file"]
    target_tokenizer = output_dir / metadata["tokenizer"]["file"]
    assert target_tokenizer.read_bytes() == source_tokenizer.read_bytes()
    assert metadata["dataset"] == {
        "name": "SimpleStories/SimpleStories",
        "requested_revision": DEFAULT_SIMPLESTORIES_DATASET_REVISION,
        "resolved_revision": DEFAULT_SIMPLESTORIES_DATASET_REVISION,
    }
    assert metadata["config"]["text_field"] == "story"
    assert metadata["splits"]["train"]["stories"] == 2
    assert metadata["splits"]["validation"]["stories"] == 1
    assert [call["split"] for call in calls] == ["train", "test"]
    assert all(call["streaming"] is True for call in calls)
    assert metadata["tokenizer"]["reused_from"] == {
        "dataset_fingerprint": source_metadata["fingerprint"],
        "tokenizer_sha256": source_metadata["tokenizer"]["sha256"],
    }


def test_simplestories_injected_loader_requires_resolved_revision(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    _prepare_test_data(source_dir)

    with pytest.raises(ValueError, match="resolved_revision is required"):
        prepare_simplestories(
            tmp_path / "target",
            tokenizer_from=source_dir,
            revision="main",
            load_dataset_fn=lambda *_args, **_kwargs: (),
        )


def test_frozen_tokenizer_reuses_exact_bytes_ids_and_stable_provenance(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    source_metadata = _prepare_test_data(source_dir)
    stories = ["A new story.", "A second new story."]
    validation = ["A frozen-tokenizer validation story."]

    first = prepare_from_stories(
        first_dir,
        stories,
        validation,
        vocab_size=TEST_VOCAB_SIZE,
        min_frequency=1,
        tokenizer_from=source_dir,
    )
    second = prepare_from_stories(
        second_dir,
        stories,
        validation,
        vocab_size=TEST_VOCAB_SIZE,
        min_frequency=1,
        tokenizer_from=source_dir,
    )
    source_bytes = (source_dir / source_metadata["tokenizer"]["file"]).read_bytes()
    first_bytes = (first_dir / first["tokenizer"]["file"]).read_bytes()
    prepared = PreparedTokenData(first_dir)

    assert first_bytes == source_bytes
    assert first["tokenizer"]["sha256"] == source_metadata["tokenizer"]["sha256"]
    assert first["tokenizer"]["special_tokens"] == source_metadata["tokenizer"][
        "special_tokens"
    ]
    assert first["tokenizer"]["reused_from"] == {
        "dataset_fingerprint": source_metadata["fingerprint"],
        "tokenizer_sha256": source_metadata["tokenizer"]["sha256"],
    }
    assert first["fingerprint"] == second["fingerprint"]
    assert str(source_dir) not in json.dumps(first)
    assert prepared.tokens("train").tolist() == [
        token_id
        for story in stories
        for token_id in prepared.tokenizer.encode(
            story,
            add_bos=True,
            add_eos=True,
        )
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"vocab_size": TEST_VOCAB_SIZE + 1}, "vocab_size conflicts"),
        ({"min_frequency": 2}, "min_frequency conflicts"),
    ],
)
def test_frozen_tokenizer_rejects_conflicting_options(
    tmp_path: Path,
    overrides: dict[str, int],
    message: str,
) -> None:
    source_dir = tmp_path / "source"
    _prepare_test_data(source_dir)
    options = {
        "vocab_size": TEST_VOCAB_SIZE,
        "min_frequency": 1,
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        prepare_from_stories(
            tmp_path / "target",
            ["train"],
            ["validation"],
            tokenizer_from=source_dir,
            **options,
        )


def test_frozen_tokenizer_rejects_corrupt_source_and_same_output(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    metadata = _prepare_test_data(source_dir)
    tokenizer_path = source_dir / metadata["tokenizer"]["file"]
    tokenizer_path.write_bytes(tokenizer_path.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="tokenizer checksum mismatch"):
        prepare_from_stories(
            tmp_path / "target",
            ["train"],
            ["validation"],
            vocab_size=TEST_VOCAB_SIZE,
            min_frequency=1,
            tokenizer_from=source_dir,
        )

    with pytest.raises(ValueError, match="source and preparation output"):
        prepare_from_stories(
            source_dir,
            ["train"],
            ["validation"],
            vocab_size=TEST_VOCAB_SIZE,
            min_frequency=1,
            tokenizer_from=source_dir,
            force=True,
        )


def test_frozen_tokenizer_streaming_loads_each_split_once(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    _prepare_test_data(source_dir)
    rows = {
        "train": [{"text": "first"}, {"text": "second"}],
        "validation": [{"text": "validation"}],
    }
    calls: list[str] = []

    def fake_load_dataset(_name: str, **kwargs):
        calls.append(kwargs["split"])
        return iter(rows[kwargs["split"]])

    metadata = prepare_tinystories(
        tmp_path / "target",
        revision="revision-name",
        resolved_revision="resolved-sha",
        train_limit=2,
        validation_limit=1,
        vocab_size=TEST_VOCAB_SIZE,
        min_frequency=1,
        show_progress=False,
        load_dataset_fn=fake_load_dataset,
        tokenizer_from=source_dir,
    )

    assert calls == ["train", "validation"]
    assert metadata["splits"]["train"]["stories"] == 2
    assert metadata["splits"]["validation"]["stories"] == 1


def test_portable_tokenizer_bundle_reuses_exact_bytes_and_provenance(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    bundle_dir = tmp_path / "bundle"
    target_dir = tmp_path / "target"
    source_metadata = _prepare_test_data(source_dir)
    bundle = export_tokenizer_bundle(source_dir, bundle_dir)

    target_metadata = prepare_from_stories(
        target_dir,
        ["A remotely prepared story."],
        ["A validation story."],
        vocab_size=TEST_VOCAB_SIZE,
        min_frequency=1,
        tokenizer_from=bundle_dir,
    )

    source_bytes = (source_dir / source_metadata["tokenizer"]["file"]).read_bytes()
    target_bytes = (target_dir / target_metadata["tokenizer"]["file"]).read_bytes()
    assert target_bytes == source_bytes
    assert target_metadata["tokenizer"]["reused_from"] == {
        "dataset_fingerprint": source_metadata["fingerprint"],
        "tokenizer_sha256": source_metadata["tokenizer"]["sha256"],
    }
    assert bundle["source_dataset_fingerprint"] == source_metadata["fingerprint"]
    assert str(source_dir) not in json.dumps(bundle)


def test_portable_tokenizer_bundle_rejects_corruption_and_existing_target(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    bundle_dir = tmp_path / "bundle"
    _prepare_test_data(source_dir)
    bundle = export_tokenizer_bundle(source_dir, bundle_dir)

    with pytest.raises(FileExistsError, match="force=True"):
        export_tokenizer_bundle(source_dir, bundle_dir)

    tokenizer_path = bundle_dir / bundle["tokenizer"]["file"]
    tokenizer_path.write_bytes(tokenizer_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="bundle checksum mismatch"):
        prepare_from_stories(
            tmp_path / "target",
            ["train"],
            ["validation"],
            vocab_size=TEST_VOCAB_SIZE,
            min_frequency=1,
            tokenizer_from=bundle_dir,
        )


def test_portable_tokenizer_bundle_rejects_manifest_tampering(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    bundle_dir = tmp_path / "bundle"
    _prepare_test_data(source_dir)
    export_tokenizer_bundle(source_dir, bundle_dir)
    manifest_path = bundle_dir / "tokenizer-bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_dataset_fingerprint"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="bundle fingerprint mismatch"):
        prepare_from_stories(
            tmp_path / "target",
            ["train"],
            ["validation"],
            vocab_size=TEST_VOCAB_SIZE,
            min_frequency=1,
            tokenizer_from=bundle_dir,
        )


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
