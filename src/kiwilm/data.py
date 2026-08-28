"""TinyStories preparation and deterministic packed-token sampling."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import tempfile
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from kiwilm.tokenizer import (
    MAX_UINT16_VOCAB_SIZE,
    SPECIAL_TOKENS,
    ByteBPETokenizer,
    ReservedTokenError,
)

DEFAULT_DATASET_NAME = "roneneldan/TinyStories"
DEFAULT_DATASET_REVISION = "main"
DEFAULT_TRAIN_LIMIT = 25_000
DEFAULT_VALIDATION_LIMIT = 2_000
DEFAULT_SIMPLESTORIES_DATASET_NAME = "SimpleStories/SimpleStories"
DEFAULT_SIMPLESTORIES_DATASET_REVISION = "e63b8adc3b1a1bdc7cac5b500d150b71346b0628"
DEFAULT_SIMPLESTORIES_TRAIN_LIMIT = 250_000
DEFAULT_SIMPLESTORIES_VALIDATION_LIMIT = 10_000
DEFAULT_VOCAB_SIZE = 8_192
DEFAULT_SMOLLM_DATASET_NAME = "HuggingFaceTB/smollm-corpus"
DEFAULT_SMOLLM_DATASET_REVISION = "main"
DEFAULT_SMOLLM_FINEWEB_CONFIG = "fineweb-edu-dedup"
DEFAULT_SMOLLM_COSMOPEDIA_CONFIG = "cosmopedia-v2"
DEFAULT_SMOLLM_VOCAB_SIZE = 32_000
METADATA_SCHEMA_VERSION = 1
TOKENIZER_BUNDLE_SCHEMA_VERSION = 1
TOKENIZER_BUNDLE_FILE = "tokenizer-bundle.json"
TOKEN_DTYPE = "uint16"
TOKEN_BYTE_ORDER = "little"
SPLITS = ("train", "validation")

Story = str | Mapping[str, Any]
StoryFactory = Callable[[], Iterable[Story]]
SplitName = Literal["train", "validation"]
BatchMode = Literal["packed", "story"]


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def metadata_fingerprint(metadata: Mapping[str, Any]) -> str:
    """Return the deterministic SHA-256 fingerprint for metadata contents."""

    payload = dict(metadata)
    payload.pop("fingerprint", None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def tokenizer_bundle_fingerprint(bundle: Mapping[str, Any]) -> str:
    """Return the deterministic SHA-256 fingerprint for a tokenizer bundle."""

    payload = dict(bundle)
    payload.pop("fingerprint", None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(handle)
    return Path(name)


def _write_bytes(path: Path, value: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _write_bytes(path, encoded)


def _referenced_artifacts(metadata_path: Path) -> set[Path]:
    """Return safe artifact paths referenced by an existing metadata file."""

    if not metadata_path.exists():
        return set()
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        tokenizer = _require_mapping(metadata.get("tokenizer"), "tokenizer")
        splits = _require_mapping(metadata.get("splits"), "splits")
        names = [
            tokenizer.get("file"),
            _require_mapping(splits.get("train"), "splits.train").get("file"),
            _require_mapping(splits.get("validation"), "splits.validation").get("file"),
        ]
    except (OSError, json.JSONDecodeError, ValueError, AttributeError, TypeError):
        return set()
    paths = set()
    for name in names:
        if isinstance(name, str) and name and Path(name).name == name:
            paths.add(metadata_path.parent / name)
    return paths


def _validate_limit(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative; 0 means unlimited")


def _validate_output_target(output_dir: Path, force: bool) -> None:
    if not isinstance(force, bool):
        raise TypeError("force must be a boolean")
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and not force:
        raise FileExistsError(
            f"prepared data already exists ({metadata_path.name}); pass force=True to replace it"
        )


def _story_text(story: Story, *, text_field: str) -> str:
    if isinstance(story, str):
        return story
    if not isinstance(story, Mapping):
        raise TypeError(f"stories must be strings or mappings, found {type(story).__name__}")
    if text_field not in story:
        raise ValueError(f"story row is missing text field {text_field!r}")
    text = story[text_field]
    if not isinstance(text, str):
        raise TypeError(f"story field {text_field!r} must be a string")
    return text


def _limited_texts(
    stories: Iterable[Story],
    *,
    limit: int,
    text_field: str,
) -> Iterable[str]:
    for index, story in enumerate(stories):
        if limit and index >= limit:
            break
        yield _story_text(story, text_field=text_field)


def _write_packed_split(
    path: Path,
    texts: Iterable[str],
    tokenizer: ByteBPETokenizer,
    *,
    split: str,
    show_progress: bool,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    if show_progress:
        from tqdm.auto import tqdm

        texts = tqdm(
            texts,
            desc=f"Packing {split}",
            unit="stories",
            mininterval=1.0,
        )
    digest = hashlib.sha256()
    story_count = 0
    skipped_reserved_token_stories = 0
    token_count = 0
    with path.open("wb") as stream:
        for text in texts:
            try:
                token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            except ReservedTokenError:
                # Corpus text must never be able to inject control-token IDs.
                # Skip the complete document and keep streaming until the exact
                # requested token budget is filled.
                skipped_reserved_token_stories += 1
                continue
            if max_tokens is not None:
                remaining = max_tokens - token_count
                if remaining <= 0:
                    break
                token_ids = token_ids[:remaining]
            if any(token_id < 0 or token_id > MAX_UINT16_VOCAB_SIZE for token_id in token_ids):
                raise ValueError("tokenizer emitted an ID outside the uint16 range")
            encoded = np.asarray(token_ids, dtype="<u2").tobytes()
            stream.write(encoded)
            digest.update(encoded)
            story_count += 1
            token_count += len(token_ids)
            if max_tokens is not None and token_count == max_tokens:
                break
        stream.flush()
        os.fsync(stream.fileno())
    if story_count == 0:
        raise ValueError("each prepared split must contain at least one story")
    if max_tokens is not None and token_count != max_tokens:
        raise ValueError(
            f"{split} source ended at {token_count} tokens before the requested "
            f"exact budget of {max_tokens}"
        )
    details = {
        "file": path.name.removeprefix(".").split(".", maxsplit=1)[0],
        "stories": story_count,
        "tokens": token_count,
        "bytes": token_count * np.dtype("<u2").itemsize,
        "sha256": digest.hexdigest(),
    }
    if skipped_reserved_token_stories:
        details["skipped_reserved_token_stories"] = skipped_reserved_token_stories
    return details


def _prepare(
    output_dir: Path,
    *,
    train_factory: StoryFactory,
    validation_factory: StoryFactory,
    dataset_name: str,
    requested_revision: str | None,
    resolved_revision: str | None,
    text_field: str,
    train_limit: int,
    validation_limit: int,
    vocab_size: int,
    min_frequency: int,
    show_progress: bool,
    force: bool,
    streaming: bool,
    tokenizer_from: str | Path | None,
    tokenizer_train_limit: int | None = None,
    train_token_limit: int | None = None,
    validation_token_limit: int | None = None,
    extra_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_limit("train_limit", train_limit)
    _validate_limit("validation_limit", validation_limit)
    if tokenizer_train_limit is not None:
        _validate_limit("tokenizer_train_limit", tokenizer_train_limit)
    for name, value in (
        ("train_token_limit", train_token_limit),
        ("validation_token_limit", validation_token_limit),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    _validate_output_target(output_dir, force)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_destination = output_dir / "metadata.json"
    previous_artifacts = _referenced_artifacts(metadata_destination)

    tokenizer_source: dict[str, str] | None = None
    if tokenizer_from is None:
        tokenizer = ByteBPETokenizer.train(
            _limited_texts(
                train_factory(),
                limit=(tokenizer_train_limit if tokenizer_train_limit is not None else train_limit),
                text_field=text_field,
            ),
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
        )
        tokenizer_json = (tokenizer.to_json() + "\n").encode("utf-8")
    else:
        tokenizer, tokenizer_json, tokenizer_source = _load_reused_tokenizer(
            Path(tokenizer_from),
            output_dir=output_dir,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )
    if tokenizer.vocab_size > MAX_UINT16_VOCAB_SIZE:
        raise ValueError("prepared token IDs require a vocabulary no larger than 65535")

    temporary_paths = {
        "tokenizer": _temporary_path(output_dir / "tokenizer.json"),
        "train": _temporary_path(output_dir / "train.bin"),
        "validation": _temporary_path(output_dir / "validation.bin"),
        "metadata": _temporary_path(metadata_destination),
    }
    try:
        tokenizer_sha256 = _sha256_bytes(tokenizer_json)
        _write_bytes(temporary_paths["tokenizer"], tokenizer_json)
        train_details = _write_packed_split(
            temporary_paths["train"],
            _limited_texts(
                train_factory(),
                limit=train_limit,
                text_field=text_field,
            ),
            tokenizer,
            split="train",
            show_progress=show_progress,
            max_tokens=train_token_limit,
        )
        validation_details = _write_packed_split(
            temporary_paths["validation"],
            _limited_texts(
                validation_factory(),
                limit=validation_limit,
                text_field=text_field,
            ),
            tokenizer,
            split="validation",
            show_progress=show_progress,
            max_tokens=validation_token_limit,
        )
        artifact_destinations = {
            "tokenizer": output_dir / f"tokenizer-{tokenizer_sha256}.json",
            "train": output_dir / f"train-{train_details['sha256']}.bin",
            "validation": output_dir / f"validation-{validation_details['sha256']}.bin",
        }
        train_details["file"] = artifact_destinations["train"].name
        validation_details["file"] = artifact_destinations["validation"].name

        tokenizer_metadata: dict[str, Any] = {
            "file": artifact_destinations["tokenizer"].name,
            "sha256": tokenizer_sha256,
            "requested_vocab_size": vocab_size,
            "vocab_size": tokenizer.vocab_size,
            "min_frequency": min_frequency,
            "special_tokens": {
                token: token_id
                for token, token_id in zip(
                    SPECIAL_TOKENS,
                    (
                        tokenizer.pad_id,
                        tokenizer.unk_id,
                        tokenizer.bos_id,
                        tokenizer.eos_id,
                    ),
                    strict=True,
                )
            },
        }
        if tokenizer_source is not None:
            tokenizer_metadata["reused_from"] = tokenizer_source

        metadata: dict[str, Any] = {
            "schema_version": METADATA_SCHEMA_VERSION,
            "dataset": {
                "name": dataset_name,
                "requested_revision": requested_revision,
                "resolved_revision": resolved_revision,
            },
            "dtype": TOKEN_DTYPE,
            "byte_order": TOKEN_BYTE_ORDER,
            "tokenizer": tokenizer_metadata,
            "splits": {
                "train": train_details,
                "validation": validation_details,
            },
            "config": {
                "text_field": text_field,
                "train_limit": train_limit,
                "validation_limit": validation_limit,
                "streaming": streaming,
            },
        }
        if tokenizer_train_limit is not None:
            metadata["config"]["tokenizer_train_limit"] = tokenizer_train_limit
        if train_token_limit is not None:
            metadata["config"]["train_token_limit"] = train_token_limit
        if validation_token_limit is not None:
            metadata["config"]["validation_token_limit"] = validation_token_limit
        if extra_config:
            metadata["config"].update(extra_config)
        metadata["fingerprint"] = metadata_fingerprint(metadata)
        _write_json(temporary_paths["metadata"], metadata)

        # Content-addressed artifact names leave the previous generation intact
        # until one atomic metadata replacement commits the new generation.
        for name in ("tokenizer", "train", "validation"):
            os.replace(temporary_paths[name], artifact_destinations[name])
        os.replace(temporary_paths["metadata"], metadata_destination)
        current_artifacts = set(artifact_destinations.values())
        for obsolete_path in previous_artifacts - current_artifacts:
            # The new generation is already committed. An orphan is safer than
            # reporting preparation failure after a successful swap.
            with suppress(OSError):
                obsolete_path.unlink(missing_ok=True)
        return metadata
    except BaseException:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise


def _load_reused_tokenizer(
    source_dir: Path,
    *,
    output_dir: Path,
    vocab_size: int,
    min_frequency: int,
) -> tuple[ByteBPETokenizer, bytes, dict[str, str]]:
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("tokenizer source and preparation output must differ")
    bundle_path = source_dir / TOKENIZER_BUNDLE_FILE
    if bundle_path.is_file():
        return _load_tokenizer_bundle(
            bundle_path,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )
    source = PreparedTokenData(source_dir)
    details = _require_mapping(source.metadata.get("tokenizer"), "tokenizer")
    source_vocab_size = details.get("requested_vocab_size")
    if source_vocab_size != vocab_size:
        raise ValueError(
            "vocab_size conflicts with the tokenizer source: "
            f"expected {source_vocab_size}, found {vocab_size}"
        )
    source_min_frequency = details.get("min_frequency")
    if source_min_frequency != min_frequency:
        raise ValueError(
            "min_frequency conflicts with the tokenizer source: "
            f"expected {source_min_frequency}, found {min_frequency}"
        )
    file_name = details.get("file")
    tokenizer_sha256 = details.get("sha256")
    if not isinstance(file_name, str) or not isinstance(tokenizer_sha256, str):
        raise ValueError("tokenizer source metadata is incomplete")
    try:
        tokenizer_json = (source_dir / file_name).read_bytes()
    except OSError as error:
        raise ValueError("cannot read the tokenizer source artifact") from error
    if _sha256_bytes(tokenizer_json) != tokenizer_sha256:
        raise ValueError("prepared tokenizer checksum mismatch")
    return (
        source.tokenizer,
        tokenizer_json,
        {
            "dataset_fingerprint": source.fingerprint,
            "tokenizer_sha256": tokenizer_sha256,
        },
    )


def export_tokenizer_bundle(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Export a prepared dataset's tokenizer as a small portable bundle."""

    source = PreparedTokenData(data_dir)
    destination = Path(output_dir)
    manifest_path = destination / TOKENIZER_BUNDLE_FILE
    if manifest_path.exists() and not force:
        raise FileExistsError(
            f"tokenizer bundle already exists ({manifest_path.name}); pass force=True to replace it"
        )
    destination.mkdir(parents=True, exist_ok=True)
    details = _require_mapping(source.metadata.get("tokenizer"), "tokenizer")
    tokenizer_file = details.get("file")
    tokenizer_sha256 = details.get("sha256")
    if not isinstance(tokenizer_file, str) or not isinstance(tokenizer_sha256, str):
        raise ValueError("prepared tokenizer metadata is incomplete")
    tokenizer_bytes = (source.data_dir / tokenizer_file).read_bytes()
    if _sha256_bytes(tokenizer_bytes) != tokenizer_sha256:
        raise ValueError("prepared tokenizer checksum mismatch")

    artifact_name = f"tokenizer-{tokenizer_sha256}.json"
    bundle: dict[str, Any] = {
        "schema_version": TOKENIZER_BUNDLE_SCHEMA_VERSION,
        "source_dataset_fingerprint": source.fingerprint,
        "tokenizer": {
            "file": artifact_name,
            "sha256": tokenizer_sha256,
            "vocab_size": details.get("vocab_size"),
            "requested_vocab_size": details.get("requested_vocab_size"),
            "min_frequency": details.get("min_frequency"),
            "special_tokens": details.get("special_tokens"),
        },
    }
    bundle["fingerprint"] = tokenizer_bundle_fingerprint(bundle)

    tokenizer_destination = destination / artifact_name
    temporary_tokenizer = _temporary_path(tokenizer_destination)
    temporary_manifest = _temporary_path(manifest_path)
    try:
        _write_bytes(temporary_tokenizer, tokenizer_bytes)
        _write_json(temporary_manifest, bundle)
        os.replace(temporary_tokenizer, tokenizer_destination)
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_tokenizer.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
    return bundle


def _load_tokenizer_bundle(
    bundle_path: Path,
    *,
    vocab_size: int,
    min_frequency: int,
) -> tuple[ByteBPETokenizer, bytes, dict[str, str]]:
    try:
        loaded = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read tokenizer bundle at {bundle_path}") from error
    bundle = _require_mapping(loaded, "tokenizer bundle")
    if bundle.get("schema_version") != TOKENIZER_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported tokenizer bundle schema version")
    fingerprint = bundle.get("fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != tokenizer_bundle_fingerprint(bundle):
        raise ValueError("tokenizer bundle fingerprint mismatch")
    source_fingerprint = bundle.get("source_dataset_fingerprint")
    if not isinstance(source_fingerprint, str) or len(source_fingerprint) != 64:
        raise ValueError("tokenizer bundle source fingerprint is invalid")

    details = _require_mapping(bundle.get("tokenizer"), "tokenizer bundle tokenizer")
    if details.get("requested_vocab_size") != vocab_size:
        raise ValueError(
            "vocab_size conflicts with the tokenizer source: "
            f"expected {details.get('requested_vocab_size')}, found {vocab_size}"
        )
    if details.get("min_frequency") != min_frequency:
        raise ValueError(
            "min_frequency conflicts with the tokenizer source: "
            f"expected {details.get('min_frequency')}, found {min_frequency}"
        )
    file_name = details.get("file")
    tokenizer_sha256 = details.get("sha256")
    if (
        not isinstance(file_name, str)
        or Path(file_name).name != file_name
        or not isinstance(tokenizer_sha256, str)
    ):
        raise ValueError("tokenizer bundle metadata is incomplete")
    try:
        tokenizer_bytes = (bundle_path.parent / file_name).read_bytes()
    except OSError as error:
        raise ValueError("cannot read tokenizer bundle artifact") from error
    if _sha256_bytes(tokenizer_bytes) != tokenizer_sha256:
        raise ValueError("tokenizer bundle checksum mismatch")
    tokenizer = ByteBPETokenizer.from_json(tokenizer_bytes.decode("utf-8"))
    if tokenizer.vocab_size != details.get("vocab_size"):
        raise ValueError("tokenizer bundle vocabulary size mismatch")
    special_tokens = {
        token: token_id
        for token, token_id in zip(
            SPECIAL_TOKENS,
            (
                tokenizer.pad_id,
                tokenizer.unk_id,
                tokenizer.bos_id,
                tokenizer.eos_id,
            ),
            strict=True,
        )
    }
    if special_tokens != details.get("special_tokens"):
        raise ValueError("tokenizer bundle special token IDs mismatch")
    return (
        tokenizer,
        tokenizer_bytes,
        {
            "dataset_fingerprint": source_fingerprint,
            "tokenizer_sha256": tokenizer_sha256,
        },
    )


def prepare_from_stories(
    output_dir: str | Path,
    train_stories: Iterable[Story],
    validation_stories: Iterable[Story],
    *,
    dataset_name: str = "in_memory",
    requested_revision: str | None = None,
    resolved_revision: str | None = None,
    text_field: str = "text",
    train_limit: int = 0,
    validation_limit: int = 0,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = 2,
    show_progress: bool = False,
    force: bool = False,
    tokenizer_from: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare packed data from local iterables without requiring network access."""

    _validate_limit("train_limit", train_limit)
    _validate_limit("validation_limit", validation_limit)
    destination = Path(output_dir)
    _validate_output_target(destination, force)
    # Materialization makes one-shot iterables reusable for tokenizer training
    # and packing. Apply caps first so bounded preparation does not consume an
    # entire large (or infinite) source.
    training_rows = list(
        itertools.islice(train_stories, train_limit) if train_limit else train_stories
    )
    validation_rows = list(
        itertools.islice(validation_stories, validation_limit)
        if validation_limit
        else validation_stories
    )
    return _prepare(
        destination,
        train_factory=lambda: iter(training_rows),
        validation_factory=lambda: iter(validation_rows),
        dataset_name=dataset_name,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
        text_field=text_field,
        train_limit=train_limit,
        validation_limit=validation_limit,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=show_progress,
        force=force,
        streaming=False,
        tokenizer_from=tokenizer_from,
    )


def _resolve_revision(
    dataset_name: str,
    requested_revision: str | None,
) -> str:
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(
            repo_id=dataset_name,
            revision=requested_revision,
        )
    except Exception as error:  # pragma: no cover - depends on network/cache state
        raise RuntimeError(
            f"could not resolve an immutable revision for dataset {dataset_name!r}"
        ) from error
    resolved = getattr(info, "sha", None)
    if not isinstance(resolved, str) or not resolved:
        raise RuntimeError(f"dataset service did not return a revision SHA for {dataset_name!r}")
    return resolved


def prepare_tinystories(
    output_dir: str | Path,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    revision: str | None = DEFAULT_DATASET_REVISION,
    resolved_revision: str | None = None,
    text_field: str = "text",
    train_limit: int = DEFAULT_TRAIN_LIMIT,
    validation_limit: int = DEFAULT_VALIDATION_LIMIT,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = 2,
    show_progress: bool = True,
    force: bool = False,
    load_dataset_fn: Callable[..., Iterable[Story]] | None = None,
    tokenizer_from: str | Path | None = None,
) -> dict[str, Any]:
    """Stream TinyStories and pack both splits with a new or reused tokenizer."""

    destination = Path(output_dir)
    _validate_output_target(destination, force)
    using_default_loader = load_dataset_fn is None
    if load_dataset_fn is None:
        try:
            from datasets import load_dataset
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("TinyStories preparation requires the `datasets` package") from error
        load_dataset_fn = load_dataset

    if resolved_revision is None:
        if not using_default_loader:
            raise ValueError("resolved_revision is required when injecting a dataset loader")
        resolved_revision = _resolve_revision(dataset_name, revision)

    def load_split(split: SplitName) -> Iterable[Story]:
        assert load_dataset_fn is not None
        return load_dataset_fn(
            dataset_name,
            split=split,
            streaming=True,
            revision=resolved_revision,
        )

    return _prepare(
        destination,
        train_factory=lambda: load_split("train"),
        validation_factory=lambda: load_split("validation"),
        dataset_name=dataset_name,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        text_field=text_field,
        train_limit=train_limit,
        validation_limit=validation_limit,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=show_progress,
        force=force,
        streaming=True,
        tokenizer_from=tokenizer_from,
    )


def _interleave_sources(
    first: Iterable[Story],
    second: Iterable[Story],
    *,
    first_probability: float,
    seed: int,
) -> Iterable[Story]:
    """Deterministically interleave two streams without materializing them."""

    if not 0.0 < first_probability < 1.0:
        raise ValueError("first_probability must be in (0, 1)")
    generators = [iter(first), iter(second)]
    active = [True, True]
    rng = random.Random(seed)
    while any(active):
        selected = 0 if rng.random() < first_probability else 1
        if not active[selected]:
            selected = 1 - selected
        try:
            yield next(generators[selected])
        except StopIteration:
            active[selected] = False


def prepare_smollm_corpus(
    output_dir: str | Path,
    *,
    dataset_name: str = DEFAULT_SMOLLM_DATASET_NAME,
    revision: str | None = DEFAULT_SMOLLM_DATASET_REVISION,
    resolved_revision: str | None = None,
    train_tokens: int = 50_000_000,
    validation_tokens: int = 2_000_000,
    tokenizer_train_documents: int = 100_000,
    validation_documents_per_source: int = 10_000,
    vocab_size: int = DEFAULT_SMOLLM_VOCAB_SIZE,
    min_frequency: int = 2,
    fineweb_probability: float = 0.7,
    seed: int = 42,
    show_progress: bool = True,
    force: bool = False,
    load_dataset_fn: Callable[..., Iterable[Story]] | None = None,
    tokenizer_from: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare an exact-token FineWeb-Edu/Cosmopedia SmolLM subset.

    The first fixed number of documents from each immutable source stream forms
    validation; training skips those documents. Both streams are then mixed by a
    seeded Bernoulli schedule. Python-Edu is deliberately absent from the recipe.
    """

    destination = Path(output_dir)
    _validate_output_target(destination, force)
    _validate_limit("tokenizer_train_documents", tokenizer_train_documents)
    _validate_limit("validation_documents_per_source", validation_documents_per_source)
    if tokenizer_train_documents == 0:
        raise ValueError("tokenizer_train_documents must be positive")
    if validation_documents_per_source == 0:
        raise ValueError("validation_documents_per_source must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    using_default_loader = load_dataset_fn is None
    if load_dataset_fn is None:
        try:
            from datasets import load_dataset
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "SmolLM-Corpus preparation requires the `datasets` package"
            ) from error
        load_dataset_fn = load_dataset
    if resolved_revision is None:
        if not using_default_loader:
            raise ValueError("resolved_revision is required when injecting a dataset loader")
        resolved_revision = _resolve_revision(dataset_name, revision)

    def load_source(config_name: str) -> Iterable[Story]:
        assert load_dataset_fn is not None
        return load_dataset_fn(
            dataset_name,
            config_name,
            split="train",
            streaming=True,
            revision=resolved_revision,
        )

    def mixed(*, validation: bool) -> Iterable[Story]:
        fineweb = load_source(DEFAULT_SMOLLM_FINEWEB_CONFIG)
        cosmopedia = load_source(DEFAULT_SMOLLM_COSMOPEDIA_CONFIG)
        if validation:
            fineweb = itertools.islice(fineweb, validation_documents_per_source)
            cosmopedia = itertools.islice(cosmopedia, validation_documents_per_source)
        else:
            fineweb = itertools.islice(fineweb, validation_documents_per_source, None)
            cosmopedia = itertools.islice(cosmopedia, validation_documents_per_source, None)
        return _interleave_sources(
            fineweb,
            cosmopedia,
            first_probability=fineweb_probability,
            seed=seed + (1 if validation else 0),
        )

    return _prepare(
        destination,
        train_factory=lambda: mixed(validation=False),
        validation_factory=lambda: mixed(validation=True),
        dataset_name=dataset_name,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        text_field="text",
        train_limit=0,
        validation_limit=0,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=show_progress,
        force=force,
        streaming=True,
        tokenizer_from=tokenizer_from,
        tokenizer_train_limit=tokenizer_train_documents,
        train_token_limit=train_tokens,
        validation_token_limit=validation_tokens,
        extra_config={
            "source_configs": [
                DEFAULT_SMOLLM_FINEWEB_CONFIG,
                DEFAULT_SMOLLM_COSMOPEDIA_CONFIG,
            ],
            "fineweb_probability": fineweb_probability,
            "seed": seed,
            "validation_documents_per_source": validation_documents_per_source,
            "python_edu_included": False,
        },
    )


def prepare_simplestories(
    output_dir: str | Path,
    *,
    tokenizer_from: str | Path,
    dataset_name: str = DEFAULT_SIMPLESTORIES_DATASET_NAME,
    revision: str | None = DEFAULT_SIMPLESTORIES_DATASET_REVISION,
    resolved_revision: str | None = None,
    text_field: str = "story",
    train_limit: int = DEFAULT_SIMPLESTORIES_TRAIN_LIMIT,
    validation_limit: int = DEFAULT_SIMPLESTORIES_VALIDATION_LIMIT,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = 2,
    show_progress: bool = True,
    force: bool = False,
    load_dataset_fn: Callable[..., Iterable[Story]] | None = None,
) -> dict[str, Any]:
    """Prepare SimpleStories with a validated, frozen KiwiLM tokenizer."""

    destination = Path(output_dir)
    _validate_output_target(destination, force)
    using_default_loader = load_dataset_fn is None
    if load_dataset_fn is None:
        try:
            from datasets import load_dataset
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "SimpleStories preparation requires the `datasets` package"
            ) from error
        load_dataset_fn = load_dataset

    if resolved_revision is None:
        if not using_default_loader:
            raise ValueError("resolved_revision is required when injecting a dataset loader")
        resolved_revision = _resolve_revision(dataset_name, revision)

    def load_split(split: str) -> Iterable[Story]:
        assert load_dataset_fn is not None
        return load_dataset_fn(
            dataset_name,
            split=split,
            streaming=True,
            revision=resolved_revision,
        )

    return _prepare(
        destination,
        train_factory=lambda: load_split("train"),
        validation_factory=lambda: load_split("test"),
        dataset_name=dataset_name,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        text_field=text_field,
        train_limit=train_limit,
        validation_limit=validation_limit,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=show_progress,
        force=force,
        streaming=True,
        tokenizer_from=tokenizer_from,
    )


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"prepared metadata field {name!r} must be an object")
    return value


class PreparedTokenData:
    """Validated memory-mapped splits with deterministic next-token batches."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        seed: int = 1337,
        expected_fingerprint: str | None = None,
        verify_integrity: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        metadata_path = self.data_dir / "metadata.json"
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read prepared metadata at {metadata_path}") from error
        if not isinstance(loaded, Mapping):
            raise ValueError("prepared metadata must be a JSON object")
        self.metadata = dict(loaded)

        if self.metadata.get("schema_version") != METADATA_SCHEMA_VERSION:
            raise ValueError(
                "unsupported prepared metadata schema version "
                f"{self.metadata.get('schema_version')!r}"
            )
        if self.metadata.get("dtype") != TOKEN_DTYPE:
            raise ValueError(f"prepared dtype must be {TOKEN_DTYPE}")
        if self.metadata.get("byte_order") != TOKEN_BYTE_ORDER:
            raise ValueError(f"prepared byte order must be {TOKEN_BYTE_ORDER}")

        fingerprint = self.metadata.get("fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("prepared metadata has an invalid fingerprint")
        computed_fingerprint = metadata_fingerprint(self.metadata)
        if fingerprint != computed_fingerprint:
            raise ValueError("prepared metadata fingerprint mismatch")
        if expected_fingerprint is not None and fingerprint != expected_fingerprint:
            raise ValueError(
                "prepared data fingerprint mismatch: "
                f"expected {expected_fingerprint}, found {fingerprint}"
            )
        self.fingerprint = fingerprint

        tokenizer_details = _require_mapping(
            self.metadata.get("tokenizer"),
            "tokenizer",
        )
        tokenizer_file = tokenizer_details.get("file")
        if not isinstance(tokenizer_file, str):
            raise ValueError("prepared tokenizer file must be a string")
        tokenizer_path = self.data_dir / tokenizer_file
        if verify_integrity and _sha256_file(tokenizer_path) != tokenizer_details.get("sha256"):
            raise ValueError("prepared tokenizer checksum mismatch")
        self.tokenizer = ByteBPETokenizer.load(tokenizer_path)
        if self.tokenizer.vocab_size != tokenizer_details.get("vocab_size"):
            raise ValueError("prepared tokenizer vocabulary size mismatch")

        split_details = _require_mapping(self.metadata.get("splits"), "splits")
        self._tokens: dict[str, np.memmap] = {}
        for split in SPLITS:
            details = _require_mapping(split_details.get(split), f"splits.{split}")
            file_name = details.get("file")
            token_count = details.get("tokens")
            byte_count = details.get("bytes")
            if not isinstance(file_name, str):
                raise ValueError(f"prepared {split} filename must be a string")
            if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 1:
                raise ValueError(f"prepared {split} token count is invalid")
            if byte_count != token_count * np.dtype("<u2").itemsize:
                raise ValueError(f"prepared {split} byte count is inconsistent")
            path = self.data_dir / file_name
            try:
                actual_size = path.stat().st_size
            except OSError as error:
                raise ValueError(f"cannot read prepared {split} split") from error
            if actual_size != byte_count:
                raise ValueError(f"prepared {split} file size mismatch")
            if verify_integrity and _sha256_file(path) != details.get("sha256"):
                raise ValueError(f"prepared {split} checksum mismatch")
            self._tokens[split] = np.memmap(
                path,
                dtype="<u2",
                mode="r",
                shape=(token_count,),
            )

        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)
        self._story_offsets: dict[str, np.ndarray] = {}
        self._story_chunks: dict[tuple[str, int], np.ndarray] = {}

    def tokens(self, split: SplitName) -> np.memmap:
        try:
            return self._tokens[split]
        except KeyError as error:
            raise ValueError(f"split must be one of {SPLITS}, found {split!r}") from error

    def get_batch(
        self,
        split: SplitName,
        *,
        batch_size: int,
        context_length: int,
        device: str | torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if isinstance(context_length, bool) or not isinstance(context_length, int):
            raise TypeError("context_length must be an integer")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if context_length < 1:
            raise ValueError("context_length must be at least 1")

        tokens = self.tokens(split)
        possible_starts = len(tokens) - context_length
        if possible_starts < 1:
            raise ValueError(
                f"{split} contains {len(tokens)} tokens, but a context length of "
                f"{context_length} needs at least {context_length + 1}"
            )
        starts = torch.randint(
            0,
            possible_starts,
            (batch_size,),
            generator=generator if generator is not None else self._generator,
            device="cpu",
        )
        windows = np.stack(
            [
                np.asarray(
                    tokens[int(start) : int(start) + context_length + 1],
                    dtype=np.int64,
                )
                for start in starts.tolist()
            ]
        )
        batch = torch.from_numpy(windows)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        if device is not None:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
        return inputs, targets

    def story_offsets(self, split: SplitName) -> np.ndarray:
        """Return validated ``[start, end)`` story boundaries for a split."""

        if split in self._story_offsets:
            return self._story_offsets[split]
        details = _require_mapping(
            _require_mapping(self.metadata.get("splits"), "splits").get(split),
            f"splits.{split}",
        )
        split_sha = details.get("sha256")
        if not isinstance(split_sha, str):
            raise ValueError(f"prepared {split} checksum is invalid")
        cache_path = self.data_dir / f"{split}-story-offsets-{split_sha}.npy"
        offsets: np.ndarray | None = None
        try:
            loaded = np.load(cache_path, allow_pickle=False)
            offsets = np.asarray(loaded, dtype=np.int64)
            self._validate_story_offsets(split, offsets)
        except (OSError, ValueError, TypeError):
            offsets = self._scan_story_offsets(split)
            temporary = _temporary_path(cache_path)
            try:
                with temporary.open("wb") as stream:
                    np.save(stream, offsets, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, cache_path)
            finally:
                temporary.unlink(missing_ok=True)
        self._story_offsets[split] = offsets
        return offsets

    def story_chunks(self, split: SplitName, context_length: int) -> np.ndarray:
        """Return ``[input_start, valid_targets]`` chunks for story-safe batching."""

        if isinstance(context_length, bool) or not isinstance(context_length, int):
            raise TypeError("context_length must be an integer")
        if context_length < 1:
            raise ValueError("context_length must be at least 1")
        cache_key = (split, context_length)
        if cache_key in self._story_chunks:
            return self._story_chunks[cache_key]
        chunks: list[tuple[int, int]] = []
        for start, end in self.story_offsets(split):
            target_count = int(end - start - 1)
            for offset in range(0, target_count, context_length):
                chunks.append((int(start + offset), min(context_length, target_count - offset)))
        if not chunks:
            raise ValueError(f"{split} contains no next-token story targets")
        result = np.asarray(chunks, dtype=np.int64)
        self._story_chunks[cache_key] = result
        return result

    def story_batch(
        self,
        split: SplitName,
        chunk_indices: torch.Tensor | list[int],
        *,
        context_length: int,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize selected story chunks with masked right padding."""

        chunks = self.story_chunks(split, context_length)
        indices = (
            chunk_indices.detach().cpu().tolist()
            if isinstance(chunk_indices, torch.Tensor)
            else list(chunk_indices)
        )
        pad_id = self.tokenizer.pad_id
        inputs = np.full((len(indices), context_length), pad_id, dtype=np.int64)
        targets = np.full((len(indices), context_length), -100, dtype=np.int64)
        tokens = self.tokens(split)
        for row, index in enumerate(indices):
            start, valid = chunks[int(index)]
            window = np.asarray(tokens[start : start + valid + 1], dtype=np.int64)
            inputs[row, :valid] = window[:-1]
            targets[row, :valid] = window[1:]
        input_tensor = torch.from_numpy(inputs)
        target_tensor = torch.from_numpy(targets)
        if device is not None:
            input_tensor = input_tensor.to(device, non_blocking=True)
            target_tensor = target_tensor.to(device, non_blocking=True)
        return input_tensor, target_tensor

    def _scan_story_offsets(self, split: SplitName) -> np.ndarray:
        tokens = self.tokens(split)
        bos_id = self.tokenizer.bos_id
        eos_id = self.tokenizer.eos_id
        starts = np.flatnonzero(tokens == bos_id)
        ends = np.flatnonzero(tokens == eos_id) + 1
        if len(starts) != len(ends):
            raise ValueError(f"{split} has mismatched BOS/EOS story markers")
        offsets = np.column_stack((starts, ends)).astype(np.int64, copy=False)
        self._validate_story_offsets(split, offsets)
        return offsets

    def _validate_story_offsets(self, split: SplitName, offsets: np.ndarray) -> None:
        tokens = self.tokens(split)
        if offsets.ndim != 2 or offsets.shape[1] != 2 or len(offsets) == 0:
            raise ValueError(f"{split} story offset cache has an invalid shape")
        starts, ends = offsets[:, 0], offsets[:, 1]
        if (
            starts[0] != 0
            or ends[-1] != len(tokens)
            or np.any(ends <= starts)
            or np.any(starts[1:] != ends[:-1])
            or np.any(tokens[starts] != self.tokenizer.bos_id)
            or np.any(tokens[ends - 1] != self.tokenizer.eos_id)
        ):
            raise ValueError(f"{split} story offset cache is inconsistent")
        details = _require_mapping(
            _require_mapping(self.metadata.get("splits"), "splits").get(split),
            f"splits.{split}",
        )
        if details.get("stories") != len(offsets):
            raise ValueError(f"{split} story offset count is inconsistent")

    def get_rng_state(self) -> torch.Tensor:
        return self._generator.get_state().clone()

    def set_rng_state(self, state: torch.Tensor) -> None:
        if not isinstance(state, torch.Tensor):
            raise TypeError("generator state must be a torch.Tensor")
        self._generator.set_state(state.detach().cpu())

    def state_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "generator_state": self.get_rng_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("fingerprint") != self.fingerprint:
            raise ValueError("prepared data state fingerprint mismatch")
        generator_state = state.get("generator_state")
        if not isinstance(generator_state, torch.Tensor):
            raise ValueError("prepared data state is missing generator_state")
        self.set_rng_state(generator_state)


class StoryBatchSampler:
    """Deterministic epoch-shuffled sampler over non-overlapping story chunks."""

    def __init__(
        self,
        data: PreparedTokenData,
        split: SplitName,
        *,
        context_length: int,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        self.data = data
        self.split = split
        self.context_length = context_length
        self.seed = seed
        self.shuffle = shuffle
        self.chunk_count = len(data.story_chunks(split, context_length))
        self.epoch = 0
        self.cursor = 0
        self._order = self._make_order()

    def next_indices(self, count: int) -> torch.Tensor:
        _positive_batch_size(count)
        selected: list[torch.Tensor] = []
        remaining = count
        while remaining:
            available = self.chunk_count - self.cursor
            take = min(remaining, available)
            selected.append(self._order[self.cursor : self.cursor + take])
            self.cursor += take
            remaining -= take
            if self.cursor == self.chunk_count:
                self.epoch += 1
                self.cursor = 0
                self._order = self._make_order()
        return torch.cat(selected)

    def get_batch(
        self,
        *,
        batch_size: int,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data.story_batch(
            self.split,
            self.next_indices(batch_size),
            context_length=self.context_length,
            device=device,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.data.fingerprint,
            "split": self.split,
            "context_length": self.context_length,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "epoch": self.epoch,
            "cursor": self.cursor,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "fingerprint": self.data.fingerprint,
            "split": self.split,
            "context_length": self.context_length,
            "seed": self.seed,
            "shuffle": self.shuffle,
        }
        if any(state.get(name) != value for name, value in expected.items()):
            raise ValueError("story sampler state is incompatible")
        epoch = state.get("epoch")
        cursor = state.get("cursor")
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("story sampler epoch is invalid")
        if not isinstance(cursor, int) or not 0 <= cursor < self.chunk_count:
            raise ValueError("story sampler cursor is invalid")
        self.epoch = epoch
        self.cursor = cursor
        self._order = self._make_order()

    def _make_order(self) -> torch.Tensor:
        if not self.shuffle:
            return torch.arange(self.chunk_count)
        generator = torch.Generator(device="cpu").manual_seed(self.seed + self.epoch)
        return torch.randperm(self.chunk_count, generator=generator)


def _positive_batch_size(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("batch size must be an integer")
    if value < 1:
        raise ValueError("batch size must be at least 1")


__all__ = [
    "DEFAULT_DATASET_NAME",
    "DEFAULT_DATASET_REVISION",
    "DEFAULT_SIMPLESTORIES_DATASET_NAME",
    "DEFAULT_SIMPLESTORIES_DATASET_REVISION",
    "DEFAULT_SIMPLESTORIES_TRAIN_LIMIT",
    "DEFAULT_SIMPLESTORIES_VALIDATION_LIMIT",
    "DEFAULT_TRAIN_LIMIT",
    "DEFAULT_VALIDATION_LIMIT",
    "DEFAULT_VOCAB_SIZE",
    "PreparedTokenData",
    "StoryBatchSampler",
    "metadata_fingerprint",
    "prepare_from_stories",
    "prepare_simplestories",
    "prepare_tinystories",
]
