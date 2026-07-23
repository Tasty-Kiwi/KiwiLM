"""TinyStories preparation and deterministic packed-token sampling."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
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
)

DEFAULT_DATASET_NAME = "roneneldan/TinyStories"
DEFAULT_DATASET_REVISION = "main"
DEFAULT_TRAIN_LIMIT = 25_000
DEFAULT_VALIDATION_LIMIT = 2_000
DEFAULT_VOCAB_SIZE = 8_192
METADATA_SCHEMA_VERSION = 1
TOKEN_DTYPE = "uint16"
TOKEN_BYTE_ORDER = "little"
SPLITS = ("train", "validation")

Story = str | Mapping[str, Any]
StoryFactory = Callable[[], Iterable[Story]]
SplitName = Literal["train", "validation"]


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
            _require_mapping(
                splits.get("validation"), "splits.validation"
            ).get("file"),
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
            "prepared data already exists "
            f"({metadata_path.name}); pass force=True to replace it"
        )


def _story_text(story: Story, *, text_field: str) -> str:
    if isinstance(story, str):
        return story
    if not isinstance(story, Mapping):
        raise TypeError(
            "stories must be strings or mappings, "
            f"found {type(story).__name__}"
        )
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
) -> dict[str, Any]:
    digest = hashlib.sha256()
    story_count = 0
    token_count = 0
    with path.open("wb") as stream:
        for text in texts:
            token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            if any(
                token_id < 0 or token_id > MAX_UINT16_VOCAB_SIZE
                for token_id in token_ids
            ):
                raise ValueError("tokenizer emitted an ID outside the uint16 range")
            encoded = np.asarray(token_ids, dtype="<u2").tobytes()
            stream.write(encoded)
            digest.update(encoded)
            story_count += 1
            token_count += len(token_ids)
        stream.flush()
        os.fsync(stream.fileno())
    if story_count == 0:
        raise ValueError("each prepared split must contain at least one story")
    return {
        "file": path.name.removeprefix(".").split(".", maxsplit=1)[0],
        "stories": story_count,
        "tokens": token_count,
        "bytes": token_count * np.dtype("<u2").itemsize,
        "sha256": digest.hexdigest(),
    }


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
) -> dict[str, Any]:
    _validate_limit("train_limit", train_limit)
    _validate_limit("validation_limit", validation_limit)
    _validate_output_target(output_dir, force)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_destination = output_dir / "metadata.json"
    previous_artifacts = _referenced_artifacts(metadata_destination)

    tokenizer = ByteBPETokenizer.train(
        _limited_texts(
            train_factory(),
            limit=train_limit,
            text_field=text_field,
        ),
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=show_progress,
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
        tokenizer_json = (tokenizer.to_json() + "\n").encode("utf-8")
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
        )
        validation_details = _write_packed_split(
            temporary_paths["validation"],
            _limited_texts(
                validation_factory(),
                limit=validation_limit,
                text_field=text_field,
            ),
            tokenizer,
        )
        artifact_destinations = {
            "tokenizer": output_dir / f"tokenizer-{tokenizer_sha256}.json",
            "train": output_dir / f"train-{train_details['sha256']}.bin",
            "validation": output_dir
            / f"validation-{validation_details['sha256']}.bin",
        }
        train_details["file"] = artifact_destinations["train"].name
        validation_details["file"] = artifact_destinations["validation"].name

        metadata: dict[str, Any] = {
            "schema_version": METADATA_SCHEMA_VERSION,
            "dataset": {
                "name": dataset_name,
                "requested_revision": requested_revision,
                "resolved_revision": resolved_revision,
            },
            "dtype": TOKEN_DTYPE,
            "byte_order": TOKEN_BYTE_ORDER,
            "tokenizer": {
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
            },
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
        raise RuntimeError(
            f"dataset service did not return a revision SHA for {dataset_name!r}"
        )
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
) -> dict[str, Any]:
    """Stream TinyStories, train on its training split, and pack both splits."""

    destination = Path(output_dir)
    _validate_output_target(destination, force)
    using_default_loader = load_dataset_fn is None
    if load_dataset_fn is None:
        try:
            from datasets import load_dataset
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "TinyStories preparation requires the `datasets` package"
            ) from error
        load_dataset_fn = load_dataset

    if resolved_revision is None:
        if not using_default_loader:
            raise ValueError(
                "resolved_revision is required when injecting a dataset loader"
            )
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
        if verify_integrity and _sha256_file(tokenizer_path) != tokenizer_details.get(
            "sha256"
        ):
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
            if (
                isinstance(token_count, bool)
                or not isinstance(token_count, int)
                or token_count < 1
            ):
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


__all__ = [
    "DEFAULT_DATASET_NAME",
    "DEFAULT_DATASET_REVISION",
    "DEFAULT_TRAIN_LIMIT",
    "DEFAULT_VALIDATION_LIMIT",
    "DEFAULT_VOCAB_SIZE",
    "PreparedTokenData",
    "metadata_fingerprint",
    "prepare_from_stories",
    "prepare_tinystories",
]
