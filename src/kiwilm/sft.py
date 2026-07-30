"""TinyStoriesInstruct preparation and response-masked batch sampling."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from kiwilm.data import PreparedTokenData, metadata_fingerprint
from kiwilm.tokenizer import ByteBPETokenizer

DEFAULT_INSTRUCT_DATASET = "roneneldan/TinyStoriesInstruct"
DEFAULT_INSTRUCT_REVISION = "1282fc1059eaf0aad1f30459a25530f44343f2a2"
DEFAULT_INSTRUCT_TRAIN_FILE = "TinyStories-Instruct-train.txt"
DEFAULT_INSTRUCT_VALIDATION_FILE = "TinyStories-Instruct-valid.txt"
DEFAULT_INSTRUCT_TRAIN_LIMIT = 50_000
DEFAULT_INSTRUCT_VALIDATION_LIMIT = 5_000
INSTRUCT_RECORD_SEPARATOR = "<|endoftext|>"
SFT_METADATA_SCHEMA_VERSION = 1
SFT_TASK = "conditional_story_sft"
SFT_SOURCE_DECODING = "utf-8-replace"
SFT_SPLITS = ("train", "validation")
SFT_INDEX_COLUMNS = 3
SFT_FORMAT_V1 = "v1"
SFT_FORMAT_V2 = "v2"
SFT_FORMATS = (SFT_FORMAT_V1, SFT_FORMAT_V2)
SFT_V2_INSTRUCTION = (
    "Instruction: Write a story that follows every provided condition. "
    "Use every requested word exactly as written.\n"
)
DEFAULT_REQUIRED_WORD_WEIGHT = 3.0

SFTSplit = Literal["train", "validation"]
_FIELD_PATTERN = re.compile(
    r"^(Features|Words|Summary|Random sentence):\s*(.*)$",
    flags=re.IGNORECASE,
)
_FIELD_ORDER = ("Features", "Words", "Summary", "Random sentence")


@dataclass(frozen=True, slots=True)
class InstructionExample:
    """Canonical prompt and target story extracted from one raw record."""

    prompt: str
    response: str
    required_words: tuple[str, ...]


def parse_tinystories_instruct_record(
    record: str,
    *,
    sft_format: str = SFT_FORMAT_V1,
) -> InstructionExample:
    """Canonicalize one raw TinyStoriesInstruct record."""

    if not isinstance(record, str):
        raise TypeError("instruction record must be a string")
    _validate_sft_format(sft_format)
    normalized = record.replace("\r\n", "\n").replace("\r", "\n").strip()
    if normalized.endswith(INSTRUCT_RECORD_SEPARATOR):
        normalized = normalized[: -len(INSTRUCT_RECORD_SEPARATOR)].rstrip()
    story_match = re.search(r"(?im)^Story:\s*", normalized)
    if story_match is None:
        raise ValueError("instruction record is missing a Story field")

    fields: dict[str, str] = {}
    for line in normalized.splitlines():
        stripped = line.strip()
        if stripped.lower() == "story:":
            continue
        field_match = _FIELD_PATTERN.match(stripped)
        if field_match is not None:
            canonical_name = next(
                name
                for name in _FIELD_ORDER
                if name.lower() == field_match.group(1).lower()
            )
            value = field_match.group(2).strip()
            if value:
                fields[canonical_name] = value
            continue

    # Locate the response directly as well, then remove any trailing metadata
    # that some source records place after the story.
    response_region = normalized[story_match.end() :].strip()
    cleaned_response_lines = [
        line
        for line in response_region.splitlines()
        if _FIELD_PATTERN.match(line.strip()) is None
    ]
    response = "\n".join(cleaned_response_lines).strip()
    if not response:
        raise ValueError("instruction record has an empty Story response")

    prompt_lines = [
        f"{field}: {fields[field]}" for field in _FIELD_ORDER if field in fields
    ]
    if not prompt_lines:
        raise ValueError("instruction record has no conditioning fields")
    prompt_lines.append("Story:")
    required_words = tuple(
        word.strip()
        for word in fields.get("Words", "").split(",")
        if word.strip()
    )
    return InstructionExample(
        prompt=(
            (SFT_V2_INSTRUCTION if sft_format == SFT_FORMAT_V2 else "")
            + "\n".join(prompt_lines)
            + "\n"
        ),
        response=response,
        required_words=required_words,
    )


def iter_raw_instruct_records(path: str | Path) -> Iterable[str]:
    """Stream records while replacing malformed UTF-8 in the official files."""

    source = Path(path)
    buffer = ""
    # The pinned official training file contains a small number of truncated
    # curly-quote byte sequences (for example ``e2 80`` without the final
    # byte). Replacement preserves all ASCII delimiters and surrounding text
    # while making preparation deterministic across platforms.
    with source.open("r", encoding="utf-8", errors="replace") as stream:
        while chunk := stream.read(1024 * 1024):
            buffer += chunk
            while INSTRUCT_RECORD_SEPARATOR in buffer:
                record, buffer = buffer.split(INSTRUCT_RECORD_SEPARATOR, maxsplit=1)
                if record.strip():
                    yield record
        if buffer.strip():
            yield buffer


def prepare_tinystories_instruct(
    output_dir: str | Path,
    *,
    tokenizer_from: str | Path,
    dataset_name: str = DEFAULT_INSTRUCT_DATASET,
    revision: str = DEFAULT_INSTRUCT_REVISION,
    train_limit: int = DEFAULT_INSTRUCT_TRAIN_LIMIT,
    validation_limit: int = DEFAULT_INSTRUCT_VALIDATION_LIMIT,
    train_file: str | Path | None = None,
    validation_file: str | Path | None = None,
    sft_format: str = SFT_FORMAT_V1,
    required_word_weight: float = DEFAULT_REQUIRED_WORD_WEIGHT,
    show_progress: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Download, parse, and tokenize a bounded TinyStoriesInstruct subset."""

    if (train_file is None) != (validation_file is None):
        raise ValueError("train_file and validation_file must be provided together")
    if train_file is None:
        from huggingface_hub import hf_hub_download

        train_file = hf_hub_download(
            repo_id=dataset_name,
            filename=DEFAULT_INSTRUCT_TRAIN_FILE,
            repo_type="dataset",
            revision=revision,
        )
        validation_file = hf_hub_download(
            repo_id=dataset_name,
            filename=DEFAULT_INSTRUCT_VALIDATION_FILE,
            repo_type="dataset",
            revision=revision,
        )
    assert validation_file is not None
    return prepare_instruct_from_records(
        output_dir,
        train_records=iter_raw_instruct_records(train_file),
        validation_records=iter_raw_instruct_records(validation_file),
        tokenizer_from=tokenizer_from,
        dataset_name=dataset_name,
        revision=revision,
        train_limit=train_limit,
        validation_limit=validation_limit,
        sft_format=sft_format,
        required_word_weight=required_word_weight,
        show_progress=show_progress,
        force=force,
    )


def prepare_instruct_from_records(
    output_dir: str | Path,
    *,
    train_records: Iterable[str],
    validation_records: Iterable[str],
    tokenizer_from: str | Path,
    dataset_name: str = "in_memory_instruct",
    revision: str | None = None,
    train_limit: int = 0,
    validation_limit: int = 0,
    sft_format: str = SFT_FORMAT_V1,
    required_word_weight: float = DEFAULT_REQUIRED_WORD_WEIGHT,
    show_progress: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Prepare response-masked SFT artifacts from raw record iterables."""

    _validate_limit("train_limit", train_limit)
    _validate_limit("validation_limit", validation_limit)
    _validate_sft_format(sft_format)
    _validate_required_word_weight(required_word_weight)
    destination = Path(output_dir)
    metadata_path = destination / "metadata.json"
    if metadata_path.exists() and not force:
        raise FileExistsError(
            "prepared SFT data already exists (metadata.json); "
            "pass force=True to replace it"
        )
    tokenizer_source = PreparedTokenData(tokenizer_from)
    tokenizer_details = _mapping(
        tokenizer_source.metadata.get("tokenizer"),
        "tokenizer",
    )
    tokenizer_file = tokenizer_details.get("file")
    tokenizer_sha256 = tokenizer_details.get("sha256")
    if not isinstance(tokenizer_file, str) or not isinstance(tokenizer_sha256, str):
        raise ValueError("tokenizer source metadata is incomplete")
    tokenizer_bytes = (Path(tokenizer_from) / tokenizer_file).read_bytes()
    if _sha256_bytes(tokenizer_bytes) != tokenizer_sha256:
        raise ValueError("tokenizer source checksum mismatch")

    destination.mkdir(parents=True, exist_ok=True)
    previous_artifacts = _referenced_artifacts(metadata_path)
    temporary = {
        "tokenizer": _temporary_path(destination / "tokenizer.json"),
        "train_tokens": _temporary_path(destination / "train.tokens"),
        "train_index": _temporary_path(destination / "train.index"),
        "validation_tokens": _temporary_path(destination / "validation.tokens"),
        "validation_index": _temporary_path(destination / "validation.index"),
        "metadata": _temporary_path(metadata_path),
    }
    if sft_format == SFT_FORMAT_V2:
        temporary["train_constraint_mask"] = _temporary_path(
            destination / "train.constraint-mask"
        )
        temporary["validation_constraint_mask"] = _temporary_path(
            destination / "validation.constraint-mask"
        )
    try:
        _write_bytes(temporary["tokenizer"], tokenizer_bytes)
        train_details = _write_sft_split(
            temporary["train_tokens"],
            temporary["train_index"],
            train_records,
            tokenizer_source.tokenizer,
            constraint_mask_path=temporary.get("train_constraint_mask"),
            sft_format=sft_format,
            split="train",
            limit=train_limit,
            show_progress=show_progress,
        )
        validation_details = _write_sft_split(
            temporary["validation_tokens"],
            temporary["validation_index"],
            validation_records,
            tokenizer_source.tokenizer,
            constraint_mask_path=temporary.get("validation_constraint_mask"),
            sft_format=sft_format,
            split="validation",
            limit=validation_limit,
            show_progress=show_progress,
        )
        artifacts = {
            "tokenizer": destination / f"tokenizer-{tokenizer_sha256}.json",
            "train_tokens": destination
            / f"train-tokens-{train_details['token_sha256']}.bin",
            "train_index": destination
            / f"train-index-{train_details['index_sha256']}.bin",
            "validation_tokens": destination
            / f"validation-tokens-{validation_details['token_sha256']}.bin",
            "validation_index": destination
            / f"validation-index-{validation_details['index_sha256']}.bin",
        }
        if sft_format == SFT_FORMAT_V2:
            artifacts["train_constraint_mask"] = destination / (
                "train-constraint-mask-"
                f"{train_details['constraint_mask_sha256']}.bin"
            )
            artifacts["validation_constraint_mask"] = destination / (
                "validation-constraint-mask-"
                f"{validation_details['constraint_mask_sha256']}.bin"
            )
        train_details["token_file"] = artifacts["train_tokens"].name
        train_details["index_file"] = artifacts["train_index"].name
        validation_details["token_file"] = artifacts["validation_tokens"].name
        validation_details["index_file"] = artifacts["validation_index"].name
        if sft_format == SFT_FORMAT_V2:
            train_details["constraint_mask_file"] = artifacts[
                "train_constraint_mask"
            ].name
            validation_details["constraint_mask_file"] = artifacts[
                "validation_constraint_mask"
            ].name
        copied_tokenizer = dict(tokenizer_details)
        copied_tokenizer["file"] = artifacts["tokenizer"].name
        copied_tokenizer["reused_from"] = {
            "dataset_fingerprint": tokenizer_source.fingerprint,
            "tokenizer_sha256": tokenizer_sha256,
        }
        metadata: dict[str, Any] = {
            "schema_version": SFT_METADATA_SCHEMA_VERSION,
            "task": SFT_TASK,
            "dataset": {
                "name": dataset_name,
                "revision": revision,
            },
            "tokenizer": copied_tokenizer,
            "splits": {
                "train": train_details,
                "validation": validation_details,
            },
            "config": {
                "train_limit": train_limit,
                "validation_limit": validation_limit,
                "target_mask": "response_only",
                "prompt_format": [*_FIELD_ORDER, "Story"],
                "source_decoding": SFT_SOURCE_DECODING,
            },
        }
        if sft_format == SFT_FORMAT_V2:
            metadata["config"].update(
                {
                    "sft_format": SFT_FORMAT_V2,
                    "instruction_prefix": SFT_V2_INSTRUCTION,
                    "required_word_weight": float(required_word_weight),
                    "constraint_mask": "required_word_response_tokens",
                }
            )
        metadata["fingerprint"] = metadata_fingerprint(metadata)
        _write_json(temporary["metadata"], metadata)
        for name, artifact in artifacts.items():
            os.replace(temporary[name], artifact)
        os.replace(temporary["metadata"], metadata_path)
        current_artifacts = set(artifacts.values())
        for obsolete in previous_artifacts - current_artifacts:
            with suppress(OSError):
                obsolete.unlink(missing_ok=True)
        return metadata
    except BaseException:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise


def _write_sft_split(
    token_path: Path,
    index_path: Path,
    records: Iterable[str],
    tokenizer: ByteBPETokenizer,
    *,
    constraint_mask_path: Path | None,
    sft_format: str,
    split: str,
    limit: int,
    show_progress: bool,
) -> dict[str, Any]:
    selected = records
    if show_progress:
        from tqdm.auto import tqdm

        selected = tqdm(
            selected,
            desc=f"Preparing SFT {split}",
            total=limit or None,
            unit="records",
        )
    token_digest = hashlib.sha256()
    index_digest = hashlib.sha256()
    constraint_mask_digest = hashlib.sha256()
    record_count = 0
    skipped_records = 0
    token_count = 0
    response_targets = 0
    constraint_targets = 0
    required_words = 0
    matched_required_words = 0
    mask_stream = (
        constraint_mask_path.open("wb") if constraint_mask_path is not None else None
    )
    try:
        token_stream = token_path.open("wb")
        index_stream = index_path.open("wb")
        try:
            for raw_record in selected:
                if limit and record_count >= limit:
                    break
                try:
                    example = parse_tinystories_instruct_record(
                        raw_record,
                        sft_format=sft_format,
                    )
                except ValueError:
                    # The pinned validation file begins with a truncated fragment
                    # from a preceding record. Keep preparation robust to such
                    # structurally incomplete source fragments and record how many
                    # were excluded.
                    skipped_records += 1
                    continue
                prompt_ids = tokenizer.encode(example.prompt)
                if mask_stream is None:
                    response_ids = tokenizer.encode(example.response)
                    response_offsets: list[tuple[int, int]] = []
                else:
                    response_ids, response_offsets = tokenizer.encode_with_offsets(
                        example.response
                    )
                sequence = [
                    tokenizer.bos_id,
                    *prompt_ids,
                    *response_ids,
                    tokenizer.eos_id,
                ]
                response_start = 1 + len(prompt_ids)
                encoded_tokens = np.asarray(sequence, dtype="<u2").tobytes()
                encoded_index = np.asarray(
                    (token_count, response_start, len(sequence)),
                    dtype="<u8",
                ).tobytes()
                token_stream.write(encoded_tokens)
                index_stream.write(encoded_index)
                token_digest.update(encoded_tokens)
                index_digest.update(encoded_index)
                if mask_stream is not None:
                    constraint_mask, matched = _constraint_mask(
                        example,
                        response_offsets,
                        sequence_length=len(sequence),
                        response_start=response_start,
                    )
                    encoded_mask = constraint_mask.tobytes()
                    mask_stream.write(encoded_mask)
                    constraint_mask_digest.update(encoded_mask)
                    constraint_targets += int(constraint_mask.sum())
                    required_words += len(example.required_words)
                    matched_required_words += matched
                record_count += 1
                token_count += len(sequence)
                response_targets += len(sequence) - response_start
            token_stream.flush()
            index_stream.flush()
            os.fsync(token_stream.fileno())
            os.fsync(index_stream.fileno())
            if mask_stream is not None:
                mask_stream.flush()
                os.fsync(mask_stream.fileno())
        finally:
            token_stream.close()
            index_stream.close()
    finally:
        if mask_stream is not None:
            mask_stream.close()
    if record_count == 0:
        raise ValueError(f"{split} contains no valid instruction records")
    details = {
        "records": record_count,
        "skipped_records": skipped_records,
        "tokens": token_count,
        "response_targets": response_targets,
        "token_bytes": token_count * np.dtype("<u2").itemsize,
        "index_bytes": record_count * SFT_INDEX_COLUMNS * np.dtype("<u8").itemsize,
        "token_sha256": token_digest.hexdigest(),
        "index_sha256": index_digest.hexdigest(),
    }
    if mask_stream is not None:
        details.update(
            {
                "constraint_mask_bytes": token_count,
                "constraint_mask_sha256": constraint_mask_digest.hexdigest(),
                "constraint_targets": constraint_targets,
                "required_words": required_words,
                "matched_required_words": matched_required_words,
            }
        )
    return details


class PreparedSFTData:
    """Validated response-masked SFT data with deterministic chunk sampling."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        seed: int = 42,
        verify_integrity: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        try:
            loaded = json.loads(
                (self.data_dir / "metadata.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("cannot read prepared SFT metadata") from error
        self.metadata = _mapping(loaded, "metadata")
        if self.metadata.get("schema_version") != SFT_METADATA_SCHEMA_VERSION:
            raise ValueError("unsupported prepared SFT schema version")
        if self.metadata.get("task") != SFT_TASK:
            raise ValueError("prepared data is not a TinyStoriesInstruct SFT dataset")
        fingerprint = self.metadata.get("fingerprint")
        if (
            not isinstance(fingerprint, str)
            or fingerprint != metadata_fingerprint(self.metadata)
        ):
            raise ValueError("prepared SFT fingerprint mismatch")
        self.fingerprint = fingerprint
        tokenizer_details = _mapping(self.metadata.get("tokenizer"), "tokenizer")
        tokenizer_file = _safe_name(tokenizer_details.get("file"), "tokenizer file")
        tokenizer_path = self.data_dir / tokenizer_file
        if verify_integrity and _sha256_file(tokenizer_path) != tokenizer_details.get(
            "sha256"
        ):
            raise ValueError("prepared SFT tokenizer checksum mismatch")
        self.tokenizer = ByteBPETokenizer.load(tokenizer_path)
        config = _mapping(self.metadata.get("config"), "config")
        self.sft_format = config.get("sft_format", SFT_FORMAT_V1)
        _validate_sft_format(self.sft_format)
        raw_required_word_weight = config.get("required_word_weight", 1.0)
        _validate_required_word_weight(raw_required_word_weight)
        self.required_word_weight = float(raw_required_word_weight)
        if (
            self.sft_format == SFT_FORMAT_V2
            and config.get("instruction_prefix") != SFT_V2_INSTRUCTION
        ):
            raise ValueError("prepared SFT v2 instruction prefix is incompatible")
        self._tokens: dict[str, np.memmap] = {}
        self._indices: dict[str, np.memmap] = {}
        self._constraint_masks: dict[str, np.memmap] = {}
        self._chunks: dict[tuple[str, int], np.ndarray] = {}
        splits = _mapping(self.metadata.get("splits"), "splits")
        for split in SFT_SPLITS:
            details = _mapping(splits.get(split), f"splits.{split}")
            records = _positive_metadata_int(details.get("records"), "records")
            tokens = _positive_metadata_int(details.get("tokens"), "tokens")
            token_path = self.data_dir / _safe_name(
                details.get("token_file"),
                "token file",
            )
            index_path = self.data_dir / _safe_name(
                details.get("index_file"),
                "index file",
            )
            if verify_integrity:
                if _sha256_file(token_path) != details.get("token_sha256"):
                    raise ValueError(f"prepared SFT {split} token checksum mismatch")
                if _sha256_file(index_path) != details.get("index_sha256"):
                    raise ValueError(f"prepared SFT {split} index checksum mismatch")
            if token_path.stat().st_size != tokens * np.dtype("<u2").itemsize:
                raise ValueError(f"prepared SFT {split} token size mismatch")
            expected_index_bytes = (
                records * SFT_INDEX_COLUMNS * np.dtype("<u8").itemsize
            )
            if index_path.stat().st_size != expected_index_bytes:
                raise ValueError(f"prepared SFT {split} index size mismatch")
            self._tokens[split] = np.memmap(
                token_path,
                dtype="<u2",
                mode="r",
                shape=(tokens,),
            )
            self._indices[split] = np.memmap(
                index_path,
                dtype="<u8",
                mode="r",
                shape=(records, SFT_INDEX_COLUMNS),
            )
            if self.sft_format == SFT_FORMAT_V2:
                mask_path = self.data_dir / _safe_name(
                    details.get("constraint_mask_file"),
                    "constraint mask file",
                )
                if verify_integrity and _sha256_file(mask_path) != details.get(
                    "constraint_mask_sha256"
                ):
                    raise ValueError(
                        f"prepared SFT {split} constraint mask checksum mismatch"
                    )
                if mask_path.stat().st_size != tokens:
                    raise ValueError(
                        f"prepared SFT {split} constraint mask size mismatch"
                    )
                self._constraint_masks[split] = np.memmap(
                    mask_path,
                    dtype=np.uint8,
                    mode="r",
                    shape=(tokens,),
                )
        self._generator = torch.Generator(device="cpu").manual_seed(seed)

    def chunks(self, split: SFTSplit, context_length: int) -> np.ndarray:
        """Return record/chunk offsets covering every response target once."""

        if context_length < 1:
            raise ValueError("context_length must be positive")
        key = (split, context_length)
        if key in self._chunks:
            return self._chunks[key]
        chunks: list[tuple[int, int]] = []
        for record_index, (_, response_start, total_length) in enumerate(
            self._indices[split]
        ):
            first_offset = ((int(response_start) - 1) // context_length) * context_length
            for offset in range(first_offset, int(total_length) - 1, context_length):
                chunks.append((record_index, offset))
        if not chunks:
            raise ValueError(f"{split} has no supervised response chunks")
        result = np.asarray(chunks, dtype=np.int64)
        self._chunks[key] = result
        return result

    def sft_batch(
        self,
        split: SFTSplit,
        chunk_indices: torch.Tensor | list[int],
        *,
        context_length: int,
        device: str | torch.device | None = None,
        include_loss_weights: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Materialize chunks with prompt and padding targets masked to -100."""

        chunks = self.chunks(split, context_length)
        selected = (
            chunk_indices.detach().cpu().tolist()
            if isinstance(chunk_indices, torch.Tensor)
            else list(chunk_indices)
        )
        inputs = np.full(
            (len(selected), context_length),
            self.tokenizer.pad_id,
            dtype=np.int64,
        )
        targets = np.full((len(selected), context_length), -100, dtype=np.int64)
        loss_weights = (
            np.zeros(
                (len(selected), context_length),
                dtype=np.float32,
            )
            if include_loss_weights
            else None
        )
        for row, chunk_index in enumerate(selected):
            record_index, local_offset = chunks[int(chunk_index)]
            start, response_start, total_length = (
                int(value) for value in self._indices[split][record_index]
            )
            valid = min(context_length, total_length - 1 - local_offset)
            sequence = np.asarray(
                self._tokens[split][
                    start + local_offset : start + local_offset + valid + 1
                ],
                dtype=np.int64,
            )
            inputs[row, :valid] = sequence[:-1]
            target_values = sequence[1:]
            target_positions = local_offset + 1 + np.arange(valid)
            supervised = target_positions >= response_start
            targets[row, :valid][supervised] = target_values[supervised]
            if loss_weights is not None:
                loss_weights[row, :valid][supervised] = 1.0
            if self.sft_format == SFT_FORMAT_V2 and loss_weights is not None:
                constraint_mask = np.asarray(
                    self._constraint_masks[split][
                        start
                        + local_offset
                        + 1 : start
                        + local_offset
                        + valid
                        + 1
                    ],
                    dtype=np.bool_,
                )
                loss_weights[row, :valid][
                    supervised & constraint_mask
                ] = self.required_word_weight
        input_tensor = torch.from_numpy(inputs)
        target_tensor = torch.from_numpy(targets)
        if device is not None:
            input_tensor = input_tensor.to(device, non_blocking=True)
            target_tensor = target_tensor.to(device, non_blocking=True)
        if include_loss_weights:
            assert loss_weights is not None
            loss_weight_tensor = torch.from_numpy(loss_weights)
            if device is not None:
                loss_weight_tensor = loss_weight_tensor.to(
                    device,
                    non_blocking=True,
                )
            return input_tensor, target_tensor, loss_weight_tensor
        return input_tensor, target_tensor

    def format_prompt(self, prompt: str) -> str:
        """Apply the prepared dataset's instruction prefix exactly once."""

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if self.sft_format == SFT_FORMAT_V2 and not prompt.startswith(
            SFT_V2_INSTRUCTION
        ):
            return SFT_V2_INSTRUCTION + prompt
        return prompt

    def get_batch(
        self,
        split: SFTSplit,
        *,
        batch_size: int,
        context_length: int,
        device: str | torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chunks = self.chunks(split, context_length)
        selected = torch.randint(
            0,
            len(chunks),
            (batch_size,),
            generator=generator if generator is not None else self._generator,
        )
        return self.sft_batch(
            split,
            selected,
            context_length=context_length,
            device=device,
        )

    def state_dict(self) -> dict[str, Any]:
        return {"fingerprint": self.fingerprint}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("fingerprint") != self.fingerprint:
            raise ValueError("prepared SFT data state fingerprint mismatch")


class SFTBatchSampler:
    """Deterministic epoch-shuffled sampler over response-supervised chunks."""

    def __init__(
        self,
        data: PreparedSFTData,
        split: SFTSplit,
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
        self.chunk_count = len(data.chunks(split, context_length))
        self.epoch = 0
        self.cursor = 0
        self._order = self._make_order()

    def next_indices(self, count: int) -> torch.Tensor:
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("batch size must be an integer")
        if count < 1:
            raise ValueError("batch size must be at least 1")
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
        return self.data.sft_batch(
            self.split,
            self.next_indices(batch_size),
            context_length=self.context_length,
            device=device,
        )

    def get_training_batch(
        self,
        *,
        batch_size: int,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result = self.data.sft_batch(
            self.split,
            self.next_indices(batch_size),
            context_length=self.context_length,
            device=device,
            include_loss_weights=True,
        )
        assert len(result) == 3
        return result

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
            raise ValueError("SFT sampler state is incompatible")
        epoch = state.get("epoch")
        cursor = state.get("cursor")
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("SFT sampler epoch is invalid")
        if not isinstance(cursor, int) or not 0 <= cursor < self.chunk_count:
            raise ValueError("SFT sampler cursor is invalid")
        self.epoch = epoch
        self.cursor = cursor
        self._order = self._make_order()

    def _make_order(self) -> torch.Tensor:
        if not self.shuffle:
            return torch.arange(self.chunk_count)
        generator = torch.Generator(device="cpu").manual_seed(self.seed + self.epoch)
        return torch.randperm(self.chunk_count, generator=generator)


def load_prepared_data(
    data_dir: str | Path,
    *,
    seed: int = 42,
) -> PreparedTokenData | PreparedSFTData:
    """Load either pretraining or SFT prepared data from its metadata task."""

    metadata_path = Path(data_dir) / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read prepared metadata at {metadata_path}") from error
    if isinstance(metadata, Mapping) and metadata.get("task") == SFT_TASK:
        return PreparedSFTData(data_dir, seed=seed)
    return PreparedTokenData(data_dir, seed=seed)


def _constraint_mask(
    example: InstructionExample,
    response_offsets: list[tuple[int, int]],
    *,
    sequence_length: int,
    response_start: int,
) -> tuple[np.ndarray, int]:
    mask = np.zeros(sequence_length, dtype=np.uint8)
    matched_words = 0
    for required_word in example.required_words:
        pattern = re.compile(
            r"(?<![A-Za-z])" + re.escape(required_word) + r"(?![A-Za-z])",
            flags=re.IGNORECASE,
        )
        matches = list(pattern.finditer(example.response))
        if matches:
            matched_words += 1
        for match in matches:
            for token_index, (start, end) in enumerate(response_offsets):
                if end > match.start() and start < match.end():
                    mask[response_start + token_index] = 1
    return mask, matched_words


def _referenced_artifacts(metadata_path: Path) -> set[Path]:
    if not metadata_path.exists():
        return set()
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        tokenizer = _mapping(metadata.get("tokenizer"), "tokenizer")
        splits = _mapping(metadata.get("splits"), "splits")
        names = [tokenizer.get("file")]
        for split in SFT_SPLITS:
            details = _mapping(splits.get(split), f"splits.{split}")
            names.extend((details.get("token_file"), details.get("index_file")))
            names.append(details.get("constraint_mask_file"))
    except (OSError, json.JSONDecodeError, ValueError):
        return set()
    return {
        metadata_path.parent / name
        for name in names
        if isinstance(name, str) and Path(name).name == name
    }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _safe_name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"prepared SFT {name} is invalid")
    return value


def _positive_metadata_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"prepared SFT {name} must be positive")
    return value


def _validate_limit(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_sft_format(value: str) -> None:
    if value not in SFT_FORMATS:
        raise ValueError(f"sft_format must be one of {SFT_FORMATS}")


def _validate_required_word_weight(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("required_word_weight must be numeric")
    if not np.isfinite(value) or value < 1:
        raise ValueError("required_word_weight must be finite and at least 1")


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
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
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


__all__ = [
    "DEFAULT_INSTRUCT_DATASET",
    "DEFAULT_INSTRUCT_REVISION",
    "DEFAULT_INSTRUCT_TRAIN_LIMIT",
    "DEFAULT_INSTRUCT_VALIDATION_LIMIT",
    "DEFAULT_REQUIRED_WORD_WEIGHT",
    "INSTRUCT_RECORD_SEPARATOR",
    "SFT_FORMATS",
    "SFT_FORMAT_V1",
    "SFT_FORMAT_V2",
    "SFT_SOURCE_DECODING",
    "SFT_V2_INSTRUCTION",
    "InstructionExample",
    "PreparedSFTData",
    "SFTBatchSampler",
    "iter_raw_instruct_records",
    "load_prepared_data",
    "parse_tinystories_instruct_record",
    "prepare_instruct_from_records",
    "prepare_tinystories_instruct",
]
