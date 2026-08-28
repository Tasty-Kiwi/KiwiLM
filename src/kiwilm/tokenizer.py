"""Byte-level BPE tokenization for KiwiLM language models."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)
MAX_UINT16_VOCAB_SIZE = 65_535
MIN_BYTE_BPE_VOCAB_SIZE = len(SPECIAL_TOKENS) + 256


class ReservedTokenError(ValueError):
    """Raised when untrusted source text contains a tokenizer control token."""

    def __init__(self, token: str) -> None:
        self.token = token
        super().__init__(f"text contains reserved tokenizer control token {token!r}")


def _tokenizers_api() -> dict[str, Any]:
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "Byte BPE tokenization requires the `tokenizers` package"
        ) from error
    return {
        "Tokenizer": Tokenizer,
        "decoders": decoders,
        "models": models,
        "pre_tokenizers": pre_tokenizers,
        "trainers": trainers,
    }


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _validate_vocab_size(vocab_size: int) -> None:
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise TypeError("vocab_size must be an integer")
    if vocab_size < MIN_BYTE_BPE_VOCAB_SIZE:
        raise ValueError(
            f"vocab_size must be at least {MIN_BYTE_BPE_VOCAB_SIZE} "
            "to include the byte alphabet and special tokens"
        )
    if vocab_size > MAX_UINT16_VOCAB_SIZE:
        raise ValueError(
            f"vocab_size must not exceed {MAX_UINT16_VOCAB_SIZE} "
            "because prepared token IDs use uint16"
        )


class ByteBPETokenizer:
    """A small Hugging Face byte-level BPE wrapper with explicit boundaries."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self.pad_id = self._required_id(PAD_TOKEN)
        self.unk_id = self._required_id(UNK_TOKEN)
        self.bos_id = self._required_id(BOS_TOKEN)
        self.eos_id = self._required_id(EOS_TOKEN)
        actual_ids = (self.pad_id, self.unk_id, self.bos_id, self.eos_id)
        expected_ids = tuple(range(len(SPECIAL_TOKENS)))
        if actual_ids != expected_ids:
            raise ValueError(
                "tokenizer special-token IDs must be "
                f"{expected_ids}, found {actual_ids}"
            )
        if self.vocab_size > MAX_UINT16_VOCAB_SIZE:
            raise ValueError(
                f"tokenizer vocabulary has {self.vocab_size} entries; "
                f"the uint16 limit is {MAX_UINT16_VOCAB_SIZE}"
            )

    def _required_id(self, token: str) -> int:
        value = self._tokenizer.token_to_id(token)
        if value is None:
            raise ValueError(f"tokenizer is missing required token {token}")
        return int(value)

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int = 8192,
        min_frequency: int = 2,
        show_progress: bool = False,
    ) -> ByteBPETokenizer:
        """Train only on ``texts`` and return a deterministic byte BPE tokenizer."""

        _validate_vocab_size(vocab_size)
        if isinstance(min_frequency, bool) or not isinstance(min_frequency, int):
            raise TypeError("min_frequency must be an integer")
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")

        api = _tokenizers_api()
        tokenizer = api["Tokenizer"](
            api["models"].BPE(unk_token=UNK_TOKEN, byte_fallback=True)
        )
        tokenizer.pre_tokenizer = api["pre_tokenizers"].ByteLevel(
            add_prefix_space=False
        )
        tokenizer.decoder = api["decoders"].ByteLevel()
        trainer = api["trainers"].BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=api["pre_tokenizers"].ByteLevel.alphabet(),
            show_progress=show_progress,
        )

        seen = 0

        def checked_texts() -> Iterable[str]:
            nonlocal seen
            for text in texts:
                if not isinstance(text, str):
                    raise TypeError(
                        "tokenizer training texts must be strings, "
                        f"found {type(text).__name__}"
                    )
                seen += 1
                yield text

        tokenizer.train_from_iterator(checked_texts(), trainer=trainer)
        if seen == 0:
            raise ValueError("at least one tokenizer training text is required")
        return cls(tokenizer)

    @classmethod
    def from_json(cls, value: str) -> ByteBPETokenizer:
        if not isinstance(value, str):
            raise TypeError("tokenizer JSON must be a string")
        try:
            tokenizer = _tokenizers_api()["Tokenizer"].from_str(value)
        except Exception as error:
            raise ValueError("invalid tokenizer JSON") from error
        return cls(tokenizer)

    @classmethod
    def load(cls, path: str | Path) -> ByteBPETokenizer:
        tokenizer_path = Path(path)
        try:
            value = tokenizer_path.read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError(f"cannot read tokenizer from {tokenizer_path}") from error
        return cls.from_json(value)

    def to_json(self, *, pretty: bool = True) -> str:
        return str(self._tokenizer.to_str(pretty=pretty))

    def save(self, path: str | Path) -> None:
        _atomic_write_text(Path(path), self.to_json() + "\n")

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        self._validate_text(text)
        ids = [int(token_id) for token_id in self._tokenizer.encode(text).ids]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Encode text and return source-character offsets for each token."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        self._validate_text(text)
        encoding = self._tokenizer.encode(text)
        return (
            [int(token_id) for token_id in encoding.ids],
            [(int(start), int(end)) for start, end in encoding.offsets],
        )

    def decode(
        self,
        ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        token_ids = [int(token_id) for token_id in ids]
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in token_ids):
            raise ValueError("token IDs must be within the tokenizer vocabulary")
        return str(
            self._tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
            )
        )

    def decode_stream(
        self,
        ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> Iterable[str]:
        """Incrementally decode token IDs without corrupting split UTF-8 bytes."""

        api = _tokenizers_api()
        stream = api["decoders"].DecodeStream(
            skip_special_tokens=skip_special_tokens
        )
        for token_id in ids:
            resolved_id = int(token_id)
            if resolved_id < 0 or resolved_id >= self.vocab_size:
                raise ValueError("token IDs must be within the tokenizer vocabulary")
            chunk = stream.step(self._tokenizer, resolved_id)
            if chunk is not None:
                yield str(chunk)

    @property
    def vocab_size(self) -> int:
        return int(self._tokenizer.get_vocab_size(with_added_tokens=True))

    @staticmethod
    def _validate_text(text: str) -> None:
        reserved = next((token for token in SPECIAL_TOKENS if token in text), None)
        if reserved is not None:
            raise ReservedTokenError(reserved)


# The shorter alias is useful in type annotations while keeping the byte-level
# implementation explicit at public call sites.
BPETokenizer = ByteBPETokenizer


__all__ = [
    "BOS_TOKEN",
    "EOS_TOKEN",
    "MAX_UINT16_VOCAB_SIZE",
    "PAD_TOKEN",
    "SPECIAL_TOKENS",
    "UNK_TOKEN",
    "BPETokenizer",
    "ByteBPETokenizer",
    "ReservedTokenError",
]
