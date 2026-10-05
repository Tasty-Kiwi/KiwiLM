"""V3 BPE with a real serialized MASK token, independent of V2 artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from tokenizers import AddedToken, Tokenizer

from kiwilm.tokenizer import ByteBPETokenizer, ReservedTokenError

MASK_TOKEN = "[MASK]"


class MaskBPETokenizer(ByteBPETokenizer):
    """Append one special token; old content/control IDs and BPE merges remain intact."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__(tokenizer)
        self.mask_id = self._required_id(MASK_TOKEN)
        description = json.loads(self.to_json())
        added = [entry for entry in description["added_tokens"] if entry["content"] == MASK_TOKEN]
        if (
            self.mask_id != self.vocab_size - 1
            or len(added) != 1
            or not added[0]["special"]
            or added[0]["normalized"]
            or added[0]["lstrip"]
            or added[0]["rstrip"]
            or added[0]["single_word"]
            or MASK_TOKEN in description["model"]["vocab"]
        ):
            raise ValueError("V3 MASK must be an appended, unnormalized special token")

    @classmethod
    def from_base(cls, base: ByteBPETokenizer) -> MaskBPETokenizer:
        tokenizer = Tokenizer.from_str(base.to_json())
        if tokenizer.token_to_id(MASK_TOKEN) is not None:
            raise ValueError("base tokenizer already contains MASK; load the V3 tokenizer instead")
        tokenizer.add_special_tokens([AddedToken(MASK_TOKEN, normalized=False, special=True)])
        result = cls(tokenizer)
        result.assert_base_compatible(base)
        return result

    def assert_base_compatible(self, base: ByteBPETokenizer) -> None:
        description = json.loads(self.to_json())
        description["added_tokens"] = [
            entry for entry in description["added_tokens"] if entry["content"] != MASK_TOKEN
        ]
        if description != json.loads(base.to_json()):
            raise ValueError("V3 tokenizer is not an ID-preserving extension of prepared data")

    @property
    def fingerprint(self) -> str:
        """Canonical tokenizer identity, distinct from the original prepared-data hash."""
        return hashlib.sha256(self.to_json(pretty=False).encode()).hexdigest()

    @staticmethod
    def _validate_text(text: str) -> None:
        ByteBPETokenizer._validate_text(text)
        if MASK_TOKEN in text:
            raise ReservedTokenError(MASK_TOKEN)

    def save(self, path: str | Path) -> None:
        """Create a separate local artifact, never overwrite a V2 tokenizer."""
        destination = Path(path)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"tokenizer already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".v3-tokenizer-", dir=destination.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(self.to_json() + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
