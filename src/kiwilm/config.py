"""Configuration objects shared by KiwiLM model variants."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Self, cast


def _require_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Settings shared by every causal language-model architecture."""

    architecture: str = "gated_cnn"
    vocab_size: int = 8192
    context_length: int = 256
    d_model: int = 256
    dropout: float = 0.1
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.architecture, str) or not self.architecture.strip():
            raise ValueError("architecture must be a non-empty string")
        _require_positive_int("vocab_size", self.vocab_size)
        _require_positive_int("context_length", self.context_length)
        _require_positive_int("d_model", self.d_model)
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(self.dropout)
            or not 0.0 <= self.dropout < 1.0
        ):
            raise ValueError("dropout must be finite and in [0, 1)")
        if not isinstance(self.tie_embeddings, bool):
            raise ValueError("tie_embeddings must be a boolean")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible configuration mapping."""

        result: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            result[field.name] = list(value) if isinstance(value, tuple) else value
        return result

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        """Reconstruct a config, dispatching the built-in architecture when possible."""

        data = dict(values)
        if cls is ModelConfig and data.get("architecture", "gated_cnn") == "gated_cnn":
            return cast(Self, GatedCNNConfig.from_dict(data))
        return cls(**data)


@dataclass(frozen=True, slots=True)
class GatedCNNConfig(ModelConfig):
    """Configuration for the causal gated-convolution baseline."""

    architecture: str = "gated_cnn"
    num_layers: int = 8
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)

    def __post_init__(self) -> None:
        super(GatedCNNConfig, self).__post_init__()
        if self.architecture != "gated_cnn":
            raise ValueError("GatedCNNConfig architecture must be 'gated_cnn'")
        _require_positive_int("num_layers", self.num_layers)
        _require_positive_int("kernel_size", self.kernel_size)

        if not isinstance(self.dilations, tuple):
            try:
                object.__setattr__(self, "dilations", tuple(self.dilations))
            except TypeError as error:
                raise ValueError("dilations must be a sequence of positive integers") from error
        if len(self.dilations) != self.num_layers:
            raise ValueError("dilations must contain exactly num_layers entries")
        for dilation in self.dilations:
            _require_positive_int("each dilation", dilation)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        """Reconstruct a gated-CNN config from a plain mapping."""

        data = dict(values)
        if "dilations" in data:
            raw_dilations = data["dilations"]
            if isinstance(raw_dilations, (str, bytes)):
                raise ValueError("dilations must be a sequence of positive integers")
            try:
                data["dilations"] = tuple(raw_dilations)  # type: ignore[arg-type]
            except TypeError as error:
                raise ValueError("dilations must be a sequence of positive integers") from error
        return cls(**data)
