"""Configuration objects for the KiwiLM 2 model family."""

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
    """Settings shared by the active KiwiLM 2 variants."""

    architecture: str = "kiwilm2"
    vocab_size: int = 32_000
    context_length: int = 512
    d_model: int = 512
    dropout: float = 0.0
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
        """Reconstruct one of the active KiwiLM 2 configurations."""

        data = dict(values)
        if cls is not ModelConfig:
            return cls(**data)
        architecture = data.get("architecture", "kiwilm2")
        if architecture == "kiwilm2":
            return cast(Self, KiwiLM2Config.from_dict(data))
        if architecture == "kiwilm2_slim":
            return cast(Self, KiwiLM2SlimConfig.from_dict(data))
        if architecture == "kiwilm2_slim_v3":
            return cast(Self, KiwiLM2SlimV3Config.from_dict(data))
        raise ValueError(
            f"unsupported model architecture {architecture!r} on master; "
            "check out the legacy branch to load historical KiwiLM checkpoints"
        )


KIWILM2_MIXER_SCHEDULE = (
    "gqa",
    "conv",
    "conv",
    "gqa",
    "conv",
    "conv",
    "gqa",
    "conv",
    "conv",
    "gqa",
)


@dataclass(frozen=True, slots=True)
class KiwiLM2Config(ModelConfig):
    """Frozen KiwiLM 2 backbone with SwiGLU feed-forward blocks."""

    architecture: str = "kiwilm2"
    mixer_schedule: tuple[str, ...] = KIWILM2_MIXER_SCHEDULE
    conv_kernel_sizes: tuple[int, ...] = (31, 63, 31, 63, 31, 63)
    num_query_heads: int = 8
    num_kv_heads: int = 2
    swiglu_dim: int = 1_536
    bigram_buckets: int = 16_384
    trigram_buckets: int = 16_384
    rms_norm_eps: float = 1e-6
    rope_base: float = 10_000.0

    def __post_init__(self) -> None:
        super(KiwiLM2Config, self).__post_init__()
        if self.architecture != "kiwilm2":
            raise ValueError("KiwiLM2Config architecture must be 'kiwilm2'")
        self._validate_kiwilm2_fields()

    def _validate_kiwilm2_fields(self) -> None:
        if not isinstance(self.mixer_schedule, tuple):
            object.__setattr__(self, "mixer_schedule", tuple(self.mixer_schedule))
        if self.mixer_schedule != KIWILM2_MIXER_SCHEDULE:
            raise ValueError("mixer_schedule must match the frozen KiwiLM 2 schedule")
        if not isinstance(self.conv_kernel_sizes, tuple):
            object.__setattr__(self, "conv_kernel_sizes", tuple(self.conv_kernel_sizes))
        if len(self.conv_kernel_sizes) != self.mixer_schedule.count("conv"):
            raise ValueError("conv_kernel_sizes must contain one entry per conv block")
        for kernel_size in self.conv_kernel_sizes:
            _require_positive_int("each conv kernel size", kernel_size)
            if kernel_size % 2 == 0:
                raise ValueError("each conv kernel size must be odd")
        _require_positive_int("num_query_heads", self.num_query_heads)
        _require_positive_int("num_kv_heads", self.num_kv_heads)
        _require_positive_int("swiglu_dim", self.swiglu_dim)
        _require_positive_int("bigram_buckets", self.bigram_buckets)
        _require_positive_int("trigram_buckets", self.trigram_buckets)
        if self.d_model % self.num_query_heads != 0:
            raise ValueError("d_model must be divisible by num_query_heads")
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")
        if (self.d_model // self.num_query_heads) % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        for name in ("rms_norm_eps", "rope_base"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        data = dict(values)
        for name in ("mixer_schedule", "conv_kernel_sizes"):
            if name in data:
                raw = data[name]
                if isinstance(raw, (str, bytes)):
                    raise ValueError(f"{name} must be a sequence")
                data[name] = tuple(raw)  # type: ignore[arg-type]
        return cls(**data)


@dataclass(frozen=True, slots=True)
class KiwiLM2SlimConfig(KiwiLM2Config):
    """KiwiLM 2 backbone with structured width-preserving Hadamard MLPs."""

    architecture: str = "kiwilm2_slim"
    hadamard_variant: str = "gated_v2"

    def __post_init__(self) -> None:
        ModelConfig.__post_init__(self)
        if self.architecture != "kiwilm2_slim":
            raise ValueError("KiwiLM2SlimConfig architecture must be 'kiwilm2_slim'")
        self._validate_kiwilm2_fields()
        if self.d_model & (self.d_model - 1):
            raise ValueError("Hadamard MLP requires d_model to be a power of two")
        if self.hadamard_variant not in {"minimal_v1", "gated_v2"}:
            raise ValueError("hadamard_variant must be 'minimal_v1' or 'gated_v2'")

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        data = dict(values)
        # Checkpoints produced before gated Slim v2 predate this discriminator.
        # Treating a missing value as v1 preserves their exact state-dict shape.
        data.setdefault("hadamard_variant", "minimal_v1")
        for name in ("mixer_schedule", "conv_kernel_sizes"):
            if name in data:
                raw = data[name]
                if isinstance(raw, (str, bytes)):
                    raise ValueError(f"{name} must be a sequence")
                data[name] = tuple(raw)  # type: ignore[arg-type]
        return cls(**data)


@dataclass(frozen=True, slots=True)
class KiwiLM2SlimV3Config(KiwiLM2Config):
    """Hybrid Slim v3 with Hadamard lower blocks and dense upper FFNs."""

    architecture: str = "kiwilm2_slim_v3"
    hadamard_variant: str = "gated_v2"
    upper_swiglu_blocks: int = 4

    def __post_init__(self) -> None:
        ModelConfig.__post_init__(self)
        if self.architecture != "kiwilm2_slim_v3":
            raise ValueError(
                "KiwiLM2SlimV3Config architecture must be 'kiwilm2_slim_v3'"
            )
        self._validate_kiwilm2_fields()
        if self.d_model & (self.d_model - 1):
            raise ValueError("Hadamard MLP requires d_model to be a power of two")
        if self.hadamard_variant != "gated_v2":
            raise ValueError("Slim v3 requires hadamard_variant='gated_v2'")
        if (
            isinstance(self.upper_swiglu_blocks, bool)
            or not isinstance(self.upper_swiglu_blocks, int)
            or self.upper_swiglu_blocks not in {3, 4}
        ):
            raise ValueError("upper_swiglu_blocks must be 3 or 4")

    @property
    def mlp_schedule(self) -> tuple[str, ...]:
        """Return the frozen contiguous lower-Hadamard/upper-SwiGLU schedule."""

        hadamard_blocks = len(self.mixer_schedule) - self.upper_swiglu_blocks
        return ("hadamard",) * hadamard_blocks + (
            "swiglu",
        ) * self.upper_swiglu_blocks

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        data = dict(values)
        for name in ("mixer_schedule", "conv_kernel_sizes"):
            if name in data:
                raw = data[name]
                if isinstance(raw, (str, bytes)):
                    raise ValueError(f"{name} must be a sequence")
                data[name] = tuple(raw)  # type: ignore[arg-type]
        return cls(**data)


__all__ = [
    "KIWILM2_MIXER_SCHEDULE",
    "KiwiLM2Config",
    "KiwiLM2SlimConfig",
    "KiwiLM2SlimV3Config",
    "ModelConfig",
]
