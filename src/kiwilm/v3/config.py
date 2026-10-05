"""Independent, versioned configuration for the bidirectional M3 backbone."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Self


@dataclass(frozen=True, slots=True)
class KiwiLM3Config:
    architecture: str = "kiwilm3"
    schema_version: int = 1
    vocab_size: int = 32_000
    context_length: int = 512
    d_model: int = 512
    num_blocks: int = 12
    num_heads: int = 8
    swiglu_dim: int = 2_048
    dropout: float = 0.0
    pad_token_id: int | None = 0
    noise_embedding_dim: int = 64
    rms_norm_eps: float = 1e-6
    rope_base: float = 10_000.0
    mixer_schedule: tuple[str, ...] | None = None
    conv_kernel_sizes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if (
            self.architecture != "kiwilm3"
            or type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise ValueError("expected kiwilm3 configuration schema version 1")
        for name in (
            "vocab_size",
            "context_length",
            "d_model",
            "num_blocks",
            "num_heads",
            "swiglu_dim",
            "noise_embedding_dim",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.num_heads or (self.d_model // self.num_heads) % 2:
            raise ValueError("d_model must divide evenly into heads with even RoPE head width")
        if self.noise_embedding_dim % 2:
            raise ValueError("noise_embedding_dim must be even")
        if self.pad_token_id is not None and (
            type(self.pad_token_id) is not int or not 0 <= self.pad_token_id < self.vocab_size
        ):
            raise ValueError("pad_token_id must be None or an ID inside the vocabulary")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("dropout must be finite and in [0, 1)")
        for name in ("rms_norm_eps", "rope_base"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        schedule = self.mixer_schedule
        if schedule is None:
            cycle = ("attention", "biconv", "biconv")
            schedule = tuple(cycle[index % 3] for index in range(self.num_blocks))
        elif isinstance(schedule, (str, bytes)):
            raise ValueError("mixer_schedule must be a sequence, not a string")
        else:
            schedule = tuple(schedule)
        if len(schedule) != self.num_blocks or any(
            mixer not in {"attention", "biconv"} for mixer in schedule
        ):
            raise ValueError("mixer_schedule must contain num_blocks attention/biconv entries")
        object.__setattr__(self, "mixer_schedule", schedule)
        kernels = self.conv_kernel_sizes
        if kernels is None:
            kernels = tuple((31, 63)[index % 2] for index in range(schedule.count("biconv")))
        elif isinstance(kernels, (str, bytes)):
            raise ValueError("conv_kernel_sizes must be a sequence, not a string")
        else:
            kernels = tuple(kernels)
        if len(kernels) != schedule.count("biconv") or any(
            type(kernel) is not int or kernel < 1 or kernel % 2 == 0 for kernel in kernels
        ):
            raise ValueError("conv_kernel_sizes needs one positive odd kernel per BiConv")
        object.__setattr__(self, "conv_kernel_sizes", kernels)

    def to_dict(self) -> dict:
        values = asdict(self)
        values["mixer_schedule"] = list(self.mixer_schedule)
        values["conv_kernel_sizes"] = list(self.conv_kernel_sizes)
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        return cls(**dict(values))
