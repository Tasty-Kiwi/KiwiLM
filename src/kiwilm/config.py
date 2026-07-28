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
        if cls is ModelConfig:
            architecture = data.get("architecture", "gated_cnn")
            if architecture == "gated_cnn":
                return cast(Self, GatedCNNConfig.from_dict(data))
            if architecture == "cnn_attention":
                return cast(Self, CNNAttentionConfig.from_dict(data))
            if architecture == "cnn_attention_ffn":
                return cast(Self, CNNFFNAttentionConfig.from_dict(data))
            if architecture == "cnn_dual_attention":
                return cast(Self, CNNDualAttentionConfig.from_dict(data))
            if architecture == "cnn_attention_mamba":
                return cast(Self, CNNAttentionMambaConfig.from_dict(data))
            if architecture == "cnn_interleaved_attention":
                return cast(Self, CNNInterleavedAttentionConfig.from_dict(data))
            if architecture == "cnn_deep_interleaved_attention":
                return cast(
                    Self,
                    CNNDeepInterleavedAttentionConfig.from_dict(data),
                )
            if architecture == "transformer":
                return cast(Self, TransformerConfig.from_dict(data))
            if architecture == "model_x":
                return cast(Self, ModelXConfig.from_dict(data))
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


@dataclass(frozen=True, slots=True)
class CNNAttentionConfig(ModelConfig):
    """Configuration for the CNN-attention-CNN comparison model."""

    architecture: str = "cnn_attention"
    kernel_size: int = 3
    pre_attention_dilations: tuple[int, ...] = (1, 2, 4)
    post_attention_dilations: tuple[int, ...] = (8, 16, 32)
    num_heads: int = 8
    feedforward_dim: int = 1024

    def __post_init__(self) -> None:
        super(CNNAttentionConfig, self).__post_init__()
        if self.architecture != "cnn_attention":
            raise ValueError("CNNAttentionConfig architecture must be 'cnn_attention'")
        _validate_cnn_attention_fields(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        """Reconstruct a CNN-attention config from a plain mapping."""

        return cls(**_normalize_cnn_attention_data(values))


@dataclass(frozen=True, slots=True)
class CNNFFNAttentionConfig(CNNAttentionConfig):
    """Configuration for Model G with an FFN after every gated CNN."""

    architecture: str = "cnn_attention_ffn"

    def __post_init__(self) -> None:
        ModelConfig.__post_init__(self)
        if self.architecture != "cnn_attention_ffn":
            raise ValueError(
                "CNNFFNAttentionConfig architecture must be 'cnn_attention_ffn'"
            )
        _validate_cnn_attention_fields(self)


@dataclass(frozen=True, slots=True)
class CNNDualAttentionConfig(CNNAttentionConfig):
    """Configuration for Model C with a second attention block."""

    architecture: str = "cnn_dual_attention"

    def __post_init__(self) -> None:
        ModelConfig.__post_init__(self)
        if self.architecture != "cnn_dual_attention":
            raise ValueError(
                "CNNDualAttentionConfig architecture must be 'cnn_dual_attention'"
            )
        _validate_cnn_attention_fields(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        """Reconstruct a dual-attention config from a plain mapping."""

        return cls(**_normalize_cnn_attention_data(values))


@dataclass(frozen=True, slots=True)
class CNNAttentionMambaConfig(CNNAttentionConfig):
    """Configuration for Model D with a final selective state-space block."""

    architecture: str = "cnn_attention_mamba"
    mamba_inner_dim: int = 896
    mamba_state_dim: int = 16
    mamba_conv_kernel: int = 4
    mamba_dt_rank: int = 16

    def __post_init__(self) -> None:
        ModelConfig.__post_init__(self)
        if self.architecture != "cnn_attention_mamba":
            raise ValueError(
                "CNNAttentionMambaConfig architecture must be "
                "'cnn_attention_mamba'"
            )
        _validate_cnn_attention_fields(self)
        _require_positive_int("mamba_inner_dim", self.mamba_inner_dim)
        _require_positive_int("mamba_state_dim", self.mamba_state_dim)
        _require_positive_int("mamba_conv_kernel", self.mamba_conv_kernel)
        _require_positive_int("mamba_dt_rank", self.mamba_dt_rank)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        """Reconstruct an attention-Mamba config from a plain mapping."""

        return cls(**_normalize_cnn_attention_data(values))


@dataclass(frozen=True, slots=True)
class CNNInterleavedAttentionConfig(ModelConfig):
    """Configuration for Model E with attention between pairs of CNN blocks."""

    architecture: str = "cnn_interleaved_attention"
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    num_heads: int = 8
    feedforward_dim: int = 1024

    def __post_init__(self) -> None:
        super(CNNInterleavedAttentionConfig, self).__post_init__()
        if self.architecture != "cnn_interleaved_attention":
            raise ValueError(
                "CNNInterleavedAttentionConfig architecture must be "
                "'cnn_interleaved_attention'"
            )
        _validate_interleaved_attention_fields(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        """Reconstruct an interleaved-attention config from a mapping."""

        return cls(**_normalize_dilation_data(values, ("dilations",)))


@dataclass(frozen=True, slots=True)
class CNNDeepInterleavedAttentionConfig(CNNInterleavedAttentionConfig):
    """Configuration for Model F's final CNN-attention refinement stage."""

    architecture: str = "cnn_deep_interleaved_attention"
    refinement_dilations: tuple[int, ...] = (1, 2, 4)

    def __post_init__(self) -> None:
        ModelConfig.__post_init__(self)
        if self.architecture != "cnn_deep_interleaved_attention":
            raise ValueError(
                "CNNDeepInterleavedAttentionConfig architecture must be "
                "'cnn_deep_interleaved_attention'"
            )
        _validate_interleaved_attention_fields(self)
        _validate_dilations(
            self,
            "refinement_dilations",
            expected_length=3,
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        """Reconstruct a deep interleaved-attention config from a mapping."""

        return cls(
            **_normalize_dilation_data(
                values,
                ("dilations", "refinement_dilations"),
            )
        )


@dataclass(frozen=True, slots=True)
class TransformerConfig(ModelConfig):
    """Configuration for the controlled decoder-only Transformer baseline."""

    architecture: str = "transformer"
    num_layers: int = 4
    num_heads: int = 8
    feedforward_dim: int = 1024

    def __post_init__(self) -> None:
        super(TransformerConfig, self).__post_init__()
        if self.architecture != "transformer":
            raise ValueError("TransformerConfig architecture must be 'transformer'")
        _require_positive_int("num_layers", self.num_layers)
        _require_positive_int("num_heads", self.num_heads)
        _require_positive_int("feedforward_dim", self.feedforward_dim)
        _validate_attention_dimensions(self)


@dataclass(frozen=True, slots=True)
class ModelXConfig(ModelConfig):
    """Configuration for the alternating local/global Model X hybrid."""

    architecture: str = "model_x"
    kernel_size: int = 3
    cnn_dilations: tuple[int, ...] = (1, 2)
    num_heads: int = 8
    swiglu_dim: int = 640
    rms_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        super(ModelXConfig, self).__post_init__()
        if self.architecture != "model_x":
            raise ValueError("ModelXConfig architecture must be 'model_x'")
        _require_positive_int("kernel_size", self.kernel_size)
        _require_positive_int("num_heads", self.num_heads)
        _require_positive_int("swiglu_dim", self.swiglu_dim)
        _validate_dilations(self, "cnn_dilations", expected_length=2)
        _validate_attention_dimensions(self)
        if (
            isinstance(self.rms_norm_eps, bool)
            or not isinstance(self.rms_norm_eps, (int, float))
            or not math.isfinite(self.rms_norm_eps)
            or self.rms_norm_eps <= 0
        ):
            raise ValueError("rms_norm_eps must be finite and positive")

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        """Reconstruct Model X from a JSON-compatible mapping."""

        return cls(**_normalize_dilation_data(values, ("cnn_dilations",)))


def _validate_interleaved_attention_fields(
    config: CNNInterleavedAttentionConfig,
) -> None:
    _require_positive_int("kernel_size", config.kernel_size)
    _require_positive_int("num_heads", config.num_heads)
    _require_positive_int("feedforward_dim", config.feedforward_dim)
    _validate_dilations(config, "dilations", expected_length=6)
    _validate_attention_dimensions(config)


def _validate_dilations(
    config: ModelConfig,
    name: str,
    *,
    expected_length: int,
) -> None:
    values = getattr(config, name)
    if not isinstance(values, tuple):
        try:
            values = tuple(values)
            object.__setattr__(config, name, values)
        except TypeError as error:
            raise ValueError(
                f"{name} must be a sequence of positive integers"
            ) from error
    if len(values) != expected_length:
        raise ValueError(
            f"{name} must contain exactly {expected_length} entries"
        )
    for dilation in values:
        _require_positive_int(f"each {name}", dilation)


def _normalize_dilation_data(
    values: Mapping[str, object],
    names: tuple[str, ...],
) -> dict[str, object]:
    data = dict(values)
    for name in names:
        if name not in data:
            continue
        raw_dilations = data[name]
        if isinstance(raw_dilations, (str, bytes)):
            raise ValueError(
                f"{name} must be a sequence of positive integers"
            )
        try:
            data[name] = tuple(raw_dilations)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError(
                f"{name} must be a sequence of positive integers"
            ) from error
    return data


def _normalize_cnn_attention_data(
    values: Mapping[str, object],
) -> dict[str, object]:
    data = dict(values)
    for name in ("pre_attention_dilations", "post_attention_dilations"):
        if name not in data:
            continue
        raw_dilations = data[name]
        if isinstance(raw_dilations, (str, bytes)):
            raise ValueError(f"{name} must be a sequence of positive integers")
        try:
            data[name] = tuple(raw_dilations)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError(
                f"{name} must be a sequence of positive integers"
            ) from error
    return data


def _validate_cnn_attention_fields(config: CNNAttentionConfig) -> None:
    _require_positive_int("kernel_size", config.kernel_size)
    _require_positive_int("num_heads", config.num_heads)
    _require_positive_int("feedforward_dim", config.feedforward_dim)
    for name in ("pre_attention_dilations", "post_attention_dilations"):
        values = getattr(config, name)
        if not isinstance(values, tuple):
            try:
                values = tuple(values)
                object.__setattr__(config, name, values)
            except TypeError as error:
                raise ValueError(
                    f"{name} must be a sequence of positive integers"
                ) from error
        if len(values) != 3:
            raise ValueError(f"{name} must contain exactly 3 entries")
        for dilation in values:
            _require_positive_int(f"each {name} dilation", dilation)
    _validate_attention_dimensions(config)


def _validate_attention_dimensions(
    config: (
        CNNAttentionConfig
        | CNNInterleavedAttentionConfig
        | ModelXConfig
        | TransformerConfig
    ),
) -> None:
    if config.d_model % config.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    if (config.d_model // config.num_heads) % 2 != 0:
        raise ValueError("attention head dimension must be even for RoPE")
