"""Causal gated-convolution language model."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F

from kiwilm.config import GatedCNNConfig, ModelConfig
from kiwilm.models.base import CausalLanguageModel
from kiwilm.models.registry import register_model


class CausalConv1d(nn.Module):
    """A length-preserving convolution padded strictly on the left."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int,
    ) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.conv(F.pad(values, (self.left_padding, 0)))


class GatedCNNBlock(nn.Module):
    """Pre-normalized residual block with a dense causal convolution and GLU."""

    def __init__(
        self,
        d_model: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.conv = CausalConv1d(
            d_model,
            2 * d_model,
            kernel_size,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        residual = values
        convolved = self.conv(self.norm(values).transpose(1, 2))
        gated = F.glu(convolved, dim=1).transpose(1, 2)
        return residual + self.dropout(gated)


class GatedCNNLM(CausalLanguageModel):
    """Embedding, eight causal gated CNN blocks, and a language-model head."""

    def __init__(self, config: GatedCNNConfig | ModelConfig | None = None) -> None:
        super().__init__()
        if config is None:
            gated_config = GatedCNNConfig()
        elif isinstance(config, GatedCNNConfig):
            gated_config = config
        elif type(config) is ModelConfig and config.architecture == "gated_cnn":
            gated_config = GatedCNNConfig.from_dict(config.to_dict())
        else:
            raise TypeError("GatedCNNLM requires a GatedCNNConfig")

        self.config = gated_config
        self.token_embedding = nn.Embedding(gated_config.vocab_size, gated_config.d_model)
        self.blocks = nn.ModuleList(
            GatedCNNBlock(
                gated_config.d_model,
                kernel_size=gated_config.kernel_size,
                dilation=dilation,
                dropout=gated_config.dropout,
            )
            for dilation in gated_config.dilations
        )
        self.final_norm = nn.LayerNorm(gated_config.d_model)
        self.lm_head = nn.Linear(
            gated_config.d_model,
            gated_config.vocab_size,
            bias=True,
        )
        self.apply(self._initialize_weights)
        if gated_config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids sequence length must be positive")
        if input_ids.shape[1] > self.config.context_length:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds context length "
                f"{self.config.context_length}"
            )

        values = self.token_embedding(input_ids)
        for block in self.blocks:
            values = block(values)
        return self.lm_head(self.final_norm(values))


def _build_gated_cnn(config: ModelConfig) -> CausalLanguageModel:
    return GatedCNNLM(config)


register_model("gated_cnn", _build_gated_cnn)
