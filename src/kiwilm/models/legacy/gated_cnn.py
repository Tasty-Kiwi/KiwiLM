"""Legacy causal gated-convolution language model."""

from __future__ import annotations

from torch import Tensor, nn

from kiwilm.config import GatedCNNConfig, ModelConfig
from kiwilm.models.base import CausalLanguageModel
from kiwilm.models.components import (
    CausalConv1d,
    GatedCNNBlock,
    initialize_weights,
    validate_input_ids,
)
from kiwilm.models.registry import register_model

__all__ = ["CausalConv1d", "GatedCNNBlock", "GatedCNNLM"]


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
        self.apply(initialize_weights)
        if gated_config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        """Backward-compatible alias for the shared initializer."""

        initialize_weights(module)

    def forward(self, input_ids: Tensor) -> Tensor:
        validate_input_ids(input_ids, context_length=self.config.context_length)

        values = self.token_embedding(input_ids)
        for block in self.blocks:
            values = block(values)
        return self.lm_head(self.final_norm(values))


def _build_gated_cnn(config: ModelConfig) -> CausalLanguageModel:
    return GatedCNNLM(config)


register_model("gated_cnn", _build_gated_cnn)
