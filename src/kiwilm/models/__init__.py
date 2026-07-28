"""Model implementations and architecture registry."""

from kiwilm.models.attention import CausalSelfAttention
from kiwilm.models.base import CausalLanguageModel
from kiwilm.models.components import CausalConv1d, GatedCNNBlock
from kiwilm.models.legacy import (
    CNNAttentionCache,
    CNNAttentionLM,
    CNNAttentionMambaLM,
    CNNDeepInterleavedAttentionLM,
    CNNDualAttentionLM,
    CNNFFNAttentionLM,
    CNNInterleavedAttentionCache,
    CNNInterleavedAttentionLM,
    GatedCNNLM,
    MambaBlock,
    ResidualFeedForwardBlock,
    SelectiveStateSpace,
    TransformerAttentionBlock,
    TransformerCache,
    TransformerLM,
)
from kiwilm.models.model_x import (
    ModelXCache,
    ModelXLM,
    ResidualSwiGLUBlock,
    RMSAttentionBlock,
    RMSGatedCNNBlock,
)
from kiwilm.models.model_y import (
    ModelYBlock,
    ModelYCache,
    ModelYLM,
)
from kiwilm.models.registry import build_model, register_model

__all__ = [
    "CNNAttentionCache",
    "CNNAttentionLM",
    "CNNAttentionMambaLM",
    "CNNDeepInterleavedAttentionLM",
    "CNNDualAttentionLM",
    "CNNFFNAttentionLM",
    "CNNInterleavedAttentionCache",
    "CNNInterleavedAttentionLM",
    "CausalConv1d",
    "CausalLanguageModel",
    "CausalSelfAttention",
    "GatedCNNBlock",
    "GatedCNNLM",
    "MambaBlock",
    "ModelXCache",
    "ModelXLM",
    "ModelYBlock",
    "ModelYCache",
    "ModelYLM",
    "RMSAttentionBlock",
    "RMSGatedCNNBlock",
    "ResidualFeedForwardBlock",
    "ResidualSwiGLUBlock",
    "SelectiveStateSpace",
    "TransformerAttentionBlock",
    "TransformerCache",
    "TransformerLM",
    "build_model",
    "register_model",
]
