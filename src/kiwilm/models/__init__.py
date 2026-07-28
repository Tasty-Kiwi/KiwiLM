"""Model implementations and architecture registry."""

from kiwilm.models.base import CausalLanguageModel
from kiwilm.models.cnn_attention import (
    CausalSelfAttention,
    CNNAttentionCache,
    CNNAttentionLM,
    CNNAttentionMambaLM,
    CNNDeepInterleavedAttentionLM,
    CNNDualAttentionLM,
    CNNFFNAttentionLM,
    CNNInterleavedAttentionCache,
    CNNInterleavedAttentionLM,
    ResidualFeedForwardBlock,
    TransformerAttentionBlock,
)
from kiwilm.models.components import CausalConv1d, GatedCNNBlock
from kiwilm.models.gated_cnn import GatedCNNLM
from kiwilm.models.mamba import MambaBlock, SelectiveStateSpace
from kiwilm.models.model_x import (
    ModelXCache,
    ModelXLM,
    ResidualSwiGLUBlock,
    RMSAttentionBlock,
    RMSGatedCNNBlock,
)
from kiwilm.models.modern_transformer import (
    ModernTransformerBlock,
    ModernTransformerCache,
    ModernTransformerLM,
)
from kiwilm.models.registry import build_model, register_model
from kiwilm.models.transformer import TransformerCache, TransformerLM

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
    "ModernTransformerBlock",
    "ModernTransformerCache",
    "ModernTransformerLM",
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
