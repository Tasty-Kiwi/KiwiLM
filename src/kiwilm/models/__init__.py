"""Model implementations and architecture registry."""

from kiwilm.models.base import CausalLanguageModel
from kiwilm.models.cnn_attention import (
    CausalSelfAttention,
    CNNAttentionLM,
    CNNAttentionMambaLM,
    CNNDeepInterleavedAttentionLM,
    CNNDualAttentionLM,
    CNNInterleavedAttentionCache,
    CNNInterleavedAttentionLM,
    TransformerAttentionBlock,
)
from kiwilm.models.components import CausalConv1d, GatedCNNBlock
from kiwilm.models.gated_cnn import GatedCNNLM
from kiwilm.models.mamba import MambaBlock, SelectiveStateSpace
from kiwilm.models.registry import build_model, register_model
from kiwilm.models.transformer import TransformerCache, TransformerLM

__all__ = [
    "CNNAttentionLM",
    "CNNAttentionMambaLM",
    "CNNDeepInterleavedAttentionLM",
    "CNNDualAttentionLM",
    "CNNInterleavedAttentionCache",
    "CNNInterleavedAttentionLM",
    "CausalConv1d",
    "CausalLanguageModel",
    "CausalSelfAttention",
    "GatedCNNBlock",
    "GatedCNNLM",
    "MambaBlock",
    "SelectiveStateSpace",
    "TransformerAttentionBlock",
    "TransformerCache",
    "TransformerLM",
    "build_model",
    "register_model",
]
