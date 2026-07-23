"""Model implementations and architecture registry."""

from kiwilm.models.base import CausalLanguageModel
from kiwilm.models.gated_cnn import CausalConv1d, GatedCNNBlock, GatedCNNLM
from kiwilm.models.registry import build_model, register_model

__all__ = [
    "CausalConv1d",
    "CausalLanguageModel",
    "GatedCNNBlock",
    "GatedCNNLM",
    "build_model",
    "register_model",
]
