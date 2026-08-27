"""KiwiLM 2 model implementations and architecture registry."""

from kiwilm.models.base import CausalLanguageModel
from kiwilm.models.kiwilm2 import (
    CachedRotaryEmbedding,
    HadamardMLP,
    KiwiLM2Block,
    KiwiLM2Cache,
    KiwiLM2GQA,
    KiwiLM2LM,
    NGramEmbedding,
    RMSNorm,
    SwiGLU,
    XXLCausalGatedConv,
    fast_walsh_hadamard,
)
from kiwilm.models.registry import build_model, register_model

__all__ = [
    "CachedRotaryEmbedding",
    "CausalLanguageModel",
    "HadamardMLP",
    "KiwiLM2Block",
    "KiwiLM2Cache",
    "KiwiLM2GQA",
    "KiwiLM2LM",
    "NGramEmbedding",
    "RMSNorm",
    "SwiGLU",
    "XXLCausalGatedConv",
    "build_model",
    "fast_walsh_hadamard",
    "register_model",
]
