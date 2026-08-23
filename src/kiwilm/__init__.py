"""KiwiLM: small, modular causal language-model experiments."""

from kiwilm.config import GatedCNNConfig, KiwiLMSANConfig, ModelConfig
from kiwilm.models import build_model

__all__ = ["GatedCNNConfig", "KiwiLMSANConfig", "ModelConfig", "build_model"]
__version__ = "0.1.0"
