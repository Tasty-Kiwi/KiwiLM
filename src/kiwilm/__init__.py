"""KiwiLM 2 hybrid causal language-model research toolkit."""

from kiwilm.config import KiwiLM2Config, KiwiLM2SlimConfig, ModelConfig
from kiwilm.models import build_model

__all__ = ["KiwiLM2Config", "KiwiLM2SlimConfig", "ModelConfig", "build_model"]
__version__ = "0.1.0"
