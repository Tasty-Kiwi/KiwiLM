"""Model architecture registry."""

from __future__ import annotations

from collections.abc import Callable

from kiwilm.config import ModelConfig
from kiwilm.models.base import CausalLanguageModel

ModelFactory = Callable[[ModelConfig], CausalLanguageModel]

_MODEL_REGISTRY: dict[str, ModelFactory] = {}


def register_model(
    architecture: str,
    factory: ModelFactory,
    *,
    replace: bool = False,
) -> None:
    """Register a model factory under a stable architecture identifier."""

    if not isinstance(architecture, str) or not architecture.strip():
        raise ValueError("architecture must be a non-empty string")
    if not callable(factory):
        raise TypeError("factory must be callable")
    if architecture in _MODEL_REGISTRY and not replace:
        raise ValueError(f"model architecture is already registered: {architecture}")
    _MODEL_REGISTRY[architecture] = factory


def build_model(config: ModelConfig) -> CausalLanguageModel:
    """Build the model selected by ``config.architecture``."""

    try:
        factory = _MODEL_REGISTRY[config.architecture]
    except KeyError as error:
        available = ", ".join(sorted(_MODEL_REGISTRY)) or "none"
        raise ValueError(
            f"unknown model architecture {config.architecture!r}; available: {available}"
        ) from error
    model = factory(config)
    if not isinstance(model, CausalLanguageModel):
        raise TypeError(
            f"factory for {config.architecture!r} did not return a CausalLanguageModel"
        )
    return model
