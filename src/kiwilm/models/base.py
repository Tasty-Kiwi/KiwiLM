"""Frozen causal-model contract; bidirectional encoders must use a separate interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn

from kiwilm.config import ModelConfig


class CausalLanguageModel(nn.Module, ABC):
    """V2 next-token interface, not a requirement for future diffusion/encoder models."""

    config: ModelConfig

    @abstractmethod
    def forward(self, input_ids: Tensor) -> Tensor:
        """Return next-token logits with shape ``[batch, sequence, vocabulary]``."""
