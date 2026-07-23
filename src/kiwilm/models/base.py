"""Shared causal language-model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn

from kiwilm.config import ModelConfig


class CausalLanguageModel(nn.Module, ABC):
    """Base class implemented by all KiwiLM architectures."""

    config: ModelConfig

    @abstractmethod
    def forward(self, input_ids: Tensor) -> Tensor:
        """Return next-token logits with shape ``[batch, sequence, vocabulary]``."""

