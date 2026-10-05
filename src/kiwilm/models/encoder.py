"""Non-causal hidden-state interface; never route encoders through V2 generation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class BidirectionalEncoder(nn.Module, ABC):
    @abstractmethod
    def encode(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        noise_level: Tensor | float | None = None,
    ) -> Tensor:
        """Return contextual states [batch, sequence, width]; True means a valid position."""

    @abstractmethod
    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        noise_level: Tensor | float | None = None,
    ) -> Tensor:
        """Return reconstruction logits [batch, sequence, vocabulary], not next-token logits."""
