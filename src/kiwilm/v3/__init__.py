"""KiwiLM 3 encoder surface, independent of the frozen causal V2 registry."""

from kiwilm.v3.config import KiwiLM3Config


def build_encoder(config: KiwiLM3Config):
    """Construct V3 without registering it as a causal V2 language model."""
    from kiwilm.models.kiwilm3 import KiwiLM3Encoder

    return KiwiLM3Encoder(config)


__all__ = ["KiwiLM3Config", "build_encoder"]
