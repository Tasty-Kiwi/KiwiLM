"""Static parameter, cache, and compute accounting for KiwiLM 2."""

from __future__ import annotations

from typing import Any

from torch import nn

from kiwilm.config import KiwiLM2SlimConfig
from kiwilm.models.kiwilm2 import KiwiLM2LM


def _unique_parameters(module: nn.Module) -> dict[int, nn.Parameter]:
    return {id(parameter): parameter for parameter in module.parameters()}


def profile_kiwilm2(
    model: KiwiLM2LM,
    *,
    sequence_length: int | None = None,
    cache_dtype_bytes: int = 2,
) -> dict[str, Any]:
    """Return explicit, non-benchmark model accounting.

    FLOPs use a multiply-add=2 convention and assume a full cached context for
    attention. Embedding lookups, normalization, activations, and hashing are
    omitted, so the value is a transparent estimate rather than profiler output.
    """

    if not isinstance(model, KiwiLM2LM):
        raise TypeError("profile_kiwilm2 requires a KiwiLM2LM")
    config = model.config
    length = sequence_length or config.context_length
    if length < 1 or length > config.context_length:
        raise ValueError("sequence_length must be within the configured context")
    unique = _unique_parameters(model)
    total = sum(parameter.numel() for parameter in unique.values())
    token_ids = {id(model.token_embedding.weight)}
    ngram_ids = {
        id(model.ngram_embedding.bigram.weight),
        id(model.ngram_embedding.trigram.weight),
    }
    token_parameters = sum(unique[key].numel() for key in token_ids)
    ngram_parameters = sum(unique[key].numel() for key in ngram_ids)
    dense_parameters = total - token_parameters - ngram_parameters

    d_model = config.d_model
    head_dim = d_model // config.num_query_heads
    gqa_layers = config.mixer_schedule.count("gqa")
    kv_elements = 2 * gqa_layers * config.num_kv_heads * length * head_dim
    kv_cache_bytes = kv_elements * cache_dtype_bytes

    projection_flops = 0
    attention_flops = 0
    convolution_flops = 0
    mlp_flops = 0
    kernel_index = 0
    for mixer in config.mixer_schedule:
        if mixer == "gqa":
            kv_width = config.num_kv_heads * head_dim
            projection_flops += 2 * (d_model * d_model + 2 * d_model * kv_width + d_model * d_model)
            attention_flops += 4 * config.num_query_heads * head_dim * length
        else:
            kernel = config.conv_kernel_sizes[kernel_index]
            kernel_index += 1
            projection_flops += 2 * (2 * d_model * d_model + d_model * d_model)
            convolution_flops += 2 * d_model * kernel
        if isinstance(config, KiwiLM2SlimConfig):
            # Two FWHTs: width*log2(width) butterflies, each with add/sub.
            mlp_flops += 4 * d_model * (d_model.bit_length() - 1)
        else:
            mlp_flops += 2 * 3 * d_model * config.swiglu_dim
    lm_head_flops = 2 * d_model * config.vocab_size
    flops_per_token = (
        projection_flops + attention_flops + convolution_flops + mlp_flops + lm_head_flops
    )
    return {
        "architecture": config.architecture,
        "parameters": {
            "total": total,
            "dense_non_embedding": dense_parameters,
            "token_embedding": token_parameters,
            "ngram": ngram_parameters,
        },
        "kv_cache": {
            "sequence_length": length,
            "dtype_bytes": cache_dtype_bytes,
            "elements": kv_elements,
            "bytes": kv_cache_bytes,
        },
        "estimated_flops_per_token": {
            "projection": projection_flops,
            "attention_at_sequence_length": attention_flops,
            "depthwise_convolution": convolution_flops,
            "mlp": mlp_flops,
            "lm_head": lm_head_flops,
            "total": flops_per_token,
            "multiply_add_convention": 2,
        },
    }


__all__ = ["profile_kiwilm2"]
