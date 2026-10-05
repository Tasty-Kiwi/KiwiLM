"""Static M3 accounting; full bidirectional forward FLOPs, not a speed benchmark."""

from __future__ import annotations

import torch

from kiwilm.models.kiwilm3 import KiwiLM3Encoder
from kiwilm.v3.config import KiwiLM3Config


def profile_encoder(config: KiwiLM3Config, *, sequence_length: int | None = None) -> dict:
    length = config.context_length if sequence_length is None else sequence_length
    if type(length) is not int or not 1 <= length <= config.context_length:
        raise ValueError("profile sequence length must be inside the configured context")
    with torch.device("meta"):
        model = KiwiLM3Encoder(config)
    total = sum(parameter.numel() for parameter in model.parameters())
    embedding = model.token_embedding.weight.numel()
    noise_parameters = sum(parameter.numel() for parameter in model.noise_conditioning.parameters())
    kernels = iter(config.conv_kernel_sizes)
    rows = []
    for index, (kind, block) in enumerate(zip(config.mixer_schedule, model.blocks, strict=True)):
        kernel = next(kernels) if kind == "biconv" else None
        if kind == "attention":
            mixer_flops = 8 * config.d_model**2 + 4 * length * config.d_model
        else:
            mixer_flops = 6 * config.d_model**2 + 2 * config.d_model * kernel
        mlp_flops = 6 * config.d_model * config.swiglu_dim
        rows.append(
            {
                "index": index,
                "mixer": kind,
                "kernel_size": kernel,
                "mlp": "swiglu",
                "parameters": sum(parameter.numel() for parameter in block.parameters()),
                "forward_flops_per_token": mixer_flops + mlp_flops,
            }
        )
    noise_flops = 2 * (config.noise_embedding_dim * config.d_model + config.d_model**2)
    encoder_flops = sum(row["forward_flops_per_token"] for row in rows) + noise_flops / length
    head_flops = 2 * config.d_model * config.vocab_size
    return {
        "architecture": config.architecture,
        "config": config.to_dict(),
        "parameters": total,
        "embedding_parameters": embedding,
        "non_embedding_parameters": total - embedding,
        "noise_conditioning_parameters": noise_parameters,
        "blocks": rows,
        "attention_blocks": config.mixer_schedule.count("attention"),
        "biconv_blocks": config.mixer_schedule.count("biconv"),
        "sequence_length": length,
        "encoder_forward_flops_per_token": encoder_flops,
        "reconstruction_head_flops_per_token": head_flops,
        "forward_flops_per_token": encoder_flops + head_flops,
        "kv_cache": None,
        "is_throughput_benchmark": False,
        "flops_convention": (
            "multiply-add=2; full bidirectional attention; omit norms, RoPE, biases, "
            "nonlinearities, dropout and lookups; no backward estimate"
        ),
    }
