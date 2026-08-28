#!/usr/bin/env python3
# ruff: noqa: E501, RUF001
"""Generate Graphviz sources, SVGs, and PNG architecture diagrams for KiwiLM 2."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from kiwilm.config import KiwiLM2Config, KiwiLM2SlimConfig
from kiwilm.model_profile import profile_kiwilm2
from kiwilm.models import KiwiLM2LM


def _format_count(value: int) -> str:
    return f"{value:,}"


def build_dot(config: KiwiLM2Config) -> str:
    """Build a DOT graph directly from one frozen KiwiLM 2 configuration."""

    slim = isinstance(config, KiwiLM2SlimConfig)
    variant = "KiwiLM 2 Slim" if slim else "KiwiLM 2"
    feed_forward = (
        "Gated Hadamard MLP 512-wide\\n"
        "independent signed D₁/D₂/D₃ diagonals\\n"
        "H(D₃(SiLU(H(D₁x)) ⊙ H(D₂x))) × learned α=0.224"
        if slim
        else "SwiGLU FFN\\n512 → 1,536 gate/up → 512"
    )
    model = KiwiLM2LM(config)
    profile = profile_kiwilm2(model)
    parameters = profile["parameters"]
    flops = profile["estimated_flops_per_token"]["total"]
    cache_bytes = profile["kv_cache"]["bytes"]

    block_lines: list[str] = []
    edge_lines: list[str] = []
    previous = "embedding_sum"
    conv_index = 0
    for index, mixer in enumerate(config.mixer_schedule, start=1):
        node = f"block_{index}"
        if mixer == "gqa":
            mixer_label = (
                f"GQA • {config.num_query_heads}Q/{config.num_kv_heads}KV heads\\n"
                "cached RoPE • causal global mixing"
            )
            fill = "#EDE9FE"
            color = "#8B5CF6"
        else:
            kernel = config.conv_kernel_sizes[conv_index]
            conv_index += 1
            mixer_label = (
                f"XXL causal gated convolution • k={kernel}\\n"
                "pointwise 512→1,024 • depthwise conv • gated output"
            )
            fill = "#FEF3C7"
            color = "#F59E0B"
        block_lines.append(
            f'    {node} [label="Block {index}\\nPre-RMSNorm → {mixer_label} → residual\\n'
            f'Pre-RMSNorm → {feed_forward} → residual", fillcolor="{fill}", '
            f'color="{color}"];'
        )
        edge_lines.append(f"    {previous} -> {node};")
        previous = node

    title = f"{variant} — Frozen Hybrid Language Model"
    summary = (
        f"{_format_count(parameters['total'])} parameters\\n"
        f"{_format_count(parameters['dense_non_embedding'])} dense/non-embedding\\n"
        f"{_format_count(parameters['ngram'])} n-gram memory\\n"
        f"{_format_count(flops)} estimated FLOPs/token\\n"
        f"{cache_bytes // (1024 * 1024)} MiB fp16 KV cache at T=512"
    )
    block_text = "\n".join(block_lines)
    edge_text = "\n".join(edge_lines)
    return f'''digraph KiwiLM2 {{
    graph [
        rankdir=TB,
        bgcolor="#FFFFFF",
        pad=0.28,
        nodesep=0.24,
        ranksep=0.40,
        splines=ortho,
        fontname="Helvetica",
        fontsize=18,
        label="{title}",
        labelloc=t
    ];

    node [
        shape=box,
        style="rounded,filled",
        color="#64748B",
        fillcolor="#F8FAFC",
        fontcolor="#0F172A",
        fontname="Helvetica",
        fontsize=10,
        margin="0.18,0.10",
        penwidth=1.2
    ];

    edge [color="#475569", arrowsize=0.7, penwidth=1.3];

    input [label="Input token IDs\\n[B, T] • T ≤ {config.context_length}", fillcolor="#DBEAFE", color="#3B82F6"];
    token_embedding [label="Token embedding\\n{config.vocab_size:,} × {config.d_model}", fillcolor="#DCFCE7", color="#22C55E"];
    ngram_memory [label="Explicit n-gram memory\\nhashed bigram + trigram tables\\n{config.bigram_buckets:,} × {config.d_model} each", fillcolor="#CCFBF1", color="#14B8A6"];
    embedding_sum [label="Sum and scale by √({config.d_model}/3)\\n[B, T, {config.d_model}]", fillcolor="#ECFCCB", color="#84CC16"];

{block_text}

    final_norm [label="Final RMSNorm\\n{config.d_model} channels", fillcolor="#E2E8F0", color="#64748B"];
    lm_head [label="Tied language-model head\\nLinear {config.d_model} → {config.vocab_size:,} • no bias", fillcolor="#F3E8FF", color="#A855F7"];
    logits [label="Next-token logits\\n[B, T, {config.vocab_size:,}]", fillcolor="#FFE4E6", color="#F43F5E"];
    summary [shape=note, style="filled", label="{summary}", fillcolor="#F1F5F9", color="#94A3B8"];
    cache [shape=note, style="filled", label="Cached autoregressive decoding\\n4 GQA KV states + 6 convolution histories\\ncontext rollover triggers a bounded prefill", fillcolor="#EFF6FF", color="#60A5FA"];

    input -> token_embedding;
    input -> ngram_memory;
    token_embedding -> embedding_sum;
    ngram_memory -> embedding_sum;
{edge_text}
    {previous} -> final_norm -> lm_head -> logits;
    token_embedding -> lm_head [xlabel="shared weights", style=dashed, color="#9333EA", fontcolor="#7E22CE", constraint=false];
    {{ rank=same; embedding_sum; cache; }}
    cache -> embedding_sum [style=invis];
    {{ rank=same; final_norm; summary; }}
    summary -> final_norm [style=invis];
}}
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    parser.add_argument("--dot-bin", default="dot")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variants = (
        ("kiwilm2", KiwiLM2Config()),
        ("kiwilm2-slim", KiwiLM2SlimConfig()),
    )
    for name, config in variants:
        dot_path = args.output_dir / f"{name}.dot"
        svg_path = args.output_dir / f"{name}.svg"
        png_path = args.output_dir / f"{name}.png"
        dot_path.write_text(build_dot(config), encoding="utf-8")
        for output_format, output_path in (("svg", svg_path), ("png", png_path)):
            subprocess.run(
                [
                    args.dot_bin,
                    f"-T{output_format}",
                    str(dot_path),
                    "-o",
                    str(output_path),
                ],
                check=True,
            )
        print(f"wrote {dot_path}, {svg_path}, and {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
