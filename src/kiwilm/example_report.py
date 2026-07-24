"""Single-checkpoint generation reports for the checked-in prompt suite."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from kiwilm.comparison import load_prompt_suite
from kiwilm.data import PreparedTokenData
from kiwilm.generation import generate
from kiwilm.inference import load_trained_model


def generate_example_report(
    checkpoint_path: str | Path,
    *,
    data: PreparedTokenData,
    suite_path: str | Path,
    output_path: str | Path,
    device: torch.device,
    title: str,
) -> dict[str, Any]:
    """Generate every prompt/profile pair and write a readable Markdown report."""

    checkpoint = Path(checkpoint_path)
    suite = load_prompt_suite(suite_path)
    model, config = load_trained_model(
        checkpoint,
        data_fingerprint=data.fingerprint,
        device=device,
    )

    rows: list[dict[str, Any]] = []
    for prompt_case in suite["prompts"]:
        for profile in suite["sampling_profiles"]:
            max_new_tokens = profile.get(
                "max_new_tokens",
                suite["max_new_tokens"],
            )
            top_k = profile["top_k"]
            text = generate(
                model,
                data.tokenizer,
                prompt_case["prompt"],
                max_new_tokens=max_new_tokens,
                context_length=config.context_length,
                temperature=profile["temperature"],
                top_k=None if top_k == 0 else top_k,
                seed=profile["seed"],
                device=device,
                cache="off",
            )
            rows.append(
                {
                    "case_id": prompt_case["id"],
                    "profile_id": profile["id"],
                    "prompt": prompt_case["prompt"],
                    "max_new_tokens": max_new_tokens,
                    "temperature": profile["temperature"],
                    "top_k": top_k,
                    "seed": profile["seed"],
                    "cache": "off",
                    "text": text,
                }
            )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(destination, render_example_report(title, rows))
    return {
        "architecture": config.architecture,
        "checkpoint": str(checkpoint.resolve()),
        "data_fingerprint": data.fingerprint,
        "device": str(device),
        "generation_count": len(rows),
        "report_path": str(destination.resolve()),
    }


def render_example_report(
    title: str,
    rows: list[Mapping[str, Any]],
) -> str:
    """Render generated rows in the format used by the examples directory."""

    lines = [f"# {title}", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['case_id']} / {row['profile_id']}",
                "",
                str(row["text"]).rstrip(),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _atomic_write(path: Path, contents: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(contents)
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


__all__ = ["generate_example_report", "render_example_report"]
