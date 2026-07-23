"""Reproducible side-by-side text generation for model variants."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from kiwilm.data import PreparedTokenData
from kiwilm.generation import generate
from kiwilm.inference import load_trained_model


def compare_checkpoints(
    checkpoint_a: str | Path,
    checkpoint_b: str | Path,
    *,
    data: PreparedTokenData,
    suite_path: str | Path,
    output_dir: str | Path,
    device: torch.device,
    label_a: str | None = None,
    label_b: str | None = None,
) -> dict[str, Any]:
    """Generate every suite case from both checkpoints with identical seeds."""

    suite = _load_suite(suite_path)
    models = []
    for checkpoint, label in (
        (Path(checkpoint_a), label_a),
        (Path(checkpoint_b), label_b),
    ):
        model, config = load_trained_model(
            checkpoint,
            data_fingerprint=data.fingerprint,
            device=device,
        )
        models.append(
            (
                model,
                config,
                label or f"{config.architecture} ({checkpoint.stem})",
                checkpoint,
            )
        )

    rows: list[dict[str, Any]] = []
    for prompt_case in suite["prompts"]:
        for profile in suite["sampling_profiles"]:
            for model, config, label, checkpoint in models:
                top_k = profile["top_k"]
                text = generate(
                    model,
                    data.tokenizer,
                    prompt_case["prompt"],
                    max_new_tokens=profile.get(
                        "max_new_tokens",
                        suite["max_new_tokens"],
                    ),
                    context_length=config.context_length,
                    temperature=profile["temperature"],
                    top_k=None if top_k == 0 else top_k,
                    seed=profile["seed"],
                    device=device,
                )
                rows.append(
                    {
                        "suite_version": suite["suite_version"],
                        "case_id": prompt_case["id"],
                        "profile_id": profile["id"],
                        "model_label": label,
                        "checkpoint": str(checkpoint.resolve()),
                        "architecture": config.architecture,
                        "model_config": config.to_dict(),
                        "data_fingerprint": data.fingerprint,
                        "prompt": prompt_case["prompt"],
                        "max_new_tokens": profile.get(
                            "max_new_tokens",
                            suite["max_new_tokens"],
                        ),
                        "temperature": profile["temperature"],
                        "top_k": top_k,
                        "seed": profile["seed"],
                        "text": text,
                    }
                )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results_path = destination / "results.jsonl"
    report_path = destination / "report.md"
    _atomic_write(
        results_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    _atomic_write(report_path, _render_report(rows, models))
    return {
        "data_fingerprint": data.fingerprint,
        "device": str(device),
        "generation_count": len(rows),
        "results_path": str(results_path.resolve()),
        "report_path": str(report_path.resolve()),
    }


def _load_suite(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        suite = json.load(handle)
    if not isinstance(suite, dict):
        raise ValueError("comparison suite must contain an object")
    for key in (
        "suite_version",
        "max_new_tokens",
        "sampling_profiles",
        "prompts",
    ):
        if key not in suite:
            raise ValueError(f"comparison suite is missing {key}")
    if not suite["sampling_profiles"] or not suite["prompts"]:
        raise ValueError("comparison suite requires profiles and prompts")
    return suite


def _render_report(
    rows: list[dict[str, Any]],
    models: list[tuple[torch.nn.Module, Any, str, Path]],
) -> str:
    labels = [models[0][2], models[1][2]]
    lines = [
        "# KiwiLM A/B generation report",
        "",
        "The outputs below use the same prompt and sampling seed for both models.",
        "",
    ]
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["case_id"], row["profile_id"]), {})[row["model_label"]] = row
    for (case_id, profile_id), pair in grouped.items():
        first = pair[labels[0]]
        lines.extend(
            [
                f"## {case_id} / {profile_id}",
                "",
                f"Prompt: `{_escape(first['prompt'])}`",
                "",
                f"| {_escape(labels[0])} | {_escape(labels[1])} |",
                "| --- | --- |",
                f"| {_cell(pair[labels[0]]['text'])} | {_cell(pair[labels[1]]['text'])} |",
                "",
            ]
        )
    return "\n".join(lines)


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _cell(value: object) -> str:
    return (
        str(value)
        .replace("\r", "")
        .replace("|", "\\|")
        .replace("\n", "<br>")
    )


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
