"""Reproducible side-by-side text generation for model variants."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from kiwilm.data import PreparedTokenData
from kiwilm.generation import generate
from kiwilm.inference import load_trained_model


def compare_checkpoints(
    checkpoints: str | Path | Sequence[str | Path],
    checkpoint_b: str | Path | None = None,
    *,
    data: PreparedTokenData,
    suite_path: str | Path,
    output_dir: str | Path,
    device: torch.device,
    labels: Sequence[str | None] | None = None,
    label_a: str | None = None,
    label_b: str | None = None,
) -> dict[str, Any]:
    """Generate every suite case from two or more checkpoints."""

    suite = load_prompt_suite(suite_path)
    if isinstance(checkpoints, (str, Path)):
        if checkpoint_b is None:
            raise ValueError("a second checkpoint is required")
        resolved_checkpoints = [Path(checkpoints), Path(checkpoint_b)]
        resolved_labels = [label_a, label_b]
        if labels is not None:
            raise ValueError("labels cannot be combined with label_a and label_b")
    else:
        if checkpoint_b is not None:
            raise ValueError(
                "checkpoint_b cannot be combined with a checkpoint sequence"
            )
        if label_a is not None or label_b is not None:
            raise ValueError(
                "label_a and label_b cannot be combined with a checkpoint sequence"
            )
        resolved_checkpoints = [Path(checkpoint) for checkpoint in checkpoints]
        resolved_labels = (
            list(labels)
            if labels is not None
            else [None] * len(resolved_checkpoints)
        )
    if len(resolved_checkpoints) < 2:
        raise ValueError("comparison requires at least two checkpoints")
    if len(resolved_labels) != len(resolved_checkpoints):
        raise ValueError("labels must match the number of checkpoints")

    models = []
    for checkpoint, label in zip(
        resolved_checkpoints,
        resolved_labels,
        strict=True,
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
    model_labels = [model[2] for model in models]
    if len(set(model_labels)) != len(model_labels):
        raise ValueError("comparison labels must be unique")

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
                    cache="off",
                )
                rows.append(
                    {
                        "suite_version": suite["suite_version"],
                        "case_id": prompt_case["id"],
                        "profile_id": profile["id"],
                        "model_label": label,
                        "checkpoint": str(checkpoint),
                        "architecture": config.architecture,
                        "cache": "off",
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


def load_prompt_suite(path: str | Path) -> dict[str, Any]:
    """Load and validate a generation prompt suite."""

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
    labels = [model[2] for model in models]
    lines = [
        "# KiwiLM generation comparison report",
        "",
        "The outputs below use the same prompt and sampling seed for every model.",
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
                "| " + " | ".join(_escape(label) for label in labels) + " |",
                "| " + " | ".join("---" for _ in labels) + " |",
                "| "
                + " | ".join(_cell(pair[label]["text"]) for label in labels)
                + " |",
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
