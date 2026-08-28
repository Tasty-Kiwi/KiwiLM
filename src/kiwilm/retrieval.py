"""Deterministic counterfactual context-retrieval probes for causal LMs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

RETRIEVAL_SUITE_VERSION = 1
DEFAULT_RETRIEVAL_DISTANCES = (32, 128, 256, 384, 448)
DEFAULT_RETRIEVAL_PAIRS_PER_DISTANCE = 32

_CANDIDATE_POOL = (
    " red",
    " blue",
    " green",
    " yellow",
    " pink",
    " purple",
    " orange",
    " brown",
    " white",
    " black",
    " gray",
    " gold",
)

_TEMPLATES = (
    {
        "id": "lantern",
        "binding_prefix": "Mila carried a little lantern. The lantern was",
        "binding_suffix": ".",
        "query": " Later, the lantern was",
    },
    {
        "id": "ribbon",
        "binding_prefix": "Noah tied a ribbon to his kite. Noah's ribbon was",
        "binding_suffix": ".",
        "query": " Later, Noah's ribbon was",
    },
    {
        "id": "gate",
        "binding_prefix": "Lily showed Ben a secret garden gate. The gate was",
        "binding_suffix": ".",
        "query": " Later, the secret gate was",
    },
    {
        "id": "blanket",
        "binding_prefix": "A puppy slept beneath a soft blanket. The blanket was",
        "binding_suffix": ".",
        "query": " Later, the puppy's blanket was",
    },
)

_FILLER_PASSAGES = (
    " The friends walked through the sunny garden. They listened to birds and "
    "told a gentle story before lunch.",
    " A small family played beside the quiet house. Everyone smiled, shared a "
    "snack, and watched the clouds.",
    " The children followed a winding path near the trees. They sang a song and "
    "helped each other along the way.",
    " Morning came to the peaceful village. The animals woke up, the flowers "
    "opened, and the day felt warm.",
)


def build_retrieval_suite(
    tokenizer: Any,
    *,
    context_length: int = 512,
    distances: Sequence[int] = DEFAULT_RETRIEVAL_DISTANCES,
    pairs_per_distance: int = DEFAULT_RETRIEVAL_PAIRS_PER_DISTANCE,
    seed: int = 42,
    candidate_values: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build balanced, single-token, counterfactual cloze probes.

    Distance is measured from the bound answer token to the final input token,
    whose logits predict the answer to the repeated cloze query.
    """

    _require_positive_int("context_length", context_length)
    _require_positive_int("pairs_per_distance", pairs_per_distance)
    if pairs_per_distance % len(_TEMPLATES) != 0:
        raise ValueError(f"pairs_per_distance must be divisible by {len(_TEMPLATES)}")
    resolved_distances = _validate_distances(distances)
    candidates = _resolve_candidates(tokenizer, candidate_values)
    candidate_ids = [candidate["token_id"] for candidate in candidates]
    filler_passages = [
        _safe_filler_ids(tokenizer, text, candidate_ids) for text in _FILLER_PASSAGES
    ]

    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []
    for distance_index, distance in enumerate(resolved_distances):
        distance_pairs: list[dict[str, Any]] = []
        for pair_index in range(pairs_per_distance):
            template_index = pair_index % len(_TEMPLATES)
            template = _TEMPLATES[template_index]
            target_a = (pair_index // len(_TEMPLATES) + template_index + distance_index) % len(
                candidates
            )
            target_b = (target_a + 1) % len(candidates)
            pair_id = f"d{distance:03d}-{template['id']}-{pair_index:03d}"
            filler_ids = filler_passages[(pair_index + distance_index) % len(filler_passages)]
            distance_pairs.append(
                _build_pair(
                    tokenizer,
                    pair_id=pair_id,
                    template=template,
                    distance=distance,
                    target_a=target_a,
                    target_b=target_b,
                    candidates=candidates,
                    filler_ids=filler_ids,
                    context_length=context_length,
                )
            )
        rng.shuffle(distance_pairs)
        pairs.extend(distance_pairs)

    suite = {
        "suite_version": RETRIEVAL_SUITE_VERSION,
        "seed": int(seed),
        "context_length": context_length,
        "distances": list(resolved_distances),
        "pairs_per_distance": pairs_per_distance,
        "pair_count": len(pairs),
        "case_count": len(pairs) * 2,
        "distance_definition": ("final_input_position_minus_bound_answer_position"),
        "tokenizer": _tokenizer_metadata(tokenizer),
        "candidates": candidates,
        "templates": [template["id"] for template in _TEMPLATES],
        "pairs": pairs,
    }
    _validate_suite(suite)
    return suite


def evaluate_retrieval_model(
    model: nn.Module,
    suite: Mapping[str, Any],
    *,
    label: str,
    device: torch.device,
    batch_size: int = 32,
    architecture: str | None = None,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate one model and return summary plus per-case diagnostics."""

    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a non-empty string")
    _require_positive_int("batch_size", batch_size)
    _validate_suite(suite)

    candidates = _require_sequence(suite["candidates"], "candidates")
    candidate_ids = [
        _require_int(candidate["token_id"], "candidate token_id") for candidate in candidates
    ]
    jobs: list[tuple[str, str, list[int]]] = []
    for pair in _require_sequence(suite["pairs"], "pairs"):
        pair_id = str(pair["id"])
        variants = _require_mapping(pair["variants"], "pair variants")
        for variant_name in ("a", "b", "control"):
            variant = _require_mapping(
                variants[variant_name], f"pair {pair_id} variant {variant_name}"
            )
            jobs.append((pair_id, variant_name, list(variant["input_ids"])))

    scores = _score_jobs(
        model,
        jobs,
        candidate_ids=candidate_ids,
        device=device,
        batch_size=batch_size,
    )
    cases: list[dict[str, Any]] = []
    pair_results: list[dict[str, Any]] = []
    candidate_texts = [str(candidate["text"]) for candidate in candidates]
    for pair in _require_sequence(suite["pairs"], "pairs"):
        pair_id = str(pair["id"])
        control_logits = scores[(pair_id, "control")]
        variants = _require_mapping(pair["variants"], "pair variants")
        pair_correct = True
        for variant_name in ("a", "b"):
            variant = _require_mapping(variants[variant_name], "bound variant")
            target_index = _require_int(variant["target_index"], "target_index")
            candidate_logits = scores[(pair_id, variant_name)]
            predicted_index = max(range(len(candidate_logits)), key=candidate_logits.__getitem__)
            distractor_score = max(
                score for index, score in enumerate(candidate_logits) if index != target_index
            )
            correct = predicted_index == target_index
            pair_correct = pair_correct and correct
            cases.append(
                {
                    "pair_id": pair_id,
                    "variant": variant_name,
                    "template": pair["template"],
                    "distance": pair["measured_distance"],
                    "target_index": target_index,
                    "target": candidate_texts[target_index],
                    "predicted_index": predicted_index,
                    "predicted": candidate_texts[predicted_index],
                    "correct": correct,
                    "candidate_logits": candidate_logits,
                    "target_vs_best_distractor_margin": (
                        candidate_logits[target_index] - distractor_score
                    ),
                    "contextual_logit_lift": (
                        candidate_logits[target_index] - control_logits[target_index]
                    ),
                }
            )
        pair_results.append(
            {
                "pair_id": pair_id,
                "template": pair["template"],
                "distance": pair["measured_distance"],
                "paired_flip_correct": pair_correct,
            }
        )

    resolved_architecture = architecture
    if resolved_architecture is None:
        resolved_architecture = getattr(getattr(model, "config", None), "architecture", None)
    summary = {
        "label": label,
        "architecture": resolved_architecture,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        **_summarize(cases, pair_results),
        "by_distance": {
            str(distance): _summarize(
                [case for case in cases if case["distance"] == distance],
                [pair for pair in pair_results if pair["distance"] == distance],
            )
            for distance in suite["distances"]
        },
        "by_template": {
            str(template): _summarize(
                [case for case in cases if case["template"] == template],
                [pair for pair in pair_results if pair["template"] == template],
            )
            for template in suite["templates"]
        },
    }
    return {"summary": summary, "cases": cases, "pairs": pair_results}


def write_retrieval_artifacts(
    output_dir: str | Path,
    *,
    suite: Mapping[str, Any],
    evaluations: Sequence[Mapping[str, Any]],
    title: str = "Context Retrieval Benchmark",
) -> dict[str, Any]:
    """Atomically write the reusable suite, machine results, and Markdown report."""

    _validate_suite(suite)
    if not evaluations:
        raise ValueError("at least one retrieval evaluation is required")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    suite_path = destination / "suite.json"
    results_path = destination / "results.jsonl"
    report_path = destination / "report.md"
    _atomic_write_text(
        suite_path,
        json.dumps(suite, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    _atomic_write_text(
        results_path,
        "".join(
            json.dumps(
                dict(evaluation),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for evaluation in evaluations
        ),
    )
    _atomic_write_text(report_path, render_retrieval_report(title, suite, evaluations))
    return {
        "model_count": len(evaluations),
        "suite_path": str(suite_path.resolve()),
        "results_path": str(results_path.resolve()),
        "report_path": str(report_path.resolve()),
    }


def render_retrieval_report(
    title: str,
    suite: Mapping[str, Any],
    evaluations: Sequence[Mapping[str, Any]],
) -> str:
    """Render retrieval summaries and distance/template breakdowns."""

    lines = [
        f"# {title}",
        "",
        (
            f"{suite['pair_count']} counterfactual pairs / {suite['case_count']} "
            f"bound cases; seed {suite['seed']}; context window "
            f"{suite['context_length']}."
        ),
        "",
        (
            "Candidate accuracy scores the four-way cloze; paired flip requires both "
            "counterfactual variants to select their changed target. Margin is target "
            "minus best distractor logit, and lift is bound-target minus no-binding "
            "control logit."
        ),
        "",
        "## Overall",
        "",
        "| Model | Candidate accuracy | Paired flip | Mean margin | Mean logit lift |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for evaluation in evaluations:
        summary = _require_mapping(evaluation["summary"], "evaluation summary")
        lines.append(
            "| "
            + " | ".join(
                (
                    _escape_markdown(str(summary["label"])),
                    _percent(summary["candidate_accuracy"]),
                    _percent(summary["paired_flip_accuracy"]),
                    _decimal(summary["mean_target_vs_best_distractor_margin"]),
                    _decimal(summary["mean_contextual_logit_lift"]),
                )
            )
            + " |"
        )
    for evaluation in evaluations:
        summary = _require_mapping(evaluation["summary"], "evaluation summary")
        lines.extend(
            [
                "",
                f"## {_escape_markdown(str(summary['label']))}",
                "",
                "### By distance",
                "",
                "| Distance | Accuracy | Paired flip | Mean margin | Mean logit lift |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for distance in suite["distances"]:
            metrics = summary["by_distance"][str(distance)]
            lines.append(_breakdown_row(str(distance), metrics))
        lines.extend(
            [
                "",
                "### By template",
                "",
                "| Template | Accuracy | Paired flip | Mean margin | Mean logit lift |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for template in suite["templates"]:
            metrics = summary["by_template"][str(template)]
            lines.append(_breakdown_row(_escape_markdown(str(template)), metrics))
    return "\n".join(lines).rstrip() + "\n"


def _build_pair(
    tokenizer: Any,
    *,
    pair_id: str,
    template: Mapping[str, str],
    distance: int,
    target_a: int,
    target_b: int,
    candidates: Sequence[Mapping[str, Any]],
    filler_ids: Sequence[int],
    context_length: int,
) -> dict[str, Any]:
    prefix_ids = _encode(tokenizer, template["binding_prefix"])
    suffix_ids = _encode(tokenizer, template["binding_suffix"])
    query_ids = _encode(tokenizer, template["query"])
    filler_length = distance - len(suffix_ids) - len(query_ids)
    if filler_length < 0:
        raise ValueError(
            f"distance {distance} is too short for template {template['id']!r}; "
            f"minimum is {len(suffix_ids) + len(query_ids)}"
        )
    selected_filler = _repeat_to_length(filler_ids, filler_length)
    needle_length = len(prefix_ids) + 1 + len(suffix_ids)
    neutral_ids = _safe_filler_ids(
        tokenizer,
        " A child played quietly near the house. Nothing important was said.",
        [candidate["token_id"] for candidate in candidates],
    )
    control_prefix = _repeat_to_length(neutral_ids, needle_length)

    variants: dict[str, Any] = {}
    for name, target_index in (("a", target_a), ("b", target_b)):
        answer_id = int(candidates[target_index]["token_id"])
        input_ids = [
            *prefix_ids,
            answer_id,
            *suffix_ids,
            *selected_filler,
            *query_ids,
        ]
        answer_position = len(prefix_ids)
        measured_distance = len(input_ids) - 1 - answer_position
        if measured_distance != distance:
            raise AssertionError("retrieval distance construction is inconsistent")
        if len(input_ids) > context_length:
            raise ValueError(
                f"distance {distance} with template {template['id']!r} requires "
                f"{len(input_ids)} tokens, exceeding context_length={context_length}"
            )
        variants[name] = {
            "target_index": target_index,
            "answer_position": answer_position,
            "input_ids": input_ids,
            "text": _decode(tokenizer, input_ids),
        }

    control_ids = [*control_prefix, *selected_filler, *query_ids]
    if len(control_ids) != len(variants["a"]["input_ids"]):
        raise AssertionError("control and bound contexts must have equal lengths")
    variants["control"] = {
        "input_ids": control_ids,
        "text": _decode(tokenizer, control_ids),
    }
    return {
        "id": pair_id,
        "template": template["id"],
        "requested_distance": distance,
        "measured_distance": distance,
        "variants": variants,
    }


def _score_jobs(
    model: nn.Module,
    jobs: Sequence[tuple[str, str, list[int]]],
    *,
    candidate_ids: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> dict[tuple[str, str], list[float]]:
    grouped: dict[int, list[tuple[str, str, list[int]]]] = defaultdict(list)
    for job in jobs:
        grouped[len(job[2])].append(job)
    results: dict[tuple[str, str], list[float]] = {}
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for length in sorted(grouped):
                group = grouped[length]
                for start in range(0, len(group), batch_size):
                    batch = group[start : start + batch_size]
                    input_ids = torch.tensor(
                        [job[2] for job in batch], dtype=torch.long, device=device
                    )
                    logits = model(input_ids)
                    _validate_logits(logits, input_ids, candidate_ids)
                    selected = logits[:, -1, candidate_ids].float().cpu()
                    for job, row in zip(batch, selected, strict=True):
                        values = [float(value) for value in row.tolist()]
                        if not all(math.isfinite(value) for value in values):
                            raise ValueError("model produced non-finite retrieval logits")
                        results[(job[0], job[1])] = values
    finally:
        model.train(was_training)
    return results


def _validate_logits(
    logits: Tensor,
    input_ids: Tensor,
    candidate_ids: Sequence[int],
) -> None:
    if logits.ndim != 3:
        raise ValueError("model must return logits with shape [batch, sequence, vocab]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("model logits do not match the retrieval input shape")
    if max(candidate_ids) >= logits.shape[-1]:
        raise ValueError("retrieval candidate token is outside the model vocabulary")


def _summarize(
    cases: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not cases or not pairs:
        raise ValueError("retrieval metric groups must not be empty")
    return {
        "pair_count": len(pairs),
        "case_count": len(cases),
        "candidate_accuracy": sum(bool(case["correct"]) for case in cases) / len(cases),
        "paired_flip_accuracy": sum(bool(pair["paired_flip_correct"]) for pair in pairs)
        / len(pairs),
        "mean_target_vs_best_distractor_margin": sum(
            float(case["target_vs_best_distractor_margin"]) for case in cases
        )
        / len(cases),
        "mean_contextual_logit_lift": sum(float(case["contextual_logit_lift"]) for case in cases)
        / len(cases),
    }


def _resolve_candidates(
    tokenizer: Any,
    candidate_values: Sequence[str] | None,
) -> list[dict[str, Any]]:
    if candidate_values is not None and len(candidate_values) != 4:
        raise ValueError("candidate_values must contain exactly four strings")
    source = candidate_values if candidate_values is not None else _CANDIDATE_POOL
    candidates: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    special_ids = {
        int(value)
        for name in ("pad_id", "unk_id", "bos_id", "eos_id")
        if (value := getattr(tokenizer, name, None)) is not None
    }
    for raw_value in source:
        if not isinstance(raw_value, str) or not raw_value.strip():
            if candidate_values is not None:
                raise ValueError("candidate values must be non-empty strings")
            continue
        value = raw_value if raw_value.startswith(" ") else f" {raw_value}"
        token_ids = _encode(tokenizer, value)
        compatible = (
            len(token_ids) == 1 and token_ids[0] not in special_ids and token_ids[0] not in seen_ids
        )
        if not compatible:
            if candidate_values is not None:
                raise ValueError(
                    f"candidate {raw_value!r} must encode to one unique non-special token"
                )
            continue
        token_id = token_ids[0]
        candidates.append({"text": value, "token_id": token_id})
        seen_ids.add(token_id)
        if candidate_values is None and len(candidates) == 4:
            break
    if len(candidates) != 4:
        raise ValueError("tokenizer does not provide four compatible single-token color candidates")
    return candidates


def _tokenizer_metadata(tokenizer: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, int) and not isinstance(vocab_size, bool):
        metadata["vocab_size"] = vocab_size
    to_json = getattr(tokenizer, "to_json", None)
    if callable(to_json):
        try:
            serialized = str(to_json(pretty=False))
        except TypeError:
            serialized = str(to_json())
        metadata["sha256"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return metadata


def _safe_filler_ids(
    tokenizer: Any,
    text: str,
    candidate_ids: Sequence[int],
) -> list[int]:
    candidate_set = set(candidate_ids)
    ids = [token_id for token_id in _encode(tokenizer, text) if token_id not in candidate_set]
    if not ids:
        raise ValueError("tokenizer produced no candidate-free filler tokens")
    return ids


def _repeat_to_length(values: Sequence[int], length: int) -> list[int]:
    if length == 0:
        return []
    if not values:
        raise ValueError("cannot extend an empty filler sequence")
    repetitions, remainder = divmod(length, len(values))
    return [*values] * repetitions + list(values[:remainder])


def _encode(tokenizer: Any, text: str) -> list[int]:
    values = tokenizer.encode(text)
    if not isinstance(values, Sequence):
        raise TypeError("tokenizer.encode must return a sequence")
    token_ids = [_require_int(value, "token ID") for value in values]
    if not token_ids:
        raise ValueError(f"tokenizer encoded {text!r} to an empty sequence")
    if any(token_id < 0 for token_id in token_ids):
        raise ValueError("tokenizer emitted a negative token ID")
    return token_ids


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return str(tokenizer.decode(token_ids, skip_special_tokens=True))
    except TypeError:
        return str(tokenizer.decode(token_ids))


def _validate_suite(suite: Mapping[str, Any]) -> None:
    if suite.get("suite_version") != RETRIEVAL_SUITE_VERSION:
        raise ValueError("unsupported retrieval suite version")
    context_length = _require_int(suite.get("context_length"), "context_length")
    candidates = _require_sequence(suite.get("candidates"), "candidates")
    if len(candidates) != 4:
        raise ValueError("retrieval suite must contain four candidates")
    candidate_ids = []
    for candidate in candidates:
        candidate_mapping = _require_mapping(candidate, "candidate")
        candidate_ids.append(_require_int(candidate_mapping.get("token_id"), "candidate token_id"))
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("retrieval candidate token IDs must be unique")
    pairs = _require_sequence(suite.get("pairs"), "pairs")
    if not pairs:
        raise ValueError("retrieval suite must contain pairs")
    for pair in pairs:
        pair_mapping = _require_mapping(pair, "pair")
        distance = _require_int(pair_mapping.get("measured_distance"), "distance")
        if distance != pair_mapping.get("requested_distance"):
            raise ValueError("requested and measured retrieval distances differ")
        variants = _require_mapping(pair_mapping.get("variants"), "pair variants")
        bound_length: int | None = None
        for variant_name in ("a", "b", "control"):
            variant = _require_mapping(variants.get(variant_name), "variant")
            input_ids = _require_sequence(variant.get("input_ids"), "input_ids")
            if not input_ids or len(input_ids) > context_length:
                raise ValueError("retrieval input is empty or exceeds context_length")
            for token_id in input_ids:
                _require_int(token_id, "input token ID")
            if bound_length is None:
                bound_length = len(input_ids)
            elif len(input_ids) != bound_length:
                raise ValueError("counterfactual and control input lengths differ")
            if variant_name != "control":
                target_index = _require_int(variant.get("target_index"), "target_index")
                if target_index < 0 or target_index >= len(candidates):
                    raise ValueError("retrieval target index is out of range")
                answer_position = _require_int(variant.get("answer_position"), "answer_position")
                measured = len(input_ids) - 1 - answer_position
                if measured != distance:
                    raise ValueError("stored retrieval distance is inconsistent")


def _validate_distances(distances: Sequence[int]) -> tuple[int, ...]:
    if not distances:
        raise ValueError("at least one retrieval distance is required")
    resolved = tuple(_require_int(value, "distance") for value in distances)
    if any(value <= 0 for value in resolved):
        raise ValueError("retrieval distances must be positive")
    if len(set(resolved)) != len(resolved):
        raise ValueError("retrieval distances must be unique")
    return resolved


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")
    return value


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _percent(value: Any) -> str:
    return f"{float(value) * 100:.2f}%"


def _decimal(value: Any) -> str:
    return f"{float(value):.4f}"


def _breakdown_row(label: str, metrics: Mapping[str, Any]) -> str:
    return (
        f"| {label} | {_percent(metrics['candidate_accuracy'])} | "
        f"{_percent(metrics['paired_flip_accuracy'])} | "
        f"{_decimal(metrics['mean_target_vs_best_distractor_margin'])} | "
        f"{_decimal(metrics['mean_contextual_logit_lift'])} |"
    )


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|")


__all__ = [
    "DEFAULT_RETRIEVAL_DISTANCES",
    "DEFAULT_RETRIEVAL_PAIRS_PER_DISTANCE",
    "RETRIEVAL_SUITE_VERSION",
    "build_retrieval_suite",
    "evaluate_retrieval_model",
    "render_retrieval_report",
    "write_retrieval_artifacts",
]
