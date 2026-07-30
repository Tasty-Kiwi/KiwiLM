"""Deterministic instruction-adherence generation and scoring reports."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from kiwilm.generation import generate
from kiwilm.inference import load_trained_model
from kiwilm.sft import PreparedSFTData

_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_DIALOGUE_PATTERN = re.compile(r"""["“][^"”\n]+["”]""")
_SENTENCE_PATTERN = re.compile(r"[^.!?]+[.!?]?")


def generate_sft_adherence_report(
    checkpoints: Sequence[str | Path],
    *,
    data: PreparedSFTData,
    suite_path: str | Path,
    output_dir: str | Path,
    device: torch.device,
    labels: Sequence[str] | None = None,
    cache: str = "off",
) -> dict[str, Any]:
    """Generate and score a fixed instruction suite for one or more checkpoints."""

    resolved_checkpoints = [Path(checkpoint) for checkpoint in checkpoints]
    if not resolved_checkpoints:
        raise ValueError("SFT adherence reporting requires at least one checkpoint")
    if cache not in {"auto", "off"}:
        raise ValueError("cache must be 'auto' or 'off'")
    if labels is not None and len(labels) != len(resolved_checkpoints):
        raise ValueError("labels must match the number of checkpoints")
    suite = load_sft_adherence_suite(suite_path)

    models: list[tuple[torch.nn.Module, Any, str, Path]] = []
    for index, checkpoint in enumerate(resolved_checkpoints):
        model, config = load_trained_model(
            checkpoint,
            data_fingerprint=data.fingerprint,
            device=device,
        )
        label = (
            labels[index]
            if labels is not None
            else f"{config.architecture} ({checkpoint.stem})"
        )
        models.append((model, config, label, checkpoint))
    model_labels = [model[2] for model in models]
    if len(set(model_labels)) != len(model_labels):
        raise ValueError("SFT adherence report labels must be unique")

    rows: list[dict[str, Any]] = []
    for prompt_case in suite["prompts"]:
        for profile in suite["sampling_profiles"]:
            for model, config, label, checkpoint in models:
                top_k = profile["top_k"]
                response = generate(
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
                    cache=cache,
                    include_prompt=False,
                )
                scores = score_instruction_response(prompt_case, response)
                rows.append(
                    {
                        "suite_version": suite["suite_version"],
                        "case_id": prompt_case["id"],
                        "profile_id": profile["id"],
                        "model_label": label,
                        "checkpoint": str(checkpoint.resolve()),
                        "architecture": config.architecture,
                        "data_fingerprint": data.fingerprint,
                        "cache": cache,
                        "prompt": prompt_case["prompt"],
                        "response": response,
                        "scores": scores,
                        "temperature": profile["temperature"],
                        "top_k": top_k,
                        "seed": profile["seed"],
                    }
                )

    aggregates = _aggregate_scores(rows)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results_path = destination / "results.jsonl"
    summary_path = destination / "summary.json"
    report_path = destination / "report.md"
    summary = {
        "suite_version": suite["suite_version"],
        "data_fingerprint": data.fingerprint,
        "device": str(device),
        "cache": cache,
        "generation_count": len(rows),
        "aggregates": aggregates,
        "results_path": str(results_path.resolve()),
        "report_path": str(report_path.resolve()),
    }
    _atomic_write(
        results_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    _atomic_write(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _atomic_write(report_path, _render_report(rows, aggregates))
    return {**summary, "summary_path": str(summary_path.resolve())}


def load_sft_adherence_suite(path: str | Path) -> dict[str, Any]:
    """Load and validate the declarative instruction-adherence suite."""

    try:
        suite = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read the SFT adherence suite") from error
    if not isinstance(suite, dict):
        raise ValueError("SFT adherence suite must contain an object")
    for key in ("suite_version", "max_new_tokens", "sampling_profiles", "prompts"):
        if key not in suite:
            raise ValueError(f"SFT adherence suite is missing {key}")
    if not isinstance(suite["max_new_tokens"], int) or suite["max_new_tokens"] < 1:
        raise ValueError("SFT adherence max_new_tokens must be positive")
    profiles = suite["sampling_profiles"]
    prompts = suite["prompts"]
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("SFT adherence suite requires sampling profiles")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("SFT adherence suite requires prompts")
    for profile in profiles:
        if not isinstance(profile, dict):
            raise ValueError("SFT sampling profiles must be objects")
        for key in ("id", "temperature", "top_k", "seed"):
            if key not in profile:
                raise ValueError(f"SFT sampling profile is missing {key}")
    for prompt in prompts:
        _validate_prompt_case(prompt)
    return suite


def score_instruction_response(
    prompt_case: Mapping[str, Any],
    response: str,
) -> dict[str, Any]:
    """Score lexical constraints and deterministic degeneration indicators."""

    if not isinstance(response, str):
        raise TypeError("response must be a string")
    words = _string_list(prompt_case.get("words", []), "words")
    features = [
        feature.casefold()
        for feature in _string_list(prompt_case.get("features", []), "features")
    ]
    entities = _string_list(prompt_case.get("entities", []), "entities")
    summary_terms = prompt_case.get("summary_terms", [])
    if not isinstance(summary_terms, list):
        raise ValueError("summary_terms must be a list")
    term_groups = [
        _string_list(group, "summary term group") for group in summary_terms
    ]
    if any(not group for group in term_groups):
        raise ValueError("summary term groups cannot be empty")

    matched_words = [word for word in words if _contains_phrase(response, word)]
    matched_entities = [
        entity for entity in entities if _contains_phrase(response, entity)
    ]
    matched_summary_groups = [
        index
        for index, group in enumerate(term_groups)
        if any(_contains_phrase(response, term) for term in group)
    ]
    feature_results = {
        feature: _feature_matches(feature, response) for feature in features
    }
    word_coverage = _coverage(len(matched_words), len(words))
    summary_coverage = _coverage(len(matched_summary_groups), len(term_groups))
    feature_coverage = _coverage(
        sum(feature_results.values()),
        len(feature_results),
    )
    entity_coverage = _coverage(len(matched_entities), len(entities))
    adherence_components = [
        value
        for value in (
            word_coverage,
            summary_coverage,
            feature_coverage,
            entity_coverage,
        )
        if value is not None
    ]
    return {
        "required_word_coverage": word_coverage,
        "matched_words": matched_words,
        "summary_term_coverage": summary_coverage,
        "matched_summary_groups": matched_summary_groups,
        "feature_coverage": feature_coverage,
        "feature_results": feature_results,
        "entity_coverage": entity_coverage,
        "matched_entities": matched_entities,
        "adherence_score": (
            sum(adherence_components) / len(adherence_components)
            if adherence_components
            else None
        ),
        "repeated_4gram_fraction": _repeated_ngram_fraction(response, 4),
        "repeated_sentence_fraction": _repeated_sentence_fraction(response),
        "response_words": len(_words(response)),
    }


def _validate_prompt_case(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("SFT prompt cases must be objects")
    for key in ("id", "prompt"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"SFT prompt case requires a non-empty {key}")
    _string_list(value.get("words", []), "words")
    features = _string_list(value.get("features", []), "features")
    unsupported = {
        feature for feature in features if feature.casefold() not in {"dialogue"}
    }
    if unsupported:
        raise ValueError(f"unsupported automatically scored features: {unsupported}")
    _string_list(value.get("entities", []), "entities")
    summary_terms = value.get("summary_terms", [])
    if not isinstance(summary_terms, list):
        raise ValueError("summary_terms must be a list")
    for group in summary_terms:
        if not _string_list(group, "summary term group"):
            raise ValueError("summary term groups cannot be empty")


def _feature_matches(feature: str, response: str) -> bool:
    if feature == "dialogue":
        return bool(_DIALOGUE_PATTERN.search(response))
    raise ValueError(f"unsupported automatically scored feature: {feature}")


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = r"(?<![A-Za-z])" + re.escape(phrase) + r"(?![A-Za-z])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _words(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD_PATTERN.finditer(text)]


def _repeated_ngram_fraction(text: str, size: int) -> float:
    words = _words(text)
    if len(words) < size:
        return 0.0
    ngrams = [tuple(words[index : index + size]) for index in range(len(words) - size + 1)]
    return (len(ngrams) - len(set(ngrams))) / len(ngrams)


def _repeated_sentence_fraction(text: str) -> float:
    sentences = [
        " ".join(_words(match.group(0)))
        for match in _SENTENCE_PATTERN.finditer(text)
        if _words(match.group(0))
    ]
    if not sentences:
        return 0.0
    return (len(sentences) - len(set(sentences))) / len(sentences)


def _coverage(matches: int, total: int) -> float | None:
    return matches / total if total else None


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{name} must be a list of non-empty strings")
    return value


def _aggregate_scores(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["model_label"]), str(row["profile_id"])),
            [],
        ).append(row["scores"])
    aggregates: list[dict[str, Any]] = []
    metric_names = (
        "required_word_coverage",
        "summary_term_coverage",
        "feature_coverage",
        "entity_coverage",
        "adherence_score",
        "repeated_4gram_fraction",
        "repeated_sentence_fraction",
    )
    for (model_label, profile_id), scores in grouped.items():
        aggregate: dict[str, Any] = {
            "model_label": model_label,
            "profile_id": profile_id,
            "cases": len(scores),
        }
        for name in metric_names:
            values = [
                float(score[name])
                for score in scores
                if score.get(name) is not None
            ]
            aggregate[name] = sum(values) / len(values) if values else None
        aggregates.append(aggregate)
    return aggregates


def _render_report(
    rows: Sequence[Mapping[str, Any]],
    aggregates: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# KiwiLM SFT instruction-adherence report",
        "",
        "Scores use deterministic lexical checks. Lower repetition is better.",
        "",
        "## Aggregate scores",
        "",
        "| Model | Profile | Words | Summary | Features | Entities | Adherence | Repeat-4 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for aggregate in aggregates:
        lines.append(
            "| "
            + " | ".join(
                (
                    _escape(aggregate["model_label"]),
                    _escape(aggregate["profile_id"]),
                    _percent(aggregate["required_word_coverage"]),
                    _percent(aggregate["summary_term_coverage"]),
                    _percent(aggregate["feature_coverage"]),
                    _percent(aggregate["entity_coverage"]),
                    _percent(aggregate["adherence_score"]),
                    _percent(aggregate["repeated_4gram_fraction"]),
                )
            )
            + " |"
        )
    lines.append("")
    for row in rows:
        scores = row["scores"]
        lines.extend(
            [
                f"## {_escape(row['case_id'])} / {_escape(row['profile_id'])} / "
                f"{_escape(row['model_label'])}",
                "",
                f"- Words: {_percent(scores['required_word_coverage'])} "
                f"({_escape(', '.join(scores['matched_words']) or 'none')})",
                f"- Summary terms: {_percent(scores['summary_term_coverage'])}",
                f"- Features: {_percent(scores['feature_coverage'])}",
                f"- Entities: {_percent(scores['entity_coverage'])}",
                f"- Repeated 4-grams: {_percent(scores['repeated_4gram_fraction'])}",
                "",
                "```text",
                str(row["response"]).replace("```", "'''").rstrip(),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _percent(value: Any) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number * 100:.1f}%"


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _atomic_write(path: Path, contents: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


__all__ = [
    "generate_sft_adherence_report",
    "load_sft_adherence_suite",
    "score_instruction_response",
]
