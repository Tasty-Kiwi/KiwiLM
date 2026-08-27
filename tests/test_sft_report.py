"""Instruction-adherence scoring and report coverage."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

import kiwilm.sft_report as sft_report
from kiwilm.sft_report import (
    generate_sft_adherence_report,
    load_sft_adherence_suite,
    score_instruction_response,
)


def _case() -> dict:
    return {
        "id": "bridge",
        "prompt": (
            "Features: Dialogue\n"
            "Words: lantern, bridge, careful\n"
            "Summary: Mia helps Noah cross a bridge.\n"
            "Story:\n"
        ),
        "features": ["Dialogue"],
        "words": ["lantern", "bridge", "careful"],
        "entities": ["Mia", "Noah"],
        "summary_terms": [
            ["help", "helps", "helped"],
            ["cross", "crosses", "crossed"],
            ["bridge"],
        ],
    }


def test_instruction_scoring_measures_constraints_and_repetition() -> None:
    response = (
        'Mia held the lantern. "Be careful on the bridge," she told Noah. '
        "Mia helped Noah cross the bridge. "
        "Mia helped Noah cross the bridge."
    )

    scores = score_instruction_response(_case(), response)

    assert scores["required_word_coverage"] == 1.0
    assert scores["summary_term_coverage"] == 1.0
    assert scores["feature_coverage"] == 1.0
    assert scores["entity_coverage"] == 1.0
    assert scores["adherence_score"] == 1.0
    assert scores["repeated_4gram_fraction"] > 0
    assert scores["repeated_sentence_fraction"] > 0


def test_sft_report_writes_scored_machine_and_human_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "suite_version": 1,
                "max_new_tokens": 32,
                "sampling_profiles": [
                    {
                        "id": "greedy",
                        "temperature": 0.0,
                        "top_k": 0,
                        "seed": 42,
                    }
                ],
                "prompts": [_case()],
            }
        ),
        encoding="utf-8",
    )
    data = SimpleNamespace(
        fingerprint="a" * 64,
        tokenizer=object(),
        format_prompt=lambda prompt: prompt,
    )
    config = SimpleNamespace(architecture="kiwilm2", context_length=32)
    monkeypatch.setattr(
        sft_report,
        "load_trained_model",
        lambda *_args, **_kwargs: (torch.nn.Identity(), config),
    )
    monkeypatch.setattr(
        sft_report,
        "generate",
        lambda *_args, **_kwargs: (
            'Mia carried a lantern. "Be careful," she said to Noah. '
            "They crossed the bridge and helped each other."
        ),
    )

    summary = generate_sft_adherence_report(
        [tmp_path / "model.pt"],
        data=data,
        suite_path=suite_path,
        output_dir=tmp_path / "report",
        device=torch.device("cpu"),
        labels=["Model X latest"],
    )

    assert summary["generation_count"] == 1
    assert summary["aggregates"][0]["required_word_coverage"] == 1.0
    assert Path(summary["results_path"]).is_file()
    assert Path(summary["summary_path"]).is_file()
    report = Path(summary["report_path"]).read_text(encoding="utf-8")
    assert "Model X latest" in report
    assert "Aggregate scores" in report
    assert "100.0%" in report
    assert load_sft_adherence_suite(suite_path)["suite_version"] == 1


def test_sft_suite_rejects_unsupported_feature(tmp_path: Path) -> None:
    case = _case()
    case["features"] = ["BadEnding"]
    suite_path = tmp_path / "bad-suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "suite_version": 1,
                "max_new_tokens": 10,
                "sampling_profiles": [
                    {
                        "id": "greedy",
                        "temperature": 0,
                        "top_k": 0,
                        "seed": 1,
                    }
                ],
                "prompts": [case],
            }
        ),
        encoding="utf-8",
    )

    try:
        load_sft_adherence_suite(suite_path)
    except ValueError as error:
        assert "unsupported" in str(error)
    else:
        raise AssertionError("unsupported features must be rejected")
