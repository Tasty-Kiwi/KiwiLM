"""Small proofs for analysis artifact accounting; no real model evaluation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def analysis():
    path = (Path(__file__).resolve().parents[1] / "examples/comparisons/"
            "kiwilm2-final-1b-tpu-muon/evaluate.py")
    spec = importlib.util.spec_from_file_location("final_1b_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generation_group_denominators_and_matched_pairs(analysis) -> None:
    labels = list(analysis.RUNS)
    rows = []
    for label, rates, runs in ((labels[0], [0.1, 0.7], [2, 2]),
                              (labels[1], [0.2, 0.5], [2, 25])):
        for seed, rate, run in zip((42, 43), rates, runs, strict=True):
            rows.append({"model": label, "category": "story", "profile": "focused",
                         "case_id": "story", "seed": seed, "repeated_four_gram_rate": rate,
                         "maximum_consecutive_word_run": run})
    result = analysis.generation_groups(rows)
    overall = next(r for r in result["groups"] if r["model"] == labels[1]
                   and r["category"] == r["profile"] == "all")
    assert overall["samples"] == 2
    assert overall["word_collapses_20_plus"] == 1
    assert overall["samples_repetition_over_0_5"] == 0  # Strictly greater, not >=.
    paired = result["paired_repetition"]
    assert paired["pairs"] == 2
    assert paired["1b_less_repetition"] == paired["1b_more_repetition"] == 1
    assert paired["mean_change_1b_minus_500m"] == pytest.approx(-0.05)


def test_artifact_writers_are_strict_and_atomic(analysis, tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    analysis.write(path, {"loss": 3.5})
    assert json.loads(path.read_text()) == {"loss": 3.5}
    assert not path.with_suffix(".json.tmp").exists()
    with pytest.raises(ValueError):
        analysis.write(path, {"loss": float("nan")})
    assert json.loads(path.read_text()) == {"loss": 3.5}
    rows = [{"model": "500M", "case_id": "a", "profile": "focused", "seed": 42}]
    output = tmp_path / "rows.jsonl"
    analysis.write_rows(output, rows)
    assert analysis.row_key(json.loads(output.read_text())) == ("500M", "a", "focused", 42)
    assert not output.with_suffix(".tmp").exists()


def test_provenance_guards_survive_optimized_python(analysis) -> None:
    with pytest.raises(ValueError, match="changed"):
        analysis.require(False, "changed checkpoint")
