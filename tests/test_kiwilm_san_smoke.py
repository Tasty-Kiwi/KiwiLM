"""Coverage for the KiwiLM-SAN smoke benchmark runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from kiwilm.checkpoint import save_checkpoint
from kiwilm.config import ModelXConfig, ModelYConfig
from kiwilm.data import PreparedTokenData, prepare_from_stories
from kiwilm.models import build_model


def _load_runner(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    scripts_dir = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    module_path = scripts_dir / "run_kiwilm_san_smoke_benchmark.py"
    spec = importlib.util.spec_from_file_location("kiwilm_san_smoke", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_test_inputs(tmp_path: Path) -> tuple[Path, PreparedTokenData, Path]:
    data_dir = tmp_path / "data"
    prepare_from_stories(
        data_dir,
        [
            "Once there was a red bird who found a blue stone near the old tree.",
            "A green fox met a yellow duck and they walked safely home together.",
        ]
        * 12,
        [
            "The red bird remembered the blue stone and returned before dark.",
            "The green fox helped the yellow duck find the path home.",
        ]
        * 4,
        vocab_size=400,
        min_frequency=1,
    )
    data = PreparedTokenData(data_dir)
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "suite_version": 1,
                "max_new_tokens": 1,
                "sampling_profiles": [
                    {
                        "id": "focused",
                        "temperature": 0,
                        "top_k": 0,
                        "seed": 42,
                    }
                ],
                "prompts": [{"id": "once", "prompt": "Once"}],
            }
        ),
        encoding="utf-8",
    )
    return data_dir, data, suite


def _save_baselines(
    tmp_path: Path,
    data: PreparedTokenData,
    *,
    context_length: int,
) -> tuple[Path, Path]:
    shared: dict[str, Any] = {
        "vocab_size": data.tokenizer.vocab_size,
        "context_length": context_length,
        "d_model": 8,
        "dropout": 0.0,
        "num_heads": 1,
        "swiglu_dim": 12,
    }
    checkpoints = []
    for name, config in (
        ("model-x", ModelXConfig(**shared)),
        ("model-y", ModelYConfig(**shared, num_layers=1)),
    ):
        checkpoint = save_checkpoint(
            tmp_path / f"{name}.pt",
            model=build_model(config),
            step=1,
            model_config=config,
            data_fingerprint=data.fingerprint,
            training_state={"tokens_seen": 16},
        )
        checkpoints.append(checkpoint)
    return checkpoints[0], checkpoints[1]


def _stub_retrieval(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        module,
        "build_retrieval_suite",
        lambda tokenizer, **kwargs: {
            "suite_version": 1,
            "context_length": kwargs["context_length"],
            "pairs": [{"pair_id": "test"}],
        },
    )

    def evaluate_stub(
        model: object,
        suite: dict[str, Any],
        *,
        label: str,
        device: object,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "summary": {"label": label, "candidate_accuracy": 1.0},
            "cases": [{"pair_id": "test", "correct": True}],
        }

    monkeypatch.setattr(module, "evaluate_retrieval_model", evaluate_stub)

    def write_stub(
        output_dir: Path,
        *,
        suite: dict[str, Any],
        evaluations: list[dict[str, Any]],
        title: str,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        suite_path = output_dir / "suite.json"
        results_path = output_dir / "results.jsonl"
        report_path = output_dir / "report.md"
        suite_path.write_text(json.dumps(suite), encoding="utf-8")
        results_path.write_text(
            "".join(json.dumps(row["summary"]) + "\n" for row in evaluations),
            encoding="utf-8",
        )
        report_path.write_text(f"# {title}\n", encoding="utf-8")
        return {
            "model_count": len(evaluations),
            "suite_path": str(suite_path.resolve()),
            "results_path": str(results_path.resolve()),
            "report_path": str(report_path.resolve()),
        }

    monkeypatch.setattr(module, "write_retrieval_artifacts", write_stub)


def test_parser_defaults_match_the_san_smoke_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner(monkeypatch)
    args = module.build_parser().parse_args([])

    assert args.output_dir == Path("runs/benchmarks/kiwilm-san-smoke")
    assert args.model_x_checkpoint == module.DEFAULT_MODEL_X_CHECKPOINT
    assert args.model_y_checkpoint == module.DEFAULT_MODEL_Y_CHECKPOINT
    assert args.max_steps == 2_000
    assert args.batch_size == 32
    assert args.context_length == 256
    assert args.san_layers == 16
    assert args.query_heads == 8
    assert args.kv_heads == 4
    assert args.retrieval_distances == (32, 64, 128, 192)
    assert args.retrieval_pairs_per_distance == 32
    assert args.max_steps * args.batch_size * args.context_length == 16_384_000


def test_reduced_san_smoke_trains_only_san_and_writes_all_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, data, suite = _prepare_test_inputs(tmp_path)
    model_x, model_y = _save_baselines(tmp_path, data, context_length=16)
    output_dir = tmp_path / "benchmark"
    module = _load_runner(monkeypatch)
    _stub_retrieval(module, monkeypatch)

    exit_code = module.main(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--suite",
            str(suite),
            "--model-x-checkpoint",
            str(model_x),
            "--model-y-checkpoint",
            str(model_y),
            "--device",
            "cpu",
            "--expected-data-fingerprint",
            data.fingerprint,
            "--max-steps",
            "1",
            "--batch-size",
            "2",
            "--context-length",
            "16",
            "--d-model",
            "8",
            "--san-layers",
            "1",
            "--query-heads",
            "1",
            "--kv-heads",
            "1",
            "--dropout",
            "0",
            "--warmup-steps",
            "0",
            "--eval-interval",
            "1",
            "--eval-batches",
            "1",
            "--post-eval-batches",
            "1",
            "--checkpoint-interval",
            "1",
            "--log-interval",
            "0",
            "--sample-tokens",
            "0",
            "--generation-tokens",
            "1",
            "--generation-repeats",
            "1",
            "--retrieval-distances",
            "4",
            "--retrieval-pairs-per-distance",
            "1",
            "--retrieval-batch-size",
            "1",
        ]
    )
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert summary["models"]["kiwilm_san"]["training"]["step"] == 1
    assert "training" not in summary["models"]["model_x"]
    assert "training" not in summary["models"]["model_y"]
    assert summary["settings"]["training_targets"] == 32
    assert summary["training_comparison"]["valid"] is False
    assert summary["comparison"]["generation_count"] == 3
    assert summary["retrieval"]["model_count"] == 3
    assert (output_dir / "kiwilm-san" / "best.pt").is_file()
    assert (output_dir / "comparison" / "report.md").is_file()
    assert (output_dir / "comparison" / "results.jsonl").is_file()
    assert (output_dir / "retrieval" / "report.md").is_file()
    assert (output_dir / "retrieval" / "results.jsonl").is_file()
    assert (output_dir / "retrieval" / "suite.json").is_file()


def test_baseline_checkpoint_validation_precedes_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, data, suite = _prepare_test_inputs(tmp_path)
    _, model_y = _save_baselines(tmp_path, data, context_length=16)
    output_dir = tmp_path / "benchmark"
    module = _load_runner(monkeypatch)

    with pytest.raises(ValueError, match="expected 'model_x'"):
        module.main(
            [
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(output_dir),
                "--suite",
                str(suite),
                "--model-x-checkpoint",
                str(model_y),
                "--model-y-checkpoint",
                str(model_y),
                "--device",
                "cpu",
                "--expected-data-fingerprint",
                data.fingerprint,
                "--context-length",
                "16",
            ]
        )

    assert not output_dir.exists()
