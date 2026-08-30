#!/usr/bin/env python3
"""Build the normalized health/quality summary for gated Slim v3 smoke runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from kiwilm.config import KiwiLM2SlimV3Config, ModelConfig
from kiwilm.data import PreparedTokenData
from kiwilm.diagnostics import (
    aggregate_health_reports,
    cached_generation_parity_report,
    model_health_report,
)
from kiwilm.inference import load_trained_model
from kiwilm.models import KiwiLM2LM, SwiGLU
from kiwilm.training import choose_device, evaluate

ROLES = ("control", "gate_025", "gate_050")
EXPECTED_GATES = {"control": None, "gate_025": 0.25, "gate_050": 0.5}
EXPECTED_TOKENS = 50_000_000


def _checkpoint_record(path: Path, *, fingerprint: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path} does not contain a mapping")
    if payload.get("data_fingerprint") != fingerprint:
        raise ValueError(f"checkpoint {path} has a different data fingerprint")
    state = payload.get("training_state")
    if not isinstance(state, dict) or state.get("tokens_seen") != EXPECTED_TOKENS:
        raise ValueError(f"checkpoint {path} must contain exactly 50M tokens")
    config_raw = payload.get("model_config")
    train_config = payload.get("train_config")
    if not isinstance(config_raw, dict) or not isinstance(train_config, dict):
        raise ValueError(f"checkpoint {path} lacks frozen configuration metadata")
    config = ModelConfig.from_dict(config_raw)
    return {
        "checkpoint": str(path),
        "config": config,
        "train_config": train_config,
        "step": payload.get("step"),
        "tokens_seen": EXPECTED_TOKENS,
    }


def _runtime_metrics(path: Path) -> dict[str, float | int | None]:
    metrics_path = path.parent / "metrics.jsonl"
    rows = (
        [json.loads(line) for line in metrics_path.read_text().splitlines()]
        if metrics_path.is_file()
        else []
    )
    train_rows = [row for row in rows if row.get("event") == "train"]
    diagnostic_rows = [
        row for row in rows if row.get("event") == "validation_diagnostic"
    ]
    steady = train_rows[max(1, len(train_rows) // 10) :] if len(train_rows) > 1 else train_rows
    throughput = [
        float(row["model_tokens_per_second"])
        for row in steady
        if isinstance(row.get("model_tokens_per_second"), (int, float))
        and row["model_tokens_per_second"] > 0
    ]
    memory = [
        int(row["accelerator_memory_bytes"])
        for row in train_rows
        if isinstance(row.get("accelerator_memory_bytes"), int)
    ]
    peak_memory = max(memory) if memory else None
    summary_path = path.parent / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        candidates = (
            summary.get("peak_cuda_memory_bytes"),
            summary.get("peak_accelerator_memory_bytes"),
            summary.get("training", {}).get("peak_accelerator_memory_bytes")
            if isinstance(summary.get("training"), dict)
            else None,
            summary.get("summary", {}).get("peak_accelerator_memory_bytes")
            if isinstance(summary.get("summary"), dict)
            else None,
        )
        peak_memory = next(
            (
                int(value)
                for value in candidates
                if isinstance(value, (int, float)) and value > 0
            ),
            peak_memory,
        )
    return {
        "tokens_per_second": statistics.median(throughput) if throughput else None,
        "peak_memory_bytes": peak_memory,
        "alpha_trajectory": [
            {
                "step": row.get("step"),
                "tokens_seen": row.get("tokens_seen"),
                "alphas": [
                    block["mlp_residual_scale"]
                    for block in row.get("blocks", [])
                    if block.get("mlp_type") == "swiglu"
                    and block.get("mlp_residual_scale") is not None
                ],
            }
            for row in diagnostic_rows
        ],
    }


def _dtype(precision: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]


def _evaluate_role(
    path: Path,
    *,
    data_dir: Path,
    fingerprint: str,
    device: torch.device,
    precision: str,
    validation_batches: int,
) -> dict[str, Any]:
    model, config = load_trained_model(
        path, data_fingerprint=fingerprint, device=device
    )
    if not isinstance(model, KiwiLM2LM) or not isinstance(
        config, KiwiLM2SlimV3Config
    ):
        raise ValueError("all residual-gate smoke checkpoints must be Slim v3")
    model.to(dtype=_dtype(precision))
    fixed_data = PreparedTokenData(data_dir, seed=143)
    fixed = evaluate(
        model,
        fixed_data,
        batch_size=2,
        context_length=512,
        num_batches=validation_batches,
        device=device,
        generator=torch.Generator(device="cpu").manual_seed(143),
        precision=precision,
    )
    reports = []
    parity_inputs = None
    for seed in (141, 142):
        data = PreparedTokenData(data_dir, seed=seed, expected_fingerprint=fingerprint)
        for _ in range(50):
            inputs, targets = data.get_batch(
                "validation", batch_size=2, context_length=512, device=device
            )
            if parity_inputs is None:
                parity_inputs = inputs
            reports.append(model_health_report(model, inputs, targets))
    assert parity_inputs is not None
    alphas = [
        float(block.mlp.effective_residual_scale.detach())
        for block in model.blocks
        if isinstance(block.mlp, SwiGLU)
        and block.mlp.effective_residual_scale is not None
    ]
    result = {
        "model_config": config.to_dict(),
        "validation_loss": fixed["validation_loss"],
        "validation_perplexity": fixed["perplexity"],
        "health": aggregate_health_reports(reports),
        "cached_generation": cached_generation_parity_report(model, parity_inputs),
        "effective_alphas": alphas,
        **_runtime_metrics(path),
    }
    del model
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--gate-025", type=Path, required=True)
    parser.add_argument("--gate-050", type=Path, required=True)
    parser.add_argument("--generation-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="fp32"
    )
    parser.add_argument("--validation-batches", type=int, default=200)
    args = parser.parse_args()
    if args.validation_batches < 1:
        raise ValueError("validation_batches must be positive")
    source = PreparedTokenData(args.data_dir)
    paths = {
        "control": args.control,
        "gate_025": args.gate_025,
        "gate_050": args.gate_050,
    }
    provenance = {
        role: _checkpoint_record(path, fingerprint=source.fingerprint)
        for role, path in paths.items()
    }
    reference_train = provenance["control"]["train_config"]
    reference_config = provenance["control"]["config"].to_dict()
    reference_config.pop("swiglu_residual_gate_init", None)
    for role, record in provenance.items():
        config = record["config"]
        if (
            not isinstance(config, KiwiLM2SlimV3Config)
            or config.upper_swiglu_blocks != 4
            or config.swiglu_residual_gate_init != EXPECTED_GATES[role]
        ):
            raise ValueError(f"{role} has the wrong H6/S4 residual-gate config")
        comparable = config.to_dict()
        comparable.pop("swiglu_residual_gate_init", None)
        if comparable != reference_config:
            raise ValueError(f"{role} does not share the frozen H6/S4 backbone")
        if record["train_config"] != reference_train:
            raise ValueError(f"{role} does not share the frozen smoke controls")

    device = choose_device(args.device)
    models = {
        role: _evaluate_role(
            path,
            data_dir=args.data_dir,
            fingerprint=source.fingerprint,
            device=device,
            precision=args.precision,
            validation_batches=args.validation_batches,
        )
        for role, path in paths.items()
    }
    if args.generation_summary is not None:
        generation_summary = json.loads(args.generation_summary.read_text())
        if (
            generation_summary.get("data_fingerprint") != source.fingerprint
            or generation_summary.get("sampling_seeds") != [42, 43, 44, 45, 46]
            or generation_summary.get("prompt_count") != 6
            or generation_summary.get("sampling_profile_count") != 5
            or generation_summary.get("generation_count") != 90
        ):
            raise ValueError(
                "generation summary does not contain the frozen 3-model seeds 42-46 suite"
            )
        generation = generation_summary["generation"]
        if set(generation) != set(ROLES):
            raise ValueError("generation summary labels must match the three smoke roles")
        for role in ROLES:
            models[role]["generation"] = generation.get(role)
    result = {
        "schema_version": 1,
        "data_fingerprint": source.fingerprint,
        "audit": {
            "seeds": [141, 142],
            "batches_per_seed": 50,
            "total_batches": 100,
            "batch_size": 2,
            "context_length": 512,
            "precision": args.precision,
            "device": str(device),
            "fixed_validation_batches": args.validation_batches,
            "fixed_validation_seed": 143,
        },
        "provenance": {
            role: {
                key: value
                for key, value in record.items()
                if key not in {"config", "train_config"}
            }
            for role, record in provenance.items()
        },
        "control": models["control"],
        "candidates": {
            "gate_025": models["gate_025"],
            "gate_050": models["gate_050"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
