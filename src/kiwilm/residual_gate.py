"""Promotion rules for the Slim v3 bounded-residual-gate experiment."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

GATE_LABELS = ("gate_025", "gate_050")


def validate_residual_audit_authorization(
    path: str | Path, *, fingerprint: str
) -> dict[str, Any]:
    """Validate the exact pre-smoke audit and return its portable record."""

    audit = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(audit, dict) or audit.get("schema_version") != 1:
        raise ValueError("residual audit has an unsupported schema")
    if audit.get("data_fingerprint") != fingerprint:
        raise ValueError("residual audit uses a different data fingerprint")
    settings = audit.get("audit")
    required = {
        "seeds": [141, 142],
        "batches_per_seed": 50,
        "total_batches": 100,
        "batch_size": 2,
        "context_length": 512,
        "residual_threshold": 1.5,
        "minimum_failures": 10,
    }
    if not isinstance(settings, dict) or any(
        settings.get(name) != value for name, value in required.items()
    ):
        raise ValueError("residual audit does not use the frozen 100-batch controls")
    if audit.get("residual_growth_reproduced") is not True:
        raise ValueError("residual growth was not reproduced; gated smoke is cancelled")
    if audit.get("gated_smoke_authorized") is not True:
        raise ValueError("residual audit does not authorize gated smoke training")
    return audit


def _positive_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _candidate_eligibility(
    candidate: Mapping[str, Any], control: Mapping[str, Any]
) -> dict[str, Any]:
    reasons: list[str] = []
    health = candidate.get("health")
    parity = candidate.get("cached_generation")
    generation = candidate.get("generation")
    if not isinstance(health, Mapping):
        reasons.append("missing 100-batch health aggregate")
        health = {}
    if not isinstance(parity, Mapping):
        reasons.append("missing cached-generation parity")
        parity = {}
    if not isinstance(generation, Mapping):
        reasons.append("missing generation metrics")
        generation = {}

    if health.get("batch_count") != 100:
        reasons.append("health aggregate must contain exactly 100 batches")
    if health.get("all_finite") is not True:
        reasons.append("activations or gradients are non-finite")
    if health.get("all_nonzero_gradients") is not True:
        reasons.append("one or more gradients are zero")
    if int(health.get("health_pass_count", 0)) < 95:
        reasons.append("fewer than 95 health batches passed")
    family_ratios = health.get("minimum_mlp_family_gradient_ratios")
    if not isinstance(family_ratios, Mapping) or set(family_ratios) != {
        "hadamard",
        "swiglu",
    }:
        reasons.append("both MLP-family gradient ratios are required")
    elif any(float(value) < 0.1 for value in family_ratios.values()):
        reasons.append("an MLP-family gradient ratio is below 0.1")

    blocks = health.get("blocks")
    block9 = blocks[9] if isinstance(blocks, list) and len(blocks) == 10 else None
    try:
        amplification = block9["metrics"]["residual_amplification"]
        if float(amplification["p90"]) > 1.5:
            reasons.append("block-9 residual-amplification p90 exceeds 1.5")
        if float(amplification["maximum"]) > 1.65:
            reasons.append("block-9 residual-amplification maximum exceeds 1.65")
    except (KeyError, TypeError, ValueError):
        reasons.append("block-9 residual-amplification distribution is missing")

    alphas = candidate.get("effective_alphas")
    if (
        not isinstance(alphas, list)
        or len(alphas) != 4
        or any(not _positive_finite(value) or float(value) >= 1 for value in alphas)
    ):
        reasons.append("exactly four effective alphas in (0, 1) are required")
    trajectory = candidate.get("alpha_trajectory")
    if (
        not isinstance(trajectory, list)
        or not trajectory
        or any(
            not isinstance(point, Mapping)
            or not isinstance(point.get("alphas"), list)
            or len(point["alphas"]) != 4
            or any(
                not _positive_finite(value) or float(value) >= 1
                for value in point["alphas"]
            )
            for point in trajectory
        )
    ):
        reasons.append("all recorded alpha trajectories must remain within (0, 1)")
    if parity.get("direct_passed") is not True:
        reasons.append("direct cached-generation parity failed")
    if parity.get("rollover_passed") is not True:
        reasons.append("rollover cached-generation parity failed")

    loss = candidate.get("validation_loss")
    control_loss = control.get("validation_loss")
    if not _positive_finite(loss) or not _positive_finite(control_loss):
        reasons.append("fixed validation losses are missing")
    elif float(loss) > float(control_loss) + 0.03:
        reasons.append("validation loss is more than 0.03 above ungated H6/S4")

    throughput = candidate.get("tokens_per_second")
    control_throughput = control.get("tokens_per_second")
    if not _positive_finite(throughput) or not _positive_finite(control_throughput):
        reasons.append("throughput measurements are missing")
    elif float(throughput) < 0.95 * float(control_throughput):
        reasons.append("throughput is below 95% of ungated H6/S4")

    memory = candidate.get("peak_memory_bytes")
    control_memory = control.get("peak_memory_bytes")
    if not _positive_finite(memory) or not _positive_finite(control_memory):
        reasons.append("peak-memory measurements are missing")
    elif float(memory) > 1.02 * float(control_memory):
        reasons.append("peak memory is more than 2% above ungated H6/S4")

    if int(generation.get("maximum_consecutive_word_run", 20)) >= 20:
        reasons.append("a generation contains 20 consecutive copies of one word")
    repeated = generation.get("repeated_four_gram_rate")
    control_generation = control.get("generation")
    control_repeated = (
        control_generation.get("repeated_four_gram_rate")
        if isinstance(control_generation, Mapping)
        else None
    )
    if not isinstance(repeated, (int, float)) or not isinstance(
        control_repeated, (int, float)
    ):
        reasons.append("repeated-four-gram measurements are missing")
    elif float(repeated) > float(control_repeated) + 0.05:
        reasons.append("repeated-four-gram rate exceeds the control by more than 0.05")
    return {"eligible": not reasons, "reasons": reasons}


def select_residual_gate_candidate(
    *,
    control: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen smoke promotion rules to alpha=0.25 and alpha=0.5."""

    if set(candidates) != set(GATE_LABELS):
        raise ValueError(f"candidates must contain exactly: {', '.join(GATE_LABELS)}")
    results = {
        label: _candidate_eligibility(candidate, control)
        for label, candidate in candidates.items()
    }
    eligible = [label for label in GATE_LABELS if results[label]["eligible"]]
    selected = None
    reason = "neither gated candidate passed every promotion rule"
    if eligible:
        if len(eligible) == 1:
            selected = eligible[0]
            reason = "only one gated candidate passed every promotion rule"
        else:
            loss_025 = float(candidates["gate_025"]["validation_loss"])
            loss_050 = float(candidates["gate_050"]["validation_loss"])
            if abs(loss_025 - loss_050) < 0.01:
                selected = "gate_050"
                reason = "losses differ by less than 0.01; alpha=0.5 wins the tie"
            else:
                selected = (
                    "gate_025" if loss_025 < loss_050 else "gate_050"
                )
                reason = "selected the eligible candidate with lower validation loss"
    return {
        "selected": selected,
        "reason": reason,
        "candidates": results,
        "next_stage": (
            "fresh controlled 250M confirmation" if selected is not None else None
        ),
    }


__all__ = [
    "GATE_LABELS",
    "select_residual_gate_candidate",
    "validate_residual_audit_authorization",
]
