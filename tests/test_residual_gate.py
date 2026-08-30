"""Residual-gate smoke promotion-rule coverage."""

from __future__ import annotations

import json
import runpy
from copy import deepcopy
from pathlib import Path

import pytest

from kiwilm.config import KiwiLM2Config, KiwiLM2SlimV3Config
from kiwilm.residual_gate import (
    select_residual_gate_candidate,
    validate_residual_audit_authorization,
)


def _health() -> dict[str, object]:
    blocks = [
        {
            "index": index,
            "metrics": {
                "residual_amplification": {
                    "p90": 1.2,
                    "maximum": 1.4,
                }
            },
        }
        for index in range(10)
    ]
    return {
        "batch_count": 100,
        "health_pass_count": 100,
        "all_finite": True,
        "all_nonzero_gradients": True,
        "minimum_mlp_family_gradient_ratios": {
            "hadamard": 0.2,
            "swiglu": 0.3,
        },
        "blocks": blocks,
    }


def _candidate(loss: float) -> dict[str, object]:
    return {
        "validation_loss": loss,
        "tokens_per_second": 19_500,
        "peak_memory_bytes": 1_010,
        "health": _health(),
        "effective_alphas": [0.3, 0.4, 0.5, 0.6],
        "alpha_trajectory": [
            {"step": 500, "tokens_seen": 8_192_000, "alphas": [0.3, 0.4, 0.5, 0.6]}
        ],
        "cached_generation": {
            "direct_passed": True,
            "rollover_passed": True,
        },
        "generation": {
            "maximum_consecutive_word_run": 3,
            "repeated_four_gram_rate": 0.08,
        },
    }


def _control() -> dict[str, object]:
    return {
        "validation_loss": 4.5,
        "tokens_per_second": 20_000,
        "peak_memory_bytes": 1_000,
        "generation": {"repeated_four_gram_rate": 0.05},
    }


def test_residual_gate_selects_lower_loss_and_prefers_half_on_tie() -> None:
    result = select_residual_gate_candidate(
        control=_control(),
        candidates={
            "gate_025": _candidate(4.51),
            "gate_050": _candidate(4.53),
        },
    )
    assert result["selected"] == "gate_025"

    tied = select_residual_gate_candidate(
        control=_control(),
        candidates={
            "gate_025": _candidate(4.510),
            "gate_050": _candidate(4.519),
        },
    )
    assert tied["selected"] == "gate_050"


def test_residual_gate_records_no_winner_when_any_required_gate_fails() -> None:
    gate025 = _candidate(4.51)
    gate050 = deepcopy(gate025)
    gate025["tokens_per_second"] = 18_000
    gate050["health"]["blocks"][9]["metrics"]["residual_amplification"] = {
        "p90": 1.51,
        "maximum": 1.6,
    }
    result = select_residual_gate_candidate(
        control=_control(),
        candidates={"gate_025": gate025, "gate_050": gate050},
    )
    assert result["selected"] is None
    assert any("throughput" in reason for reason in result["candidates"]["gate_025"]["reasons"])
    assert any("p90" in reason for reason in result["candidates"]["gate_050"]["reasons"])


def test_residual_audit_authorization_requires_the_exact_frozen_controls(
    tmp_path: Path,
) -> None:
    audit = {
        "schema_version": 1,
        "data_fingerprint": "a" * 64,
        "authorized_smoke_data": {
            "fingerprint": "b" * 64,
            "train_tokens": 50_000_000,
        },
        "audit": {
            "seeds": [141, 142],
            "batches_per_seed": 50,
            "total_batches": 100,
            "batch_size": 2,
            "context_length": 512,
            "residual_threshold": 1.5,
            "minimum_failures": 10,
        },
        "residual_growth_reproduced": True,
        "gated_smoke_authorized": True,
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit), encoding="utf-8")
    assert validate_residual_audit_authorization(
        path, fingerprint="b" * 64
    ) == audit
    with pytest.raises(ValueError, match="different 50M smoke dataset"):
        validate_residual_audit_authorization(path, fingerprint="c" * 64)
    audit["audit"]["batches_per_seed"] = 49
    path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen 100-batch"):
        validate_residual_audit_authorization(path, fingerprint="b" * 64)


def test_experiment_candidates_drop_h7_and_isolate_both_gate_initializers() -> None:
    namespace = runpy.run_path(
        Path(__file__).parents[1] / "scripts" / "run_kiwilm2_experiment.py"
    )
    candidates = namespace["CANDIDATES"]
    assert "slim-v3-h7s3" not in candidates
    assert "slim-v3-h6s4-gate-025" in candidates
    assert "slim-v3-h6s4-gate-050" in candidates
    uses_residual_gate = namespace["_uses_residual_gate"]
    assert not uses_residual_gate(KiwiLM2Config(vocab_size=256))
    assert not uses_residual_gate(
        KiwiLM2SlimV3Config(vocab_size=256, upper_swiglu_blocks=4)
    )
    assert uses_residual_gate(
        KiwiLM2SlimV3Config(
            vocab_size=256,
            upper_swiglu_blocks=4,
            swiglu_residual_gate_init=0.5,
        )
    )
