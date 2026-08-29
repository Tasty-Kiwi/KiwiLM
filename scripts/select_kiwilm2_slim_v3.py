#!/usr/bin/env python3
"""Apply the frozen H7/S3 versus H6/S4 promotion rule to a comparison summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kiwilm.slim_v3 import select_slim_v3_candidate

DEFAULT_LABELS = {
    "dense": "KiwiLM 2 Dense",
    "h7s3": "KiwiLM 2 Slim v3 H7/S3",
    "h6s4": "KiwiLM 2 Slim v3 H6/S4",
}


def _model_value(summary: dict[str, Any], section: str, label: str, key: str) -> Any:
    try:
        return summary[section]["models"][label][key]
    except KeyError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dense-label", default=DEFAULT_LABELS["dense"])
    parser.add_argument("--h7s3-label", default=DEFAULT_LABELS["h7s3"])
    parser.add_argument("--h6s4-label", default=DEFAULT_LABELS["h6s4"])
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = select_slim_v3_candidate(
        dense_tokens_per_second=_model_value(
            summary, "training_logs", args.dense_label, "median_tokens_per_second"
        ),
        h7s3_validation_loss=_model_value(
            summary, "fixed_validation", args.h7s3_label, "validation_loss"
        ),
        h6s4_validation_loss=_model_value(
            summary, "fixed_validation", args.h6s4_label, "validation_loss"
        ),
        h6s4_tokens_per_second=_model_value(
            summary, "training_logs", args.h6s4_label, "median_tokens_per_second"
        ),
        h7s3_health_passed=bool(
            _model_value(summary, "health_probe", args.h7s3_label, "health_passed")
        ),
        h6s4_health_passed=bool(
            _model_value(summary, "health_probe", args.h6s4_label, "health_passed")
        ),
        h7s3_parity_passed=bool(
            _model_value(
                summary, "health_probe", args.h7s3_label, "cached_generation_parity"
            )
        ),
        h6s4_parity_passed=bool(
            _model_value(
                summary, "health_probe", args.h6s4_label, "cached_generation_parity"
            )
        ),
    )
    if result["selected"] is not None:
        candidate = f"slim-v3-{result['selected']}"
        upper = 3 if result["selected"] == "h7s3" else 4
        result["promotion_commands"] = {
            "windows_powershell": (
                "uv run --locked python scripts\\run_kiwilm2_experiment.py `\n"
                "  --phase architecture `\n"
                '  --data-dir "data\\smollm-architecture" `\n'
                '  --output-dir "runs\\kiwilm2-slim-v3-architecture" `\n'
                f"  --candidates {candidate} `\n"
                "  --device cuda `\n"
                "  --precision bf16 `\n"
                "  --batch-size 8 `\n"
                "  --grad-accum-steps 4 `\n"
                "  --resume-existing"
            ),
            "colab_bash": (
                "KIWILM2_PHASE=architecture \\\n"
                "KIWILM2_VARIANT=kiwilm2_slim_v3 \\\n"
                f"KIWILM2_UPPER_SWIGLU_BLOCKS={upper} \\\n"
                "COLAB_GPU=L4 \\\n"
                "KIWILM2_PRECISION=bf16 \\\n"
                "scripts/run_colab_kiwilm2.sh"
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["selected"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
