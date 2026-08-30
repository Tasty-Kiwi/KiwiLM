#!/usr/bin/env python3
"""Apply the frozen Slim v3 residual-gate smoke promotion rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kiwilm.residual_gate import select_residual_gate_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("summary must contain a JSON object")
    result = select_residual_gate_candidate(
        control=summary.get("control", {}),
        candidates=summary.get("candidates", {}),
    )
    if result["selected"] is not None:
        gate = "0.25" if result["selected"] == "gate_025" else "0.5"
        result["confirmation_commands"] = {
            "windows_powershell": (
                "uv run --locked python scripts\\run_kiwilm2_experiment.py `\n"
                "  --phase architecture `\n"
                '  --data-dir "data\\smollm-architecture" `\n'
                '  --output-dir "runs\\kiwilm2-slim-v3-gated-architecture" `\n'
                f"  --candidates slim-v3-h6s4-gate-{'025' if gate == '0.25' else '050'} `\n"
                "  --residual-audit "
                '"examples\\comparisons\\kiwilm2-slim-v3-residual-audit\\audit.json" `\n'
                "  --device cuda --precision bf16 --batch-size 8 --grad-accum-steps 4"
            ),
            "colab_bash": (
                "KIWILM2_PHASE=architecture \\\n"
                "KIWILM2_VARIANT=kiwilm2_slim_v3 \\\n"
                "KIWILM2_UPPER_SWIGLU_BLOCKS=4 \\\n"
                f"KIWILM2_SWIGLU_RESIDUAL_GATE_INIT={gate} \\\n"
                "KIWILM2_RESIDUAL_AUDIT="
                "examples/comparisons/kiwilm2-slim-v3-residual-audit/audit.json \\\n"
                "COLAB_GPU=L4 KIWILM2_PRECISION=bf16 scripts/run_colab_kiwilm2.sh"
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
