"""Fresh 5M-token hybrid-12 L4/BF16 LR-scale diagnostic. Default: plan, no cloud actions."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from kiwilm.v3.accelerator import AcceleratorTrainConfig
from kiwilm.v3.colab_cli import main as colab_main

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ROOT / "configs/kiwilm3-lr-diagnostic-5m.json"
STATE = "runs/colab/v3-gpu-hybrid-12-lr3e4-diagnostic-5m"
DATA_FINGERPRINT = "66b9899b879a5aba9eabdd4a40a54ab9ede62fdd1070f43be9b4c5b0e0e9714b"
TOKENIZER_SHA256 = "7fc721c3c0dfc77ca5786756b5821d292fad399f62298b46638fc83c8f680a8a"
DATA_CACHE = f"/content/drive/MyDrive/KiwiLM2/data/tpu-smoke-{DATA_FINGERPRINT}"


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv) or ["plan"]
    action, *options = arguments
    if action == "plan":
        config = AcceleratorTrainConfig(**json.loads(CONTROLS.read_text()))
        print(
            json.dumps(
                {
                    "training_started": False,
                    "cloud_actions": False,
                    "candidate": "hybrid-12",
                    "config": config.to_dict(),
                    "collapse_diagnostics": True,
                    "state_dir": STATE,
                    "data_cache": DATA_CACHE,
                    "command": "bash scripts/run_colab_kiwilm3_lr_diagnostic.sh run",
                    "note": "Fresh initialization; original 50M controls/checkpoint unchanged. "
                    "run explicitly allocates L4, authorizes Drive, trains, collects and stops. "
                    "This short cosine schedule is a diagnostic, not a peak-only LR ablation.",
                },
                indent=2,
            )
        )
        return 0
    defaults = ["--state-dir", STATE]
    if action in {"setup", "run"}:
        locked = {
            "--profile",
            "--candidate",
            "--precision",
            "--controls",
            "--wheel",
            "--stop-after-step",
            "--tpu",
            "--expected-data-fingerprint",
            "--expected-tokenizer-sha256",
        }
        if any(option.split("=", 1)[0] in locked for option in options):
            raise ValueError(
                "diagnostic controls are locked; use the generic GPU launcher for other experiments"
            )
        defaults += [
            "--session",
            f"kiwilm3-lr3e4-5m-{uuid.uuid4().hex[:8]}",
            "--profile",
            "smoke",
            "--candidate",
            "hybrid-12",
            "--precision",
            "bf16",
            "--controls",
            str(CONTROLS),
            "--run-name",
            "lr3e4-diagnostic-5m",
            "--collapse-diagnostics",
            "--data-cache",
            DATA_CACHE,
            "--expected-data-fingerprint",
            DATA_FINGERPRINT,
            "--expected-tokenizer-sha256",
            TOKENIZER_SHA256,
        ]
    return colab_main([action, *defaults, *options], default_gpu="L4")


if __name__ == "__main__":
    raise SystemExit(main())
