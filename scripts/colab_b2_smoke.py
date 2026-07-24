"""Non-interactive Model B2 T4 smoke workload executed by the Colab CLI."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

DATA_DIR = Path("/content/kiwilm-b2-smoke/data")
RUN_DIR = Path("/content/kiwilm-b2-smoke/run")
SUMMARY = Path("/content/kiwilm-b2-smoke-summary.json")


def run(*command: str, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )


wheels = sorted(Path("/content").glob("kiwilm-*.whl"))
if len(wheels) != 1:
    raise RuntimeError(f"expected one KiwiLM wheel in /content, found {wheels}")
run("python", "-m", "pip", "install", str(wheels[0]))
import torch  # noqa: E402

from kiwilm.cli import main as kiwilm_main  # noqa: E402

gpu = run(
    "nvidia-smi",
    "--query-gpu=name,memory.total,driver_version",
    "--format=csv,noheader",
    capture=True,
).stdout.strip()
torch_info = run(
    "python",
    "-c",
    (
        "import torch; "
        "print(torch.__version__, torch.cuda.get_device_name(0), "
        "torch.cuda.is_available())"
    ),
    capture=True,
).stdout.strip()

run(
    "kiwilm",
    "prepare",
    "--output-dir",
    str(DATA_DIR),
    "--train-limit",
    "5000",
    "--validation-limit",
    "500",
    "--vocab-size",
    "8192",
    "--quiet",
)

started = time.perf_counter()
torch.cuda.reset_peak_memory_stats()
exit_code = kiwilm_main(
    [
        "train",
        "--architecture",
        "cnn_attention",
        "--data-dir",
        str(DATA_DIR),
        "--output-dir",
        str(RUN_DIR),
        "--device",
        "cuda",
        "--batch-mode",
        "story",
        "--precision",
        "fp16",
        "--max-tokens",
        "1000000",
        "--warmup-tokens",
        "50000",
        "--max-steps",
        "200",
        "--batch-size",
        "64",
        "--eval-mode",
        "both",
        "--eval-interval",
        "50",
        "--eval-batches",
        "10",
        "--checkpoint-interval",
        "50",
        "--log-interval",
        "10",
        "--sample-tokens",
        "16",
        "--seed",
        "42",
    ]
)
if exit_code:
    raise RuntimeError(f"KiwiLM training exited with status {exit_code}")
elapsed = time.perf_counter() - started
peak_memory = torch.cuda.max_memory_allocated()

generated = run(
    "kiwilm",
    "generate",
    "--data-dir",
    str(DATA_DIR),
    "--checkpoint",
    str(RUN_DIR / "best.pt"),
    "--prompt",
    "Once upon a time",
    "--device",
    "cuda",
    "--max-new-tokens",
    "16",
    "--temperature",
    "0",
    "--cache",
    "auto",
    capture=True,
).stdout.strip()

metrics = [
    json.loads(line)
    for line in (RUN_DIR / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
]
final_train = next(row for row in reversed(metrics) if row["event"] == "train")
SUMMARY.write_text(
    json.dumps(
        {
            "gpu": gpu,
            "torch": torch_info,
            "elapsed_seconds": elapsed,
            "peak_cuda_memory_bytes": peak_memory,
            "valid_tokens_per_second": final_train["valid_tokens_per_second"],
            "model_tokens_per_second": final_train["model_tokens_per_second"],
            "padding_fraction": final_train["padding_fraction"],
            "generated": generated,
            "latest_checkpoint_exists": (RUN_DIR / "latest.pt").is_file(),
            "best_checkpoint_exists": (RUN_DIR / "best.pt").is_file(),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
print(SUMMARY.read_text(encoding="utf-8"))
