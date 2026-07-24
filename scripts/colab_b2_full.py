"""Full Model B2 training workload executed non-interactively on Colab."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

EXPECTED_DATA_FINGERPRINT = (
    "80e38a5ec88a3d4a543aaadfd6cfcbb8e1fc41e1987c382e9d2f8f7db9d843ad"
)
DATA_DIR = Path("/content/kiwilm-b2-full/data")
RUN_DIR = Path("/content/kiwilm-b2-full/run")
SUMMARY = Path("/content/kiwilm-b2-full-summary.json")


def run(*command: str, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )


def install_package() -> None:
    wheels = sorted(Path("/content").glob("kiwilm-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one KiwiLM wheel in /content, found {wheels}")
    run("python", "-m", "pip", "install", str(wheels[0]))


def reassemble_uploads() -> None:
    grouped: dict[str, list[Path]] = {}
    for part in sorted(Path("/content").glob("*.part-*")):
        target_name, separator, _suffix = part.name.rpartition(".part-")
        if not separator or not target_name:
            raise RuntimeError(f"invalid upload chunk name: {part.name}")
        grouped.setdefault(target_name, []).append(part)
    if not grouped:
        raise RuntimeError("no prepared-data upload chunks found in /content")
    for target_name, parts in grouped.items():
        destination = Path("/content") / target_name
        with destination.open("wb") as output:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)


def stage_prepared_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    patterns = (
        "metadata.json",
        "tokenizer-*.json",
        "train-*.bin",
        "validation-*.bin",
        "*-story-offsets-*.npy",
    )
    staged: list[Path] = []
    for pattern in patterns:
        staged.extend(sorted(Path("/content").glob(pattern)))
    for source in staged:
        shutil.move(str(source), DATA_DIR / source.name)


install_package()
reassemble_uploads()
stage_prepared_data()

import torch  # noqa: E402

from kiwilm.cli import main as kiwilm_main  # noqa: E402
from kiwilm.data import PreparedTokenData  # noqa: E402

data = PreparedTokenData(
    DATA_DIR,
    expected_fingerprint=EXPECTED_DATA_FINGERPRINT,
)
gpu = run(
    "nvidia-smi",
    "--query-gpu=name,memory.total,driver_version",
    "--format=csv,noheader",
    capture=True,
).stdout.strip()
torch_info = (
    f"{torch.__version__} {torch.cuda.get_device_name(0)} "
    f"cuda={torch.cuda.is_available()}"
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
        "105221120",
        "--warmup-tokens",
        "5261056",
        "--max-steps",
        "12000",
        "--batch-size",
        "64",
        "--eval-mode",
        "both",
        "--eval-interval",
        "500",
        "--eval-batches",
        "50",
        "--checkpoint-interval",
        "500",
        "--log-interval",
        "10",
        "--sample-tokens",
        "64",
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
    "160",
    "--temperature",
    "0.8",
    "--top-k",
    "40",
    "--seed",
    "42",
    "--cache",
    "auto",
    capture=True,
).stdout.strip()

metrics = [
    json.loads(line)
    for line in (RUN_DIR / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
]
final_train = next(row for row in reversed(metrics) if row["event"] == "train")
best_checkpoint = torch.load(
    RUN_DIR / "best.pt",
    map_location="cpu",
    weights_only=True,
)
best_metrics = best_checkpoint.get("metrics") or {}
SUMMARY.write_text(
    json.dumps(
        {
            "data_fingerprint": data.fingerprint,
            "gpu": gpu,
            "torch": torch_info,
            "elapsed_seconds": elapsed,
            "peak_cuda_memory_bytes": peak_memory,
            "step": final_train["step"],
            "tokens_seen": final_train["tokens_seen"],
            "valid_tokens_per_second": final_train["valid_tokens_per_second"],
            "model_tokens_per_second": final_train["model_tokens_per_second"],
            "padding_fraction": final_train["padding_fraction"],
            "best_validation_loss": best_metrics.get("best_validation_loss"),
            "best_validation_perplexity": best_metrics.get(
                "best_validation_perplexity"
            ),
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
