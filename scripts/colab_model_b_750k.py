"""Matched-budget Model B training workload executed non-interactively on Colab."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

EXPECTED_DATA_FINGERPRINT = (
    "6b2687870c402c5e70e677e8a6c88bb854786c8dcb963f9c734feb022862ed82"
)
EXPECTED_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
EXPECTED_SOURCE_FINGERPRINT = (
    "d2f500e2a85cf7c1a1c1b292b2f186c04782e9443312aaea5f1dc08a561dc764"
)
EXPECTED_TOKENIZER_SHA256 = (
    "0127391ca334542dd206b0bef735b571d3739e5a399e89bbe0b42e79a09d9226"
)
EXPECTED_PARAMETER_COUNT = 5_261_056
DATA_DIR = Path("/content/kiwilm-model-b-750k/data")
RUN_DIR = Path("/content/kiwilm-model-b-750k/run")
SUITE = Path("/content/story-consistency-prompts.json")
EXAMPLES = Path("/content/kiwilm-model-b-750k-examples.md")
SUMMARY = Path("/content/kiwilm-model-b-750k-summary.json")


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


def validate_data(data: Any) -> None:
    metadata = data.metadata
    tokenizer = metadata["tokenizer"]
    reused_from = tokenizer.get("reused_from")
    if metadata["dataset"].get("resolved_revision") != EXPECTED_REVISION:
        raise RuntimeError("prepared data uses the wrong TinyStories revision")
    if metadata["splits"]["train"].get("stories") != 750_000:
        raise RuntimeError("prepared data must contain 750,000 training stories")
    if metadata["splits"]["validation"].get("stories") != 10_000:
        raise RuntimeError("prepared data must contain 10,000 validation stories")
    if tokenizer.get("sha256") != EXPECTED_TOKENIZER_SHA256:
        raise RuntimeError("prepared data does not use the frozen 550k tokenizer")
    if reused_from != {
        "dataset_fingerprint": EXPECTED_SOURCE_FINGERPRINT,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
    }:
        raise RuntimeError("prepared data has invalid frozen-tokenizer provenance")


install_package()
reassemble_uploads()
stage_prepared_data()

import torch  # noqa: E402

from kiwilm.cli import main as kiwilm_main  # noqa: E402
from kiwilm.config import CNNAttentionConfig  # noqa: E402
from kiwilm.data import PreparedTokenData  # noqa: E402
from kiwilm.example_report import generate_example_report  # noqa: E402
from kiwilm.models import build_model  # noqa: E402

data = PreparedTokenData(
    DATA_DIR,
    expected_fingerprint=EXPECTED_DATA_FINGERPRINT,
)
validate_data(data)
model_config = CNNAttentionConfig(vocab_size=data.tokenizer.vocab_size)
parameter_count = sum(
    parameter.numel() for parameter in build_model(model_config).parameters()
)
if parameter_count != EXPECTED_PARAMETER_COUNT:
    raise RuntimeError(
        f"expected {EXPECTED_PARAMETER_COUNT} Model B parameters, "
        f"found {parameter_count}"
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
        "160465920",
        "--warmup-tokens",
        "8023296",
        "--max-steps",
        "17000",
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

report_summary = generate_example_report(
    RUN_DIR / "best.pt",
    data=data,
    suite_path=SUITE,
    output_path=EXAMPLES,
    device=torch.device("cuda"),
    title=(
        "Model B Matched Model F Data "
        "(trained on 750k stories, 30.5:1 ratio) Examples"
    ),
)
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
            "parameter_count": parameter_count,
            "step": final_train["step"],
            "tokens_seen": final_train["tokens_seen"],
            "tokens_per_parameter": final_train["tokens_seen"] / parameter_count,
            "valid_tokens_per_second": final_train["valid_tokens_per_second"],
            "model_tokens_per_second": final_train["model_tokens_per_second"],
            "padding_fraction": final_train["padding_fraction"],
            "best_validation_loss": best_metrics.get("best_validation_loss"),
            "best_validation_perplexity": best_metrics.get(
                "best_validation_perplexity"
            ),
            "generated": generated,
            "example_report": report_summary,
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
