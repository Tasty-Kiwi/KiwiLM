"""Full-split Model B training workload executed non-interactively on Colab."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

EXPECTED_DATASET = "roneneldan/TinyStories"
EXPECTED_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
EXPECTED_SOURCE_FINGERPRINT = "d2f500e2a85cf7c1a1c1b292b2f186c04782e9443312aaea5f1dc08a561dc764"
EXPECTED_TOKENIZER_SHA256 = "0127391ca334542dd206b0bef735b571d3739e5a399e89bbe0b42e79a09d9226"
EXPECTED_PARAMETER_COUNT = 5_261_056
EXPECTED_VALIDATION_STORIES = 10_000
BATCH_SIZE = 64
DATA_DIR = Path("/content/kiwilm-model-b-2m/data")
RUN_DIR = Path("/content/kiwilm-model-b-2m/run")
BUNDLE_DIR = Path("/content/kiwilm-model-b-2m/tokenizer-bundle")
SUITE = Path("/content/story-consistency-prompts.json")
EXAMPLES = Path("/content/kiwilm-model-b-2m-examples.md")
SUMMARY = Path("/content/kiwilm-model-b-2m-summary.json")


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


def stage_tokenizer_bundle() -> None:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = Path("/content/tokenizer-bundle.json")
    tokenizers = sorted(
        path for path in Path("/content").glob("tokenizer-*.json") if path.name != manifest.name
    )
    if not manifest.is_file() or len(tokenizers) != 1:
        raise RuntimeError("expected one tokenizer bundle manifest and one tokenizer artifact")
    shutil.move(str(manifest), BUNDLE_DIR / manifest.name)
    shutil.move(str(tokenizers[0]), BUNDLE_DIR / tokenizers[0].name)


def validate_data(data: Any) -> None:
    metadata = data.metadata
    dataset = metadata["dataset"]
    config = metadata["config"]
    tokenizer = metadata["tokenizer"]
    train = metadata["splits"]["train"]
    validation = metadata["splits"]["validation"]
    if dataset.get("name") != EXPECTED_DATASET:
        raise RuntimeError("prepared data uses the wrong dataset")
    if dataset.get("resolved_revision") != EXPECTED_REVISION:
        raise RuntimeError("prepared data uses the wrong TinyStories revision")
    if config.get("train_limit") != 0:
        raise RuntimeError("prepared data is not the complete training split")
    if config.get("validation_limit") != EXPECTED_VALIDATION_STORIES:
        raise RuntimeError("prepared data uses the wrong validation limit")
    if train["stories"] < 2_000_000:
        raise RuntimeError("prepared data contains fewer than two million stories")
    if validation["stories"] != EXPECTED_VALIDATION_STORIES:
        raise RuntimeError("prepared data uses the wrong validation split size")
    if tokenizer.get("sha256") != EXPECTED_TOKENIZER_SHA256:
        raise RuntimeError("prepared data does not use the frozen 550k tokenizer")
    if tokenizer.get("reused_from") != {
        "dataset_fingerprint": EXPECTED_SOURCE_FINGERPRINT,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
    }:
        raise RuntimeError("prepared data has invalid frozen-tokenizer provenance")


install_package()
stage_tokenizer_bundle()

import torch  # noqa: E402

from kiwilm.cli import main as kiwilm_main  # noqa: E402
from kiwilm.config import CNNAttentionConfig  # noqa: E402
from kiwilm.data import PreparedTokenData  # noqa: E402
from kiwilm.example_report import generate_example_report  # noqa: E402
from kiwilm.models import build_model  # noqa: E402

prepare_started = time.perf_counter()
exit_code = kiwilm_main(
    [
        "prepare",
        "--output-dir",
        str(DATA_DIR),
        "--revision",
        EXPECTED_REVISION,
        "--train-limit",
        "0",
        "--validation-limit",
        str(EXPECTED_VALIDATION_STORIES),
        "--tokenizer-from",
        str(BUNDLE_DIR),
    ]
)
if exit_code:
    raise RuntimeError(f"KiwiLM preparation exited with status {exit_code}")
prepare_elapsed = time.perf_counter() - prepare_started

data = PreparedTokenData(DATA_DIR)
validate_data(data)
model_config = CNNAttentionConfig(vocab_size=data.tokenizer.vocab_size)
parameter_count = sum(parameter.numel() for parameter in build_model(model_config).parameters())
if parameter_count != EXPECTED_PARAMETER_COUNT:
    raise RuntimeError(
        f"expected {EXPECTED_PARAMETER_COUNT} Model B parameters, found {parameter_count}"
    )

train_metadata = data.metadata["splits"]["train"]
max_tokens = int(train_metadata["tokens"] - train_metadata["stories"])
warmup_tokens = max_tokens // 20
story_chunks = len(data.story_chunks("train", model_config.context_length))
expected_steps = math.ceil(story_chunks / BATCH_SIZE)
max_steps = expected_steps + 1
print(
    json.dumps(
        {
            "event": "full_corpus_plan",
            "data_fingerprint": data.fingerprint,
            "prepare_elapsed_seconds": prepare_elapsed,
            "train_stories": train_metadata["stories"],
            "max_tokens": max_tokens,
            "warmup_tokens": warmup_tokens,
            "story_chunks": story_chunks,
            "expected_steps": expected_steps,
            "max_steps": max_steps,
            "tokens_per_parameter": max_tokens / parameter_count,
        },
        sort_keys=True,
    )
)

gpu = run(
    "nvidia-smi",
    "--query-gpu=name,memory.total,driver_version",
    "--format=csv,noheader",
    capture=True,
).stdout.strip()
torch_info = f"{torch.__version__} {torch.cuda.get_device_name(0)} cuda={torch.cuda.is_available()}"

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
        str(max_tokens),
        "--warmup-tokens",
        str(warmup_tokens),
        "--max-steps",
        str(max_steps),
        "--batch-size",
        str(BATCH_SIZE),
        "--eval-mode",
        "both",
        "--eval-interval",
        "1000",
        "--eval-batches",
        "50",
        "--checkpoint-interval",
        "1000",
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
        "Model B Full TinyStories "
        f"({train_metadata['stories']:,} stories, one complete epoch) Examples"
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
            "prepare_elapsed_seconds": prepare_elapsed,
            "elapsed_seconds": elapsed,
            "peak_cuda_memory_bytes": peak_memory,
            "parameter_count": parameter_count,
            "train_stories": train_metadata["stories"],
            "train_targets": max_tokens,
            "warmup_tokens": warmup_tokens,
            "expected_steps": expected_steps,
            "step": final_train["step"],
            "tokens_seen": final_train["tokens_seen"],
            "tokens_per_parameter": final_train["tokens_seen"] / parameter_count,
            "valid_tokens_per_second": final_train["valid_tokens_per_second"],
            "model_tokens_per_second": final_train["model_tokens_per_second"],
            "padding_fraction": final_train["padding_fraction"],
            "best_validation_loss": best_metrics.get("best_validation_loss"),
            "best_validation_perplexity": best_metrics.get("best_validation_perplexity"),
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
