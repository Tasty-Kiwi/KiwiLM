"""Full-split Model B training workload executed non-interactively on Colab."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tarfile
import threading
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

# Colab's UI-only secret provider blocks headless kernels when Hugging Face
# probes for HF_TOKEN. This is a public, commit-pinned dataset, so force
# anonymous HTTP and use the stable HTTP bridge instead of Xet.
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

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
RAW_DATA_DIR = Path("/content/kiwilm-model-b-2m/raw-parquet")
SUITE = Path("/content/story-consistency-prompts.json")
EXAMPLES = Path("/content/kiwilm-model-b-2m-examples.md")
SUMMARY = Path("/content/kiwilm-model-b-2m-summary.json")
ARTIFACT_MANIFEST = Path("/content/kiwilm-model-b-2m-artifacts.json")
ARTIFACT_ARCHIVE = Path("/content/kiwilm-model-b-2m-artifacts.tar")
ARTIFACT_PART_BYTES = 24 * 1024 * 1024
BACKUP_CONFIG = Path("/content/kiwilm-drive-backup.json")
DRIVE_ROOT = Path("/content/drive/MyDrive")
BACKUP_INTERVAL_SECONDS = 20
DATA_BASE_URL = (
    f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{EXPECTED_REVISION}/data"
)
PINNED_SHARDS = {
    "train": [
        "train-00000-of-00004-2d5a1467fff1081b.parquet",
        "train-00001-of-00004-5852b56a2bd28fd9.parquet",
        "train-00002-of-00004-a26307300439e943.parquet",
        "train-00003-of-00004-d243063613e5a057.parquet",
    ],
    "validation": [
        "validation-00000-of-00001-869c898b519ad725.parquet",
    ],
}
LOCAL_DATA_FILES: dict[str, list[str]] = {}


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


def load_drive_backup_directory() -> Path:
    if not BACKUP_CONFIG.is_file():
        raise RuntimeError(f"Drive backup configuration is missing: {BACKUP_CONFIG}")
    config = json.loads(BACKUP_CONFIG.read_text(encoding="utf-8"))
    configured = config.get("backup_dir")
    if not isinstance(configured, str) or not configured:
        raise RuntimeError("Drive backup configuration has no backup_dir")
    if not DRIVE_ROOT.is_dir():
        raise RuntimeError(
            "Google Drive is not mounted at /content/drive; "
            "run colab drivemount before training"
        )
    drive_root = DRIVE_ROOT.resolve()
    backup_directory = Path(configured).resolve()
    try:
        backup_directory.relative_to(drive_root)
    except ValueError as error:
        raise RuntimeError(
            f"Drive backup directory must be inside {drive_root}: {backup_directory}"
        ) from error
    return backup_directory


def download_pinned_shards() -> dict[str, list[str]]:
    free_bytes = shutil.disk_usage("/content").free
    if free_bytes < 4 * 1024**3:
        raise RuntimeError(
            "at least 4 GiB of free Colab disk is required for raw and "
            f"prepared TinyStories data; found {free_bytes / 1024**3:.1f} GiB"
        )
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    local_files: dict[str, list[str]] = {}
    total_shards = sum(len(names) for names in PINNED_SHARDS.values())
    shard_number = 0
    for split, names in PINNED_SHARDS.items():
        local_files[split] = []
        for name in names:
            shard_number += 1
            destination = RAW_DATA_DIR / name
            partial = destination.with_suffix(f"{destination.suffix}.part")
            print(
                f"[download {shard_number}/{total_shards}] {name}",
                flush=True,
            )
            run(
                "curl",
                "--fail",
                "--location",
                "--retry",
                "8",
                "--retry-all-errors",
                "--retry-delay",
                "5",
                "--connect-timeout",
                "30",
                "--speed-limit",
                "1024",
                "--speed-time",
                "60",
                "--max-time",
                "3600",
                "--continue-at",
                "-",
                "--progress-bar",
                "--output",
                str(partial),
                f"{DATA_BASE_URL}/{name}",
            )
            if partial.stat().st_size < 1024 * 1024:
                raise RuntimeError(f"downloaded shard is unexpectedly small: {name}")
            partial.replace(destination)
            local_files[split].append(str(destination))
    return local_files


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


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stage_artifacts() -> None:
    sources = {
        "best.pt": RUN_DIR / "best.pt",
        "latest.pt": RUN_DIR / "latest.pt",
        "metrics.jsonl": RUN_DIR / "metrics.jsonl",
        "summary.json": SUMMARY,
        "examples.md": EXAMPLES,
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"cannot stage missing training artifacts: {missing}")

    with tarfile.open(ARTIFACT_ARCHIVE, mode="w") as archive:
        for name, path in sources.items():
            archive.add(path, arcname=name, recursive=False)

    parts: list[dict[str, Any]] = []
    with ARTIFACT_ARCHIVE.open("rb") as source:
        part_index = 0
        while chunk := source.read(ARTIFACT_PART_BYTES):
            part_name = f"{ARTIFACT_ARCHIVE.name}.part-{part_index:04d}"
            part_path = ARTIFACT_ARCHIVE.parent / part_name
            part_path.write_bytes(chunk)
            parts.append(
                {
                    "name": part_name,
                    "bytes": len(chunk),
                    "sha256": sha256(chunk).hexdigest(),
                }
            )
            part_index += 1
    manifest = {
        "schema_version": 1,
        "archive": {
            "name": ARTIFACT_ARCHIVE.name,
            "bytes": ARTIFACT_ARCHIVE.stat().st_size,
            "sha256": file_sha256(ARTIFACT_ARCHIVE),
        },
        "files": {
            name: {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for name, path in sources.items()
        },
        "parts": parts,
    }
    ARTIFACT_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[artifacts] Staged {len(parts)} verified root-level parts "
        f"({ARTIFACT_ARCHIVE.stat().st_size / 1024**2:.1f} MiB).",
        flush=True,
    )


install_package()
stage_tokenizer_bundle()

import torch  # noqa: E402

from kiwilm.backup import VerifiedDirectoryBackup  # noqa: E402
from kiwilm.cli import main as kiwilm_main  # noqa: E402
from kiwilm.config import CNNAttentionConfig  # noqa: E402
from kiwilm.data import PreparedTokenData, prepare_tinystories  # noqa: E402
from kiwilm.example_report import generate_example_report  # noqa: E402
from kiwilm.models import build_model  # noqa: E402


class DriveBackupMonitor:
    """Periodically copy newly published checkpoints to mounted Google Drive."""

    def __init__(
        self,
        directory: Path,
        sources: dict[str, Path],
        metadata: dict[str, Any],
    ) -> None:
        self.backup = VerifiedDirectoryBackup(directory)
        self.sources = sources
        self.metadata = metadata
        self.signatures: dict[str, tuple[int, int]] = {}
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name="kiwilm-drive-backup",
            daemon=True,
        )

    def start(self) -> None:
        self.sync_changed(force=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join()

    def sync_changed(
        self,
        *,
        force: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        changed: dict[str, Path] = {}
        next_signatures = dict(self.signatures)
        for name, path in self.sources.items():
            if not path.is_file():
                continue
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if force or self.signatures.get(name) != signature:
                changed[name] = path
                next_signatures[name] = signature
        if not changed:
            return
        manifest = self.backup.sync(
            changed,
            metadata={**self.metadata, **dict(metadata or {})},
        )
        self.signatures = next_signatures
        print(
            f"[drive] Backed up {', '.join(sorted(changed))} to "
            f"{self.backup.directory} ({len(manifest['files'])} files tracked).",
            flush=True,
        )

    def add_sources(self, sources: dict[str, Path]) -> None:
        self.sources.update(sources)

    def _run(self) -> None:
        while not self.stop_event.wait(BACKUP_INTERVAL_SECONDS):
            try:
                self.sync_changed()
            except Exception as error:
                print(f"[drive] Periodic backup failed; will retry: {error}", flush=True)


def load_pinned_parquet(_dataset_name: str, **kwargs: Any) -> Any:
    import pyarrow.parquet as parquet

    split = kwargs.get("split")
    if split not in LOCAL_DATA_FILES:
        raise RuntimeError(f"unexpected TinyStories split: {split!r}")
    if kwargs.get("revision") != EXPECTED_REVISION or kwargs.get("streaming") is not True:
        raise RuntimeError("TinyStories loader lost its pinned streaming contract")

    def rows() -> Any:
        for file_name in LOCAL_DATA_FILES[split]:
            print(f"[prepare] Reading {Path(file_name).name}", flush=True)
            parquet_file = parquet.ParquetFile(file_name)
            if "text" not in parquet_file.schema_arrow.names:
                raise RuntimeError(f"parquet shard has no text column: {file_name}")
            for batch in parquet_file.iter_batches(
                batch_size=8_192,
                columns=["text"],
                use_threads=True,
            ):
                yield from batch.column(0).to_pylist()

    return rows()


download_started = time.perf_counter()
print(
    "[download] Fetching pinned TinyStories parquet shards before preparation...",
    flush=True,
)
LOCAL_DATA_FILES.update(download_pinned_shards())
download_elapsed = time.perf_counter() - download_started
print(f"[download] Complete in {download_elapsed:.1f}s.", flush=True)

prepare_started = time.perf_counter()
print(
    "[prepare] Packing the downloaded parquet shards with the frozen tokenizer...",
    flush=True,
)
prepared_metadata = prepare_tinystories(
    DATA_DIR,
    dataset_name=EXPECTED_DATASET,
    revision=EXPECTED_REVISION,
    resolved_revision=EXPECTED_REVISION,
    train_limit=0,
    validation_limit=EXPECTED_VALIDATION_STORIES,
    tokenizer_from=BUNDLE_DIR,
    load_dataset_fn=load_pinned_parquet,
)
prepare_elapsed = time.perf_counter() - prepare_started
print(
    "[prepare] Complete: "
    f"{prepared_metadata['splits']['train']['stories']:,} train stories in "
    f"{prepare_elapsed:.1f}s."
)

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
            "download_elapsed_seconds": download_elapsed,
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

drive_backup_directory = load_drive_backup_directory()
tokenizer_file = DATA_DIR / data.metadata["tokenizer"]["file"]
backup_monitor = DriveBackupMonitor(
    drive_backup_directory,
    {
        "data-metadata.json": DATA_DIR / "metadata.json",
        tokenizer_file.name: tokenizer_file,
        "best.pt": RUN_DIR / "best.pt",
        "latest.pt": RUN_DIR / "latest.pt",
        "metrics.jsonl": RUN_DIR / "metrics.jsonl",
    },
    {
        "architecture": "cnn_attention",
        "complete": False,
        "data_fingerprint": data.fingerprint,
        "parameter_count": parameter_count,
        "train_stories": train_metadata["stories"],
        "train_targets": max_tokens,
    },
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
backup_monitor.start()
try:
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
finally:
    backup_monitor.stop()
    backup_monitor.sync_changed(force=True, metadata={"phase": "training-stopped"})
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
            "download_elapsed_seconds": download_elapsed,
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
            "drive_backup_dir": str(drive_backup_directory),
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
backup_monitor.add_sources(
    {
        "summary.json": SUMMARY,
        "examples.md": EXAMPLES,
    }
)
backup_monitor.sync_changed(
    force=True,
    metadata={
        "complete": True,
        "phase": "complete",
        "step": final_train["step"],
        "tokens_seen": final_train["tokens_seen"],
    },
)
backup_monitor.backup.verify()
print(f"[drive] Verified complete backup at {drive_backup_directory}.", flush=True)
stage_artifacts()
