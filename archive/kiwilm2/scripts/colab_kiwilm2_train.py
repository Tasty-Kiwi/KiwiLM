"""Prepare data and execute one KiwiLM 2 training job on a Colab GPU."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

CONTENT = Path("/content")
JOB_PATH = CONTENT / "kiwilm2-job.json"
WORK_DIR = CONTENT / "kiwilm2-colab"
DATA_DIR = WORK_DIR / "data"
RUN_DIR = WORK_DIR / "run"
SUMMARY_PATH = WORK_DIR / "summary.json"
ARTIFACT_DIR = CONTENT / "kiwilm2-artifacts"


def run(*command: str, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )


def install_package() -> None:
    wheels = sorted(CONTENT.glob("kiwilm-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one KiwiLM wheel in /content, found {wheels}")
    run("python", "-m", "pip", "install", str(wheels[0]))


job = json.loads(JOB_PATH.read_text(encoding="utf-8"))
if job.get("schema_version") != 4:
    raise RuntimeError("unsupported KiwiLM 2 Colab job schema")
install_package()

import torch  # noqa: E402

from kiwilm.colab_artifacts import create_colab_artifacts  # noqa: E402
from kiwilm.colab_drive import (  # noqa: E402
    CheckpointBackup,
    cache_prepared_data,
    restore_checkpoint_backup,
    restore_prepared_data,
)
from kiwilm.colab_kiwilm2 import (  # noqa: E402
    build_colab_model_config,
    checkpoint_backup_key,
)
from kiwilm.compile_benchmark import benchmark_slim_runtime  # noqa: E402
from kiwilm.data import (  # noqa: E402
    TOKENIZER_BUNDLE_FILE,
    PreparedTokenData,
    export_tokenizer_bundle,
    prepare_smollm_corpus,
)
from kiwilm.diagnostics import (  # noqa: E402
    cached_generation_parity_report,
    model_health_report,
    model_residual_report,
)
from kiwilm.generation import generate  # noqa: E402
from kiwilm.inference import load_trained_model  # noqa: E402
from kiwilm.model_profile import profile_kiwilm2  # noqa: E402
from kiwilm.models import KiwiLM2LM  # noqa: E402
from kiwilm.training import TrainConfig, train  # noqa: E402


def validate_prepared_data(data: PreparedTokenData) -> None:
    config = data.metadata.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("prepared SmolLM data has no configuration metadata")
    if config.get("python_edu_included") is not False:
        raise RuntimeError("prepared SmolLM data must explicitly exclude Python-Edu")
    if config.get("source_configs") != ["fineweb-edu-dedup", "cosmopedia-v2"]:
        raise RuntimeError("prepared SmolLM data has the wrong source configuration")
    if config.get("seed") != job["seed"]:
        raise RuntimeError("prepared SmolLM data has the wrong source-mixing seed")
    if config.get("fineweb_probability") != job["fineweb_probability"]:
        raise RuntimeError("prepared SmolLM data has the wrong source-mixing probability")
    if (
        config.get("validation_documents_per_source")
        != job["validation_documents_per_source"]
    ):
        raise RuntimeError("prepared SmolLM data has the wrong validation prefix")
    if data.tokenizer.vocab_size != 32_000:
        raise RuntimeError("prepared SmolLM data does not use the frozen 32K tokenizer")
    if data.metadata["splits"]["train"]["tokens"] != job["prepared_train_tokens"]:
        raise RuntimeError("prepared training split does not match the job specification")
    if data.metadata["splits"]["validation"]["tokens"] != job["validation_tokens"]:
        raise RuntimeError("prepared validation split does not match the job specification")


def prepare_data(drive_root: Path | None) -> PreparedTokenData:
    cache_dir = (
        drive_root / "data" / str(job["data_cache_key"])
        if drive_root is not None
        else None
    )
    if cache_dir is not None and restore_prepared_data(
        cache_dir, DATA_DIR,
        required=job.get("require_resume", False),
        storage_root=drive_root,
    ):
        print(f"Restored prepared SmolLM data from {cache_dir}", flush=True)
    else:
        DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
        tokenizer_dir = drive_root / "tokenizer" if drive_root is not None else None
        tokenizer_from = (
            tokenizer_dir
            if tokenizer_dir is not None
            and (tokenizer_dir / TOKENIZER_BUNDLE_FILE).is_file()
            else None
        )
        print(
            f"Preparing {job['prepared_train_tokens']:,} SmolLM tokens inside Colab...",
            flush=True,
        )
        prepare_smollm_corpus(
            DATA_DIR,
            train_tokens=job["prepared_train_tokens"],
            validation_tokens=job["validation_tokens"],
            tokenizer_train_documents=job["tokenizer_train_documents"],
            validation_documents_per_source=job["validation_documents_per_source"],
            vocab_size=32_000,
            fineweb_probability=job["fineweb_probability"],
            seed=job["seed"],
            tokenizer_from=tokenizer_from,
            show_progress=True,
        )
        prepared = PreparedTokenData(DATA_DIR, seed=job["seed"])
        validate_prepared_data(prepared)
        if tokenizer_dir is not None and not (tokenizer_dir / TOKENIZER_BUNDLE_FILE).is_file():
            export_tokenizer_bundle(DATA_DIR, tokenizer_dir)
            print(f"Saved the shared tokenizer to {tokenizer_dir}", flush=True)
        if cache_dir is not None:
            created = cache_prepared_data(DATA_DIR, cache_dir)
            action = "Cached" if created else "Found concurrently cached"
            print(f"{action} prepared data at {cache_dir}", flush=True)
        return prepared
    prepared = PreparedTokenData(DATA_DIR, seed=job["seed"])
    validate_prepared_data(prepared)
    return prepared


def stage_resume(backup_dir: Path | None) -> Path | None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    explicit = CONTENT / "resume.pt"
    if explicit.is_file():
        for source_name, destination_name in (
            ("resume-best.pt", "best.pt"),
            ("resume-metrics.jsonl", "metrics.jsonl"),
        ):
            source = CONTENT / source_name
            if source.is_file():
                shutil.copy2(source, RUN_DIR / destination_name)
        print("Resuming from the explicitly uploaded checkpoint", flush=True)
        return explicit
    if backup_dir is None:
        if job.get("require_resume", False):
            raise RuntimeError(
                "Resume is required but no uploaded checkpoint or Drive backup is available; "
                "refusing to train from scratch"
            )
        return None
    restored = restore_checkpoint_backup(
        backup_dir, RUN_DIR,
        required=job.get("require_resume", False),
        storage_root=Path(job["drive_root"]),
    )
    if restored is not None:
        print(f"Resuming from Google Drive checkpoint {backup_dir / 'latest.pt'}", flush=True)
    return restored


drive_root: Path | None = None
backup_dir: Path | None = None
if job["drive_backups"]:
    if not (CONTENT / "drive" / "MyDrive").is_dir():
        raise RuntimeError(
            "Google Drive is not mounted; run the launcher interactively or set "
            "KIWILM2_DRIVE_BACKUPS=0"
        )
    drive_root = Path(job["drive_root"])
    drive_root.mkdir(parents=True, exist_ok=True)
    backup_dir = drive_root / "checkpoints" / checkpoint_backup_key(job)

# Fail closed before expensive data restoration/preparation or any backup writes.
resume_path = stage_resume(backup_dir)
data = prepare_data(drive_root)
tokenizer_metadata = data.metadata["tokenizer"]
job["data_fingerprint"] = data.fingerprint
job["tokenizer_sha256"] = tokenizer_metadata["sha256"]
job["resolved_dataset_revision"] = data.metadata["dataset"]["resolved_revision"]
JOB_PATH.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")

if not torch.cuda.is_available():
    raise RuntimeError("KiwiLM 2 Colab training requires a CUDA GPU")
device = torch.device("cuda")

from kiwilm.config import (  # noqa: E402
    KiwiLM2Config,
    KiwiLM2SlimConfig,
    KiwiLM2SlimV3Config,
)

model_config = build_colab_model_config(job, vocab_size=data.tokenizer.vocab_size)
eval_interval = 250 if job["phase"] == "smoke" else 500
if job["phase"].startswith("final"):
    eval_interval = 1_000
settings = TrainConfig(
    max_steps=job["max_steps"],
    max_tokens=job["max_tokens"],
    warmup_tokens=job["warmup_tokens"],
    batch_size=job["batch_size"],
    grad_accum_steps=job["grad_accum_steps"],
    lr=job["learning_rate"],
    min_lr=job["min_learning_rate"],
    precision=job["precision"],
    optimizer=job["optimizer"],
    muon_lr=job["muon_lr"],
    eval_interval=eval_interval,
    eval_batches=job.get("eval_batches", 50),
    checkpoint_interval=eval_interval,
    log_interval=10,
    sample_tokens=64,
    seed=job["seed"],
)
compile_benchmark = None
compile_model = False
compile_policy = job["compile_policy"]
if compile_policy == "compiled":
    compile_model = True
elif compile_policy == "auto" and isinstance(
    model_config, (KiwiLM2SlimConfig, KiwiLM2SlimV3Config)
):
    dense_benchmark_config = KiwiLM2Config(vocab_size=data.tokenizer.vocab_size)
    print("Benchmarking Dense eager and gated Slim eager/compiled...", flush=True)
    compile_benchmark = benchmark_slim_runtime(
        dense_benchmark_config,
        model_config,
        device=device,
        batch_size=job["batch_size"],
        precision=job["precision"],
    )
    compile_model = compile_benchmark["selected_runtime"] == "compiled"
    print(json.dumps({"compile_benchmark": compile_benchmark}, indent=2), flush=True)
runtime = "compiled" if compile_model else "eager"
gpu = run(
    "nvidia-smi",
    "--query-gpu=name,memory.total,driver_version",
    "--format=csv,noheader",
    capture=True,
).stdout.strip()
torch_info = {
    "version": torch.__version__,
    "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
}

backup = (
    CheckpointBackup(
        RUN_DIR,
        backup_dir,
        job_path=JOB_PATH,
        interval_seconds=30.0,
        error_callback=lambda error: print(
            f"Google Drive checkpoint backup failed: {error}",
            file=sys.stderr,
            flush=True,
        ),
    )
    if backup_dir is not None
    else None
)
if backup is not None:
    backup.start()

telemetry_generator = torch.Generator(device="cpu").manual_seed(141)
telemetry_inputs, _ = data.get_batch(
    "validation",
    batch_size=2,
    context_length=model_config.context_length,
    device=device,
    generator=telemetry_generator,
)


def validation_diagnostic(
    network: torch.nn.Module, step: int, tokens_seen: int
) -> dict[str, object] | None:
    if step % 500 and tokens_seen < job["max_tokens"]:
        return None
    if not isinstance(network, KiwiLM2LM):
        raise TypeError("residual telemetry requires KiwiLM2LM")
    return model_residual_report(network, telemetry_inputs)

try:
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    training_summary = train(
        model_config,
        data,
        RUN_DIR,
        settings,
        device=device,
        resume_from=resume_path,
        compile_model=compile_model,
        validation_diagnostic_fn=(
            validation_diagnostic
            if job.get("swiglu_residual_gate_init") is not None
            else None
        ),
    )
    elapsed = time.perf_counter() - started

    model, loaded_config = load_trained_model(
        RUN_DIR / "latest.pt",
        data_fingerprint=data.fingerprint,
        device=device,
    )
    if not isinstance(model, KiwiLM2LM):
        raise RuntimeError("checkpoint did not reconstruct a KiwiLM 2 model")
    generator = torch.Generator(device="cpu").manual_seed(job["seed"] + 99)
    diagnostic_inputs, diagnostic_targets = data.get_batch(
        "validation",
        batch_size=1,
        context_length=loaded_config.context_length,
        device=device,
        generator=generator,
    )
    health = model_health_report(model, diagnostic_inputs, diagnostic_targets)
    cached_generation = cached_generation_parity_report(model, diagnostic_inputs)
    profile = profile_kiwilm2(model)
    sample = generate(
        model,
        data.tokenizer,
        "Once upon a time",
        max_new_tokens=160,
        temperature=0.8,
        top_k=40,
        seed=job["seed"],
        device=device,
        cache="auto",
    )
    del model

    metrics = [
        json.loads(line)
        for line in (RUN_DIR / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    final_train = next(row for row in reversed(metrics) if row["event"] == "train")
    summary = {
        "job": job,
        "data_fingerprint": data.fingerprint,
        "gpu": gpu,
        "torch": torch_info,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "training": training_summary,
        "runtime": runtime,
        "compile_policy": compile_policy,
        "compile_benchmark": compile_benchmark,
        "final_train_metrics": final_train,
        "profile": profile,
        "health": health,
        "cached_generation": cached_generation,
        "sample": sample,
        "resumed": resume_path is not None,
        "drive_checkpoint_dir": str(backup_dir) if backup_dir is not None else None,
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = {
        "latest.pt": RUN_DIR / "latest.pt",
        "metrics.jsonl": RUN_DIR / "metrics.jsonl",
        "summary.json": SUMMARY_PATH,
        "job.json": JOB_PATH,
    }
    if (RUN_DIR / "best.pt").is_file():
        files["best.pt"] = RUN_DIR / "best.pt"
    manifest_path = create_colab_artifacts(
        files,
        ARTIFACT_DIR,
        archive_name="kiwilm2-artifacts.tar",
    )
    print(SUMMARY_PATH.read_text(encoding="utf-8"))
    print(f"artifact_manifest={manifest_path}")
finally:
    if backup is not None:
        active_error = sys.exc_info()[0] is not None
        extra_files = (("summary.json", SUMMARY_PATH),) if SUMMARY_PATH.is_file() else ()
        try:
            backup.stop(
                final_sync=(RUN_DIR / "latest.pt").is_file(),
                extra_files=extra_files,
            )
        except Exception as backup_error:
            if not active_error:
                raise
            print(
                f"Final Google Drive checkpoint backup failed: {backup_error}",
                file=sys.stderr,
                flush=True,
            )
