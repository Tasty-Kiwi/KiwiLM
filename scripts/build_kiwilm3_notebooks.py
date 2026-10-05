"""Generate thin M2 notebooks. No training, VM allocation, or network access."""
# Embedded notebook prose/cells retain their own readable wrapping.
# ruff: noqa: E501

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


def notebooks() -> dict[str, nbformat.NotebookNode]:
    output = {}
    for kind in ("smoke", "train", "evaluate"):
        cells = []

        def markdown(text: str, target: list = cells) -> None:
            target.append(nbformat.v4.new_markdown_cell(text))

        def code(text: str, target: list = cells) -> None:
            target.append(nbformat.v4.new_code_cell(text))

        markdown(
            f"# KiwiLM 3 — {kind} workflow qualification (M2)\n\n"
            "## Goal\n\nTest notebook control and checkpoint recovery with a **tiny V2 causal model**. "
            "This is not KiwiLM 3 training: the encoder and masked-diffusion objective are later phases. "
            "All installation, Drive, training and evaluation actions are **off by default**.\n\n"
            "The synthetic corpus and 16-wide model measure infrastructure, not model quality or production throughput."
        )
        markdown(
            "## Setup\n\nOpen in a manually selected Colab CPU/GPU/single-chip TPU runtime. "
            "No CLI provisioning is used. Build the reviewed wheel locally with `uv build --wheel`, "
            "then upload it to the path below. For TPU use `DEVICE='xla', PRECISION='bf16'`; "
            "for a T4 choose `DEVICE='cuda', PRECISION='fp16'`, not BF16. "
            "CPU uses FP32. The worker runs in isolated Python 3.12; system/kernel torch is not replaced. "
            "See `docs/notebook-workflow.md` for the two-runtime recovery procedure."
        )
        code(
            """from pathlib import Path
import json
import subprocess
import sys

IN_COLAB = Path("/content").is_dir()
WORKSPACE = Path("/content/kiwilm-m2") if IN_COLAB else Path("runs/notebook-m2")
DRIVE_MOUNT = Path("/content/drive")
BACKUP_ROOT = DRIVE_MOUNT / "MyDrive/KiwiLM3" if IN_COLAB else Path("runs/notebook-backups")
WHEEL = BACKUP_ROOT / "wheels" / "kiwilm-0.1.0-py3-none-any.whl"
ENVIRONMENT = Path("/content/kiwilm-m2-env")
WORKER_PYTHON = Path(sys.executable)

DEVICE = "cpu"                    # cpu / cuda / xla — select manually
PRECISION = "fp32"                # CPU fp32; GPU fp16/bf16/fp32; TPU bf16
MOUNT_DRIVE = False
SETUP_ENVIRONMENT = False
PREPARE_DATA = False
RUN_PREFLIGHT = False
START_TRAINING = False
RUN_EVALUATION = False
RESTORE_FOR_EVALUATION = False
EXPORT_SAFETENSORS = False
MODE = "fresh"                    # change to resume after VM replacement
STOP_AFTER_STEP = """
            + ("4" if kind == "smoke" else "None")
            + """     # full locked budget is 8 steps / 256 tokens

print("Actions are disabled until explicitly enabled. No VM allocation or training has started.")"""
        )
        markdown(
            "### 1. Mount Drive and install the worker\n\nEnable these two switches only in Colab. "
            "Grant Drive access yourself. Setup can install packages, but never starts training. "
            "Use the same reviewed wheel on a replacement runtime; changed code or runtime versions refuse resume."
        )
        code("""if MOUNT_DRIVE:
    if not IN_COLAB:
        raise RuntimeError("Drive mounting is only for a manually opened Colab runtime")
    from google.colab import drive
    drive.mount(str(DRIVE_MOUNT))
else:
    print("Drive mount skipped.")

if SETUP_ENVIRONMENT:
    if not IN_COLAB:
        raise RuntimeError("Locally use uv sync --locked --extra notebooks instead")
    if not WHEEL.is_file():
        raise FileNotFoundError("Upload the reviewed wheel and mount Drive first")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", str(WHEEL)], check=True)
    from kiwilm.notebook_setup import prepare_environment
    WORKER_PYTHON = prepare_environment(WHEEL, ENVIRONMENT, DEVICE)
    print(f"Isolated worker ready: {WORKER_PYTHON}")
else:
    print("Environment installation skipped; local actions use the current installed kiwilm environment.")""")
        markdown(
            "## Steps\n\n### 2. Freeze the experiment\n\nCopy this configuration unchanged when resuming "
            "or evaluating. Each backup namespace includes its contract digest. Fresh and resume are explicit; "
            "there is no warm-start fallback. A new empty local run directory is required on each invocation."
        )
        code("""CONFIG = {
    "experiment": "m2-recovery-smoke",
    "device": DEVICE, "precision": PRECISION,
    "max_tokens": 256, "warmup_tokens": 64,
    "batch_size": 2, "grad_accum_steps": 1, "context_length": 16,
    "seed": 42, "noise_seed": 142,
    "checkpoint_interval": 2, "eval_interval": 2, "eval_batches": 2, "lr": 0.001,
}
DATA_DIR = WORKSPACE / "data"
RUN_DIR = WORKSPACE / "run-1"       # change to run-2 if this local directory already has state
REQUEST = {
    "config": CONFIG, "data_dir": str(DATA_DIR), "run_dir": str(RUN_DIR),
    "backup_root": str(BACKUP_ROOT),
    "storage_root": str(DRIVE_MOUNT) if IN_COLAB else None,
}
print(json.dumps({"config": CONFIG, "mode": MODE, "start_training": START_TRAINING}, indent=2))""")
        markdown(
            "### 3. Prepare bounded data and check the backend\n\nThe local synthetic corpus is deterministic "
            "and regenerated in a fresh VM. Preparation does not download SmolLM or overwrite an existing corpus. "
            "Preflight checks integrity, precision, backend availability and a small matrix operation. "
            "There is no automatic CPU fallback."
        )
        code("""if PREPARE_DATA:
    from kiwilm.notebook_setup import run_notebook_action
    prepared = run_notebook_action(WORKER_PYTHON, "prepare", REQUEST)
    print("Prepared fingerprint:", prepared["data"]["fingerprint"])
else:
    print("Data preparation skipped.")

if RUN_PREFLIGHT:
    from kiwilm.notebook_setup import run_notebook_action
    checked = run_notebook_action(WORKER_PYTHON, "preflight", REQUEST)
    print(json.dumps({
        "identity": checked["identity"], "backend": checked["backend"],
        "versions": checked["contract"]["runtime_versions"], "counters": checked["counters"],
    }, indent=2))
else:
    print("Backend preflight skipped.")""")
        markdown(
            "### 4. Explicit train or resume\n\nSet `START_TRAINING=True` only when ready. "
            "Metrics stream below; publication waits for verified Drive readback and retains latest/previous. "
            "Interrupting a worker preserves the last committed boundary, not necessarily its last printed step. "
            "With the smoke default, stop at step 4, replace the VM, choose `MODE='resume'` and "
            "`STOP_AFTER_STEP=None` to finish the same 256-token schedule."
        )
        code("""if START_TRAINING:
    from kiwilm.notebook_setup import run_notebook_action
    trained = run_notebook_action(WORKER_PYTHON, "train", {
        **REQUEST, "mode": MODE, "start_training": True,
        "stop_after_step": STOP_AFTER_STEP,
    })
    print(json.dumps(trained, indent=2))
else:
    print("Training disabled. No optimizer updates were performed.")""")
        markdown(
            "## Checks\n\n### 5. Restore, evaluate and optionally export\n\nFor a separate evaluation VM "
            "set `RESTORE_FOR_EVALUATION=True` with an empty local run directory and the original configuration. "
            "Evaluation is CPU/FP32 and reports causal validation, health, cached direct/rollover parity, "
            "and a short sample. These are workflow checks, not diffusion or quality benchmarks. "
            "Export is inference-only BF16 Safetensors and refuses an existing output; it is not a resume checkpoint."
        )
        code("""if RESTORE_FOR_EVALUATION:
    from kiwilm.notebook_setup import run_notebook_action
    restored = run_notebook_action(WORKER_PYTHON, "restore", REQUEST)
    print("Restored committed step:", restored["step"])

if RUN_EVALUATION:
    from kiwilm.notebook_setup import run_notebook_action
    report = run_notebook_action(WORKER_PYTHON, "evaluate", {
        **REQUEST, "checkpoint": str(RUN_DIR / "latest.pt"),
        "export_dir": str(WORKSPACE / "probe-bf16") if EXPORT_SAFETENSORS else None,
    })
    print(json.dumps({key: value for key, value in report.items() if key != "health"}, indent=2))
    print("Per-block health is available in report['health'].")
else:
    print("Evaluation/export skipped; no checkpoint was read or published.")""")
        markdown(
            "## Next Steps\n\nRecord the contract identity, committed step/token counts, worker versions, "
            "and parity result for both runtimes. Local CPU continuation is regression-tested; actual "
            "Colab GPU/TPU plus Drive recovery must be qualified by running this procedure yourself. "
            "Do not infer production TPU performance from the tiny probe. The next model phase adds "
            "the bidirectional encoder; masked-diffusion loss/noise and retrieval evaluation follow separately. "
            "Do not start a long training run from this M2 qualification notebook."
        )
        book = nbformat.v4.new_notebook(cells=cells)
        book.metadata = {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
            "kiwilm_phase": "M2",
            "kiwilm_engine": "m2-v2-causal-recovery-probe",
        }
        for index, cell in enumerate(book.cells):
            cell.id = f"{kind}-{index:02d}"
        output[f"kiwilm3-{kind}.ipynb"] = book
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("notebooks"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, book in notebooks().items():
        nbformat.validate(book)
        nbformat.write(book, args.output_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
