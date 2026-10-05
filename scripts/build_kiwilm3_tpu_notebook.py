"""Generate the opt-in V3 TPU/Drive qualification notebook (never allocate a VM)."""
# ruff: noqa: E501

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


def notebook() -> nbformat.NotebookNode:
    markdown, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [
        markdown("""# KiwiLM 3 — TPU/BF16 and Drive continuation

## Goal

Qualify the **real V3 dense denoising trainer**, not the M2 V2 causal probe. First run a tiny synthetic model, pause at step 8, then resume on a **different manually selected Colab TPU VM**. This notebook does not allocate a session. All actions default off; running all cells unchanged produces only skip messages.

B/C only: hybrid/attention-only x 12/16 blocks; dense SwiGLU throughout. M4 variable-noise unweighted masked CE is unchanged, not an ELBO/full diffusion likelihood. No corpus quality or TPU performance result is claimed yet.

## Setup

Select a single-device TPU runtime yourself. Build a reviewed wheel locally (`uv build --wheel`), then upload that small wheel once to Drive, e.g. `MyDrive/KiwiLM3/packages/kiwilm-0.1.0-py3-none-any.whl`. Reuse the **identical wheel** on both VMs. Do not upload a virtual environment. Setup downloads pinned dependencies into an isolated Python 3.12 worker (torch/XLA 2.9.0); it does not replace Colab kernel torch. No package installation or Drive authorization happens until explicitly enabled."""),
        code("""from pathlib import Path
import json
import subprocess
import sys

MOUNT_DRIVE = False
SETUP_ENVIRONMENT = False
PREPARE_DEMO = False
RESTORE_DATA = False
RUN_PREFLIGHT = False
START_TRAINING = False

STORAGE_ROOT = Path('/content/drive')
BACKUP_ROOT = STORAGE_ROOT / 'MyDrive/KiwiLM3'
WHEEL = BACKUP_ROOT / 'packages/kiwilm-0.1.0-py3-none-any.whl'
WORKSPACE = Path('/content/kiwilm-v3-tpu')
ENVIRONMENT = Path('/content/kiwilm-v3-tpu-env')
WORKER_PYTHON = ENVIRONMENT / 'bin/python'
DATA_DIR = WORKSPACE / 'prepared-data'
TOKENIZER_PATH = WORKSPACE / 'v3-tokenizer.json'
DATA_CACHE = BACKUP_ROOT / 'data/reviewed-smoke'
QUALIFICATION = True       # synthetic width16/context16; False: width512/context512
CANDIDATE = 'hybrid-12'    # hybrid-12 / attention-12 / hybrid-16 / attention-16
RUN_NAME = 'recovery-qualification'
MODE = 'fresh'             # replacement VM: 'resume'; never implicit fresh fallback
REQUIRE_NEW_VM = False     # replacement VM qualification: True
STOP_AFTER_STEP = 8        # replacement VM: None (same total budget)
RUN_DIR = WORKSPACE / 'run' # must be empty/new; change explicitly on same-VM retries
CONFIG = {
    'device': 'xla', 'precision': 'bf16',
    'max_tokens': 4096, 'warmup_tokens': 512,
    'batch_size': 2, 'grad_accum_steps': 2,
    'learning_rate': 0.001, 'min_learning_rate': 0.0001,
    'weight_decay': 0.01, 'gradient_clip': 1.0,
    'seed': 42, 'data_seed': 42, 'noise_seed': 142, 'validation_seed': 242,
    'checkpoint_interval': 4, 'eval_interval': 4, 'eval_batches': 2,
}
result = None
print('All actions off. No session allocation, Drive writes, downloads or training.')"""),
        markdown("""### 1. Explicit Drive mount and isolated environment

Mount authorization is interactive in the notebook, never remote CLI auth. A disconnected mount is an error, not permission to start without backups. Package setup installs only the reviewed wheel without dependencies in the notebook kernel; all training dependencies live in the isolated worker."""),
        code("""if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount(str(STORAGE_ROOT))
else:
    print('Drive mount skipped.')

if SETUP_ENVIRONMENT:
    if not STORAGE_ROOT.is_mount() or not WHEEL.is_file():
        raise RuntimeError('Mount Drive and supply the reviewed wheel first')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', str(WHEEL)], check=True)
    from kiwilm.notebook_setup import prepare_environment
    WORKER_PYTHON = prepare_environment(WHEEL, ENVIRONMENT, CONFIG['device'])
    print('Worker ready:', WORKER_PYTHON)
else:
    print('Environment setup skipped.')"""),
        markdown("""## Steps

### 2. Explicit VM-local data preparation

For the tiny two-VM test enable `PREPARE_DEMO=True` on each fresh VM; deterministic synthetic data/tokenizer are regenerated locally. Existing paths refuse overwrite. This is infrastructure qualification, not architecture evidence.

For later full-width B/C smoke, set `QUALIFICATION=False`, disable demo preparation, point `DATA_CACHE` at a complete verified prepared-data cache, and enable `RESTORE_DATA`. Set `TOKENIZER_PATH` to its already converted **real V3 MASK tokenizer** in Drive. Data are copied locally for training; no hidden Hugging Face download/rebuild. `docs/kiwilm3-tpu.md` describes the proposed 50M controls; review/freeze them only after recovery passes."""),
        code("""request = {
    'config': CONFIG, 'data_dir': str(DATA_DIR), 'tokenizer_path': str(TOKENIZER_PATH),
    'candidate': CANDIDATE, 'qualification': QUALIFICATION, 'run_name': RUN_NAME,
    'run_dir': str(RUN_DIR), 'backup_root': str(BACKUP_ROOT),
    'storage_root': str(STORAGE_ROOT), 'data_cache': str(DATA_CACHE),
    'mode': MODE, 'stop_after_step': STOP_AFTER_STEP, 'require_new_vm': REQUIRE_NEW_VM,
}
if PREPARE_DEMO or RESTORE_DATA:
    from kiwilm.v3.notebook import run_action
    if PREPARE_DEMO and RESTORE_DATA:
        raise ValueError('Select demo preparation OR prepared cache restore')
    if PREPARE_DEMO and not QUALIFICATION:
        raise ValueError('Synthetic demo is qualification-only')
    action = 'prepare-demo' if PREPARE_DEMO else 'restore-data'
    print(json.dumps(run_action(WORKER_PYTHON, action, request), indent=2))
else:
    print('Data preparation/restore skipped.')"""),
        markdown("""### 3. Read-only backend/model preflight

This explicitly allocates real model weights and runs a small forward on the selected backend, but no optimizer step or backup publication. XLA BF16 requires a real single-device TPU; there is no CPU fallback. Confirm hardware, precision, tokenizer/data/code/runtime identity, model profile and compilation/fallback counters."""),
        code("""if RUN_PREFLIGHT:
    from kiwilm.v3.notebook import run_action
    print(json.dumps(run_action(WORKER_PYTHON, 'preflight', request), indent=2))
else:
    print('Preflight skipped.')"""),
        markdown("""### 4. Explicit fresh training or verified continuation

Set `START_TRAINING=True` only when ready. Fresh commits step zero before training; checkpoint publication is synchronous, hash/read-back verified, with latest + previous **distinct** committed steps retained. A failed upload stops training and leaves VM-local `latest.pt`; old committed Drive progress remains safe. At most a checkpoint interval of new work can be lost on VM failure.

Resume restores model, FP32 AdamW moments, token schedule/progress, global/dropout/backend RNG and separate data/noise streams plus the exact metrics prefix. Model/objective/tokenizer/data/code/runtime/controls are locked. No training checkpoint is mistaken for inference weights. Do not run two writers against the same namespace. Do not delete or reset backup pointers."""),
        code("""if START_TRAINING:
    from kiwilm.v3.notebook import run_action
    result = run_action(WORKER_PYTHON, 'train', {**request, 'start_training': True})
    print(json.dumps(result, indent=2))
else:
    print('Training/resume skipped.')"""),
        markdown("""## Checks

Before terminating VM 1, save its printed result and confirm `status=paused`, step 8, token count, checkpoint receipt and backup namespace. In Drive, `latest.json` should identify step 8 and `previous.json` step 4. Fresh VM 2 must show a different Linux boot ID and restore that exact step/token count before any update. Full recovery is **not** proven merely by a finite loss.

Keep the same wheel, CONFIG, candidate, run name, data and tokenizer. Change only `MODE='resume'`, `REQUIRE_NEW_VM=True`, `STOP_AFTER_STEP=None`. Enable mount/setup/demo/preflight/train deliberately on the new VM; local paths are ephemeral. Final progress should be exactly 4096 non-padding input tokens with the uninterrupted schedule, finite masked CE and no duplicated metrics steps. Retain both VM logs/receipts for review. CPU exact-equivalence tests are not TPU equivalence evidence.

Failed staged uploads can leave orphan directories; retention prunes only known superseded committed generations, never unrelated data. If latest bytes are corrupt, previous bytes may be copied for inspection, but continuing behind a newer committed head is refused; do not silently roll back.

## Next steps

See `docs/kiwilm3-tpu.md`. Review live compilation/CPU-fallback counters, finite loss/gradients, learning-rate continuity, per-step throughput and memory before a corpus smoke. Then freeze the four matched full-width B/C controls. No automated architecture winner, classifier, 50M corpus run or longer training is launched by this implementation. Existing M2/M4/M5 CPU notebooks and V2 formats remain separate."""),
    ]
    for index, cell in enumerate(cells):
        cell.id = f"v3-tpu-{index:02d}"
    result = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    nbformat.validate(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("notebooks/kiwilm3-tpu.ipynb"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook(), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
