"""Generate the opt-in M4 CPU companion, leaving the M2 recovery notebooks intact."""
# ruff: noqa: E501

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


def notebook() -> nbformat.NotebookNode:
    markdown, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [
        markdown("""# KiwiLM 3 — M4 denoising prototype (CPU)

## Goal

Check real `[MASK]` token handling, variable-noise reconstruction and bounded local continuation. **This is an opt-in tutorial, not a production TPU run.** No action is enabled by default. Synthetic data tests implementation, not model quality or throughput. The existing M2 notebooks remain separate V2 recovery probes.

## Setup

Locally: `uv sync --locked --extra notebooks`. Select that environment as your kernel. In a manually opened Colab CPU runtime, upload your reviewed `uv build --wheel` artifact and opt into installation below. Do not select a TPU for this CPU-only trainer. No session provisioning, Drive mounting, or downloads of training data happen here."""),
        code("""from pathlib import Path
import hashlib
import json
import subprocess
import sys

WORKSPACE = Path('/content/kiwilm-m4') if Path('/content').is_dir() else Path('runs/notebook-m4')
DATA_DIR = WORKSPACE / 'demo-data'
TOKENIZER_PATH = WORKSPACE / 'v3-tokenizer.json'
WHEEL = Path('/content/kiwilm-0.1.0-py3-none-any.whl')
INSTALL_PACKAGE = False
PREPARE_DEMO = False
RUN_PREFLIGHT = False
START_TRAINING = False
RUN_VALIDATION = False
SAVE_STATE = False
RESUME_STATE = False
DEPTH = 12                  # 12 or 16 blocks, width 16; not the full-width model
STEPS_THIS_SESSION = 4      # full locked budget: 20 local optimizer-step attempts
CHECKPOINT = WORKSPACE / 'step4.pt'
EXPECTED_SHA256 = ''        # copy the digest from the original save receipt to resume
trainer = None
print('All actions are disabled. CPU prototype only; no training has started.')"""),
        markdown(
            "### 1. Install only if explicitly selected\n\nUse the same reviewed wheel and locked runtime for continuation. Never substitute a changed package to bypass the code-identity check."
        ),
        code("""if INSTALL_PACKAGE:
    if not WHEEL.is_file():
        raise FileNotFoundError('Upload the reviewed wheel first')
    subprocess.run([sys.executable, '-m', 'pip', 'install', str(WHEEL)], check=True)
else:
    print('Package installation skipped.')"""),
        markdown("""## Steps

### 2. Prepare a tiny synthetic corpus and separate tokenizer

Existing content IDs are preserved. `[MASK]` is appended as a real special token in the new tokenizer JSON; the old tokenizer and packed files are not modified. Preparation refuses an existing destination. For actual data use `PreparedTokenData` and the tokenizer conversion script in `docs/kiwilm3-denoising.md`."""),
        code("""if PREPARE_DEMO:
    from kiwilm.data import PreparedTokenData, prepare_from_stories
    from kiwilm.v3.tokenizer import MaskBPETokenizer
    if DATA_DIR.exists() or TOKENIZER_PATH.exists():
        raise FileExistsError('Demo artifacts exist; reuse them with PREPARE_DEMO=False')
    prepare_from_stories(DATA_DIR, ['the cat saw the red ball. ' * 30] * 3,
                         ['the cat saw the red ball. ' * 30], vocab_size=300,
                         min_frequency=1, show_progress=False)
    data = PreparedTokenData(DATA_DIR)
    tokenizer = MaskBPETokenizer.from_base(data.tokenizer)
    tokenizer.save(TOKENIZER_PATH)
    print(json.dumps({'vocab_size': tokenizer.vocab_size, 'mask_id': tokenizer.mask_id,
                      'tokenizer_sha256': tokenizer.fingerprint, 'data_fingerprint': data.fingerprint}))
else:
    print('Demo preparation skipped.')"""),
        markdown(
            "### 3. Configure and preflight\n\nThe 12/16-block mixer schedule is unchanged, at tiny width 16. AdamW uses a constant LR; this prototype has no full-run token schedule, accelerator support or Drive backup. Loss is unweighted CE over masked content tokens, not an ELBO or AR perplexity."
        ),
        code("""if RUN_PREFLIGHT or START_TRAINING or RESUME_STATE or RUN_VALIDATION:
    import torch
    from kiwilm.data import PreparedTokenData
    from kiwilm.v3 import KiwiLM3Config, build_encoder
    from kiwilm.v3.tokenizer import MaskBPETokenizer
    from kiwilm.v3.trainer import DenoisingTrainConfig, DenoisingTrainer
    torch.set_num_threads(1)
    torch.manual_seed(42)
    data = PreparedTokenData(DATA_DIR)
    tokenizer = MaskBPETokenizer.load(TOKENIZER_PATH)
    config = KiwiLM3Config(vocab_size=tokenizer.vocab_size, d_model=16, num_heads=2,
                          context_length=16, swiglu_dim=48, noise_embedding_dim=16,
                          num_blocks=DEPTH, dropout=0.1)
    trainer = DenoisingTrainer(build_encoder(config), tokenizer, data,
                              DenoisingTrainConfig(max_steps=20))
    print(json.dumps({'experiment_identity': trainer.identity, 'training_started': False,
                      'device': 'cpu', 'precision': 'fp32'}))
else:
    print('Preflight skipped.')"""),
        markdown(
            "### 4. Resume only with a verified original checksum\n\nCopy the checkpoint and its receipt to the new local VM yourself. This is not automated Drive recovery. Model, objective, tokenizer, data, runtime and code must match. No missing-checkpoint fallback to a fresh run is allowed."
        ),
        code("""if RESUME_STATE:
    from kiwilm.v3.checkpoints import load_training_state
    if trainer is None or not EXPECTED_SHA256:
        raise ValueError('Preflight and an original checkpoint checksum are required')
    load_training_state(trainer, CHECKPOINT, expected_sha256=EXPECTED_SHA256)
    print(json.dumps({'restored_step': trainer.step, 'tokens_seen': trainer.tokens_seen}))
else:
    print('Resume skipped.')"""),
        markdown(
            "### 5. Opt into bounded CPU steps\n\nTraining RNGs for data windows and corruption are separate. Zero-mask batches skip AdamW entirely, including weight decay. All model math lives in package modules, not these cells."
        ),
        code("""if START_TRAINING:
    if trainer is None or not 1 <= STEPS_THIS_SESSION <= 20:
        raise ValueError('A configured trainer and 1-20 session steps are required')
    for _ in range(STEPS_THIS_SESSION):
        print(json.dumps(trainer.train_step()))
else:
    print('Training skipped.')"""),
        markdown("""## Checks

### 6. Fixed validation and optional state save

Evaluate the same validation windows and masks at noise 0.15, 0.5, 0.9 and 1.0. Metrics aggregate CE/correct predictions by masked-token count; absent masks produce `null`, not misleading accuracy/loss. Validation cannot consume training RNGs. Saved state includes optimizer, progress, data/noise/global RNGs and a resume contract. Choose a new checkpoint filename for each save; overwrite is refused."""),
        code("""if RUN_VALIDATION:
    if trainer is None:
        raise ValueError('Configure a trainer first')
    print(json.dumps(trainer.evaluate(batches=4), indent=2))
else:
    print('Validation skipped.')

if SAVE_STATE:
    from kiwilm.v3.checkpoints import save_training_state
    if trainer is None:
        raise ValueError('Configure a trainer first')
    path = save_training_state(trainer, CHECKPOINT)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps({'checkpoint': str(path), 'sha256': sha256,
                      'step': trainer.step, 'experiment_identity': trainer.identity}))
else:
    print('State save skipped.')"""),
        markdown("""## Next Steps

Run the correctness tests in `tests/test_denoising.py`. They check stable toy learning and exact CPU continuation with dropout, data ordering and noise restored. These are not evidence of corpus-scale learning or fresh-Colab/Drive reliability. GPU/TPU/production latest-previous recovery remain unqualified; iterative infilling and generation belong to M5. See `docs/kiwilm3-denoising.md` for policy and limitations."""),
    ]
    result = nbformat.v4.new_notebook(cells=cells)
    result.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    result.metadata.language_info = {"name": "python", "version": "3.13"}
    # Stable generated IDs prevent unrelated notebook JSON churn.
    for index, cell in enumerate(result.cells):
        cell.id = f"m4-cell-{index:02d}"
    nbformat.validate(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("notebooks/kiwilm3-denoising.ipynb"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook(), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
