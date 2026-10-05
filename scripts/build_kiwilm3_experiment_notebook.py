"""Generate the thin opt-in M5 B/C notebook; do not replace the M2/M4 companions."""
# ruff: noqa: E501

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


def notebook() -> nbformat.NotebookNode:
    markdown, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [
        markdown("""# KiwiLM 3 — B/C dense denoising experiments

## Goal

Prepare **B: attention-only vs hybrid BiConv** and **C: 12 vs 16 blocks**. All four candidates use dense SwiGLU and the same M4 variable-noise masked reconstruction objective. A and D are dropped. The fixed-slot sampler supports unconditional output, continuation and right-context infilling.

This is a **bounded CPU qualification tutorial**, not a pretrained model or a production TPU/Drive launcher. Tiny tests cannot choose an architecture. Actions default off; nothing allocates a cloud session, mounts Drive or downloads data.

## Setup

Locally: `uv sync --locked --extra notebooks`. Select the installed environment as kernel. On a manually selected Colab CPU runtime, upload a reviewed wheel and explicitly enable installation. Existing TPU/Drive continuation notebooks are V2 recovery probes, not V3 diffusion training."""),
        code("""from pathlib import Path
import json
import subprocess
import sys

WORKSPACE = Path('/content/kiwilm-m5') if Path('/content').is_dir() else Path('runs/notebook-m5')
DATA_DIR = WORKSPACE / 'prepared-data'
TOKENIZER_PATH = WORKSPACE / 'v3-tokenizer.json'
SUITE_PATH = WORKSPACE / 'suite.json'
WHEEL = Path('/content/kiwilm-0.1.0-py3-none-any.whl')
INSTALL_PACKAGE = False
PLAN_SUITE = False
RUN_PREFLIGHT = False
START_TRAINING = False
COMPARE_RUNS = False
GENERATE_OUTPUT = False
QUALIFICATION = True       # True: width16, 20 steps; False: width512, bounded 1000 steps
CANDIDATE = 'hybrid-12'    # hybrid-12 / attention-12 / hybrid-16 / attention-16
STOP_AFTER_STEP = None     # optional pause inside the frozen budget
RESUME_RECEIPT = None     # original step-qualified .json receipt, never implicit fresh fallback
EXPECTED_CHECKPOINT_SHA256 = None
COMPARISON_DIR = WORKSPACE / 'comparison'
INFERENCE_WEIGHTS = WORKSPACE / CANDIDATE / 'encoder.pt'
suite = None
print('All actions disabled. A/D dropped; dense SwiGLU + variable-noise reconstruction only.')"""),
        markdown(
            "### 1. Optional package install\n\nContinuation needs the same reviewed package/runtime. CPU-only; no automatic TPU selection."
        ),
        code("""if INSTALL_PACKAGE:
    if not WHEEL.is_file():
        raise FileNotFoundError('Upload the reviewed wheel first')
    subprocess.run([sys.executable, '-m', 'pip', 'install', str(WHEEL)], check=True)
else:
    print('Package install skipped.')"""),
        markdown("""## Steps

### 2. Freeze data, tokenizer and four candidates

Prepare data and convert its tokenizer separately (see `docs/kiwilm3-experiments.md`). The original packed IDs remain valid; the new tokenizer adds a real MASK token. No preparation/download action is hidden here. Suite creation refuses overwrite. The default full-width CPU budget is 1,024,000 input positions, not a 50M/1B accelerator recipe."""),
        code("""if PLAN_SUITE or RUN_PREFLIGHT or START_TRAINING or COMPARE_RUNS:
    from kiwilm.data import PreparedTokenData
    from kiwilm.v3.tokenizer import MaskBPETokenizer
    from kiwilm.v3.experiments import build_suite, validate_suite, CANDIDATES
    from kiwilm.v3.experiment_runner import write_json, preflight, run_candidate, write_comparisons
    if PLAN_SUITE:
        data = PreparedTokenData(DATA_DIR)
        tokenizer = MaskBPETokenizer.load(TOKENIZER_PATH)
        tokenizer.assert_base_compatible(data.tokenizer)
        suite = build_suite(data_fingerprint=data.fingerprint, tokenizer_sha256=tokenizer.fingerprint,
                            vocab_size=tokenizer.vocab_size, qualification=QUALIFICATION)
        for candidate in CANDIDATES:
            preflight(suite, candidate, data, tokenizer)
        write_json(SUITE_PATH, suite)
    else:
        suite = json.loads(SUITE_PATH.read_text())
    validate_suite(suite)
    print(json.dumps({'identity': suite['identity'], 'budget': suite['budget'],
                      'qualification': suite['qualification'], 'training_started': False}, indent=2))
    for name, spec in suite['candidates'].items():
        print(name, spec['profile']['parameters'], 'parameters;',
              spec['profile']['forward_flops_per_token'], 'estimated forward FLOPs/token')
else:
    print('Suite planning/loading skipped.')"""),
        markdown(
            "### 3. Read-only preflight\n\nMeta profiles do not allocate real model weights. All candidates share tokenizer, objective, data/noise seeds, batches and evaluation controls."
        ),
        code("""if RUN_PREFLIGHT:
    data = PreparedTokenData(DATA_DIR)
    tokenizer = MaskBPETokenizer.load(TOKENIZER_PATH)
    print(json.dumps(preflight(suite, CANDIDATE, data, tokenizer), indent=2))
else:
    print('Preflight skipped.')"""),
        markdown("""### 4. Explicit fresh execution or verified local continuation

One candidate at a time. Fresh execution refuses an existing output directory. Resume needs the original receipt and independently retained checkpoint digest. Pausing saves an exclusive step-qualified checkpoint; no checkpoint overwrite or automatic cleanup occurs. Metrics beyond the verified checkpoint are refused rather than silently deleted. This prototype does **not** provide Google Drive backups. On a Colab VM, manually copy verified state/receipts/metrics off ephemeral storage before termination."""),
        code("""if START_TRAINING:
    import torch
    torch.set_num_threads(1)
    data = PreparedTokenData(DATA_DIR)
    tokenizer = MaskBPETokenizer.load(TOKENIZER_PATH)
    report = run_candidate(suite, CANDIDATE, data, tokenizer, WORKSPACE / CANDIDATE,
                           start_training=True, stop_after_step=STOP_AFTER_STEP,
                           resume_receipt=RESUME_RECEIPT,
                           expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256)
    print(json.dumps({'status': report['status'], 'step': report['step'],
                      'checkpoint': report['checkpoint'],
                      'checkpoint_sha256': report['checkpoint_sha256']}, indent=2))
else:
    print('Training/continuation skipped.')"""),
        markdown("""## Checks

### 5. Compare only completed matched runs

After explicitly running all four, compare fixed masked losses/accuracy at 15/50/90/100% noise, health distributions, FP32 inference parity and seeded samples. Data/tokenizer/code/runtime/order controls must match. CPU timings exclude evaluation/checkpoint I/O; peak accelerator memory/cost is missing. No automatic winner, causal perplexity or fabricated quality metrics."""),
        code("""if COMPARE_RUNS:
    step = suite['controls']['max_steps']
    reports = {name: json.loads((WORKSPACE / name / f'report-{step:06d}.json').read_text())
               for name in CANDIDATES}
    print(json.dumps(write_comparisons(suite, reports, COMPARISON_DIR), indent=2))
else:
    print('Comparison skipped.')"""),
        markdown("""### 6. Dataset-free infilling

Inference weights and matching V3 tokenizer are sufficient. Temperature 0.8/top-k 40 is a starting sampling profile, not a quality claim. Output slots have fixed BPE length; EOS is protected rather than generated. No silent truncation, remasking, rollover or causal KV cache."""),
        code("""if GENERATE_OUTPUT:
    from kiwilm.v3.tokenizer import MaskBPETokenizer
    from kiwilm.v3.weights import load_encoder_weights
    from kiwilm.v3.sampling import SamplingConfig, generate_slots
    tokenizer = MaskBPETokenizer.load(TOKENIZER_PATH)
    model, _ = load_encoder_weights(INFERENCE_WEIGHTS,
                                   expected_tokenizer_sha256=tokenizer.fingerprint)
    print(json.dumps(generate_slots(model, tokenizer, 'The ', suffix=' cat.', output_slots=4,
                                    config=SamplingConfig(4, 0.8, 40, 42)), indent=2))
else:
    print('Generation skipped.')"""),
        markdown("""## Next steps

Review `docs/kiwilm3-experiments.md`, then qualify an accelerator/Drive adapter and a matched small corpus budget before choosing a longer run. Contextual cloze scoring is available in `kiwilm.v3.evaluation`; held-out retrieval/classification/transfer tasks and human semantic review remain separate gates. This notebook does not train a classifier or start a long run. Re-running default-disabled cells should produce only skip messages, with no filesystem writes."""),
    ]
    for index, cell in enumerate(cells):
        cell.id = f"m5-{index:02d}"
    result = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
    )
    nbformat.validate(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("notebooks/kiwilm3-experiments.ipynb"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook(), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
