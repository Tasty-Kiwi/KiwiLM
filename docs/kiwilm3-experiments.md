# KiwiLM 3 — iterative generation and B/C experiments

Phase 5 implements M5 fixed-slot generation and **prepares** matched B/C runs.
It does not select an architecture, train a corpus model or qualify live TPU/Drive
recovery. By owner decision, **A and D are dropped**: no fixed-15% training
baseline or Hadamard ablation. All candidates use the unchanged
[M4 variable-noise masked reconstruction objective](kiwilm3-denoising.md).

## Frozen matrix

Width 512, context 512, 8-head bidirectional attention/RoPE, SwiGLU width 2048,
pre-RMSNorm and tied head throughout. Hybrid repeats attention → BiConv31 →
BiConv63; attention-only replaces every mixer with attention. No n-grams/GQA,
Hadamard or residual gates. Profiles use the real MASK tokenizer's **32,001 IDs**.

| Candidate | Mixers | Blocks | Parameters | Forward FLOPs/token |
| --- | --- | ---: | ---: | ---: |
| `hybrid-12` | attention + BiConv | 12 | 65,124,352 | 133,818,496 |
| `attention-12` | attention only | 12 | 67,024,896 | 146,016,384 |
| `hybrid-16` | attention + BiConv | 16 | 81,430,528 | 168,517,760 |
| `attention-16` | attention only | 16 | 83,806,208 | 183,765,120 |

Compare B at both depths, C within each mixer family. Parameter/FLOP budgets
naturally differ; this is not parameter matching. Seeds, data/order/noise
streams, tokenizer, batches, context, training budget and evaluation controls
match. Different module shapes mean equal seeds do not imply identical tensors.
FLOPs are forward estimates, not measurements.

`build_suite` freezes the complete matrix/controls/profiles into a SHA256
identity. Comparison refuses mismatched candidate, data/tokenizer, code/runtime,
final data/noise RNG state, token count or evaluation controls. No automatic
winner is inferred from toy results or missing evidence.

## Notebook and execution boundary

Open [kiwilm3-experiments.ipynb](../notebooks/kiwilm3-experiments.ipynb).
Installation, planning, preflight, training, comparison and generation are off
by default. Cells call tested Python modules, never embed training math. Existing
M2/M4 notebooks are unchanged. No VM is allocated or Drive mounted.

This notebook's adapter is **bounded CPU FP32**, constant-LR AdamW, no accumulation.
The separate [V3 TPU notebook](../notebooks/kiwilm3-tpu.ipynb) now provides
token-based scheduling/accumulation and native Drive state; live recovery is
still pending. Its new checkpoint format does not reinterpret these CPU suites.
Full-width defaults: 1,000 step attempts × batch 2 × context 512 = **1,024,000
input positions**, not a 50M/1B TPU recipe. Non-padding tokens/optimizer updates
are separately measured; empty-mask attempts skip the optimizer. LR 0.001,
weight decay 0.01, dropout 0.1; initialization/data seed 42, noise 142, validation
242. Optional `--controls controls.json` freezes reviewed overrides.

`--qualification` retains actual depths/kernels at width 16, SwiGLU 48, heads 2,
context 16 and 20 attempts. Implementation proof only: at tiny width the XXL
kernel cost can reverse full-width cost ordering. CPU times cannot be compared
with V2 TPU results. Before corpus selection, qualify a V3 GPU/XLA adapter,
schedule/accumulation and fresh-runtime Drive continuation, then freeze a
matched small corpus budget. This phase does not bypass the pending M2 gate.

## Windows PowerShell

Use an existing prepared SmolLM subset and its original tokenizer. The Windows
lock retains CUDA 13.2 torch; **this prototype still trains on CPU**.

```powershell
uv sync --locked --extra notebooks
$SourceTokenizer = (Get-Content data/smollm-smoke/metadata.json | ConvertFrom-Json).tokenizer.file
uv run --locked python scripts/prepare_kiwilm3_tokenizer.py --source "data/smollm-smoke/$SourceTokenizer" --output artifacts/kiwilm3/tokenizer.json
uv run --locked python scripts/run_kiwilm3_experiments.py plan --qualification --data-dir data/smollm-smoke --tokenizer artifacts/kiwilm3/tokenizer.json --suite runs/kiwilm3-bc/suite.json
uv run --locked python scripts/run_kiwilm3_experiments.py preflight --suite runs/kiwilm3-bc/suite.json --data-dir data/smollm-smoke --tokenizer artifacts/kiwilm3/tokenizer.json --candidate hybrid-12
```

For full-width bounded CPU execution omit `--qualification` and use a **new**
suite/output directory. Preparation/planning refuse overwrite. The MASK
tokenizer appends a real special token without rewriting existing packed IDs.

Only when ready, explicitly execute all four qualification candidates:

```powershell
foreach ($Candidate in @("hybrid-12", "attention-12", "hybrid-16", "attention-16")) {
    uv run --locked python scripts/run_kiwilm3_experiments.py train --suite runs/kiwilm3-bc/suite.json --data-dir data/smollm-smoke --tokenizer artifacts/kiwilm3/tokenizer.json --candidate $Candidate --output-dir "runs/kiwilm3-bc/$Candidate" --start-training
    if ($LASTEXITCODE -ne 0) { throw "Stopped: $Candidate failed" }
}
```

The same one-line Python commands work on macOS/Linux. Notebook cells are the
preferred Colab interface, on a manually selected **CPU** runtime for now.

### Verified local continuation

Fresh execution refuses an existing candidate directory. `--stop-after-step 4`
pauses/saves at step four without changing the locked total budget. Retain the
original checkpoint SHA256 independently, plus receipt and exact metrics prefix:

```powershell
uv run --locked python scripts/run_kiwilm3_experiments.py train --suite runs/kiwilm3-bc/suite.json --data-dir data/smollm-smoke --tokenizer artifacts/kiwilm3/tokenizer.json --candidate hybrid-12 --output-dir runs/kiwilm3-bc/hybrid-12 --start-training --resume-receipt runs/kiwilm3-bc/hybrid-12/step-000004.json --checkpoint-sha256 ORIGINAL_CHECKPOINT_SHA256
```

Exclusive step-qualified files preserve model/AdamW/progress/global dropout RNG
and independent data/noise RNGs. Receipts bind suite and metrics checksum.
Changed code, torch/Python, tokenizer/data or controls fail closed; old M4 runs
need their original reviewed package, not relaxed identity checks. No state
overwrite, automatic checkpoint deletion or implicit fresh fallback occurs.

An interrupted attempt may leave metrics beyond the checkpoint; resume refuses
silent truncation. Preserve the interrupted directory and deliberately recover
the verified prefix into a separate working copy. A stale `.training.lock`
requires verifying its process ended before explicitly removing that single
file. This is **not production Drive backup**: manually copy state/receipts/
metrics off a Colab VM before termination.

## Fixed-slot sampler

`sample_masked` preserves visible IDs/padding, proposes content tokens at MASK
holes, then commits a linear cumulative quota of highest-confidence proposals.
Committed IDs never remask. Confidence is the sampled ID's unfiltered
content-only probability; ties break by sequence index. Noise is remaining /
initial holes per row. A local sampling RNG avoids advancing training/global
RNG; model training mode is restored. This is inspired by iterative masked
decoding ([MaskGIT paper](https://arxiv.org/abs/2202.04200)), not its complete
image sampler or a validated diffusion ancestral sampler. Unweighted M4 CE is
not claimed to be a diffusion likelihood bound/ELBO.

`generate_slots` constructs BOS + prompt + MASK slots + suffix + EOS. Slots
count **BPE tokens**, not words. PAD/UNK/BOS/EOS/MASK cannot be sampled; EOS is
fixed, not an early-stop prediction. No length model, causal cache, remasking,
rollover or silent context truncation; no accelerator-optimized generation claim.

Dataset-free infilling after training:

```powershell
uv run --locked python scripts/generate_kiwilm3.py --checkpoint runs/kiwilm3-bc/hybrid-12/encoder.pt --tokenizer artifacts/kiwilm3/tokenizer.json --prompt "The " --suffix " cat." --output-slots 4 --steps 4 --temperature 0.8 --top-k 40 --seed 42
```

Use inference `encoder.pt`/`encoder.safetensors`, not optimizer state. Matching
tokenizer identity is enforced; optionally add a separately retained
`--checkpoint-sha256`. BF16 is default Safetensors storage; exact logits/greedy
sampler parity uses lossless FP32 `encoder.pt`. No pretrained V3 quality claim.

## Evaluation and comparison

Completed reports include fixed reconstruction CE/accuracy/counts at
15/50/90/100% noise, 20 validation batches per level by default. Health uses
fixed validation windows at full eligible-token masking: per-block gradient
norms, MLP-update/post-mixer RMS contribution, post-MLP/post-mixer amplification,
median/p90/p95/max and >1.5 flag rate. All FFNs are SwiGLU; the last/first MLP
gradient ratio is meaningful within that family. Audits do not consume training
RNGs. The sampler suite covers unconditional/continuation/infilling across seeds
42–46, records traces/latency and scores **generated segments** for word streaks
and repeated four-grams. Semantic quality still needs human/task review.

Reports include static profiles, CPU step throughput (exclude the first five
steps when available), null accelerator memory and exact FP32 inference parity.
Checkpoint/validation I/O is excluded from those step times.

```powershell
uv run --locked python scripts/run_kiwilm3_experiments.py compare --suite runs/kiwilm3-bc/suite.json --reports runs/kiwilm3-bc/hybrid-12/report-000020.json runs/kiwilm3-bc/attention-12/report-000020.json runs/kiwilm3-bc/hybrid-16/report-000020.json runs/kiwilm3-bc/attention-16/report-000020.json --output-dir examples/comparisons/kiwilm3-bc-qualification
```

The writer creates `suite.json`, `results.json` (samples/health) and `analysis.md`,
refuses overwrite and preserves caller-supplied relative paths. Qualification
is not corpus architecture evidence. `validate_comparison` can separately check
either B or C pair; this command validates all four. No fabricated result files
are checked in before actual runs arrive.

`cloze_scores` supports protected left/right context and equal-BPE-length
answer choices via parallel masked pseudo-scores, **not joint likelihood**.
It reports actual length/context limit and refuses truncation. A held-out
retrieval task spanning the full 512-token context still needs to be supplied;
no V2 score is silently reused. `evaluate_transfer` supports external prepared
data with identical original BPE IDs using fixed masked reconstruction, not
causal perplexity. Transfer/classification/retrieval scores stay null until
their relevant data/tasks are supplied. Review those and measured accelerator
efficiency before selecting a model; no automatic longer run is launched.

## Verification

Tests cover all four actual depths/schedules at tiny width, deterministic
confidence reveal/top-k/greedy sampling, preserved context/padding, numerical
failure cleanup, FP32 checkpoint/Safetensors parity, cloze scoring, fixed health
statistics, exact continuous-vs-resumed weights with dropout/data/noise restored,
safe CLI/notebook defaults and provenance refusals. Only bounded synthetic
optimizer steps execute during verification.

```bash
uv run --locked --extra browser --extra notebooks pytest -q
uv run --locked ruff check src scripts archive tests
uv run --locked --extra notebooks python scripts/build_kiwilm3_experiment_notebook.py
uv run --locked --extra notebooks jupyter nbconvert --execute --to html notebooks/kiwilm3-experiments.ipynb
```

Keep action flags false for notebook execution. Its default cells and generated
sources are tested; visual HTML/Colab presentation remains a manual check
because local-file browser access was previously rejected.

### Local verification record — 2026-10-05

**497 pytest tests pass**, including 41 M5/B/C cases, with two existing ONNX
deprecation warnings. Seven playground tests, Ruff/format checks, dependency
lock validation, `git diff --check` and wheel build pass. All seven default-off
notebook code cells execute with zero errors/only skip messages; saved outputs
are retained. HTML rendering succeeds; visual review remains unverified.
Frozen V2 branch/source and M2/M4 notebooks remain unchanged. No corpus/cloud
training, commit, push, Hub update or Drive mutation was performed.
