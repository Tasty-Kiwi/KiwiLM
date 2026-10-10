# KiwiLM 3 fixed-corpus readout diagnostic

This bounded local experiment follows the
[conditioning/mixer isolation tests](../examples/comparisons/kiwilm3-lr-diagnostic-5m/analysis.md).
It tests contextual **memorization**, not corpus quality, generalization,
diffusion generation, or accelerator parity. It never allocates a VM, resumes
the saved optimizer, saves new weights, or changes production defaults.

## Run locally

```bash
uv run --locked python scripts/diagnose_kiwilm3_readout.py \
  --run-dir runs/colab/v3-gpu-hybrid-12-lr3e4-diagnostic-5m-recovered/downloads \
  --data-dir data/smollm-smoke \
  --steps 100 --seed 42 --threads 4 \
  --output runs/kiwilm3-diagnostics/readout-fixed-corpus-repeat.json
```

PowerShell can use the same command on one line with its local checkpoint
bundle path. Prepared data and tokenizer fingerprints must match the original
run. The script accepts verified native generations or completed collected
bundles, refuses existing output paths, and bounds each case to 200 updates
and PyTorch intra-op execution to at most 8 CPU threads. Completion is not a quality pass.

## Frozen task

Select four nonoverlapping 64-token windows deterministically from the first
1M training tokens: two with comma as target, two with the single token
` the` as target. The target is always at position 32 (zero-based). Exclude
windows containing controls or either target ID in visible positions. This
prevents class imbalance, target-position shortcuts and visible-label copying.
The report includes actual IDs, token windows, source offsets and task digest.
No validation or held-out generalization set is claimed for these four windows.

Only that target is masked and scored. The supplied noise level is fixed at
1/64 in every row and every condition. Batch is always four windows; each
case performs 100 updates / 25,600 repeated input tokens / 400 target exposures.
This is **not** the native corpus run's variable-noise distribution, 512-token
input length, effective batch, token budget or CUDA BF16 precision.

## Four fresh cases

| Head | Loss support | Purpose |
| --- | --- | --- |
| Native tied | Full clean vocabulary | Unchanged backbone/head and native dense selected-token CE |
| Untied clone | Full clean vocabulary | Isolate weight sharing, with exactly identical initial logits |
| Native tied | Only the two target IDs | Separate target discrimination from full-vocabulary probability allocation |
| Untied clone | Only the two target IDs | Complete the factorial control |

All cases start from fresh seed 42, not the collapsed weights. Untying clones
the embedding values into a separate trainable head parameter before optimizer
construction; it does not reinitialize the head or consume another RNG draw.
It adds parameters and is diagnostic-only: production transfers still enforce
weight tying. No initialization, mixer, conditioning or residual-scale change
is combined with these comparisons.

The CPU loop uses AdamW (single-tensor, weight decay 0.01), clip 1, dropout 0.1,
peak LR 0.0003, floor 0.00003, 10-update linear warmup and cosine decay to the
requested update count. This is a separate local overfit schedule, not a
reinterpretation of the original checkpoint's training contract. The full-loss
implementation calls the native `dense_reconstruction`; regression tests check
its loss and gradients against selected-logit CE. The two-target loss deliberately
changes prediction support and is not proposed as a production diffusion loss.

## Measurements and controls

At step zero, every 20 updates and completion, record both full-vocabulary and
two-target CE/accuracy regardless of training objective, full argmaxes, entropy,
target-pair probability mass and margins against the paired class and strongest
other vocabulary item. A balanced target-only unigram has CE `ln(2)`; this is
not a smoothed full-corpus unigram baseline.

Record pre-final-normalization RMS and centered energy (variation divided by
total mean-square energy), both over all input tokens and over the four target
representations. Variation is not semantic understanding: four distinct
contexts can simply be memorized.

- Context swap exchanges whole visible windows between opposite target classes
  at the same MASK slot, validity and noise, updating expected targets to match.
  This checks context/target pairing, not unseen semantic substitutions.
- Context erasure fills every row with identical MASK inputs while retaining
  the original slot/noise/targets. Its two-class accuracy must be 50%; full
  accuracy can also be 0% if another vocabulary item wins. This deliberately
  out-of-distribution control is not a calibrated high-noise denoising test.
- Full-task pass: 100% original/swapped full-vocabulary accuracy, original
  full CE below 0.1 and erased full accuracy 50%.
- Two-target pass: the corresponding restricted-CE/accuracy checks. Passing
  this alone does not imply a full-vocabulary pass.

The verified trained checkpoint is evaluated read-only on the same selected
task before fresh runs. Its result describes this balanced task only, not its
original full-validation accuracy. Hooks and evaluation preserve model mode
and RNG; fresh runs preserve the caller's global torch RNG. No model variant
is automatically selected or scheduled for training.

## Completed local result

The native tied/full-loss case and untied/full-loss control both pass the small
task. The two-target-only cases classify all examples correctly but fail the
full-vocabulary CE criterion. See the
[analysis](../examples/comparisons/kiwilm3-readout-fixed-corpus/analysis.md) and
[portable results](../examples/comparisons/kiwilm3-readout-fixed-corpus/results.json).
This supports retaining the native head while investigating corpus masking/noise
conditions next; it does not establish a model fix or authorize a cloud run.
