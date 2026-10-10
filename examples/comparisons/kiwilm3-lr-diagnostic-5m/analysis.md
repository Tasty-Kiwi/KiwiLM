# KiwiLM 3: lower-LR diagnostic and collapse isolation

## Decision

**Do not promote the 5M checkpoint or restart the B/C 50M matrix yet.**
Lowering the LR scale did not resolve the corpus failure. Neither removing
noise conditioning nor quarter-scaling mixer updates prevented the early
comma-only mode in a bounded fresh CPU replay. No canonical fix is selected.

The original checkpoint is unchanged. No cloud allocation, checkpoint
continuation or production architecture change was performed for this audit.

## Provenance and scope

The user-run diagnostic completed exactly **5,000,000 nonpadding input tokens**
in **306 updates** on an NVIDIA L4 with BF16 autocast and FP32 master weights.
It used hybrid-12, width 512, SwiGLU width 2048, context 512, batch 8,
accumulation 4, AdamW, seed 42, data seed 42, noise seed 142, validation seed
242, peak LR 0.0003, floor 0.00003 and 1M-token warmup.

- Checkpoint SHA256:
  `b512e0f30b304f6090fab1caf1da4f6e55fcc2d4d0b8b3edcc6ee8ad86a22f0b`.
- Native job identity:
  `78a6ce7e67c759872aec469fbeb613d53c03bf599e2b9e582747ce31ffc050c5`.
- Data fingerprint:
  `66b9899b879a5aba9eabdd4a40a54ab9ede62fdd1070f43be9b4c5b0e0e9714b`.
- V3 tokenizer SHA256:
  `7fc721c3c0dfc77ca5786756b5821d292fad399f62298b46638fc83c8f680a8a`.

The collected bundle's core file hashes, completion marker, summary/spec,
checkpoint job/contract, tied finite FP32 weights and metrics prefix were
verified before evaluation. Its final Drive commit matches the checkpoint
hash. The local audit never restores the saved optimizer or RNG into training.

This is **not** a peak-only LR ablation against the older 50M run: the floor
and cosine horizon also changed. Nor is 5M vs 50M a matched-budget comparison.
All losses below are masked-token cross-entropy, not autoregressive perplexity.

## User-run GPU results

Fixed validation uses 20 batches of 8 windows per noise level, context 512.

| Noise level | Final masked CE | Masked accuracy | Selected targets |
| --- | ---: | ---: | ---: |
| 0.15 | 7.691744 | 4.134% | 12,215 |
| 0.50 | 7.666561 | 4.217% | 40,815 |
| 0.90 | 7.663634 | 4.255% | 73,579 |
| 1.00 | 7.663304 | 4.244% | 81,715 |

All masked argmax predictions in both fixed diagnostic windows, across all
four noise levels, are token 15 (`,`), starting at **step 25 / 409,600 input
tokens** and at every recorded diagnostic point thereafter. This is the first
sampled failure, not its exact onset. At step 25 the LR is still warming up,
approximately 0.00012288; reaching peak LR is not required for the symptom.

Reversing visible content at partial noise changes zero argmaxes at those
points. The fully masked probe contains no visible content, so its lack of
context response is only a negative control. These two windows do not prove
that every position in the validation corpus predicts commas.

On the monitored first window, first-attention residual amplification grows
from **1.455x to 8.204x**; the final pre-normalization residual RMS grows from
**0.1059 to 5.0056** and mean pairwise token cosine from **0.82053 to 0.99998**.
Both attention and MLP updates grow; convolution updates remain much smaller.
These observations associate homogenization with the failure, but do not
establish which mechanism initiated it.

Final 15%-noise loss minus the prefix-fitted unigram baseline is **-0.0827**
on window 0 and **+0.0198** on window 1. This baseline uses only the first 1M
training tokens, excludes protected-token counts, and adds one-count smoothing.
It is not a full-validation unigram comparison or a promotion metric.

## Frozen-checkpoint counterfactuals

Four CPU FP32 probes use the same validation window and 60 masked targets.
Noise conditioning is either unchanged or zeroed. All attention/convolution
mixer outputs are either unchanged or multiplied by 0.25. **SwiGLU outputs
are not scaled.** Hooks are temporary and leave weights/state dictionaries
unchanged. This tests reversibility at inference, not learning-time causation.

| Intervention | Masked CE | First mixer amplification | Final residual RMS | Comma argmaxes |
| --- | ---: | ---: | ---: | ---: |
| Control | 8.5716 | 8.205x | 5.0065 | 60/60 |
| Conditioning off | 8.5758 | 3.911x | 4.5603 | 60/60 |
| Mixers ×0.25 | 8.5661 | 2.469x | 2.9539 | 60/60 |
| Both | 8.5679 | 1.441x | 2.5840 | 60/60 |

None changes an argmax when visible content is reversed. Reducing activation
scale after training does not recover useful predictions in this probe.

## Fresh local factorial replay

Each case starts from the same fresh seed and runs **32 updates / 32,768 input
tokens** on the real prepared corpus using the native trainer, real 32,001-token
vocabulary, width 512, context 512, dropout and masking objective. The total
across four cases is 131,072 input tokens, not another 5M smoke.

CPU FP32 and batch 2/accumulation 1 replace CUDA BF16 and batch 8/accumulation 4.
The LR horizon and warmup are divided by 16 to match the original GPU run's
first 32 **update-level** learning rates exactly. This does not match its
effective batch, number of examples, token schedule or accelerator numerics.
Input/masked/eligible counts and LR match across the four local cases.
Measurements at updates 0, 8, 16, 24 and 32 do not consume training RNG.

Final measurements on the same one-window, 60-target probe:

| Intervention | Masked CE | First mixer amplification | Final residual RMS | Final token cosine | Comma argmaxes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control | 9.1382 | 2.593x | 1.2023 | 0.999661 | 60/60 |
| Conditioning off | 9.1260 | 1.198x | 1.0531 | 0.998064 | 60/60 |
| Mixers ×0.25 | 9.1842 | 1.185x | 0.8974 | 0.998570 | 60/60 |
| Both | 9.1415 | 1.013x | 0.5398 | 0.965014 | 60/60 |

All cases predict only commas at the first post-initialization measurement,
update 8, and subsequent measurements. Visible-content reversal changes no
argmaxes. Losses and gradients remain finite at the measured points; disabled
conditioning parameters intentionally have zero gradients.

An important distinction: at update 8, the **combined intervention has only
0.0324 final token cosine yet already predicts 60/60 commas**. Thus a common
argmax can precede severe residual homogenization. Comma-only argmax alone is
not proof of irreversible collapse, and reducing residual growth alone is not
a sufficient quality criterion. The replay is too short and too small to
establish whether any case could eventually recover or which has best loss.

In the combined case at update 32, the final block's MLP update RMS is 0.1073
versus 0.0010 for its convolution update. Mixer-only attenuation leaves dense
MLP branches active and cannot rule out a broader residual-update mechanism.

## Next diagnostic gate

Keep the production architecture and remaining full smoke matrix unchanged.
Before another long run, test **context-conditioned learning**, not only
argmax diversity or residual thresholds. The next useful bounded diagnostic is
a fixed-corpus overfit/readout test with balanced target categories and
context-swapping controls. Track target margins, loss relative to matched
unigram predictions, and centered/token-varying hidden-state energy as well
as raw RMS. This should distinguish an output-prior/readout bottleneck from
loss of usable context before selecting an initialization or residual change.

That diagnostic is a proposed next experiment, not an implemented model fix.
No further cloud run is launched or recommended as a confirmed solution.

Follow-up (2026-10-11): the
[balanced fixed-corpus readout diagnostic](../kiwilm3-readout-fixed-corpus/analysis.md)
has now completed. The unchanged native tied/full-loss model passes that small
memorization task; this does not repair the corpus checkpoint. The next gate
is isolating corpus masking/noise conditions, with no head change selected.

## Reproduction and artifacts

From the repository root, this command repeats the bounded local tests:

```bash
uv run --locked python scripts/isolate_kiwilm3_collapse.py \
  --run-dir runs/colab/v3-gpu-hybrid-12-lr3e4-diagnostic-5m-recovered/downloads \
  --data-dir data/smollm-smoke \
  --replay-steps 32 --threads 4 \
  --output runs/kiwilm3-diagnostics/lr3e4-5m-isolation-repeat.json
```

Use `--replay-steps 0` for frozen-checkpoint counterfactuals only. Replays are
bounded to 64 updates per case; output files are never overwritten. The
original local raw report is `runs/kiwilm3-diagnostics/lr3e4-5m-isolation.json`.
[summary.json](summary.json) preserves portable source identities, GPU
validation/monitor trajectories, local probe results, final per-block
statistics and training rows; its raw-report checksum permits later matching.
It contains no model weights or machine-specific host paths.

## Implementation verification

- Complete local suite: **627 passed, 1 skipped** (original V2 artifact absent),
  with two existing ONNX deprecation warnings.
- Focused diagnostic suite: **27 passed**, including bounded/exclusive CLI
  output, paired replay controls, deterministic replay, temporary differentiable
  hook cleanup, collected-bundle provenance and unchanged checkpoint/RNG tests.
- Ruff, changed-Python formatting, `git diff --check`, offline dependency-lock
  validation (124 packages), offline wheel build and retained V3 shell syntax
  checks passed. Both diagnostic modules are included in the wheel.
- Original `latest.pt`, `job.json` and `metrics.jsonl` still match the collected
  manifest after testing. The portable report matches the raw local result's
  SHA256 and contains no absolute host/VM paths.

These checks establish local implementation/integrity behavior only; they
are not accelerator parity or model-quality acceptance. Nothing was committed
or pushed.
