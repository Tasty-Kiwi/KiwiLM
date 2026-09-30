# KiwiLM 2 Dense: corrected v6e-1 50M smoke

2026-10-01. **The corrected tied-embedding smoke passed.** This establishes
single-device training, convergence, portable checkpoint reconstruction,
same-process reload, and CPU health/cache checks. It does not yet qualify a
production 1B run: fresh-VM continuation and on-TPU cached generation remain
unverified, and no matched idle GPU benchmark was run.

Source: `runs/colab/tpu-v6e1-muon-smoke-50m-tied`.
Hardware was confirmed as **V6E1** by Colab; real XLA matrix multiplication
passed before data upload. The run started from scratch; no old TPU or GPU
weights were imported, and Google Drive was not mounted or modified.

## Frozen experiment

Dense, 64,252,416 unique parameters, tied token embedding/output head,
512-wide backbone, 1536-wide SwiGLU, context 512, 8Q/2KV, fixed ten-block
attention/convolution schedule and hashed bigram/trigram tables. Seed 42,
batch 8, accumulation 4, BF16, Muon 0.01 and auxiliary AdamW 0.0003,
50M-token schedule with 1M warmup. Fifty fixed validation batches, seed 43,
every 500 updates and at completion. Final partial update masks excess targets.

- Data fingerprint: `66b9899b879a5aba9eabdd4a40a54ab9ede62fdd1070f43be9b4c5b0e0e9714b`.
- Tokenizer SHA: `4bcfc2d969a7a8c2285b364d709917d14e17a141e281fed9d7770db00329acf3`.
- Checkpoint SHA: `c8e945ac88610a1efd06768ecbb399097d0d89bc12be52944b8696efdbfc1425`.
- Torch 2.9.0+cpu / PyTorch-XLA 2.9.0 in an isolated Python 3.12 environment.

## Results

| Measurement | Result |
| --- | ---: |
| Tokens / optimizer updates | 50,000,000 / 3,052 |
| Final XLA BF16 validation loss | 4.291772 |
| Perplexity | 73.096 |
| Ordinary CPU FP32 reconstruction, same 50 batches | 4.291665 |
| Absolute portability loss difference | 0.000107 |
| Weighted steady throughput | 67,736 tokens/s |
| Median steady step throughput | 78,227 tokens/s |
| Training and periodic I/O | 16.55 minutes |
| Session through final diagnostics | 19.37 minutes |
| Effective session throughput | 43,019 tokens/s |
| Peak XLA allocated memory | 3.52 GiB |
| Compilations after warmup / training | 2 / 5 |
| Recorded `aten::` CPU fallbacks | None |

Steady throughput excludes the first 20 updates, final partial update and
periodic validation/checkpoint/portability-check time. It **includes** the
101.78-second recompilation on step 501 after reload. The first training update
took 102.17 seconds and is included in session time, not steady time. These
compile costs explain much of the gap between weighted and median throughput.
Session time includes final validation and CPU diagnostics but excludes
environment setup, transfers and final artifact packaging/download.

Validation decreased monotonically from 5.4825 at 8.192M tokens to 4.2918 at
50M. Every logged training loss and gradient norm is finite, gradients are
nonzero, and all 3,052 training steps are present. Exact curves, configurations,
health distributions and generation are preserved in [summary.json](summary.json).

## Weight-tying and portability proof

The [previous v5e-1 smoke](../kiwilm2-tpu-50m-smoke/analysis.md) accidentally
trained independent embedding/head matrices. This run re-establishes their
parameter identity **after XLA transfer and before optimizer construction**.
The head stays in auxiliary AdamW, not Muon; parameter count remains 64.25M.
Old smoke-v1 checkpoints and unequal tied matrices are rejected.

All 172 archive chunks were downloaded and reassembled; every extracted file's
size and SHA-256 matches the final manifest. Local ordinary checkpoint loading
with Torch 2.13.0 on CPU also preserves the tied identity and successfully
generates from the frozen tokenizer. The final checkpoint remains unchanged.
The launcher terminated its TPU session; a server-side session listing confirmed
no active Colab sessions remain.

The first checkpoint, step 500, passed a five-batch TPU/CPU comparison:
absolute loss difference 0.000120 and first-batch relative-logit RMS error
0.002722. Reloading model, optimizer and data-generator state on XLA preserved
the tied identity and pre/post-reload logits **exactly**. This is same-process
proof, not a fresh VM restart.

The final ordinary CPU reconstruction passed the full 50-batch comparison,
204,800 targets: absolute loss difference 0.000107 and first-batch relative-
logit RMS error 0.003449. Both are below the predeclared 0.02 tolerances.
The final saved embedding/head matrices are equal; no untied reconstruction
or checkpoint conversion is needed.

## Health and generation

CPU FP32 health: **50/50 batches pass**, seeds 141/142, 25 batches per seed,
batch 2, context 512. All activation and gradient checks are finite/nonzero.
The minimum deepest/first SwiGLU gradient ratio is 0.8005. Block-9 residual
amplification has median 1.3724, p90 1.3980, and maximum 1.4356, with zero
batches above 1.5.

CPU cached generation passes direct and actual full-context rollover parity:
maximum direct logit difference 0.00000381 and rollover difference 0.
On-XLA cached generation was not measured.

The seed-42, temperature-0.8, top-k-40, 64-new-token story sample begins:

> Once upon a time, in a cozy town named Harmonyville, there was a friendly little little creature called the "Citerous Cotny."

It continues with invented names and inconsistent prose, but no longer collapses
into one repeated token. This one short sample does **not** establish general
language quality, factuality, retrieval, or transfer performance. No broad
generation suite, retrieval benchmark or external evaluation was run here.

## Interpretation and next gate

The hardware result is promising and the previous portability failure is fixed.
The old v5e-1 loss/speed is not a matched control: it used an 80.64M-parameter
untied model and optimized its separate head with Muon. Historical GPU smoke
reports also differ in optimizer, precision and throughput aggregation. Do not
claim a controlled speedup or a quality winner from those comparisons.

Before a larger TPU run, verify fresh-VM continuation, on-XLA direct/rollover
cache parity, and a matched idle GPU control with the same data/optimizer/loop.
Use wall time including compilation, validation, checkpoint and backup overhead
when estimating a 1B budget. No 250M or 1B run was automatically started.

## Reproduce

With the frozen local smoke data available:

```bash
KIWILM_RESULT_DIR=runs/colab/tpu-v6e1-muon-smoke-50m-tied-repeat \
COLAB_SESSION_NAME=kiwilm2-tpu-v6e1-muon-smoke-50m-tied-repeat \
bash scripts/run_colab_kiwilm2_tpu_smoke.sh
```

The launcher explicitly requests v6e-1, recovers checksummed chunks and stops
only its own session afterward. Set `COLAB_TPU=v5e1` for a corrected v5e-1
control; do not resume the historical untied weights.

Local verification: **174 pytest tests pass**, Ruff, shell syntax,
`git diff --check`, offline locked dependency validation, and wheel build pass.
Regression tests simulate alias-breaking transfer, exclude the head from Muon,
reject unequal checkpoint matrices, detect a logit mismatch even when loss is
unchanged, and check exact CPU resume and portability. Real TPU evidence is
reported separately above.
