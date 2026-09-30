# TPU v5e-1: completed 50M smoke, architecture qualification failed

Analyzed 2026-09-30. Source: `runs/colab/tpu-v5e1-muon-smoke-50m`.
**The hardware is promising, but this is not a valid tied-embedding Dense run.
Do not promote these weights or the current TPU worker to a 1B run.**

The downloaded checkpoint completed exactly 50,000,000 tokens at step 3,052.
The checkpoint and all five accompanying files match the final artifact
manifest's sizes and SHA-256 digests. Data and tokenizer match the frozen smoke:

- Data: `66b9899b879a5aba9eabdd4a40a54ab9ede62fdd1070f43be9b4c5b0e0e9714b`.
- Tokenizer: `4bcfc2d969a7a8c2285b364d709917d14e17a141e281fed9d7770db00329acf3`.
- Checkpoint: `60b08fac0d61c7e1066f9db07180ff9da883f4b1b74837ff42aa5221cb7fb11e`.

## Hardware findings

| Measurement | Result | Scope |
| --- | ---: | --- |
| Weighted steady throughput | 47,871 tokens/s | First 20 steps and final partial step excluded; periodic I/O excluded |
| Median steady step throughput | 51,652 tokens/s | Same window; not a wall-time throughput |
| Training plus periodic I/O | 22.45 minutes | Before final validation/checkpoint/diagnostics |
| Session through final diagnostics | 24.18 minutes | Excludes setup, transfer and final packaging |
| Effective throughput over that session | 34,464 tokens/s | Includes evaluation/checkpoint/CPU diagnostic overhead |
| TPU peak allocated memory | 3.86 GiB | XLA allocator accounting, not comparable directly with CUDA peak memory |
| Compilations after warmup / training | 2 / 4 | Recorded counters |
| Recorded `aten::` CPU fallbacks | None | Does not mean there is no host synchronization or data transfer |

The large discrepancy between median and weighted throughput is mostly step
501: 83.72 seconds immediately after the same-process checkpoint reload. The
initial step took 85.70 seconds and is excluded from steady throughput. Ordinary
steps remain near 51–52k tokens/s, without an obvious late-run slowdown.

All 3,052 logged losses and gradient norms are finite; gradients are nonzero.
Fixed XLA BF16 validation (seed 43, batch 8, context 512, 50 batches) improved
from 5.3369 at 8.192M tokens to 4.1640 / perplexity 64.33 at 50M.
This measures the **untied model that actually ran**, not canonical Dense.

## Critical finding: device transfer broke weight tying

The serialized configuration says `tie_embeddings=true`, but the checkpoint's
`token_embedding.weight` and `lm_head.weight` are different tensors with
different values:

| Weight evidence | Result |
| --- | ---: |
| Maximum absolute embedding/head difference | 0.405909 |
| Embedding RMS | 0.024039 |
| Output-head RMS | 0.040734 |
| Intended tied Dense parameters | 64,252,416 |
| Actual untied parameters | 80,636,416 |
| Extra output-head parameters | 16,384,000 (+25.5%) |

`KiwiLM2LM` ties weights in its CPU constructor. The TPU probe then calls
`.to(runtime.device)` before choosing optimizer groups, without re-establishing
the tie. PyTorch/XLA explicitly documents that shared parameters must be tied
**after** moving modules to XLA; otherwise independent copies are made.
[PyTorch/XLA tensor quirks](https://docs.pytorch.org/xla/release/2.2/index.html#xla-tensor-quirks).

This affects optimization as well as parameter count: the detached output head
becomes a dense linear matrix and enters Muon, whereas the intended shared
embedding/head belongs to auxiliary AdamW. The saved optimizer has 59 Muon
parameters and 36 auxiliary parameters; their shapes match the reconstructed
untied model exactly. Their element counts are 47,316,992 and 33,319,424.
This is not merely lost storage aliasing during checkpoint serialization:
the actual weight values diverged during training. The earlier 200-step TPU
checkpoint also has unequal matrices, while the existing 500M GPU checkpoint
retains equal embedding/head weights.

The ordinary loader reconstructs the configured tied model. Loading both
different matrices into that shared parameter leaves the **head matrix** in
both positions, replacing the trained input embedding. That accounts for the
bad CPU validation and token repetition. An explicitly untied diagnostic
reconstruction preserves both original matrices and restores sensible loss.
It changes no checkpoint files and is not a repair or a canonical Dense export.

The full aligned comparison confirms the diagnosis:

| Reconstruction | Device / precision | Validation loss | Perplexity |
| --- | --- | ---: | ---: |
| Model that trained, worker measurement | XLA / BF16 | 4.164010 | 64.329 |
| Ordinary configured tied model | CPU / FP32 | 15.453753 | 5,146,115 |
| Preserve separate embedding/head | CPU / FP32 | 4.163875 | 64.320 |

All three use seed 43, batch 8, context 512 and the same 50 validation batches.
The untied reconstruction differs from the XLA loss by only 0.000135; the
ordinary tied reconstruction is not evaluating the network that trained.

The accompanying [audit.json](audit.json) records aligned CPU FP32 losses on all
50 fixed validation batches (204,800 next-token targets), a CPU BF16 cross-check
on the first five batches (20,480 targets), normal and untied reconstructions,
full-512-token cache parity, and paired cached and uncached generation samples.
CPU BF16 here is a numerical cross-check, not a performance benchmark, and its
five-batch loss is not directly compared with the XLA 50-batch aggregate.
Reproduce from the repository root:

```bash
uv run python examples/comparisons/kiwilm2-tpu-50m-smoke/audit.py
```

## Health and generation caveats

The worker reported 50/50 healthy CPU batches and direct/rollover cache parity.
However, those diagnostics used the ordinary tied loader, so they describe the
**incorrectly reconstructed model**, not the trained XLA network. They cannot
qualify the checkpoint. Finiteness and cached/uncached equivalence alone do not
establish numerical portability or sensible language generation.

The worker's sample repeats `time` for the full continuation. Locally both
cache-off and cache-auto reproduce that behavior, excluding an ordinary cache-
only explanation. The diagnostic reconstruction's outputs and repetition
measurements are saved separately; they must not be presented as outputs from
the normal KiwiLM generation CLI.

The local audit generated 12 outputs: three prompts, two reconstructions, and
cache off/auto, using seed 42, temperature 0.8, top-k 40 and 64 new tokens.
Cached and uncached text matches exactly in every pair. Both reconstructions
pass direct and actual context-rollover parity at context 512 on CPU FP32.

| Prompt | Ordinary tied loading | Untied diagnostic reconstruction |
| --- | --- | --- |
| `Once upon a time` | 65 consecutive `time` words including prompt | A readable Westville/Goliath story, with semantic inconsistencies |
| `The capital of France is` | 65 consecutive `is` words including prompt | Fluent but incorrect political/geographic claims; not `Paris` |
| `Explain why the sky is blue.` | Repeated punctuation | Confused color explanation; not a correct scattering explanation |

The untied samples have no repeated four-grams and longest consecutive-word
run 1, but this tiny sample does not establish general generation quality.
The punctuation-only collapse also demonstrates a limitation of word-run and
word-four-gram metrics: they can miss severe non-word degeneration.

Same-process checkpoint reload passed on XLA at step 500, but only step/token
state was asserted. No pre/post-reload logit equality, XLA-to-CPU loss equality,
or fresh-VM restart equivalence was established. On-XLA cached-generation parity
was not measured.

## Historical GPU comparison: context only

The [original Dense smoke](../kiwilm2-smoke-dense-vs-slim/analysis.md) used AdamW
on a T4 with FP16 and reported about 22.3k steady tokens/s, loss 4.6722 and
60.9 minutes total. A [later archived Dense control](../kiwilm2-smoke-dense-vs-slim-gated-v2/summary.json)
reports a median 27.2k tokens/s and fixed validation 4.6904 under a different
evaluation seed/budget. Its runtime identity was not recorded. Those weights
are no longer available locally for freshly aligned evaluation.

The TPU's nominal 47.9k is about 2.15 times the historical T4 rate. This is
**not a matched hardware speedup**: optimizer, precision, runtime versions,
throughput aggregation and actual parameterization differ. Likewise, the TPU's
lower reported loss must not be claimed as a Dense quality improvement. No
architecture or hardware winner is selected.

## Next gate before larger training

1. Re-tie embedding/head after device transfer, before optimizer construction.
   Assert parameter identity, expected 64.25M count, and head exclusion from
   Muon. Include equivalent guards for resume and any subsequent transfers.
2. Fail checkpoint portability qualification if a tied configuration contains
   unequal matrices. Compare fixed CPU and XLA logits/loss before and after a
   portable reload, rather than only asserting metadata.
3. Run a short corrected TPU qualification, including generation and an actual
   restart test; then repeat the controlled 50M smoke if that passes.
4. Only then compare idle, matched GPU/TPU runs and consider 1B. The current
   checkpoint can be retained as a separately labeled untied diagnostic baseline,
   but there is no exact conversion into the intended tied model.

No remote allocation, training, checkpoint mutation, or worker fix was performed
for this analysis. Local verification: 170 pytest tests passed, Ruff passed,
launcher shell syntax passed, and `git diff --check` passed. These automated
tests did not detect the real XLA aliasing behavior; local CPU success is not
TPU qualification.
