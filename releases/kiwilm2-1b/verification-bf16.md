# Default BF16 release verification — 2026-10-02

New exports default to **BF16 storage**. Use `--dtype fp32` for an explicit FP32
bundle. Runtime arithmetic remains FP32 by default; native BF16 is an explicit
loader option. No upload, training, tagging, or checkpoint overwrite occurred.

Local draft: `artifacts/huggingface/KiwiLM-2-bf16`. The prior FP32 draft at
`artifacts/huggingface/KiwiLM-2` and original training checkpoint are unchanged.
The new `release.json` records all file checksums and conversion measurements.

- BF16 Safetensors size: **161,282,896 bytes** (161.3 MB), versus FP32
  **322,555,664 bytes** (322.6 MB), approximately half the storage.
- BF16 weights SHA256:
  `88a775621789bd2de37850b0403f57442827c9552a78936ade8f405b930b2908`.
- New package wheel SHA256:
  `f049a7edb4e75a8eb5c385fd1f6bd62ceba008e1af0ad28cf7c56136477027e1`.
- Exact logit parity with independently BF16-rounded reference weights at lengths
  1, 31, and 512, using FP32 CPU execution. This is **not** exact parity with the
  original FP32 weights.
- Maximum original-versus-rounded logit difference across those windows: 0.22989.
  At length 512, argmax agreement is 98.24%; rounding is not numerically invisible.
- Cached direct/rollover checks pass in FP32 execution (direct maximum difference
  0.000016212; rollover zero; established tolerance 0.002).
- All four sampled outputs match the rounded reference. All four differ from the
  original FP32 checkpoint. That is recorded, not treated as an export failure.

## Paired rounding-loss check

Frozen SmolLM validation split and tokenizer, seed 143, 32 batches of two
512-token sequences, **32,768 targets per model**, FP32 CPU arithmetic:

| Weights | Validation loss | Perplexity |
| --- | ---: | ---: |
| Original FP32 | 3.583916895 | 36.014329 |
| BF16-stored weights, executed in FP32 | 3.583986886 | 36.016850 |

Loss increases by **0.000069991**, below the builder's declared 0.01 rounding
guard. This 32-batch paired check is not the published 200-batch protocol and
does not replace its headline loss/perplexity. Native BF16 execution performance
and large-model native cached parity were not measured by this check.

Verification: **284 tests passed**, including both BF16 and FP32 round trips for
all six tested Dense/Slim/hybrid configurations, default CLI dtype, explicit native
loading, tied weights, and legacy FP32 bundle loading. Ruff, locked dependency
validation, wheel build, and `git diff --check` passed.
The newly installed wheel also reproduces the BF16 draft's seed-42/cache-auto
output without a checkout source or prepared data on the import path; dependencies
were reused from the existing environment, not freshly resolved.

## Private draft upload — 2026-10-02

Uploaded all ten release files to **Tasty-Kiwi/KiwiLM-2**, commit
`53d6abcb4d11c8a98936c19d55d2b741a99d7602`, with explicit owner authorization.
Privacy was verified before and after upload and again after fresh download.
All files were downloaded into a newly created empty cache, checked against the
local bundle's checksums/sizes, and the inference manifest passed. The pinned Hub
loader reproduced the recorded seed-42/cache-auto/32-token BF16-reference sample
exactly using FP32 CPU execution. See `private-upload.json` for the receipt.

Source commit/tag freeze remains pending; this is a **private draft**, not a
source-frozen public release. The uploaded `release.json` intentionally remains
the unchanged pre-upload build record. No source commit/tag, Git push, training,
checkpoint overwrite, or visibility change to public occurred. The owner alone
will make the repository public.
