# KiwiLM 2 Slim v3 H6/S4 residual-growth audit

## Verdict

The residual-growth signal reproduces. Launch the two gated 50M-token smoke
runs at alpha initializations 0.25 and 0.5.

H6/S4 block 9 exceeds the 1.5 residual-amplification threshold in 63 of 100
validation batches. Its p90 is 1.598, above the predeclared 1.5 trigger, and
its maximum is 1.654. The audit therefore authorizes gated training.

## Matched audit controls

Both checkpoints are exact 250M-token, seed-42 AdamW runs over data fingerprint
`d71d246e9af91a538515509c91df9ec1031e541fa51ef7009d0732e0a825c018`.
The audit uses seeds 141 and 142, 50 batches per seed, batch size 2, context 512,
CPU, and FP32 for both models. Training precision was BF16; audit precision is
intentionally matched between the compared checkpoints.

| Block-9 metric | Dense AdamW | H6/S4 ungated |
| --- | ---: | ---: |
| Median residual amplification | 1.439 | **1.526** |
| p90 | 1.514 | **1.598** |
| p95 | 1.517 | **1.606** |
| Maximum | 1.587 | **1.654** |
| Batches above 1.5 | 14/100 | **63/100** |
| Whole-batch health passes | 86/100 | **37/100** |

Dense also occasionally crosses 1.5, showing that the threshold is strict.
That does not invalidate the intervention: H6/S4 crosses it 4.5 times as often
and satisfies both predeclared trigger conditions.

## Health and localization

Both models remain finite, have nonzero gradients, and pass direct and rollover
cached-generation parity. H6/S4's minimum family gradient ratios are 0.415 for
Hadamard and 1.161 for SwiGLU, so the audit does not indicate a dead branch.

The issue is localized to the upper dense suffix. In H6/S4 block 9, the median
SwiGLU update RMS is 0.996 times the post-mixer residual RMS and the resulting
post-MLP residual is amplified by 1.526. This is exactly the branch targeted by
the bounded scalar gates; the H6/S4 schedule itself does not need to change.

## Decision

Train both gated variants from scratch on the frozen 50M smoke controls. Do not
retrain H7/S3 or ungated H6/S4. Compare alpha 0.25 and 0.5 against the existing
provenance-matched ungated smoke checkpoint using the promotion rules already
encoded by the evaluator.

The complete machine-readable evidence is in [audit.json](audit.json).
