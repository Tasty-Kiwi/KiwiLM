# KiwiLM 2 Slim v3 residual-gate smoke analysis

## Verdict

Record no canonical winner and do not launch a 250M gated run yet.

Alpha 0.5 is the only candidate worth repeating. It fixes the residual-growth
signal, stays barely within the allowed validation-loss delta, passes every
health and generation check, and is much better than alpha 0.25. It formally
misses promotion because observed throughput is 94.79% of control rather than
the required 95%, and the training runner failed to record alpha trajectories.

The trajectory omission is an instrumentation defect, not a model failure.
The runner compared generated output labels such as
`kiwilm2-slim-v3-h6-s4-gate-050-adamw` against candidate names such as
`slim-v3-h6s4-gate-050`, so the callback was never attached. The runner now
derives gated status from the model configuration and has regression coverage.

## Provenance

All three latest checkpoints match the frozen smoke fingerprint
`66b9899b879a5aba9eabdd4a40a54ab9ede62fdd1070f43be9b4c5b0e0e9714b`.
They use the same H6/S4 backbone, tokenizer, seed 42, AdamW controls, BF16
training, 512-token context, and exact 50,000,000-token budget. The only model
difference is no upper-SwiGLU gate, alpha initialization 0.25, or alpha
initialization 0.5.

## Aligned validation and efficiency

The evaluator uses 200 fixed validation batches with seed 143.

| Metric | Ungated control | Alpha 0.25 | Alpha 0.5 |
| --- | ---: | ---: | ---: |
| Validation loss | **4.78289** | 4.84814 | 4.81245 |
| Loss delta vs control | — | +0.06525 | **+0.02956** |
| Perplexity | **119.45** | 127.50 | 123.03 |
| Median logged throughput | **28.13k tok/s** | 23.63k | 26.66k |
| Throughput vs control | 100% | 84.03% | **94.79%** |
| Peak accelerator memory | 3.719 GB | **3.209 GB** | 3.622 GB |

Alpha 0.25 clearly fails the maximum +0.03 loss delta and 95% throughput rule.
Alpha 0.5 passes loss by only 0.00044 and misses throughput by only 0.21
percentage point. Because the runs do not serialize GPU identity, power state,
or foreground load, that throughput miss is too small to interpret as a stable
architectural cost—but it is still a formal failure under the frozen rule.

Both gated models have 50,115,082 parameters and an estimated 70,994,944
FLOPs/token. Each adds exactly four scalar parameters to ungated H6/S4.

## Residual health

All three models have finite activations and gradients, nonzero gradients,
100/100 health passes, family gradient ratios above 0.1, and direct plus
rollover cached-generation parity.

| Block-9 residual amplification | Control | Alpha 0.25 | Alpha 0.5 |
| --- | ---: | ---: | ---: |
| Median | 1.366 | **1.263** | 1.316 |
| p90 | 1.400 | **1.293** | 1.346 |
| p95 | 1.421 | **1.305** | 1.363 |
| Maximum | 1.428 | **1.317** | 1.374 |
| Batches above 1.5 | 0/100 | 0/100 | 0/100 |

The gate works as intended at 50M. Alpha 0.25 suppresses the branch more, but
its quality cost is too high. Alpha 0.5 provides substantial residual control
while retaining more dense-branch capacity.

The final alpha-0.25 gates are 0.317, 0.329, 0.314, and 0.318. The final
alpha-0.5 gates are 0.564, 0.581, 0.563, and 0.574. All remain strictly within
the bounded interval. No intermediate trajectory was logged, so the frozen
trajectory requirement cannot be declared passed retroactively.

## Generation

The complete generation suite contains 90 rows: three models, six prompts, and
seeds 42 through 46.

| Metric | Control | Alpha 0.25 | Alpha 0.5 |
| --- | ---: | ---: | ---: |
| Maximum identical-word run | 3 | 3 | 3 |
| Repeated-four-gram rate | 0.00285 | **0.00178** | 0.00297 |

All candidates comfortably pass the generation rules. Alpha 0.5 adds only
0.00012 repeated-four-gram rate relative to control, far below the allowed
0.05 increase. See [generation/report.md](generation/report.md) and
[generation/results.jsonl](generation/results.jsonl).

## Retrieval

All three models remain exactly at four-way chance with zero paired flips at
every tested distance. Retrieval does not select a winner at this budget.

| Supporting metric | Control | Alpha 0.25 | Alpha 0.5 |
| --- | ---: | ---: | ---: |
| Candidate accuracy | 25% | 25% | 25% |
| Paired-flip accuracy | 0% | 0% | 0% |
| Mean contextual logit lift | **0.273** | 0.107 | 0.130 |
| Mean target margin | **-0.627** | -0.771 | -0.645 |

Alpha 0.5 preserves more of the control's weak contextual response than alpha
0.25, but none has learned reliable binding. See
[retrieval/report.md](retrieval/report.md) and
[retrieval/results.jsonl](retrieval/results.jsonl).

## Formal selection and next action

The unchanged selector records:

- alpha 0.25: ineligible for missing trajectories, excessive loss delta, and
  throughput below 95% of control;
- alpha 0.5: ineligible for missing trajectories and throughput below 95%;
- selected candidate: none.

Rerun only alpha 0.5 from scratch after syncing the telemetry fix, using a new
output directory and an otherwise identical frozen smoke configuration. Do not
use `--resume-existing`, because the completed checkpoint cannot reconstruct
its missing training trajectory. If the corrected run passes every rule, it
may advance to the planned fresh 250M confirmation.

Machine-readable evidence is in [summary.json](summary.json) and the frozen
decision is in [selection.json](selection.json).
