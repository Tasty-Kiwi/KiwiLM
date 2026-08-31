# KiwiLM 2 Slim v3 residual-gate smoke analysis

## Verdict

The corrected alpha-0.5 run passes every frozen promotion criterion except peak
memory. The unchanged selector therefore records no canonical winner. The user
subsequently made an explicit manual decision to waive that single measurement
and promote alpha 0.5 to a fresh 250M confirmation.

Scientifically, alpha 0.5 is otherwise a successful intervention. It reproduces
the original model's loss, lowers block-9 residual amplification, records a
healthy seven-point alpha trajectory, runs faster than the historical control,
and passes all gradient, parity, and generation checks. Alpha 0.25 remains
dropped.

## Provenance

All three latest checkpoints match the frozen smoke fingerprint
`66b9899b879a5aba9eabdd4a40a54ab9ede62fdd1070f43be9b4c5b0e0e9714b`.
They use the same H6/S4 backbone, tokenizer, seed 42, AdamW controls, BF16
training, compiled runtime, 512-token context, and exact 50,000,000-token
budget. The corrected alpha-0.5 checkpoint is:

`runs/kiwilm2-slim-v3-residual-gate-050-smoke-rerun/kiwilm2-slim-v3-h6-s4-gate-050-adamw/latest.pt`

## Aligned validation and efficiency

The evaluator uses 200 fixed validation batches with seed 143.

| Metric | Ungated control | Alpha 0.25 | Alpha 0.5 corrected |
| --- | ---: | ---: | ---: |
| Validation loss | **4.78289** | 4.84814 | 4.81203 |
| Loss delta vs control | — | +0.06525 | **+0.02914** |
| Perplexity | **119.45** | 127.50 | 122.98 |
| Median logged throughput | 28.13k tok/s | 23.63k | **30.67k** |
| Throughput vs control | 100% | 84.03% | **109.05%** |
| Peak accelerator memory | 3.719 GB | **3.209 GB** | 3.920 GB |
| Memory vs control | 100% | 86.30% | **105.42%** |

Alpha 0.5 passes the maximum +0.03 loss delta by 0.00086 and comfortably
passes throughput. It fails the maximum +2% memory rule by 3.42 percentage
points. Both gated models have 50,115,082 parameters and an estimated
70,994,944 FLOPs/token; the gate itself adds exactly four scalar parameters.

The memory traces are not clean architectural measurements. The control's CUDA
peak rises from 3.194 GB to 3.719 GB after its first validation. The corrected
run rises from 3.475 GB to 3.537 GB after the first diagnostic and to 3.920 GB
after the second. PyTorch records cumulative `max_memory_allocated`, so compiled
validation and diagnostic graphs contribute to the run peak. This makes the
5.4% delta plausibly instrumentation- or allocator-driven, but the frozen rule
uses the observed run peak; it is not waived after seeing the result.

## Residual health and alpha trajectory

All three models have finite activations and gradients, nonzero gradients,
100/100 health passes, family gradient ratios above 0.1, and direct plus
rollover cached-generation parity.

| Block-9 residual amplification | Control | Alpha 0.25 | Alpha 0.5 corrected |
| --- | ---: | ---: | ---: |
| Median | 1.366 | **1.263** | 1.311 |
| p90 | 1.400 | **1.293** | 1.348 |
| p95 | 1.421 | **1.305** | 1.358 |
| Maximum | 1.428 | **1.317** | 1.371 |
| Batches above 1.5 | 0/100 | 0/100 | 0/100 |

The corrected alpha-0.5 trajectory contains every required validation point:

| Step | Tokens | Block 6 | Block 7 | Block 8 | Block 9 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 8.19M | 0.511 | 0.517 | 0.514 | 0.513 |
| 1000 | 16.38M | 0.527 | 0.538 | 0.532 | 0.534 |
| 1500 | 24.58M | 0.541 | 0.556 | 0.546 | 0.550 |
| 2000 | 32.77M | 0.552 | 0.568 | 0.557 | 0.562 |
| 2500 | 40.96M | 0.558 | 0.575 | 0.563 | 0.568 |
| 3000 | 49.15M | 0.562 | 0.579 | 0.567 | 0.572 |
| 3052 | 50.00M | 0.562 | 0.579 | 0.567 | 0.572 |

Every gate remains strictly within `(0, 1)` and changes smoothly. The residual
gate is functioning as intended without saturating.

## Generation

The complete generation suite contains 90 rows: three models, six prompts, and
seeds 42 through 46.

| Metric | Control | Alpha 0.25 | Alpha 0.5 corrected |
| --- | ---: | ---: | ---: |
| Maximum identical-word run | 3 | 3 | 3 |
| Repeated-four-gram rate | 0.00285 | **0.00178** | 0.00315 |

Alpha 0.5 adds only 0.00030 repeated-four-gram rate relative to control, far
below the allowed 0.05 increase. See [generation/report.md](generation/report.md)
and [generation/results.jsonl](generation/results.jsonl).

## Retrieval

All three models remain exactly at four-way chance with zero paired flips at
every tested distance. Retrieval does not select a winner at this budget.

| Supporting metric | Control | Alpha 0.25 | Alpha 0.5 corrected |
| --- | ---: | ---: | ---: |
| Candidate accuracy | 25% | 25% | 25% |
| Paired-flip accuracy | 0% | 0% | 0% |
| Mean contextual logit lift | **0.273** | 0.107 | 0.196 |
| Mean target margin | **-0.627** | -0.771 | -0.674 |

Alpha 0.5 preserves more contextual response than alpha 0.25, but none has
learned reliable binding. See [retrieval/report.md](retrieval/report.md) and
[retrieval/results.jsonl](retrieval/results.jsonl).

## Formal selection and next action

The unchanged selector records:

- alpha 0.25: ineligible for missing historical trajectories, excessive loss
  delta, and throughput below 95% of control;
- corrected alpha 0.5: ineligible only because peak memory exceeds control by
  more than 2%;
- selected candidate: none;
- next stage: none.

Do not repeat another 50M training run. The manual promotion is recorded in
[manual-promotion.json](manual-promotion.json), bound to the exact summary and
selection checksums, and authorizes only alpha 0.5 on the 250M architecture
dataset. The frozen selector remains a no-winner result; the override does not
rewrite or relax it retroactively.

Machine-readable evidence is in [summary.json](summary.json) and the frozen
decision is in [selection.json](selection.json).
