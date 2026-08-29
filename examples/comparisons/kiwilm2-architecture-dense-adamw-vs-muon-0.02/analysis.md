# KiwiLM 2 Dense AdamW vs Muon 0.02: 250M-token analysis

## Verdict

Muon 0.02 is a retrieval-positive result, but it does not pass the original
optimizer adoption rule. At exactly 250M tokens it finishes 0.0058 validation
loss ahead of AdamW and produces substantially stronger counterfactual
retrieval. It is nevertheless 23.8% slower in observed steady training
throughput and needs more tokens than AdamW to reach nearly every useful loss
threshold in this run.

Keep AdamW as the default optimizer. Preserve Muon 0.02 as evidence that the
optimizer changes the learned representation, and test Muon 0.01 next.

## Experiment validity

The two latest checkpoints serialize the same KiwiLM 2 Dense model
configuration, data fingerprint
`d71d246e9af91a538515509c91df9ec1031e541fa51ef7009d0732e0a825c018`,
tokenizer, seed, data order, BF16 precision, 512-token context, batch size,
gradient accumulation, AdamW auxiliary learning rate, schedule, and exact
250,000,000-token budget. The intentional difference is the optimizer applied
to eligible dense matrices: AdamW versus Muon at peak learning rate 0.02.

The primary generation and retrieval artifacts use `latest.pt` for both
models. This matters because the best AdamW checkpoint occurs at 229.376M
tokens while the best Muon checkpoint occurs at 245.760M. Best-checkpoint loss
is reported for completeness, but unequal-token best checkpoints are not used
for the controlled behavioral comparison.

## Validation and efficiency

| Metric | AdamW | Muon 0.02 | Muon relative to AdamW |
| --- | ---: | ---: | ---: |
| Final validation loss at 250M | 3.7757 | **3.7699** | -0.0058 |
| Final validation perplexity | 43.63 | **43.37** | 0.6% lower |
| Best validation loss | 3.7628 | **3.7582** | -0.0046 |
| Best validation perplexity | 43.07 | **42.87** | 0.5% lower |
| Median steady throughput | **27.74k tok/s** | 21.14k tok/s | 23.8% slower |
| Peak accelerator memory | 4.05 GB | **3.93 GB** | 3.1% lower |

The throughput statistic is the median logged valid-token rate from 5M tokens
to below the final partial batch, excluding validation-adjacent measurements.
Both runs record CUDA and eager execution, but not GPU identity or power state,
so this is an observed run comparison rather than a universal hardware ratio.

Muon's small final loss advantage is not enough to establish a robust optimizer
win without replicated seeds. More importantly, it does not describe the
learning curve: Muon leads early, crosses behind AdamW around 49M tokens, stays
behind through almost the entire middle of training, and only catches up during
late learning-rate decay.

## Tokens-to-loss

| Target loss | AdamW tokens | Muon 0.02 tokens | Muon overhead |
| ---: | ---: | ---: | ---: |
| 4.4 | 57.34M | 65.54M | 14.3% |
| 4.0 | 122.88M | 172.03M | 40.0% |
| 3.9 | 155.65M | 196.61M | 26.3% |
| 3.8 | 204.80M | 229.38M | 12.0% |
| 3.78 | 229.38M | 229.38M | equal at 500-step resolution |

The mid-run plateau followed by improvement as the cosine schedule lowers the
Muon learning rate is consistent with 0.02 being too aggressive. It makes 0.01
the most informative next point in the planned sweep. A 0.04 run should remain
a bounded smoke test unless its early curve is unexpectedly strong and stable.

## Model health

Both final checkpoints pass finiteness, nonzero-gradient, bounded residual-step,
and cached-generation parity checks on the same validation batch.

Muon has a more even depth profile. Its deepest-to-first SwiGLU gradient ratio
is 0.909, compared with AdamW at 2.101, and its MLP output-RMS ratio is 2.00,
compared with AdamW at 8.10. Final residual RMS is 1.405 for Muon versus 2.435
for AdamW. This is not itself a quality score, but it shows that Muon's retrieval
gain does not come from numerical instability or exploding residual additions.

The current Muon optimizer performs five Newton-Schulz matrix iterations in
ordinary PyTorch for each selected dense matrix. Merely installing Triton does
not fuse this implementation, so optimizer work remains a plausible contributor
to the observed throughput penalty.

## Generation

The fixed suite contains 24 equal-budget rows: six prompts, two sampling
profiles, shared seeds, and 160 generated tokens per row. Both models remain
unreliable story generators and frequently lose ownership, goals, speakers, or
locations.

- AdamW has the higher mean distinct-unigram fraction, 0.418 versus 0.398.
- AdamW has the lower repeated-four-gram fraction, 0.194 versus 0.272.
- Neither model has an identical-word run longer than two, but both exhibit
  severe phrase-level loops.
- Muon's focused location continuation repeats variants of “the tree was
  covered with trees”; its focused entity and ownership continuations also
  loop heavily. AdamW's corresponding location sample repeatedly lays the tree
  on the ground.

Muon therefore has no general-generation advantage matching its retrieval
gain. See the [generation report](report.md) and
[raw generation rows](results.jsonl).

## Retrieval at 512 tokens

| Distance | AdamW candidate accuracy | Muon candidate accuracy | AdamW paired flips | Muon paired flips |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 25.0% | **56.25%** | 0.0% | **25.0%** |
| 128 | 25.0% | **56.25%** | 0.0% | **12.5%** |
| 256 | 31.25% | **37.5%** | 0.0% | 0.0% |
| 384 | 25.0% | **43.75%** | 0.0% | **12.5%** |
| 448 | 25.0% | **37.5%** | 0.0% | 0.0% |

| Aggregate metric | AdamW | Muon 0.02 |
| --- | ---: | ---: |
| Four-way candidate accuracy | 26.25% | **46.25%** |
| Paired flip accuracy | 0.0% | **10.0%** |
| Mean contextual logit lift | 0.810 | **2.901** |
| Target-vs-best-distractor margin | -0.446 | **-0.152** |

This is the clearest positive Muon result. It improves candidate accuracy by 20
percentage points, produces 3.58 times AdamW's contextual logit lift, and is the
only optimizer to complete any counterfactual flips. The gain survives the
equal-token correction: all values above use the final 250M checkpoints.

It is not yet reliable long-context binding. Aggregate margin remains negative,
paired flips are only 10%, and there are no successful flips at distances 256
or 448. The result should be treated as a strong signal for follow-up rather
than proof of solved retrieval.

See the [retrieval report](retrieval/report.md),
[raw retrieval rows](retrieval/results.jsonl), and
[retrieval suite](retrieval/suite.json).

## Recommendation

1. Keep AdamW as the final-run default because Muon 0.02 fails the specified
   tokens-to-loss criterion and loses substantially on observed throughput.
2. Preserve Muon 0.02 as the retrieval-positive optimizer baseline.
3. Run Muon 0.01 next under the same frozen 250M controls; use a shorter 0.04
   smoke only as a stability boundary.
4. Add a fused or compiled Newton-Schulz update path before using Muon when
   wall-clock efficiency matters.
5. Require another seed and external transfer evaluation before interpreting
   the small final validation advantage as a general quality improvement.

Machine-readable aggregates are in [summary.json](summary.json).
