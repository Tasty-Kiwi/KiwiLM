# KiwiLM 2 Dense AdamW vs Muon 0.01: 250M-token analysis

## Verdict

Muon 0.01 is the first Muon setting that produces a meaningful late-training
quality win. At exactly 250M tokens it finishes 0.0937 validation loss ahead of
AdamW, reaches loss 3.8 with 8% fewer tokens, and is the only run to reach 3.75
or 3.70. It also substantially improves the controlled retrieval probe.

The tradeoff is systems efficiency and generation repetition. Observed median
throughput is 20.5% below AdamW, so the lower tokens-to-loss does not become a
wall-clock win on this implementation. Its repeated-four-gram rate is also
higher. Keep AdamW as the architecture-control optimizer; retain Muon 0.01 as
the leading Dense quality candidate and replicate it before making it the sole
final-run optimizer.

## Experiment validity

Both latest checkpoints serialize the same KiwiLM 2 Dense configuration, data
fingerprint
`d71d246e9af91a538515509c91df9ec1031e541fa51ef7009d0732e0a825c018`,
tokenizer, seed 42, data order, BF16 precision, 512-token context, batch size 8,
gradient accumulation 4, schedule, and exact 250,000,000-token budget. The
intentional difference is AdamW versus Muon with peak Muon learning rate 0.01;
embeddings, n-gram tables, norms, biases, and depthwise kernels remain on the
shared auxiliary AdamW optimizer.

## Validation and efficiency

| Metric | AdamW | Muon 0.01 | Muon relative to AdamW |
| --- | ---: | ---: | ---: |
| Final validation loss at 250M | 3.7757 | **3.6820** | -0.0937 |
| Final validation perplexity | 43.63 | **39.73** | 8.9% lower |
| Best validation loss | 3.7628 | **3.6693** | -0.0935 |
| Best validation perplexity | 43.07 | **39.23** | 8.9% lower |
| Median steady throughput | **27.74k tok/s** | 22.05k tok/s | 20.5% slower |
| Peak accelerator memory | 4.05 GB | **3.93 GB** | 3.1% lower |

Throughput is the median logged valid-token rate from 5M to below 250M tokens,
excluding the padded final batch. Hardware identity and foreground load are not
serialized, so this is observed-run evidence rather than a portable benchmark.

## Tokens-to-loss

| Target loss | AdamW tokens | Muon 0.01 tokens | Result |
| ---: | ---: | ---: | --- |
| 4.4 | 57.34M | **49.15M** | Muon 14.3% fewer |
| 4.0 | **122.88M** | 131.07M | Muon 6.7% more |
| 3.9 | 155.65M | 155.65M | Equal at 500-step resolution |
| 3.8 | 204.80M | **188.42M** | Muon 8.0% fewer |
| 3.75 | Not reached | **204.80M** | Muon only |
| 3.70 | Not reached | **229.38M** | Muon only |

Muon 0.01 is not uniformly more sample-efficient, but it clearly wins in the
low-loss regime that matters for extending this run. At the observed throughput
rates, reaching 3.8 still takes roughly 15.7% more wall time than AdamW. A fused
or compiled Muon update remains necessary for a systems-level win.

## Model health

Both final checkpoints pass finiteness, nonzero-gradient, residual-step, and
direct plus rollover cached-generation parity checks on the same validation
batch. Muon 0.01 has lower loss on that batch (4.141 versus 4.271), lower final
residual RMS (0.948 versus 2.322), and lower block-9 residual amplification
(1.401 versus 1.471). There is no sign that its loss improvement comes from a
numerical instability.

## Generation

The fixed generation suite contains 24 equal-budget rows across six prompts and
two sampling profiles.

- AdamW's repeated-four-gram rate is 0.194; Muon 0.01 rises to 0.245.
- The longest identical-word run is two for AdamW and five for Muon 0.01.
- Muon 0.01 improves over the earlier Muon 0.02 repetition rate of 0.272, but
  still has a severe focused persistent-goal loop.

Muon's validation advantage therefore does not yet translate into uniformly
better sampled prose. See the [generation report](generation-report.md) and
[raw rows](generation-results.jsonl).

## Retrieval at 512 tokens

| Distance | AdamW accuracy | Muon 0.01 accuracy | AdamW paired flips | Muon paired flips |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 25.0% | **75.0%** | 0.0% | **50.0%** |
| 128 | 25.0% | **62.5%** | 0.0% | **31.25%** |
| 256 | 31.25% | **37.5%** | 0.0% | **12.5%** |
| 384 | 25.0% | **31.25%** | 0.0% | **6.25%** |
| 448 | 25.0% | 25.0% | 0.0% | 0.0% |

| Aggregate metric | AdamW | Muon 0.01 |
| --- | ---: | ---: |
| Four-way candidate accuracy | 26.25% | **46.25%** |
| Paired flip accuracy | 0.0% | **20.0%** |
| Mean contextual logit lift | 0.810 | **2.164** |
| Target-vs-best-distractor margin | -0.446 | **-0.269** |

Muon 0.01 matches Muon 0.02's aggregate candidate accuracy and doubles its
paired-flip rate, though its mean lift and margin are weaker. Retrieval still
decays to chance at distance 448. See the [retrieval report](retrieval/report.md),
[raw rows](retrieval/results.jsonl), and [suite](retrieval/suite.json).

## Recommendation

1. Treat Muon 0.01 as the winning point in the tested Muon learning-rate sweep.
2. Keep AdamW for Dense-vs-Slim architecture controls because optimizer and
   architecture effects must remain separated.
3. Before a 500M-1B Dense commitment, replicate Muon 0.01 with another seed and
   prioritize a fused/compiled Muon update plus generation-quality checks.
4. Proceed now with the separately authorized H6/S4 residual-gate smoke; that
   experiment remains AdamW and does not depend on adopting Muon.

Machine-readable aggregates are in [summary.json](summary.json).
