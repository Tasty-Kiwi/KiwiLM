# KiwiLM 2 Slim v3 H6/S4: 250M-token architecture analysis

## Verdict

H6/S4 is a successful architectural improvement over all-Hadamard Slim v2 and
is worth retaining as the efficiency candidate. At exactly 250M tokens it
recovers 63.6% of Slim v2's validation-loss gap to Dense, finishes only 0.0929
loss behind Dense, runs 40.9% faster, and remains 22.0% smaller and 28.4%
lower-FLOP than Dense.

It is not yet a clean final-run winner. Dense still reaches H6/S4's best loss
with slightly less estimated wall time and compute, Dense passes every repeated
health probe, and H6/S4 develops a borderline final-block residual-growth issue
plus one severe generation loop. Keep H6/S4; do not scale it to 500M–1B without
first resolving or explicitly accepting those two risks.

## Experiment validity

Dense, Slim v2, and Slim v3 H6/S4 use the same prepared-data fingerprint
`d71d246e9af91a538515509c91df9ec1031e541fa51ef7009d0732e0a825c018`,
32K tokenizer, seed 42, 512-token context, AdamW schedule, BF16 precision,
batch size 8, gradient accumulation 4, and 250,000,000-token budget. Their final
logged validations each contain 819,200 packed targets.

All evaluation in this comparison uses `latest.pt`, not each run's independently
selected `best.pt`. This preserves the exact equal-token comparison. Generation,
retrieval, and cached-generation parity were rerun against the locally retained
architecture dataset. All paths in the artifacts are repository-relative.

## Quality and efficiency

| Metric | Dense | Slim v2 H10 | Slim v3 H6/S4 |
| --- | ---: | ---: | ---: |
| Best validation loss | **3.7628** | 4.0159 | 3.8554 |
| Final validation loss | **3.7757** | 4.0310 | 3.8686 |
| Final perplexity | **43.63** | 56.32 | 47.87 |
| Parameters | 64.25M | **40.69M** | 50.12M |
| Dense non-embedding parameters | 31.09M | **7.53M** | 16.95M |
| Estimated FLOPs/token | 99.12M | **52.24M** | 70.99M |
| Median training throughput | 27.74k tok/s | **44.11k tok/s** | 39.09k tok/s |
| Peak accelerator memory | 4.05 GB | **3.21 GB** | 3.72 GB |

The hybrid improves final loss over Slim v2 by 0.1624. Its remaining gap to
Dense is 0.0929. H6/S4 is 40.9% faster than Dense while using 8.3% less peak
accelerator memory, but it is 11.4% slower and uses 16.0% more memory than Slim
v2. The additional four SwiGLUs are buying quality, not free speed.

H6/S4 trails Dense at every validation point. The gap narrows from 0.125 near
8M tokens to roughly 0.09 by 80–115M tokens, then remains near 0.09 through the
end. There is no late evidence that the hybrid is still catching Dense.

## Tokens-to-loss

Linear interpolation puts Dense at H6/S4's best 3.8554 loss after approximately
172.1M tokens. H6/S4 reaches that loss at its 245.76M-token best checkpoint.
Using each run's steady median throughput gives:

| Route to 3.8554 loss | Tokens | Estimated steady time | Estimated compute |
| --- | ---: | ---: | ---: |
| Dense | 172.1M | **1.724 h** | **17.06 PFLOP** |
| H6/S4 | 245.76M | 1.746 h | 17.45 PFLOP |

Those differences are small enough to be within real-system measurement noise,
but they do not show a training-efficiency win for H6/S4. Its present advantages
are parameter count, per-token cost, inference footprint, and throughput at an
equal token budget—not lower measured cost to reach this particular loss.

## Model health

The original Slim v2 failure is substantially repaired. Its deepest-to-first
Hadamard gradient ratio was 0.0656; H6/S4's remaining Hadamard family reaches
0.580. The upper SwiGLU gradient ratio is also healthy at 1.434. All block
gradients are nonzero, logits and activations are finite, learned Hadamard
residual scales remain bounded, and cached decoding passes both direct and
context-rollover parity.

The new issue is upper-stack growth. The SwiGLU output-RMS last/first ratio is
4.318, and block 9 raises residual RMS from 1.360 to 2.129 on the manifest probe,
a ratio of 1.565 against the configured 1.5 limit. Across five deterministic
validation batches:

| Model | Health passes | Mean maximum residual-step ratio | Range |
| --- | ---: | ---: | ---: |
| Dense | **5/5** | 1.414 | 1.355–1.471 |
| Slim v2 | 3/5 | **1.027** | 1.026–1.027 |
| H6/S4 | 3/5 | 1.496 | 1.437–1.565 |

Slim v2's failures come from its weak Hadamard gradient ratio, not residual
growth. H6/S4 fails the residual-step threshold on two batches, always at block
9. This is a threshold-level warning rather than an activation explosion, but
it repeated and grew relative to the 50M smoke checkpoint. Detailed probe data
is in [`health.json`](health.json).

## Generation

The 36-row suite uses identical prompts and sampling seeds across the three
exact-250M checkpoints. Dense has the best aggregate repetition statistics in
this run. H6/S4 has the lowest distinct-unigram fraction and the highest
repeated-four-gram fraction, driven partly by a creative open-story sample that
repeats one word 117 consecutive times.

| Model | Distinct unigram | Mean max word run | Worst word run | Repeated 4-gram |
| --- | ---: | ---: | ---: | ---: |
| Dense | **0.418** | **1.25** | **2** | **0.194** |
| Slim v2 | 0.374 | 1.42 | **2** | 0.268 |
| H6/S4 | 0.327 | 10.92 | 117 | 0.303 |

All three still drift semantically and repeat concepts, so this suite is a
failure-mode probe rather than a general generation benchmark. The H6/S4 loop
is nevertheless serious enough to retest across several sampling seeds before
larger training. See the [generation report](generation/report.md) and
[raw rows](generation/results.jsonl).

## Retrieval at 512 tokens

| Metric | Dense | Slim v2 | H6/S4 |
| --- | ---: | ---: | ---: |
| Four-way candidate accuracy | 26.25% | **37.50%** | **37.50%** |
| Paired-flip accuracy | 0.00% | 8.75% | **10.00%** |
| Mean contextual logit lift | 0.810 | 0.683 | **1.297** |
| Target-vs-best-distractor margin | -0.446 | -0.489 | **-0.360** |

H6/S4 provides the strongest contextual response and slightly improves paired
flips over Slim v2. It reaches 43.75% accuracy and 12.5% paired flips at both
distance 256 and 384. At distance 448 it falls back to 25% accuracy and zero
paired flips, so this is encouraging medium-range binding evidence rather than
reliable full-window retrieval. See the [retrieval report](retrieval/report.md),
[raw rows](retrieval/results.jsonl), and [suite](retrieval/suite.json).

## Recommendation

1. Keep **H6/S4** as the only Slim v3 schedule; H7/S3 remains dropped.
2. Keep Dense as the quality and training-efficiency baseline.
3. Before a 500M–1B H6/S4 run, reproduce generation across multiple seeds and
   address or deliberately waive the block-9 residual-step failure.
4. If the architecture changes to control the upper SwiGLU residual, treat it
   as a new variant and rerun a short controlled smoke test before final scale.
5. Do not resume this checkpoint as a nominal 500M run: its 250M cosine schedule
   has ended. Start a fresh run with the final token schedule.

Machine-readable aggregate results are in [`summary.json`](summary.json).
