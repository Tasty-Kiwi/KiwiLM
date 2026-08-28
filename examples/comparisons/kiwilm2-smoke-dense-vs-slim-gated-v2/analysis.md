# KiwiLM 2 Dense vs Slim Gated v2: 50M-token smoke analysis

## Verdict

The gated Hadamard revision fixes the original Slim implementation's two most
important failures: it is numerically controlled, and the downloaded training
logs show an actual throughput advantage. It also narrows the loss gap relative
to the minimal Hadamard baseline. Dense remains clearly stronger in validation
loss, perplexity, and generation quality at the same 50M-token budget.

This is enough evidence to treat gated v2 as a viable 250M-token architecture
candidate, but not enough to replace Dense. The useful 250M experiment is still
a controlled Dense-vs-gated-Slim run, not a Slim-only promotion.

## Experiment validity

Both checkpoints use the same:

- SmolLM subset fingerprint
  `66b9899b879a5aba9eabdd4a40a54ab9ede62fdd1070f43be9b4c5b0e0e9714b`
- tokenizer SHA-256
  `4bcfc2d969a7a8c2285b364d709917d14e17a141e281fed9d7770db00329acf3`
- 50,000,000 training tokens, seed 42, and data order
- 512-token context, 8Q/2KV heads, batch 8, and accumulation 4
- AdamW, BF16 training, and the same learning-rate schedule

The original training logs used different validation sample sizes: 50 batches
for Dense and 20 for gated Slim. This report therefore reevaluates both best
checkpoints on the same 200 seeded FP32 batches: 819,200 validation targets per
model. The fixed evaluation ran locally on MPS; it is a loss comparison, not a
hardware-throughput benchmark.

## Quantitative comparison

| Metric | KiwiLM 2 Dense | Slim Gated v2 | Slim relative to Dense |
| --- | ---: | ---: | ---: |
| Fixed validation loss | **4.6904** | 4.9836 | +0.2933 |
| Fixed validation perplexity | **108.89** | 146.00 | 34.1% higher |
| Total parameters | 64.25M | **40.69M** | 36.7% fewer |
| Dense non-embedding parameters | 31.09M | **7.53M** | 75.8% fewer |
| Estimated FLOPs/token | 99.12M | **52.24M** | 47.3% fewer |
| KV cache at 512, BF16 | 1.00 MiB | 1.00 MiB | equal |
| Logged median training throughput | 27.18k tok/s | **32.63k tok/s** | 20.0% faster |
| Logged peak accelerator memory | 4.05 GB | **2.94 GB** | 27.4% less |

The training-log speed and memory rows are evidence from the downloaded runs.
The checkpoints do not record hardware identity or Slim's selected compile
runtime, so those figures should not be generalized to other devices. The low
first-step Slim throughput is consistent with compilation startup, but does not
prove which runtime was selected.

Slim's original 20-batch checkpoint evaluation reported 4.9036 at step 3,000,
then 4.9799 at the final step. The larger fixed evaluation scores that best
checkpoint at 4.9836, confirming that the step-3,000 dip was optimistic sampling
noise. Dense's 50-batch logged value, 4.6808, was much closer to its fixed result.

Against the retained minimal-v1 smoke result of 5.1202 versus Dense at 4.6722,
gated v2 reduces the historical loss gap by roughly 35%. This is material, but
it does not close the gap.

## Learning curve

| Step | Tokens | Dense validation loss | Slim gated-v2 validation loss |
| ---: | ---: | ---: | ---: |
| 250 | 4.10M | 6.3638 | 6.7143 |
| 500 | 8.19M | 5.8033 | 6.0476 |
| 750 | 12.29M | 5.4826 | 5.7305 |
| 1,000 | 16.38M | 5.2926 | 5.5152 |
| 1,250 | 20.48M | 5.0879 | 5.3933 |
| 1,500 | 24.58M | 4.9638 | 5.2579 |
| 1,750 | 28.67M | 4.8999 | 5.2130 |
| 2,000 | 32.77M | 4.8620 | 5.1000 |
| 2,250 | 36.86M | 4.7776 | 5.1123 |
| 2,500 | 40.96M | 4.7521 | 5.0896 |
| 2,750 | 45.06M | 4.6979 | 5.0945 |
| 3,000 | 49.15M | 4.7129 | 4.9036 |
| 3,052 | 50.00M | 4.6808 | 4.9799 |

These checkpoint-time values preserve each run's original evaluation size and
show trend rather than a perfectly matched absolute comparison. Dense leads at
every recorded point. Gated Slim is still improving near the end, but the curve
does not establish that more tokens will erase the gap.

## Model health

Both trained checkpoints pass the same seeded 512-token validation health probe:
finite logits, nonzero mixer and MLP gradients, bounded residual steps, bounded
Slim residual scales, and cached-generation parity.

- Gated Slim's maximum mixer-output RMS is 0.598, far below minimal v1's reported
  peak of 2.34. The scaling explosion is fixed.
- Its post-MLP residual ratios remain between 1.004 and 1.026. The learned
  residual scales remain controlled at 0.172-0.259, with mean 0.234.
- Its deepest-to-first MLP gradient ratio is 0.182. Gradients still attenuate
  with depth, but the ratio passes the 0.1 smoke threshold and materially improves
  on minimal v1's roughly 0.076 ratio.
- Dense remains healthier by this depth diagnostic, with a ratio of 1.273.

The health probe is one deterministic validation batch. It demonstrates that the
trained computation is well behaved; it is not a population estimate.

## Generation

The fixed report contains 24 rows: six prompts, two sampling profiles, and
identical seeds for each model.

- Both models drift away from named entities, ownership, goals, and locations.
- Dense has a lower mean maximum identical-word run, 1.75 versus Slim's 2.83,
  and a slightly higher distinct-unigram fraction, 0.481 versus 0.470.
- Gated Slim's worst identical-word run is 8, a large improvement over minimal
  v1's reported peak of 53.
- Gated Slim still has a catastrophic formatting repetition: one focused sample
  repeats `*` 178 times. Dense's largest consecutive lexical-or-punctuation run
  is 4.

Gating improves Slim's worst lexical repetition mode, but generation quality is
still below Dense and not yet reliable for either model.

See [generation report](report.md) and [raw generation rows](results.jsonl).

## Retrieval at the full context length

The updated suite uses 160 counterfactual pairs at distances 32, 128, 256, 384,
and 448, so it exercises the full 512-token configuration.

| Metric | KiwiLM 2 Dense | Slim Gated v2 |
| --- | ---: | ---: |
| Four-way candidate accuracy | 25.0% | 25.0% |
| Paired flip accuracy | 0.0% | 0.0% |
| Mean contextual logit lift | **0.0904** | 0.0766 |
| Target-vs-best-distractor margin | -0.6508 | **-0.5768** |

Both models remain exactly at four-way chance and never change their prediction
when the binding is counterfactually flipped. Dense has a larger contextual lift,
especially at distance 32. For both models, lift is near zero from distance 256
onward. Slim's less-negative margin does not outweigh chance accuracy and zero
paired flips.

See [retrieval report](retrieval/report.md),
[raw retrieval rows](retrieval/results.jsonl), and
[retrieval suite](retrieval/suite.json).

## Recommendation

Promote Dense unchanged to the 250M-token architecture run. Gated Slim v2 also
meets the previously stated promotion gate narrowly: it materially improves over
minimal v1 and realizes a logged throughput advantage. Run it at 250M only as a
controlled peer to Dense with identical data order and a shared, larger fixed
validation suite.

For that run, record hardware and compile-runtime metadata in the summary, use at
least 200 fixed validation batches throughout, and treat retrieval as diagnostic
until candidate accuracy and paired flips rise above chance. Do not promote the
minimal-v1 checkpoint; it remains the useful negative baseline.

Machine-readable aggregate results are in [summary.json](summary.json).
