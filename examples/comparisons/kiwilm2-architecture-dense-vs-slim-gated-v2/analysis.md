# KiwiLM 2 Dense vs Slim Gated v2: 250M-token architecture analysis

## Verdict

Dense wins the controlled 250M-token comparison. Slim gated v2 is now a real
efficiency implementation: it is substantially faster, smaller, and lighter on
accelerator memory. Its validation quality remains materially behind Dense,
however, and its deep Hadamard MLP gradients have weakened enough to fail the
trained-model health threshold.

Promote Dense to the next pretraining stage. Retain Slim gated v2 as the
structured-efficiency baseline, not as the primary final-run candidate.

## Experiment validity

The two checkpoints have identical serialized training configurations and the
same prepared-data fingerprint
`d71d246e9af91a538515509c91df9ec1031e541fa51ef7009d0732e0a825c018`.
Both completed exactly 250,000,000 tokens with AdamW, seed 42, BF16, batch 8,
gradient accumulation 4, a 512-token context, and 200 validation batches. Their
model configurations differ only in architecture and the gated-v2 Hadamard
variant. Every tensor in all four downloaded best/latest checkpoints is finite.

The architecture data binaries are not retained locally, so the logged
819,200-target validation was inspected rather than rerun. Generation and
retrieval use the frozen smoke tokenizer reused by architecture preparation;
checkpoint loading still validates the architecture-run data fingerprint.

## Quantitative comparison

| Metric | KiwiLM 2 Dense | Slim gated v2 | Slim relative to Dense |
| --- | ---: | ---: | ---: |
| Best validation loss | **3.7628** | 4.0159 | +0.2531 |
| Best validation perplexity | **43.07** | 55.47 | 28.8% higher |
| Total parameters | 64.25M | **40.69M** | 36.7% fewer |
| Dense non-embedding parameters | 31.09M | **7.53M** | 75.8% fewer |
| Estimated FLOPs/token | 99.12M | **52.24M** | 47.3% fewer |
| KV cache at 512, BF16 | 1.00 MiB | 1.00 MiB | equal |
| Median logged training throughput | 27.73k tok/s | **44.10k tok/s** | 59.0% faster |
| Peak accelerator memory | 4.05 GB | **3.21 GB** | 20.9% less |

The throughput median excludes compilation startup and the final partial batch.
The run records CUDA and compile state but not GPU identity, so these speed and
memory values describe this machine, not a universal device-independent ratio.
Dense ran eagerly; Slim selected compiled execution.

## Tokens-to-loss and compute-to-loss

Dense leads at every validation point. The loss gap rises as high as 0.305 near
65.5M tokens, narrows to roughly 0.25 by 130M tokens, then stays near that value
through 250M. There is no late trend suggesting that Slim is catching Dense.

Linear interpolation between fixed validation points puts Dense at Slim's best
4.0159 loss after approximately 113M tokens. At each model's measured median
throughput, reaching that loss takes roughly 1.13 hours of steady Dense training
versus 1.55 hours for Slim at its 245.76M-token best checkpoint. Dense also uses
slightly less estimated total compute to reach that quality: roughly 11.2 PFLOP
versus 12.8 PFLOP under the repository's multiply-add convention.

Slim therefore does not recover its validation deficit through its realized
throughput advantage in the quality range observed by this run.

## Model health

Dense passes every trained-model health check. Its deepest-to-first MLP gradient
ratio is 1.713, all block gradients are nonzero, and all outputs are finite. Its
last block is comparatively active: the final MLP output RMS reaches 1.396 and
the final residual RMS reaches 2.322. That deserves continued monitoring at a
larger token budget but does not trip the current bounded-step check.

Slim remains finite and its learned residual scales remain bounded, but its
deepest-to-first MLP gradient ratio falls to 0.0656 from 0.182 in the 50M smoke
probe. The structured MLP output RMS declines from 0.088 in block 0 to 0.027 in
block 9. Learned residual scales grow to 0.47-0.51 in the first two blocks while
falling to 0.175 in the final block.

The scaling fix prevented the original activation explosion, but the longer run
shows a different failure: deep Hadamard MLP contributions are being
progressively suppressed. Gated v2 is stable, but its channel-processing
capacity or parameterization remains inadequate at this depth.

## Generation

The fixed suite contains 24 rows: six prompts, two sampling profiles, and the
same sampling seeds for both best checkpoints.

- Both models frequently abandon named entities, ownership, promised goals, and
  locations. Neither is a reliable story generator at this scale.
- Dense has two severe repetition failures. The creative open-story sample
  repeats one word 146 times, and the focused location sample repeats an entire
  clause through most of its continuation.
- Slim has no identical-word run longer than four in this suite and has a lower
  repeated-four-gram fraction, 0.160 versus Dense at 0.296.
- Slim's better repetition statistics do not imply better semantic coherence;
  many outputs still loop around a subject or drift into unrelated educational
  prose.

See the [generation report](report.md) and
[raw generation rows](results.jsonl).

## Retrieval at 512 tokens

| Metric | KiwiLM 2 Dense | Slim gated v2 |
| --- | ---: | ---: |
| Four-way candidate accuracy | 25.0% | **30.0%** |
| Paired flip accuracy | 0.0% | **2.5%** |
| Mean contextual logit lift | **0.8754** | 0.6318 |
| Target-vs-best-distractor margin | -0.6516 | **-0.5451** |

Dense remains at chance with zero counterfactual flips. It produces a strong
short-distance contextual logit lift, but predicts `red` in 288 of 320 cases;
the context changes its logits without overcoming its candidate prior.

Slim predicts more than one candidate and succeeds on four of 160 complete
counterfactual pairs. Its 30% aggregate candidate accuracy is a weak positive
signal, not retrieval success: paired flips are only 2.5%, and accuracy returns
to chance at distances 384 and 448. Neither model demonstrates reliable
long-context binding.

See the [retrieval report](retrieval/report.md),
[raw retrieval rows](retrieval/results.jsonl), and
[retrieval suite](retrieval/suite.json).

## Recommendation

1. Promote Dense to the 500M-token run and preserve the fixed backbone.
2. Run the planned Dense Muon tokens-to-loss experiment before selecting the
   final optimizer.
3. Keep Slim gated v2 as the efficient structured baseline and preserve this
   checkpoint as evidence that the implementation realizes real speed gains.
4. Do not simply resume the completed Slim checkpoint for an equal-compute
   comparison: its 250M cosine schedule has already ended. A fair equal-FLOP
   test would be a fresh roughly 474M-token Slim run with a proportionally
   defined schedule.
5. Before another redesigned Slim run, address deep structured-MLP gradient
   attenuation or add more structured channel capacity without parameter
   matching Dense by depth.

Machine-readable aggregate results are in [summary.json](summary.json).
