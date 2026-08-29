# KiwiLM 2 Slim v3 smoke analysis

## Decision

The hybrid-FFN hypothesis works: both Slim v3 schedules improve materially on
all-Hadamard Slim v2. H6/S4 is the stronger model-quality result, but the frozen
promotion rule selects **H7/S3** for the next 250M-token run.

H6/S4 lowers fixed validation loss by 0.0470 relative to H7/S3, clearing the
required 0.03 loss improvement. Its median throughput is only 1.036x Dense,
however, below the required 1.10x. H7/S3 is therefore selected by the specified
fallback rule. It runs at 1.439x Dense throughput in these logs.

This is a rule-driven selection, not evidence that H7/S3 is the intrinsically
better architecture. H6/S4 is the best v3 candidate on loss, generation
stability, family-specific health, and contextual-logit lift. Its throughput is
also unusually bimodal, starting around 39k tokens/s and later settling near
27–28k. An isolated same-machine throughput rerun is warranted before the final
500M–1B decision.

## Controlled results

All primary results below use the portable `latest.pt` checkpoints containing
exactly 50M training tokens. Provenance validation confirmed the same data
fingerprint, 32K tokenizer, seed 42, packed order, 512-token context, AdamW
settings, batch size, accumulation, precision, and token schedule.

The fixed validation pass reevaluated all models on the same CPU FP32 batches:
50 batches × 8 sequences × 512 targets = 204,800 targets per model.

| Model | Validation loss | PPL | Params | FLOPs/token |
|---|---:|---:|---:|---:|
| Dense | **4.6973** | **109.65** | 64.25M | 99.12M |
| Slim v2 H10 | 4.9871 | 146.51 | **40.69M** | **52.24M** |
| Slim v3 H7/S3 | 4.8432 | 126.87 | 47.76M | 66.31M |
| Slim v3 H6/S4 | 4.7962 | 121.05 | 50.12M | 70.99M |

H7/S3 recovers 0.1439 loss from Slim v2. H6/S4 recovers 0.1909 and finishes
only 0.0989 behind Dense. H6/S4 beat H7/S3 at every logged validation point,
so its advantage is not an end-of-run fluctuation.

| Model | Median tok/s | Versus Dense | Peak memory | Versus Dense |
|---|---:|---:|---:|---:|
| Dense | 27,163 | 1.000x | 4.05 GB | — |
| Slim v2 H10 | 32,630 | 1.201x | 2.94 GB | −27.4% |
| Slim v3 H7/S3 | **39,091** | **1.439x** | 3.13 GB | −22.8% |
| Slim v3 H6/S4 | 28,153 | 1.036x | 3.72 GB | −8.3% |

Throughput is the median after 1M tokens with validation-adjacent log windows
excluded. Dense did not use compilation, while the three Slim runs did, so the
figures describe the actual run configurations rather than an isolated-kernel
benchmark. H6/S4's p10–p90 range is especially wide (27.3k–39.2k), which is why
its regression should be investigated rather than treated as an architectural
constant.

Static efficiency remains meaningful. H7/S3 is 25.7% smaller and 33.1%
lower-FLOP than Dense. H6/S4 is 22.0% smaller and 28.4% lower-FLOP. Moving from
H7/S3 to H6/S4 adds 2.36M parameters.

## Health and caching

All four models passed the same-batch health probe and cached-generation parity.
The hybrid diagnostics compare depth trends within each MLP family, avoiding a
misleading final-SwiGLU versus first-Hadamard ratio.

| Model | Hadamard grad last/first | SwiGLU grad last/first | Hadamard RMS last/first | SwiGLU RMS last/first |
|---|---:|---:|---:|---:|
| Dense | — | 1.058 | — | 3.689 |
| Slim v2 H10 | 0.651 | — | 0.683 | — |
| Slim v3 H7/S3 | 0.257 | 1.160 | 0.710 | 2.478 |
| Slim v3 H6/S4 | **1.487** | 1.137 | **0.816** | 3.173 |

H6/S4 most clearly fixes the original deep-Hadamard suppression: the remaining
lower Hadamard family no longer shows a depthwise gradient collapse. H7/S3 is
healthy by the configured thresholds, but its Hadamard gradient ratio is weaker.

## Generation and retrieval

The generation suite contains 48 rows. All four 50M-token models still produce
weak prose, so these samples are diagnostic rather than a capability claim.
H6/S4 is the cleanest v3 result: no sample has more than two consecutive copies
of a word. H7/S3 has a severe 43-word repetition loop in one named-entity
sample, raising its mean maximum word run to 5.58. This is a substantive caveat
for its rule-based promotion.

The 512-token retrieval evaluation is inconclusive. Every model remains at
four-way chance (25%) with 0% paired-flip accuracy. H6/S4 has the largest mean
contextual-logit lift (0.273, including 1.181 at distance 32), but it does not
turn that directional response into correct candidate selection. No model has
yet demonstrated retrieval.

TinyStories and SimpleStories transfer evaluations were not run because the
prepared external datasets were unavailable locally.

## Recommendation

Proceed with the specified **H7/S3 250M-token architecture run** and retain
H6/S4 as the quality-favored challenger. Before interpreting speed as a durable
architectural property, rerun isolated H7/S3 and H6/S4 throughput measurements
under the same fresh process, compile settings, and GPU state. If H6/S4 clears
the 1.10x Dense gate there, its consistently better loss and qualitative health
would justify revisiting the frozen smoke selection before a final-scale run.

The exact Windows and Colab promotion commands are recorded in
[`selection.json`](selection.json). Supporting evidence is in
[`summary.json`](summary.json), [`provenance.json`](provenance.json), the
[`generation report`](generation/report.md), and the
[`retrieval report`](retrieval/report.md).
