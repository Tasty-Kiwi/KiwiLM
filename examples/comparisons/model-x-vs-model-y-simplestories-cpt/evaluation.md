# SimpleStories CPT evaluation

All measurements use story-safe validation, FP16, seed 42, batch size 64, and
500 batches on the GTX 1050 Ti. Each before/after pair evaluates the same
targets.

## Quantitative results

| Model | Checkpoint | Evaluation data | Loss | Perplexity | Targets |
| --- | --- | --- | ---: | ---: | ---: |
| X | TinyStories 750k | SimpleStories | 3.5964 | 36.4683 | 5,814,054 |
| X | SimpleStories CPT | SimpleStories | 2.5138 | 12.3518 | 5,814,054 |
| Y | TinyStories 750k | SimpleStories | 3.5746 | 35.6794 | 5,814,054 |
| Y | SimpleStories CPT | SimpleStories | 2.4808 | 11.9513 | 5,814,054 |
| X | TinyStories 750k | TinyStories | 1.8695 | 6.4849 | 5,697,554 |
| X | SimpleStories CPT | TinyStories | 2.5408 | 12.6901 | 5,697,554 |
| Y | TinyStories 750k | TinyStories | 1.8468 | 6.3392 | 5,697,554 |
| Y | SimpleStories CPT | TinyStories | 2.5465 | 12.7624 | 5,697,554 |

## Changes caused by CPT

| Model | SimpleStories perplexity | TinyStories perplexity |
| --- | ---: | ---: |
| X | -66.13% | +95.69% |
| Y | -66.50% | +101.33% |

After CPT, Model Y has 3.24% lower SimpleStories perplexity than Model X.
Model X has 0.57% lower TinyStories perplexity than Model Y after CPT.
If the two domains are weighted equally in log-loss space, geometric-mean
perplexity improves from 15.3783 to 12.5198 for X (-18.59%) and from 15.0392
to 12.3502 for Y (-17.88%). CPT therefore improves broad two-domain modeling
despite shifting substantial capacity away from TinyStories.

## Interpretation

The 50-million-target CPT run successfully transfers both architectures to
SimpleStories, but pure sequential CPT causes substantial TinyStories
forgetting. Model Y remains the better SimpleStories language model. Model X
retains a negligible TinyStories advantage after CPT, not enough to offset
Model Y's stronger target-domain result.

The fixed generation suite also shows that neither CPT checkpoint reliably
tracks all prompt entities, ownership, locations, or persistent goals. Model Y
is usually more structurally coherent, while both models still repeat phrases,
substitute characters, and abandon prompt facts.

The next controlled experiment should mix TinyStories replay into
SimpleStories CPT rather than increasing the pure SimpleStories budget.
A reasonable first mixture is 75% SimpleStories and 25% TinyStories by valid
targets, using the same 50-million-target budget and learning-rate schedule.
