# Model Y direct SFT vs CPT-to-SFT

All perplexity measurements use FP16, seed 42, and 500 validation batches.
Story datasets use batch size 64; the response-masked SFT dataset uses batch
size 8. Each pair evaluates identical targets.

## Perplexity

| Initialization | SFT v2 | TinyStories 750k | SimpleStories 250k |
| --- | ---: | ---: | ---: |
| Direct from TinyStories 750k | **5.7595** | **6.6368** | 37.5875 |
| SimpleStories CPT, then SFT v2 | 6.3614 | 7.5989 | **16.9978** |

CPT initialization changes perplexity by +10.45% on SFT validation, +14.50%
on TinyStories, and -54.78% on SimpleStories. With all three domains weighted
equally in log-loss space, geometric-mean perplexity improves from 11.2840 to
9.3662 (-17.00%).

## Instruction adherence

| Initialization | Profile | Adherence | Required words | Summary | Features | Entities | Repeat-4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct SFT | greedy | **59.6%** | **55.6%** | **45.8%** | **83.3%** | 50.0% | 15.0% |
| CPT then SFT | greedy | 52.7% | 50.0% | 41.7% | 66.7% | 50.0% | **12.0%** |
| Direct SFT | focused | 65.2% | 55.6% | 50.0% | 100.0% | 50.0% | 5.0% |
| CPT then SFT | focused | **69.0%** | **66.7%** | **54.2%** | 100.0% | 50.0% | **3.5%** |

The direct checkpoint is stronger under greedy decoding and on in-domain
perplexity. The CPT-initialized checkpoint is stronger under the recommended
focused sampling profile, is less repetitive in both profiles, and retains
far more SimpleStories competence. Neither checkpoint improves entity coverage
beyond 50%, and individual stories still contain substantial semantic errors.

## Verdict

CPT-to-SFT is the better broad model and the better focused-sampling candidate,
but it is not a strict replacement for direct SFT. Use the direct checkpoint
when deterministic loss or greedy decoding matters; use the CPT checkpoint
when broader language modeling and focused generation matter. A replay-mixed
CPT run remains the best route to seek a single checkpoint that dominates both.
