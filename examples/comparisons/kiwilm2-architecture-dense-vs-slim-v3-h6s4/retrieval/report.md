# Context Retrieval Benchmark

160 counterfactual pairs / 320 bound cases; seed 42; context window 512.

Candidate accuracy scores the four-way cloze; paired flip requires both counterfactual variants to select their changed target. Margin is target minus best distractor logit, and lift is bound-target minus no-binding control logit.

## Overall

| Model | Candidate accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| KiwiLM 2 Dense | 26.25% | 0.00% | -0.4465 | 0.8098 |
| KiwiLM 2 Slim Gated v2 | 37.50% | 8.75% | -0.4894 | 0.6828 |
| KiwiLM 2 Slim v3 H6/S4 | 37.50% | 10.00% | -0.3604 | 1.2970 |

## KiwiLM 2 Dense

### By distance

| Distance | Accuracy | Paired flip | Mean margin | Mean logit lift |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 25.00% | 0.00% | -0.2450 | 2.5939 |
| 128 | 25.00% | 0.00% | -0.4352 | 0.7739 |
| 256 | 31.25% | 0.00% | -0.4710 | 0.4451 |
| 384 | 25.00% | 0.00% | -0.5499 | 0.1973 |
| 448 | 25.00% | 0.00% | -0.5312 | 0.0387 |

### By template

| Template | Accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| lantern | 30.00% | 0.00% | -0.3527 | 0.8503 |
| ribbon | 25.00% | 0.00% | -0.2470 | 1.3662 |
| gate | 25.00% | 0.00% | -0.8055 | 0.5544 |
| blanket | 25.00% | 0.00% | -0.3808 | 0.4683 |

## KiwiLM 2 Slim Gated v2

### By distance

| Distance | Accuracy | Paired flip | Mean margin | Mean logit lift |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 50.00% | 18.75% | -0.1739 | 1.9382 |
| 128 | 37.50% | 12.50% | -0.4681 | 0.7473 |
| 256 | 31.25% | 6.25% | -0.6004 | 0.4255 |
| 384 | 37.50% | 6.25% | -0.5867 | 0.2000 |
| 448 | 31.25% | 0.00% | -0.6180 | 0.1029 |

### By template

| Template | Accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| lantern | 45.00% | 15.00% | -0.3859 | 0.8077 |
| ribbon | 35.00% | 10.00% | -0.3082 | 0.9710 |
| gate | 35.00% | 5.00% | -0.7815 | 0.6057 |
| blanket | 35.00% | 5.00% | -0.4821 | 0.3467 |

## KiwiLM 2 Slim v3 H6/S4

### By distance

| Distance | Accuracy | Paired flip | Mean margin | Mean logit lift |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 43.75% | 25.00% | -0.3152 | 2.4882 |
| 128 | 31.25% | 0.00% | -0.3543 | 1.4794 |
| 256 | 43.75% | 12.50% | -0.2079 | 1.3538 |
| 384 | 43.75% | 12.50% | -0.3848 | 0.8589 |
| 448 | 25.00% | 0.00% | -0.5396 | 0.3045 |

### By template

| Template | Accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| lantern | 55.00% | 30.00% | 0.0654 | 1.8221 |
| ribbon | 35.00% | 10.00% | -0.5505 | 1.4163 |
| gate | 35.00% | 0.00% | -0.5400 | 1.3590 |
| blanket | 25.00% | 0.00% | -0.4165 | 0.5905 |
