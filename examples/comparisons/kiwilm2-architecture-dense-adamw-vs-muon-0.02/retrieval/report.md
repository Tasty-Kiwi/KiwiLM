# Context Retrieval Benchmark

160 counterfactual pairs / 320 bound cases; seed 42; context window 512.

Candidate accuracy scores the four-way cloze; paired flip requires both counterfactual variants to select their changed target. Margin is target minus best distractor logit, and lift is bound-target minus no-binding control logit.

## Overall

| Model | Candidate accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| AdamW-250M | 26.25% | 0.00% | -0.4465 | 0.8098 |
| Muon-0.02-250M | 46.25% | 10.00% | -0.1516 | 2.9014 |

## AdamW-250M

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

## Muon-0.02-250M

### By distance

| Distance | Accuracy | Paired flip | Mean margin | Mean logit lift |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 56.25% | 25.00% | 0.1232 | 3.8878 |
| 128 | 56.25% | 12.50% | 0.2038 | 4.1026 |
| 256 | 37.50% | 0.00% | -0.2594 | 3.0518 |
| 384 | 43.75% | 12.50% | -0.3757 | 1.8960 |
| 448 | 37.50% | 0.00% | -0.4501 | 1.5688 |

### By template

| Template | Accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| lantern | 35.00% | 0.00% | -0.3936 | 2.1188 |
| ribbon | 70.00% | 40.00% | 0.6473 | 5.3793 |
| gate | 45.00% | 0.00% | -0.3671 | 2.6896 |
| blanket | 35.00% | 0.00% | -0.4930 | 1.4180 |
