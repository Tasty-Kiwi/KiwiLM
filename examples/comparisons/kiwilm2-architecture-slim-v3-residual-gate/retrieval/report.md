# Context Retrieval Benchmark

160 counterfactual pairs / 320 bound cases; seed 42; context window 512.

Candidate accuracy scores the four-way cloze; paired flip requires both counterfactual variants to select their changed target. Margin is target minus best distractor logit, and lift is bound-target minus no-binding control logit.

## Overall

| Model | Candidate accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| dense | 26.25% | 0.00% | -0.4465 | 0.8098 |
| h6s4_ungated | 37.50% | 10.00% | -0.3604 | 1.2970 |
| h6s4_gate050 | 30.00% | 3.75% | -0.6485 | 0.8034 |

## dense

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

## h6s4_ungated

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

## h6s4_gate050

### By distance

| Distance | Accuracy | Paired flip | Mean margin | Mean logit lift |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 31.25% | 6.25% | -0.6013 | 1.5522 |
| 128 | 31.25% | 6.25% | -0.6512 | 1.1209 |
| 256 | 31.25% | 6.25% | -0.5350 | 0.9605 |
| 384 | 25.00% | 0.00% | -0.8047 | 0.2603 |
| 448 | 31.25% | 0.00% | -0.6504 | 0.1230 |

### By template

| Template | Accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| lantern | 45.00% | 15.00% | -0.3105 | 0.9093 |
| ribbon | 25.00% | 0.00% | -0.6869 | 0.7367 |
| gate | 25.00% | 0.00% | -1.0403 | 0.7558 |
| blanket | 25.00% | 0.00% | -0.5563 | 0.8118 |
