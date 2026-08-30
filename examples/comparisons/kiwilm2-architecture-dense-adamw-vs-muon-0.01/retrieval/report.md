# Context Retrieval Benchmark

160 counterfactual pairs / 320 bound cases; seed 42; context window 512.

Candidate accuracy scores the four-way cloze; paired flip requires both counterfactual variants to select their changed target. Margin is target minus best distractor logit, and lift is bound-target minus no-binding control logit.

## Overall

| Model | Candidate accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| KiwiLM 2 Dense AdamW | 26.25% | 0.00% | -0.4465 | 0.8098 |
| KiwiLM 2 Dense Muon 0.01 | 46.25% | 20.00% | -0.2686 | 2.1636 |

## KiwiLM 2 Dense AdamW

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

## KiwiLM 2 Dense Muon 0.01

### By distance

| Distance | Accuracy | Paired flip | Mean margin | Mean logit lift |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 75.00% | 50.00% | 0.4230 | 4.3324 |
| 128 | 62.50% | 31.25% | 0.1615 | 3.5486 |
| 256 | 37.50% | 12.50% | -0.5530 | 1.7687 |
| 384 | 31.25% | 6.25% | -0.7034 | 1.0056 |
| 448 | 25.00% | 0.00% | -0.6712 | 0.1629 |

### By template

| Template | Accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| lantern | 45.00% | 20.00% | -0.1701 | 2.0838 |
| ribbon | 50.00% | 30.00% | 0.0118 | 3.2362 |
| gate | 45.00% | 10.00% | -0.4400 | 2.0381 |
| blanket | 45.00% | 20.00% | -0.4762 | 1.2964 |
