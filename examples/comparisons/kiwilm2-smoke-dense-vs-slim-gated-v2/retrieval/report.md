# Context Retrieval Benchmark

160 counterfactual pairs / 320 bound cases; seed 42; context window 512.

Candidate accuracy scores the four-way cloze; paired flip requires both counterfactual variants to select their changed target. Margin is target minus best distractor logit, and lift is bound-target minus no-binding control logit.

## Overall

| Model | Candidate accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| KiwiLM 2 Dense | 25.00% | 0.00% | -0.6508 | 0.0904 |
| KiwiLM 2 Slim Gated v2 | 25.00% | 0.00% | -0.5768 | 0.0766 |

## KiwiLM 2 Dense

### By distance

| Distance | Accuracy | Paired flip | Mean margin | Mean logit lift |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 25.00% | 0.00% | -0.6182 | 0.3524 |
| 128 | 25.00% | 0.00% | -0.6520 | 0.0613 |
| 256 | 25.00% | 0.00% | -0.6560 | 0.0162 |
| 384 | 25.00% | 0.00% | -0.6709 | 0.0095 |
| 448 | 25.00% | 0.00% | -0.6571 | 0.0125 |

### By template

| Template | Accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| lantern | 25.00% | 0.00% | -0.6602 | 0.1060 |
| ribbon | 25.00% | 0.00% | -0.4991 | 0.0834 |
| gate | 25.00% | 0.00% | -0.7951 | 0.1317 |
| blanket | 25.00% | 0.00% | -0.6489 | 0.0405 |

## KiwiLM 2 Slim Gated v2

### By distance

| Distance | Accuracy | Paired flip | Mean margin | Mean logit lift |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 25.00% | 0.00% | -0.5448 | 0.2914 |
| 128 | 25.00% | 0.00% | -0.5635 | 0.0543 |
| 256 | 25.00% | 0.00% | -0.5728 | 0.0176 |
| 384 | 25.00% | 0.00% | -0.5923 | 0.0097 |
| 448 | 25.00% | 0.00% | -0.6108 | 0.0099 |

### By template

| Template | Accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| lantern | 25.00% | 0.00% | -0.7253 | 0.0967 |
| ribbon | 25.00% | 0.00% | -0.5453 | 0.0688 |
| gate | 25.00% | 0.00% | -0.4852 | 0.1022 |
| blanket | 25.00% | 0.00% | -0.5515 | 0.0386 |
