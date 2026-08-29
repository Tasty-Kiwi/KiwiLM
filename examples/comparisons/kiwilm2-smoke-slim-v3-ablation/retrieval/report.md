# Context Retrieval Benchmark

160 counterfactual pairs / 320 bound cases; seed 42; context window 512.

Candidate accuracy scores the four-way cloze; paired flip requires both counterfactual variants to select their changed target. Margin is target minus best distractor logit, and lift is bound-target minus no-binding control logit.

## Overall

| Model | Candidate accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| KiwiLM 2 Dense | 25.00% | 0.00% | -0.6508 | 0.0904 |
| KiwiLM 2 Slim Gated v2 | 25.00% | 0.00% | -0.5569 | 0.0783 |
| KiwiLM 2 Slim v3 H7/S3 | 25.00% | 0.00% | -0.8658 | 0.1098 |
| KiwiLM 2 Slim v3 H6/S4 | 25.00% | 0.00% | -0.6265 | 0.2727 |

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
| 32 | 25.00% | 0.00% | -0.5211 | 0.2963 |
| 128 | 25.00% | 0.00% | -0.5440 | 0.0561 |
| 256 | 25.00% | 0.00% | -0.5562 | 0.0182 |
| 384 | 25.00% | 0.00% | -0.5724 | 0.0103 |
| 448 | 25.00% | 0.00% | -0.5905 | 0.0104 |

### By template

| Template | Accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| lantern | 25.00% | 0.00% | -0.6957 | 0.0989 |
| ribbon | 25.00% | 0.00% | -0.5253 | 0.0699 |
| gate | 25.00% | 0.00% | -0.4699 | 0.1060 |
| blanket | 25.00% | 0.00% | -0.5365 | 0.0383 |

## KiwiLM 2 Slim v3 H7/S3

### By distance

| Distance | Accuracy | Paired flip | Mean margin | Mean logit lift |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 25.00% | 0.00% | -0.8016 | 0.4660 |
| 128 | 25.00% | 0.00% | -0.8679 | 0.0537 |
| 256 | 25.00% | 0.00% | -0.8968 | 0.0109 |
| 384 | 25.00% | 0.00% | -0.8967 | 0.0076 |
| 448 | 25.00% | 0.00% | -0.8659 | 0.0108 |

### By template

| Template | Accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| lantern | 25.00% | 0.00% | -0.9695 | 0.0692 |
| ribbon | 25.00% | 0.00% | -0.7712 | 0.0881 |
| gate | 25.00% | 0.00% | -0.9411 | 0.1667 |
| blanket | 25.00% | 0.00% | -0.7813 | 0.1151 |

## KiwiLM 2 Slim v3 H6/S4

### By distance

| Distance | Accuracy | Paired flip | Mean margin | Mean logit lift |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 25.00% | 0.00% | -0.4789 | 1.1814 |
| 128 | 25.00% | 0.00% | -0.6197 | 0.1617 |
| 256 | 25.00% | 0.00% | -0.6548 | 0.0156 |
| 384 | 25.00% | 0.00% | -0.6831 | 0.0020 |
| 448 | 25.00% | 0.00% | -0.6961 | 0.0026 |

### By template

| Template | Accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| lantern | 25.00% | 0.00% | -0.6340 | 0.2866 |
| ribbon | 25.00% | 0.00% | -0.5831 | 0.2699 |
| gate | 25.00% | 0.00% | -0.6610 | 0.3023 |
| blanket | 25.00% | 0.00% | -0.6281 | 0.2318 |
