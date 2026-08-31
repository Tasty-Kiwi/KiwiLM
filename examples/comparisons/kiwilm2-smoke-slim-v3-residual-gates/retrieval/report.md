# Context Retrieval Benchmark

160 counterfactual pairs / 320 bound cases; seed 42; context window 512.

Candidate accuracy scores the four-way cloze; paired flip requires both counterfactual variants to select their changed target. Margin is target minus best distractor logit, and lift is bound-target minus no-binding control logit.

## Overall

| Model | Candidate accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| control | 25.00% | 0.00% | -0.6265 | 0.2727 |
| gate_025 | 25.00% | 0.00% | -0.7711 | 0.1069 |
| gate_050 | 25.00% | 0.00% | -0.6743 | 0.1958 |

## control

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

## gate_025

### By distance

| Distance | Accuracy | Paired flip | Mean margin | Mean logit lift |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 25.00% | 0.00% | -0.7134 | 0.4677 |
| 128 | 25.00% | 0.00% | -0.7582 | 0.0449 |
| 256 | 25.00% | 0.00% | -0.7930 | 0.0132 |
| 384 | 25.00% | 0.00% | -0.7888 | 0.0038 |
| 448 | 25.00% | 0.00% | -0.8022 | 0.0048 |

### By template

| Template | Accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| lantern | 25.00% | 0.00% | -0.8320 | 0.0921 |
| ribbon | 25.00% | 0.00% | -0.7552 | 0.1004 |
| gate | 25.00% | 0.00% | -0.7476 | 0.1239 |
| blanket | 25.00% | 0.00% | -0.7496 | 0.1112 |

## gate_050

### By distance

| Distance | Accuracy | Paired flip | Mean margin | Mean logit lift |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 25.00% | 0.00% | -0.5643 | 0.8965 |
| 128 | 25.00% | 0.00% | -0.6850 | 0.0624 |
| 256 | 25.00% | 0.00% | -0.6916 | 0.0105 |
| 384 | 25.00% | 0.00% | -0.7322 | 0.0040 |
| 448 | 25.00% | 0.00% | -0.6981 | 0.0057 |

### By template

| Template | Accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| lantern | 25.00% | 0.00% | -0.6607 | 0.1956 |
| ribbon | 25.00% | 0.00% | -0.6059 | 0.1469 |
| gate | 25.00% | 0.00% | -0.6928 | 0.2710 |
| blanket | 25.00% | 0.00% | -0.7377 | 0.1697 |
