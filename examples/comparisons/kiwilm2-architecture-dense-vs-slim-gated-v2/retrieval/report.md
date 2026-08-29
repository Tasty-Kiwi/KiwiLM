# Context Retrieval Benchmark

160 counterfactual pairs / 320 bound cases; seed 42; context window 512.

Candidate accuracy scores the four-way cloze; paired flip requires both counterfactual variants to select their changed target. Margin is target minus best distractor logit, and lift is bound-target minus no-binding control logit.

## Overall

| Model | Candidate accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| KiwiLM 2 Dense | 25.00% | 0.00% | -0.6516 | 0.8754 |
| KiwiLM 2 Slim Gated v2 | 30.00% | 2.50% | -0.5451 | 0.6318 |

## KiwiLM 2 Dense

### By distance

| Distance | Accuracy | Paired flip | Mean margin | Mean logit lift |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 25.00% | 0.00% | -0.5588 | 2.5951 |
| 128 | 25.00% | 0.00% | -0.6326 | 0.9238 |
| 256 | 25.00% | 0.00% | -0.6546 | 0.5699 |
| 384 | 25.00% | 0.00% | -0.6906 | 0.2349 |
| 448 | 25.00% | 0.00% | -0.7215 | 0.0534 |

### By template

| Template | Accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| lantern | 25.00% | 0.00% | -0.5416 | 0.9868 |
| ribbon | 25.00% | 0.00% | -0.4757 | 1.5028 |
| gate | 25.00% | 0.00% | -0.9555 | 0.4908 |
| blanket | 25.00% | 0.00% | -0.6337 | 0.5212 |

## KiwiLM 2 Slim Gated v2

### By distance

| Distance | Accuracy | Paired flip | Mean margin | Mean logit lift |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 37.50% | 6.25% | -0.2710 | 1.7999 |
| 128 | 31.25% | 6.25% | -0.5001 | 0.6861 |
| 256 | 31.25% | 0.00% | -0.6545 | 0.3956 |
| 384 | 25.00% | 0.00% | -0.6455 | 0.1828 |
| 448 | 25.00% | 0.00% | -0.6542 | 0.0947 |

### By template

| Template | Accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| lantern | 25.00% | 0.00% | -0.4626 | 0.7442 |
| ribbon | 40.00% | 10.00% | -0.3460 | 0.8755 |
| gate | 30.00% | 0.00% | -0.8223 | 0.5701 |
| blanket | 25.00% | 0.00% | -0.5494 | 0.3374 |
