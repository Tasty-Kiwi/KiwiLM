# Context Retrieval Benchmark

128 counterfactual pairs / 256 bound cases; seed 42; context window 256.

Candidate accuracy scores the four-way cloze; paired flip requires both counterfactual variants to select their changed target. Margin is target minus best distractor logit, and lift is bound-target minus no-binding control logit.

## Overall

| Model | Candidate accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| KiwiLM 2 | 25.00% | 0.00% | -0.6350 | 0.1579 |
| KiwiLM 2 Slim | 25.00% | 0.00% | -0.4549 | 0.0894 |

## KiwiLM 2

### By distance

| Distance | Accuracy | Paired flip | Mean margin | Mean logit lift |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 25.00% | 0.00% | -0.6419 | 0.3644 |
| 64 | 25.00% | 0.00% | -0.6146 | 0.1717 |
| 128 | 25.00% | 0.00% | -0.6502 | 0.0651 |
| 192 | 25.00% | 0.00% | -0.6335 | 0.0303 |

### By template

| Template | Accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| lantern | 25.00% | 0.00% | -0.6596 | 0.2131 |
| ribbon | 25.00% | 0.00% | -0.4922 | 0.1242 |
| gate | 25.00% | 0.00% | -0.7380 | 0.2436 |
| blanket | 25.00% | 0.00% | -0.6502 | 0.0506 |

## KiwiLM 2 Slim

### By distance

| Distance | Accuracy | Paired flip | Mean margin | Mean logit lift |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 25.00% | 0.00% | -0.4505 | 0.2250 |
| 64 | 25.00% | 0.00% | -0.4437 | 0.0801 |
| 128 | 25.00% | 0.00% | -0.4532 | 0.0312 |
| 192 | 25.00% | 0.00% | -0.4722 | 0.0213 |

### By template

| Template | Accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| lantern | 25.00% | 0.00% | -0.3934 | 0.0859 |
| ribbon | 25.00% | 0.00% | -0.4010 | 0.0811 |
| gate | 25.00% | 0.00% | -0.4799 | 0.0874 |
| blanket | 25.00% | 0.00% | -0.5455 | 0.1033 |
