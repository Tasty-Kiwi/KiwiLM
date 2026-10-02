# Dense Muon scaling: 500M GPU versus 1B TPU

160 counterfactual pairs / 320 bound cases; seed 42; context window 512.

Candidate accuracy scores the four-way cloze; paired flip requires both counterfactual variants to select their changed target. Margin is target minus best distractor logit, and lift is bound-target minus no-binding control logit.

## Overall

| Model | Candidate accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| Dense Muon 0.01 500M | 46.25% | 18.75% | -0.0641 | 2.4568 |
| Dense Muon 0.01 TPU 1B | 62.50% | 46.25% | 0.2149 | 3.1291 |

## Dense Muon 0.01 500M

### By distance

| Distance | Accuracy | Paired flip | Mean margin | Mean logit lift |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 50.00% | 12.50% | 0.0886 | 3.1977 |
| 128 | 68.75% | 43.75% | 0.3659 | 4.0134 |
| 256 | 56.25% | 31.25% | 0.1252 | 3.0821 |
| 384 | 31.25% | 6.25% | -0.3199 | 1.7063 |
| 448 | 25.00% | 0.00% | -0.5804 | 0.2843 |

### By template

| Template | Accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| lantern | 50.00% | 20.00% | -0.0198 | 1.9116 |
| ribbon | 65.00% | 45.00% | 0.3617 | 3.5611 |
| gate | 35.00% | 0.00% | -0.3635 | 2.6901 |
| blanket | 35.00% | 10.00% | -0.2348 | 1.6642 |

## Dense Muon 0.01 TPU 1B

### By distance

| Distance | Accuracy | Paired flip | Mean margin | Mean logit lift |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 75.00% | 62.50% | 0.3217 | 4.8112 |
| 128 | 100.00% | 100.00% | 1.2586 | 6.2204 |
| 256 | 62.50% | 37.50% | 0.3266 | 3.1445 |
| 384 | 50.00% | 31.25% | -0.1516 | 1.4238 |
| 448 | 25.00% | 0.00% | -0.6810 | 0.0455 |

### By template

| Template | Accuracy | Paired flip | Mean margin | Mean logit lift |
| --- | ---: | ---: | ---: | ---: |
| lantern | 65.00% | 50.00% | 0.0367 | 1.8756 |
| ribbon | 75.00% | 60.00% | 0.6154 | 4.2382 |
| gate | 70.00% | 55.00% | 0.2732 | 3.6201 |
| blanket | 40.00% | 20.00% | -0.0659 | 2.7825 |
