# Slim v3 H6/S4 residual gate: 250M confirmation

## Verdict

The alpha-0.5 residual gate does not qualify for a 500M–1B run. It is a small
improvement over ungated H6/S4, but the improvement is not large enough to fix
the architecture's upper-block behavior.

At the same exact 250M-token AdamW budget, the gate improves aligned validation
loss from 3.8351 to 3.8284. It also moves block-9 residual amplification in the
right direction: p90 falls from 1.598 to 1.582, maximum falls from 1.654 to
1.625, and health passes rise from 37/100 to 46/100. Those are real but small
gains. The required result was at least 95/100 passes and block-9 p90 at most
1.5.

## Controlled quality comparison

All aligned losses use the same 200 packed validation batches, seed 143,
batch size 2, context length 512, and FP32 evaluation.

| Model | Aligned loss | Perplexity | Gap from gated H6/S4 |
| --- | ---: | ---: | ---: |
| Dense Muon 0.01 | **3.6459** | **38.32** | -0.1824 |
| Dense AdamW | 3.7425 | 42.20 | -0.0858 |
| H6/S4 gated 0.5 | 3.8284 | 45.99 | — |
| H6/S4 ungated | 3.8351 | 46.30 | +0.0067 |

The gate satisfies the within-0.1 rule against Dense AdamW by 0.0142 loss, but
does not satisfy it against the stronger completed Dense Muon 0.01 baseline.
Most importantly, it recovers only 0.0067 loss over ungated H6/S4. That is too
small to offset the failed health and generation checks.

## Residual health

The 100-batch audit uses seeds 141 and 142, 50 batches per seed. Activations and
gradients remain finite and nonzero. Family-specific gradient ratios are
healthy: 0.435 for Hadamard and 1.240 for SwiGLU. Cached direct and rollover
generation parity both pass.

Block 9 remains the failure:

| Metric | Ungated H6/S4 | Gate 0.5 | Requirement |
| --- | ---: | ---: | ---: |
| Health passes | 37/100 | 46/100 | at least 95/100 |
| Median amplification | 1.526 | 1.502 | — |
| p90 amplification | 1.598 | 1.582 | at most 1.500 |
| Maximum amplification | 1.654 | 1.625 | at most 1.650 |
| Batches above 1.5 | 63 | 54 | at most 5 |

The learned gate values end at 0.683, 0.709, 0.654, and 0.702. Each gate moves
up from its 0.5 initialization, so the bounded sigmoid prevents an unbounded
coefficient but does not keep the upper SwiGLU branch sufficiently attenuated.
The last block's median is already above the threshold.

## Throughput and memory

| Model | Median training tok/s | Peak accelerator memory |
| --- | ---: | ---: |
| Dense AdamW | 27.74k | 4.05 GB |
| H6/S4 ungated | **39.09k** | 3.72 GB |
| H6/S4 gated 0.5 | 29.64k | **3.68 GB** |

The gated run records only a 6.9% throughput advantage over Dense AdamW, below
the required 10%, and is 24.1% slower than the ungated run. Throughput is
host-load-sensitive and the log briefly reaches 40.7k tok/s, so this result
should not be interpreted as four scalar gates causing a 24% kernel slowdown.
It is still the only reproducible run-level throughput measurement available,
and the model already fails architecture-independent health and repetition
requirements.

## Generation and retrieval

The full six-prompt suite uses sampling seeds 42 through 46. Gating reduces the
repeated-four-gram rate from 0.136 to 0.085 and lowers the worst consecutive
word run from 147 to 118. It therefore helps repetition, but the required worst
run is below 20; 118 is still a severe generation failure. Dense AdamW's worst
run is 2 and its repeated-four-gram rate is 0.031.

The full 512-token retrieval probe also regresses relative to ungated H6/S4:

| Model | Candidate accuracy | Paired flips | Contextual lift |
| --- | ---: | ---: | ---: |
| Dense AdamW | 26.25% | 0.00% | 0.810 |
| H6/S4 ungated | **37.50%** | **10.00%** | **1.297** |
| H6/S4 gated 0.5 | 30.00% | 3.75% | 0.803 |

Retrieval remains supporting evidence, but it supplies no reason to override
the failed health and generation criteria.

## Recommendation

Do not promote gated H6/S4 to a final-scale run. Keep this checkpoint as the
negative result showing that a freely learned bounded scalar gate is too weak a
constraint: optimization learns to reopen the dense branches.

Dense remains the final-run baseline. Dense Muon 0.01 is the quality candidate;
Dense AdamW remains the faster optimizer/runtime control. If Slim is revisited,
the next experiment should change the constraint itself—such as a fixed or
regularized upper-block scale—not spend another 250M tokens on the current
learned-sigmoid design.

Machine-readable results are in [summary.json](summary.json), the complete
100-batch distribution is in [health.json](health.json), and generation and
retrieval details are in their respective subdirectories.
