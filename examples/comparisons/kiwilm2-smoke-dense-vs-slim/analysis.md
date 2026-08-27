# KiwiLM 2 vs KiwiLM 2 Slim: 50M-token smoke analysis

## Verdict

KiwiLM 2 Dense is the stronger model at this budget. Slim is materially smaller,
cheaper on paper, and lighter in VRAM, but the current Hadamard MLP sacrifices too
much language-model capacity and does not map its theoretical FLOP advantage to
faster GPU kernels.

This is a smoke result, not a final architecture selection. It does identify two
Slim implementation issues worth addressing before spending 250M tokens:

1. The width-preserving Hadamard path is extremely capacity-constrained: only
   learned diagonal affine parameters surround two fixed transforms.
2. The Python/PyTorch FWHT implementation is allocation-heavy and slower on T4
   than the dense SwiGLU matrix multiplications.

## Experiment validity

Both runs used the same:

- SmolLM subset fingerprint
  `66b9899b879a5aba9eabdd4a40a54ab9ede62fdd1070f43be9b4c5b0e0e9714b`
- tokenizer SHA-256
  `4bcfc2d969a7a8c2285b364d709917d14e17a141e281fed9d7770db00329acf3`
- 50,000,000 training tokens and 2,000,000 validation tokens
- seed 42, data order, 512-token context, batch 8, accumulation 4
- AdamW, fp16, Tesla T4, learning-rate schedule, and evaluation schedule

Both stopped normally at step 3,052 with `stop_reason=max_tokens`. Neither run
resumed, diverged, or produced non-finite logits.

## Quantitative comparison

| Metric | KiwiLM 2 | KiwiLM 2 Slim | Slim relative to Dense |
| --- | ---: | ---: | ---: |
| Validation loss | **4.6722** | 5.1202 | 9.6% higher |
| Validation perplexity | **106.93** | 167.37 | 56.5% higher |
| Total parameters | 64.25M | **40.68M** | 36.7% fewer |
| Dense non-embedding parameters | 31.09M | **7.52M** | 75.8% fewer |
| Estimated FLOPs/token | 99.12M | **52.12M** | 47.4% fewer |
| Peak CUDA memory | 3.78 GiB | **3.03 GiB** | 19.8% less |
| Best-checkpoint size | 739 MiB | **466 MiB** | 36.9% smaller |
| End-to-end training time | 60.9 min | **51.9 min** | 14.9% shorter |
| Clean logged training throughput, median | **22.3k tok/s** | 16.8k tok/s | 24.5% lower |

The wall-time and kernel-throughput measurements point in different directions.
Slim's training kernels are slower, but it writes much smaller AdamW checkpoints.
Because validation improved almost monotonically, both runs repeatedly wrote
`best.pt` plus periodic `latest.pt`; checkpoint I/O therefore makes Slim's total
job shorter despite lower steady-state token throughput.

The shared token and n-gram tables account for about 33.16M parameters. They are
51.6% of Dense, but 81.5% of Slim. Slim's trainable sequence-model core is thus
very small relative to its fixed embedding memory.

## Learning curve

| Step | Tokens | Dense validation loss | Slim validation loss |
| ---: | ---: | ---: | ---: |
| 250 | 4.10M | 6.3621 | 6.9430 |
| 500 | 8.19M | 5.8024 | 6.2510 |
| 750 | 12.29M | 5.4823 | 5.9148 |
| 1,000 | 16.38M | 5.2908 | 5.7173 |
| 1,250 | 20.48M | 5.0853 | 5.5240 |
| 1,500 | 24.58M | 4.9607 | 5.3873 |
| 1,750 | 28.67M | 4.8939 | 5.3278 |
| 2,000 | 32.77M | 4.8559 | 5.2876 |
| 2,250 | 36.86M | 4.7704 | 5.2054 |
| 2,500 | 40.96M | 4.7449 | 5.1740 |
| 2,750 | 45.06M | 4.6892 | 5.1323 |
| 3,000 | 49.15M | 4.7044 | 5.1345 |
| 3,052 | 50.00M | **4.6722** | **5.1202** |

Dense passed Slim's final loss between the 16.38M- and 20.48M-token evaluations.
Slim was still improving at the end, so a longer run may narrow the gap. At equal
estimated training FLOPs, Slim could see about 1.90 times as many tokens, but that
equal-compute result has not been measured and should not be inferred from this run.

## Model health

- Dense's per-block SwiGLU gradient norms remain in a compact 0.55-0.65 range.
- Slim's Hadamard MLP gradient norm falls from 0.090 in block 0 to 0.0068 in
  block 9. Absolute Dense-vs-Slim gradient norms are not directly comparable
  because the modules have very different parameter counts, but the depth trend
  within Slim is concerning.
- Slim's mixer output RMS grows from 0.79 to peaks of 2.34 in deeper blocks,
  whereas Dense's mixers stay between 0.05 and 0.34 on the same diagnostic batch.
- Slim's learned Hadamard diagonal scales remain nearly uniform after training:
  block means are roughly 0.92-0.94 with small per-channel standard deviations.
  The module mostly learned global attenuation rather than differentiated channel
  mixing.
- Both models' n-gram tables are active and finite. The observed unique fractions
  are identical because the lookup scheme and diagnostic input are shared.

## Generation

The fixed 24-generation comparison uses six prompts, two sampling profiles, and
identical sampling seeds.

- Neither model is yet good at story consistency; both frequently drift from the
  requested entity, object, goal, or location.
- Slim has a much worse catastrophic repetition mode. Across the 12 Slim outputs,
  the maximum consecutive identical-word run averages 6.0 and peaks at 53. Dense
  averages 1.5 and peaks at 3.
- The worst Slim sample repeats `ancient` dozens of times. Its final greedy sample
  also repeats `named` continuously.
- Dense is more locally fluent, but still repeats concepts and loses prompt state.

See [generation report](report.md) and [raw generation rows](results.jsonl).

## Retrieval

On 128 paired counterfactual probes across distances 32, 64, 128, and 192:

| Metric | KiwiLM 2 | KiwiLM 2 Slim |
| --- | ---: | ---: |
| Four-way candidate accuracy | 25.0% | 25.0% |
| Paired flip accuracy | 0.0% | 0.0% |
| Mean contextual logit lift | **0.1579** | 0.0894 |
| Target-vs-best-distractor margin | -0.6350 | **-0.4549** |

Both models are at the four-way chance baseline and always retain the same answer
when the contextual fact is flipped. Dense reacts more strongly to context, but
neither model performs retrieval reliably at this budget. The less-negative Slim
margin does not overturn its chance accuracy and zero paired flips.

See [retrieval report](retrieval/report.md), [raw retrieval rows](retrieval/results.jsonl),
and [retrieval suite](retrieval/suite.json).

## Recommended next step

Do not spend the 250M-token architecture budget on the current Slim implementation
unchanged. First run a small Slim ablation at the same 50M-token budget:

1. Add an explicit residual-output scale to Hadamard MLPs and initialize it on the
   same residual-depth principle as Dense's SwiGLU down projection.
2. Try additional learned diagonal/FWHT stages or a lightweight gated branch so
   the structured path has more than roughly 20k MLP parameters across all blocks.
3. Fuse or compile the FWHT path and benchmark forward/backward throughput before
   training; theoretical FLOPs are not useful if the implementation is
   memory-bound and allocation-heavy.
4. Keep the existing Slim result as the immutable baseline. Compare each ablation
   on tokens-to-loss, repetition, block health, and raw GPU throughput.

Dense is healthy enough to proceed to the planned 250M-token AdamW architecture
run. Slim remains an interesting research direction, but this smoke run says the
present Hadamard block is too weak and insufficiently optimized—not that all
structured mixing is unviable.
