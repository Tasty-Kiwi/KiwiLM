# Dense Muon: 500M GPU versus 1B TPU

## Verdict

The exact-1B checkpoint improves matched validation loss and short/mid-range retrieval, and the fresh-VM TPU continuation completed successfully. It does **not** demonstrate improved free-generation quality: repetition is essentially unchanged, story consistency remains weak, and one sample crosses the identical-word repetition guard. This is a better loss/retrieval checkpoint, not a generation-ready model.

The usual local suite is complete: two fresh fixed-validation evaluations, 200 total health batches, direct/rollover cache parity, static profiles, 640 retrieval cases, and 240 five-seed generation samples. No training, Colab allocation, or Drive mutation was performed during this analysis.

## Matched scores

Both checkpoints were loaded strictly and evaluated on the same local MPS device in FP32 with context 512. Validation used 200 batches of two sequences, seed 143, and 204,800 target positions per checkpoint from the frozen SmolLM validation split. These are sampled windows, not a complete traversal of all 2M validation tokens.

| Measurement | Dense Muon 500M | Dense TensorMuon TPU 1B |
| --- | ---: | ---: |
| Training tokens | 500,000,000 | 1,000,000,000 |
| Final step | 30,518 | 61,036 |
| Fixed FP32 validation loss | 3.536938 | 3.486474 |
| Fixed FP32 perplexity | 34.3615 | 32.6705 |
| Health batches passing | 100/100 | 99/100 |
| Minimum last/first SwiGLU-family gradient ratio | 0.7381 | 0.7667 |
| Direct cached parity | Pass | Pass |
| Rollover cached parity | Pass | Pass |
| Retrieval candidate accuracy | 46.25% | 62.50% |
| Retrieval paired flip | 18.75% | 46.25% |

The aligned improvement is **0.050464 loss**, or **4.92% lower perplexity**. The new evaluation exactly reproduces the previous 500M fixed score. All 120 regenerated 500M texts also match their previously stored counterparts byte-for-byte.

The 1B worker's final BF16/XLA validation was 3.522610 / 33.8727 perplexity; its CPU portability check was 3.522351. Those use the worker's different validation protocol and are not interchangeable with the fixed FP32 scores above. Logged losses flattened near the end: the best logged value was 3.520482 at 991.232M tokens, only 0.002128 below the final value. That small difference does not establish overfitting. The full logged curves remain in [summary.json](summary.json).

This compares two independently trained target-budget runs, not a 500M checkpoint continued to 1B. Architecture, tokenizer, source revision/mix, seed, batch size, accumulation, and nominal Muon LR agree. However, GPU FP16 versus TPU BF16, optimizer implementation/backend, warmup (10M versus 20M), evaluation cadence, and total decay horizon differ. The measured improvement cannot be attributed exclusively to token count or TPU hardware. There is one training seed, no training-seed uncertainty estimate, and no new architecture ablation here.

## Block health and cache parity

All 100 batches per checkpoint had finite activations and finite, nonzero gradients. The one 1B health failure was **block 5**, data seed 142, zero-based batch 28: post-MLP/post-mixer residual RMS amplification was **1.513788**, just above the 1.5 threshold. Its gradient-family ratio passed. This is an isolated threshold crossing, not evidence of exploding training, but it must not be reported as a perfect health pass.

| Residual amplification | 500M median | 500M p90 | 500M max | 1B median | 1B p90 | 1B max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Block 5 | 1.1055 | 1.1609 | 1.1894 | 1.3421 | 1.4396 | 1.5138 |
| Block 9 | 1.2906 | 1.3491 | 1.3788 | 1.1768 | 1.2442 | 1.2822 |

Block 9 is more controlled at 1B. Block 5 has become more dominant: median MLP-update/post-mixer-residual RMS rises from 0.368 to 0.807. Monitor that block in any later experiment; this audit alone does not justify adding gates or changing Dense. The worker's earlier 50-batch CPU audit passed all batches; the additional seed-142 batches in this 100-batch audit expose the small outlier.

Direct cached maximum logit differences were 0.00001240 (500M) and 0.00001621 (1B), below the established 0.002 absolute/relative tolerances. Rollover differences were zero for both. These are local FP32 MPS checks; cached decoding on XLA remains unmeasured.

See [health-report.md](health-report.md) for every block's contribution and amplification distributions. The two `*-health-batches.json` files retain individual seeds, batch indices, n-gram diagnostics, gradients, and health checks; aggregate quantiles also include p95 and failure rates in [summary.json](summary.json).

## Full-context retrieval

The unchanged suite uses context 512, four candidate colors, 160 counterfactual pairs / 320 bound cases per model, and 32 pairs at each distance. Candidate chance is 25%. Paired flip requires both counterfactual variants to select their changed target, making it a stronger check than candidate accuracy alone.

| Distance | 500M accuracy | 1B accuracy | 500M paired flip | 1B paired flip |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 50.00% | 75.00% | 12.50% | 62.50% |
| 128 | 68.75% | 100.00% | 43.75% | 100.00% |
| 256 | 56.25% | 62.50% | 31.25% | 37.50% |
| 384 | 31.25% | 50.00% | 6.25% | 31.25% |
| 448 | 25.00% | 25.00% | 0.00% | 0.00% |

Overall accuracy improves by **16.25 percentage points** and paired flip by **27.50 points**. The mean target-minus-best-distractor logit margin turns positive (-0.0641 to +0.2149), with mean contextual lift increasing from 2.4568 to 3.1291.

The gain is not uniform across the window. At distance 448, the 1B contextual lift is only 0.0455, versus 0.2843 at 500M; both remain at chance and fail every paired flip. Full 512-token support is therefore not reliable long-range binding. Perfect performance at distance 128 applies only to these 32 templated pairs, not arbitrary retrieval or reasoning.

See [retrieval/report.md](retrieval/report.md), [retrieval/results.jsonl](retrieval/results.jsonl), and the frozen [retrieval/suite.json](retrieval/suite.json).

## Generation: the remaining weakness

Twelve prompts (six story, six expository), two sampling profiles, seeds 42–46, and a 160-new-token cap produce **120 samples per model**. All use FP32 MPS, context 512, cache off. Focused sampling is temperature 0.4 / top-k 20; creative sampling is 0.8 / top-k 40. Repetition is calculated on continuations, excluding the prompt, using case-folded word tokens. The repeated-four-gram rate is duplicate four-grams divided by all four-grams; a severe-loop flag means rate strictly above 0.5. These are decoding diagnostics, not semantic-quality scores.

| Group | Samples/model | 500M mean repetition | 1B mean repetition | 500M severe loops | 1B severe loops |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 120 | 0.3097 | 0.3114 | 39 | 41 |
| Focused | 60 | 0.5651 | 0.5412 | 38 | 39 |
| Creative | 60 | 0.0543 | 0.0816 | 1 | 2 |
| Story, both profiles | 60 | 0.2738 | 0.2826 | 16 | 20 |
| Expository, both profiles | 60 | 0.3456 | 0.3401 | 23 | 21 |

Paired repetition decreases in 64 samples, increases in 51, and ties in five. Mean change is +0.001659; the median is -0.004430. That mixed result is not a convincing generation improvement. Seed-level mean rates (24 samples each) are:

| Seed | 500M | 1B |
| ---: | ---: | ---: |
| 42 | 0.2811 | 0.3449 |
| 43 | 0.2930 | 0.3322 |
| 44 | 0.3211 | 0.2676 |
| 45 | 0.3437 | 0.2950 |
| 46 | 0.3095 | 0.3171 |

The 1B `plant_science` focused sample at seed 46 ends in **24 consecutive `micro` word tokens**, hyphen-separated in the raw text. That exceeds the existing 20-word guard. Its overall four-gram rate is only 0.2151, demonstrating why the word-run guard must be tracked separately. There were no such failures in the 500M samples. The 1B creative maximum word run is seven; focused sampling accounts for the single guard failure.

Manual inspection gives concrete limitations even in low-repetition outputs:

- `open_story`, creative, seed 42: a fluent New York opening drifts into an invented music council and nonsensical guitar explanation (repetition 0.0615).
- `named_entity`, creative, seed 42: Fluffy is preserved, but bird behavior drifts into molecular biology; entity preservation is not plot coherence (repetition 0.0000).
- `persistent_goal`, creative, seed 42: the promised food-before-sunset goal is not resolved and the narrative drifts through incompatible fox/cage/snake descriptions (repetition 0.0226).
- `procedure`, creative, seed 42: the tea procedure includes salt, onion/garlic, sauteing, and baking (repetition 0.0391). Low repetition does not make the instructions correct.
- `object_ownership`, focused, seed 42: a sentence about getting a ball out of a box repeats throughout the continuation (repetition 0.8655).

Temperature 0.8 / top-k 40 remains much less loop-prone than 0.4 / top-k 20, but it does not repair semantic consistency. This analysis did not rerun the separate 480-sample decoding sweep or measure repetition penalties/top-p variants. It supports keeping the previous creative profile for exploration, not promising reliable output.

Every sample is retained in [generation-report.md](generation-report.md) and [generation-results.jsonl](generation-results.jsonl), with grouped and paired metrics in [generation-summary.json](generation-summary.json) and the unchanged [generation-suite.json](generation-suite.json).

## TPU execution and continuation

The checkpoint, counters, and restored metrics agree on exactly **1B tokens / step 61,036**. All 61,036 training steps are present, strictly increasing, and have the expected token counters; the final masked step contains 2,560 targets. Recorded losses and gradients remain finite. Weight tying is preserved.

The resumed VM started from step 13,000 / 212.992M tokens, restored optimizer and data-generator state, and finished the remaining 787.008M tokens. This validates practical checkpoint continuation. It is not a controlled bitwise comparison against an uninterrupted full run.

| TPU measurement | Result |
| --- | ---: |
| Whole-run weighted steady training throughput | 77,890 tokens/s |
| Whole-run median steady training throughput | 78,019 tokens/s |
| Resumed-leg weighted steady training throughput | 79,227 tokens/s |
| Peak recorded TPU tensor memory | 3,189,593,600 bytes / 2.971 GiB |
| Resumed training plus periodic I/O | 10,932.57 s / 3.037 h |
| Resumed session through final diagnostics, before final packaging | 11,374.92 s / 3.160 h |
| Resumed tokens / that session duration | 69,188 tokens/s |

Steady throughput excludes validation/checkpoint/diagnostic/transfer overhead; the session rate includes more of that overhead but excludes setup, original-VM time, downtime, and final packaging/download. Thus it is not a measured full-run wall-clock rate. The GPU baseline's mixed-session median is about 24,266 tokens/s, but hardware/load/precision differ; this is not a controlled TPU speedup benchmark. GPU allocated memory and XLA tensor memory are also not directly comparable.

The final local checkpoint SHA256 matches the recorded verified Drive commit at step 61,036:

```text
cdbea1da1f7ad6b9cc27f609cdbe35673e597b7741895905384a21e4060eb173
```

This validates the downloaded file against its saved receipt, not a live inspection of current Drive contents. No remote state was changed.

## Parameters, provenance, and missing evaluations

Both models have the same static profile: **64,252,416 parameters**, including 31,091,200 dense/non-embedding parameters, 16,384,000 token-embedding parameters, and 16,777,216 n-gram parameters. Estimated forward FLOPs/token at context 512 are 99,117,056 (multiply-add counted as two), not a measured training FLOP rate. The attention KV cache at two-byte precision is 1 MiB per batch element at 512 tokens; it excludes convolution/ngram state and doubles in FP32.

Training fingerprints differ legitimately with training-budget/preparation provenance. The analysis validates each checkpoint against its own original job instead of pretending the fingerprints match. The shared tokenizer artifact SHA is `4bcfc2d969a7a8c2285b364d709917d14e17a141e281fed9d7770db00329acf3`, and the local validation-bin SHA is `afc8a779d6d941584505c17318c24e56a6ac3e1e02b906beb33dcafb87be1e1c`. Loading verifies local data integrity. Both jobs use source revision `3ba9d605774198c5868892d7a8deda78031a781f`, FineWeb probability 0.7, seed 42, and the same reserved validation prefix. The retrieval suite uses its existing compact-JSON tokenizer hash, which was validated separately.

Prepared TinyStories/SimpleStories transfer data is absent locally, so external transfer losses were **not measured**. Story prompts are not a substitute. We also did not measure on-TPU inference/cache speed, multi-seed training variance, general factual accuracy, or an instruction-tuned model.

## Practical conclusion

Keep the 1B checkpoint as the stronger measured loss/retrieval baseline and preserve its verified backup. The TPU training and fresh-VM resume path are useful. Do not treat the generation guard failure as a successful promotion check, or assume another token-budget increase will fix loops and semantic drift. A next experiment should explicitly target those generation failures with the same multi-seed prompt suite; no further training is started here.

## Reproduction and checks

From the repository root, with the two downloaded runs and the frozen local data present:

```bash
uv run --locked python examples/comparisons/kiwilm2-final-1b-tpu-muon/evaluate.py --stage all --device mps
uv run --locked python examples/comparisons/kiwilm2-final-1b-tpu-muon/render.py
```

`--stage generation` resumes partial samples only after checkpoint/prompt/settings identity checks. To evaluate on CPU, rerun both core and generation stages on CPU; do not mix device-specific measurements. Output paths are portable repository-relative paths; no local macOS/Windows absolute checkpoint paths are saved.

Verification: **265 pytest tests passed**, repository Ruff passed, locked dependency validation passed, and `git diff --check` passed. These checks complement the actual checkpoint evaluations above; they do not replace them. No commits or pushes were made.
