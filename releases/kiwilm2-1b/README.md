---
license: mit
language:
  - en
pipeline_tag: text-generation
library_name: kiwilm
datasets:
  - HuggingFaceTB/smollm-corpus
tags:
  - pytorch
  - safetensors
  - kiwilm2
  - experimental
  - hybrid
  - base-model
---

# KiwiLM 2 — Dense, 1B training tokens

**Experimental English base language model, not a chat or instruction model.**
There are **64,252,416 parameters**; “1B” is the training-token budget, not model size.

Matched FP32 evaluation gives **3.486474 validation loss**, **32.6705 perplexity**,
and **62.5% four-choice retrieval accuracy**. Retrieval uses a small, templated
320-case evaluation, not a general reasoning benchmark. At distance 448,
retrieval remains at chance (25%).

**Generation remains unreliable:** 41 of 120 generation samples have a repeated
four-gram rate above 0.5; one sample contains 24 consecutive `micro` word tokens.
Outputs show persistent topic/semantic drift, inconsistent entities and goals,
and incorrect explanations. Low repetition does not imply factual correctness.
Do not use this model for consequential advice or unattended factual tasks.

## Download and generate without training data

This is a custom PyTorch architecture. It does not currently support
`transformers.AutoModel`, Transformers pipelines, or hosted Inference Providers.
Use the included KiwiLM wheel and bundled tokenizer. Installing the wheel runs
ordinary Python package installation; only install code from a source you trust.
The loader does not fetch or execute model-repository Python code.

After publication, copy the full 40-character **model repository commit SHA** from
Hugging Face. Keep that revision pinned in downloads (do not confuse it with the
source Git commit or checkpoint checksum).

```bash
hf download Tasty-Kiwi/KiwiLM-2 --revision FULL_HF_COMMIT_SHA --local-dir KiwiLM-2
python -m pip install ./KiwiLM-2/kiwilm-0.1.0-py3-none-any.whl
kiwilm generate --checkpoint KiwiLM-2 --prompt "Once upon a time, a fox found a box." --max-new-tokens 160 --temperature 0.8 --top-k 40 --seed 42 --cache auto --stream
```

Install the `huggingface-hub` package first if the `hf` command is unavailable.
On Windows use `python -m pip install .\KiwiLM-2\kiwilm-0.1.0-py3-none-any.whl`.
The same `kiwilm generate` command works without `--data-dir` on every platform.
The bundle loader checks the inference-file manifest before loading weights.

Or, after installing the included wheel:

```python
import torch
from kiwilm.hub import load_pretrained
from kiwilm.generation import generate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, config = load_pretrained(
    "Tasty-Kiwi/KiwiLM-2",
    revision="FULL_HF_COMMIT_SHA",  # Replace with the published model commit.
    device=device,
)
print(generate(
    model, tokenizer, "Once upon a time, a fox found a box.",
    max_new_tokens=160, context_length=config.context_length,
    temperature=0.8, top_k=40, seed=42, device=device, cache="auto",
))
```

Temperature 0.8 / top-k 40 reduces loops compared with temperature 0.4 / top-k 20,
but does not repair semantic consistency. Samples vary with device and runtime.

## Architecture

- 10 pre-RMSNorm residual blocks, width 512, context 512; tied embedding/LM head.
- Mixer schedule: GQA, Conv31, Conv63, GQA, Conv31, Conv63, GQA, Conv31, Conv63, GQA.
- Grouped-query attention: 8 query / 2 KV heads, cached RoPE on attention only.
- Large-kernel causal depthwise convolutions with gated pointwise projections.
- All ten FFNs use width-1536 SwiGLU; this is **Dense**, not Slim or Slim v3.
- Hashed bigram and trigram embeddings: 16,384 buckets each, width 512.
- Byte-level BPE, 32,000 vocabulary entries. Special IDs: PAD 0, UNK 1, BOS 2, EOS 3.
- 31,091,200 dense/non-embedding parameters, 16,384,000 token-embedding parameters,
  and 16,777,216 n-gram parameters.

Weights are exported in **BF16** Safetensors by default. The embedding/head are duplicated in
the serialized state dictionary for ordinary loading, then tied in memory.
The package uses FP32 execution by default for portability; storage precision and
execution precision are separate. BF16 rounding can change logits and sampled
text relative to the original FP32 checkpoint. The headline metrics above describe
the original FP32 checkpoint, not a new BF16 benchmark. See `release.json` for
rounded-reference parity and any paired rounding-loss check. Native BF16 loading
is available via `load_pretrained(..., dtype=torch.bfloat16)` where supported;
native BF16 performance/cache accuracy is not claimed by the FP32 export checks.
Estimated forward FLOPs/token at context 512: 99.12M, not a measured training rate.

## Training

Trained from scratch on exactly **1,000,000,000 tokens**, with seed 42. The run
resumed its own step-13,000 checkpoint on a fresh TPU VM; it is not a continuation
of the separately trained 500M GPU model. Final step: 61,036 (2,560 targets in the
final masked batch).

Data: controlled SmolLM-Corpus subset, **70% FineWeb-Edu deduplicated / 30%
Cosmopedia-v2 source sampling probability**. This is the source-selection policy,
not a measured 70/30 fraction of packed tokens. Python-Edu is excluded. The source
revision is `3ba9d605774198c5868892d7a8deda78031a781f`. Validation reserves 10,000
documents per source and packs a frozen 2M-token split. The 32K tokenizer is reused
from the controlled smoke preparation.

Hardware: single Colab **TPU v6e-1**, BF16 training, PyTorch/XLA. TensorMuon at LR
0.01 handles dense 2D projections; AdamW handles embeddings, n-gram tables,
norms, biases, and depthwise kernels. AdamW LR is 0.0003, minimum LR 0.00003;
warmup 20M tokens, weight decay 0.1, beta2 0.95, gradient clip 1.0. Batch 8,
accumulation 4, context 512 gives 16,384 targets per full optimizer step.

Whole-run weighted steady training throughput: about 77.9k tokens/s; recorded
TPU tensor peak: 2.971 GiB. These exclude various setup/I/O costs and are not a
matched GPU speed or memory comparison.

## Evaluation and limitations

All headline comparisons use the same frozen validation split and local FP32 MPS
evaluation, not the different BF16 training-worker validation metric.

| Metric | Separate Dense Muon 500M GPU control | This 1B TPU model |
| --- | ---: | ---: |
| Fixed validation loss | 3.536938 | 3.486474 |
| Fixed perplexity | 34.3615 | 32.6705 |
| Four-choice retrieval accuracy | 46.25% | 62.50% |
| Counterfactual paired flip | 18.75% | 46.25% |
| Health batches passing | 100/100 | 99/100 |
| Severe-loop generation samples | 39/120 | 41/120 |
| Mean repeated-four-gram rate | 0.3097 | 0.3114 |
| Samples with 20+ identical word tokens | 0/120 | 1/120 |

Validation: 200 batches of two 512-token sequences, seed 143, 204,800 target
positions. Health: 100 batches, seeds 141/142; finite/nonzero gradients throughout.
The single health failure is block-5 amplification 1.514 against a 1.5 threshold,
not an observed numerical explosion. Direct/rollover cached-generation parity
passes locally; on-XLA cached decoding was not measured.

Retrieval: 160 counterfactual pairs / 320 bound cases, context 512, distances
32/128/256/384/448. Paired flip requires both variants to follow the changed
binding. At distance 448, both controls remain at chance and have 0% paired flip.

Generation: 12 story/expository prompts, two profiles (0.4/top-k20 and 0.8/top-k40),
seeds 42–46, 160-new-token cap, FP32 MPS, cache off. Severe looping means duplicate
four-gram rate strictly above 0.5 on the continuation, excluding the prompt.
Identical-word runs use case-folded word tokens (including hyphen-separated words).
No automated semantic-quality score is claimed.

One training seed only. Training precision, backend/optimizer implementation,
warmup, and decay horizon differ from the GPU control; this is not a causal test
of token count or accelerator hardware. Prepared TinyStories/SimpleStories
transfer data was unavailable locally; those transfer losses were not measured.
This model is not safety-aligned, instruction-tuned, or tested for bias/toxicity
or memorization. Treat outputs as unverified synthetic text.

[Full analysis and retained raw artifacts](https://github.com/Tasty-Kiwi/KiwiLM/tree/master/examples/comparisons/kiwilm2-final-1b-tpu-muon)
and [architecture/runbook](https://github.com/Tasty-Kiwi/KiwiLM/blob/master/docs/kiwilm2.md).

## Release provenance and license

Checkpoint SHA256:
`cdbea1da1f7ad6b9cc27f609cdbe35673e597b7741895905384a21e4060eb173`.
The original optimizer/training checkpoint is preserved separately and is not in
this inference bundle. See `metadata.json` for the original data fingerprint,
tokenizer checksum, training settings and worker metrics; see `release.json` for
package/code status, file checksums, and parity verification.

Model weights and KiwiLM source code: **MIT**, as confirmed by the owner. See
`LICENSE`. Training data is not redistributed; upstream datasets have their own
licenses and provenance. No broader rights claim about the source data is made.

This card is a **local draft** until source commit/tag and Hub publication are
completed. Follow the repository release runbook before uploading.
