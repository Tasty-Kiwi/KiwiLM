---
license: mit
language:
  - en
library_name: kiwilm
pipeline_tag: text-generation
datasets:
  - roneneldan/TinyStories
tags:
  - pytorch
  - safetensors
  - causal-lm
  - hybrid-model
  - toy-model
  - research
---

# KiwiLM Model X

KiwiLM Model X is a 5.39M-parameter decoder-only hybrid causal language model
designed to combine inexpensive local mixing with content-dependent global
mixing. Its sequence is:

```text
token embedding
-> gated causal CNN -> SwiGLU
-> RoPE causal attention -> SwiGLU
-> gated causal CNN -> SwiGLU
-> RoPE causal attention -> SwiGLU
-> final RMSNorm -> tied LM head
```

This private research release contains the final TinyStories-Instruct SFT v2
checkpoint. It is a custom PyTorch/KiwiLM model, **not Transformers-native**.

## Released checkpoint

The root bundle is Model X's final SFT v2 `latest.pt` state, selected because
it achieved better fixed-prompt instruction adherence than the loss-selected
`best.pt` state in both decoding profiles.

| Metric | Value |
| --- | ---: |
| Parameters | 5,387,520 |
| Context length | 256 tokens |
| Frozen BPE vocabulary | 8,192 tokens |
| TinyStories pretraining targets | 160,465,920 |
| SFT v2 response targets | 10,000,000 |
| SFT step | 2,523 |
| Checkpoint SFT validation PPL | 6.0323 |

The matched 750k-story pretrained Model X checkpoint scored 6.4849 perplexity
on 500 deterministic TinyStories story-validation batches. That base checkpoint
is not included in this inference release.

## Instruction-adherence evaluation

The six-prompt suite uses cache-off generation with seed 42. Scores are
deterministic lexical diagnostics rather than semantic-judge scores.

| Checkpoint | Profile | Adherence | Required words | Summary terms | Features | Entities | Repeated 4-grams |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT v2 latest | greedy | 58.0% | 44.4% | 29.2% | 100.0% | 60.0% | 12.5% |
| SFT v2 latest | focused | 63.0% | 50.0% | 58.3% | 100.0% | 40.0% | 4.0% |
| SFT v2 best | greedy | 56.6% | 38.9% | 29.2% | 100.0% | 60.0% | 12.5% |
| SFT v2 best | focused | 54.6% | 50.0% | 41.7% | 83.3% | 40.0% | 7.6% |

Model X is the throughput-oriented KiwiLM finalist. Model Y achieved slightly
better validation quality and instruction adherence, while Model X trained and
generated faster in the matched experiments.

## Files

```text
model.safetensors  inference weights and embedded KiwiLM metadata
config.json        Model X architecture configuration
tokenizer.json     exact frozen byte-level BPE tokenizer
metadata.json      training lineage, metrics, fingerprints, and source hash
manifest.json      SHA-256 and size for the inference bundle
release-manifest.json  SHA-256 and size for every published artifact
```

The Safetensors file contains model weights only. Optimizer state, AMP scaler,
sampler state, and RNG snapshots from the resumable checkpoint are excluded.
Tied weights are represented under both state-dict keys for portable strict
loading; KiwiLM re-establishes parameter tying during reconstruction.

## Usage

```bash
hf download Tasty-Kiwi/KiwiLM-X --local-dir weights/KiwiLM-X
uv venv
uv pip install weights/KiwiLM-X/kiwilm-0.1.0-py3-none-any.whl
```

Generate an instructed story:

```bash
.venv/bin/kiwilm generate \
  --checkpoint weights/KiwiLM-X \
  --prompt $'Instruction: Write a story that follows every provided condition. Use every requested word exactly as written.\nFeatures: Dialogue\nWords: oak, gloomy, kind\nSummary: Two friends help each other get home before dark.\nStory:\n' \
  --max-new-tokens 200 \
  --temperature 0.4 \
  --top-k 20 \
  --cache auto \
  --stream
```

The bundled tokenizer is selected automatically when `--checkpoint` points to
the repository directory. `model.safetensors` can also be loaded directly with
`kiwilm.inference.load_trained_model`. On Windows, invoke
`.venv\Scripts\kiwilm.exe` instead.

## Training data

- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
  licensed under CDLA-Sharing-1.0.
- `roneneldan/TinyStoriesInstruct`, using the frozen pretraining tokenizer and
  response-only supervised loss.

See the [KiwiLM source repository](https://github.com/Tasty-Kiwi/KiwiLM) for
pinned revisions, preprocessing, training commands, architecture graphs, and
comparison reports.

## Intended use and limitations

This is an educational research model for studying tiny hybrid language-model
architectures, caching, and instruction fine-tuning. It is suitable for local
experiments and short synthetic-story demonstrations.

It is not suitable for factual, safety-critical, production, or child-facing
use. Its 256-token context and very small parameter count cause entity
substitution, forgotten goals, ownership errors, repetition, malformed
dialogue, and logically inconsistent stories. Outputs require review.

## License and attribution

KiwiLM code and these released weights are provided under the included MIT
license. Training-data licenses remain applicable; consult the linked dataset
card before redistribution or commercial use.
