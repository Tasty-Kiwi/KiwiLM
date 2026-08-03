---
license: mit
language:
  - en
library_name: kiwilm
pipeline_tag: text-generation
datasets:
  - roneneldan/TinyStories
  - SimpleStories/SimpleStories
tags:
  - pytorch
  - safetensors
  - causal-lm
  - toy-model
  - research
---

# KiwiLM Model Y

KiwiLM Model Y is a 5.37M-parameter decoder-only causal language model created
to compare modern Transformer mixing against gated causal convolutions at tiny
scale. It uses four pre-RMSNorm RoPE causal-attention blocks, four SwiGLU
feed-forward networks, a final RMSNorm, and tied token/LM-head weights.

This repository contains the two final checkpoints from the KiwiLM research
series. They are custom PyTorch/KiwiLM models, **not Transformers-native**.

## Variants

| Directory | Training path | Recommended use |
| --- | --- | --- |
| `direct-sft-v2` | TinyStories 750k pretraining -> TinyStories Instruct SFT v2 | Lowest instruction/TinyStories perplexity and best greedy adherence |
| `cpt-sft-v2` | TinyStories 750k -> SimpleStories CPT -> TinyStories Instruct SFT v2 | Best focused-sampling adherence, lower repetition, broader story modeling |

Both variants use the same byte-level 8,192-token BPE vocabulary, a 256-token
context window, and 5,372,160 trainable parameters.

## Final evaluation

All perplexities use FP16, seed 42, and 500 deterministic validation batches.
Story datasets use batch size 64; response-masked SFT evaluation uses batch
size 8.

| Variant | SFT v2 PPL | TinyStories PPL | SimpleStories PPL |
| --- | ---: | ---: | ---: |
| Direct SFT v2 | **5.7595** | **6.6368** | 37.5875 |
| CPT -> SFT v2 | 6.3614 | 7.5989 | **16.9978** |

| Variant | Profile | Adherence | Required words | Summary terms | Features | Entities | Repeated 4-grams |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct SFT v2 | greedy | **59.6%** | **55.6%** | **45.8%** | **83.3%** | 50.0% | 15.0% |
| CPT -> SFT v2 | greedy | 52.7% | 50.0% | 41.7% | 66.7% | 50.0% | **12.0%** |
| Direct SFT v2 | focused | 65.2% | 55.6% | 50.0% | 100.0% | 50.0% | 5.0% |
| CPT -> SFT v2 | focused | **69.0%** | **66.7%** | **54.2%** | 100.0% | 50.0% | **3.5%** |

The adherence suite contains six fixed prompts. Scores are deterministic
lexical diagnostics, not semantic-judge scores. With the three evaluation
domains weighted equally in log-loss space, CPT -> SFT improves geometric-mean
perplexity from 11.2840 to 9.3662.

## Files

Each variant is a standalone inference bundle:

```text
<variant>/
  model.safetensors  inference weights and embedded KiwiLM metadata
  config.json        Model Y architecture configuration
  tokenizer.json     exact frozen byte-level BPE tokenizer
  metadata.json      training lineage, metrics, fingerprints, and source hash
  manifest.json      SHA-256 and size for every bundle artifact
```

The Safetensors files contain model weights only. Optimizer state, AMP scaler,
sampler state, and RNG snapshots from the resumable training checkpoints are
intentionally excluded. Tied weights are represented under both state-dict
keys for portable strict loading; KiwiLM re-establishes parameter tying when
the model is reconstructed.

## Usage

Download this repository and install the bundled KiwiLM wheel:

```bash
hf download Tasty-Kiwi/KiwiLM --local-dir weights/KiwiLM
uv venv
uv pip install weights/KiwiLM/kiwilm-0.1.0-py3-none-any.whl
```

The instruction prefix used by SFT v2 is:

```text
Instruction: Write a story that follows every provided condition. Use every requested word exactly as written.
Features: Dialogue
Words: oak, gloomy, kind
Summary: Two friends help each other get home before dark.
Story:
```

Generate from the recommended broad checkpoint:

```bash
.venv/bin/kiwilm generate \
  --checkpoint weights/KiwiLM/cpt-sft-v2 \
  --prompt $'Instruction: Write a story that follows every provided condition. Use every requested word exactly as written.\nFeatures: Dialogue\nWords: oak, gloomy, kind\nSummary: Two friends help each other get home before dark.\nStory:\n' \
  --max-new-tokens 200 \
  --temperature 0.4 \
  --top-k 20 \
  --cache auto \
  --stream
```

The bundled tokenizer is selected automatically when `--checkpoint` points to
a bundle directory. `model.safetensors` can also be loaded directly with
`kiwilm.inference.load_trained_model`.

On Windows, invoke `.venv\Scripts\kiwilm.exe` instead. The complete source is
available at [Tasty-Kiwi/KiwiLM](https://github.com/Tasty-Kiwi/KiwiLM).

## Training data

- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
  licensed under CDLA-Sharing-1.0.
- [SimpleStories](https://huggingface.co/datasets/SimpleStories/SimpleStories),
  licensed under MIT. Only the CPT variant was trained on this dataset.
- `roneneldan/TinyStoriesInstruct`, using the same frozen tokenizer and
  response-only supervised loss for both variants.

See the [KiwiLM repository](https://github.com/Tasty-Kiwi/KiwiLM) for pinned
dataset revisions, preprocessing, complete training commands, architecture
graphs, and comparison reports.

## Intended use and limitations

These are educational research models for studying tiny language-model
architectures, training curricula, caching, and sampling. They are suitable
for local experiments and short synthetic-story demonstrations.

They are not suitable for factual, safety-critical, production, or
child-facing use. The 256-token context and very small parameter count cause
entity substitution, forgotten goals, ownership errors, repetition, malformed
dialogue, and logically inconsistent stories. Training data is synthetic, but
the models can still emit undesirable or biased text. Outputs require review.

## License and attribution

KiwiLM code and these released weight files are provided under the MIT license
included in this repository. Training-data licenses remain applicable to their
respective datasets; consult the linked dataset cards before redistribution or
commercial use.
