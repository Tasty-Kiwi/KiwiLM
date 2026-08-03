---
title: KiwiLM Playground
emoji: 🥝
colorFrom: green
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: mit
models:
  - Tasty-Kiwi/KiwiLM-X
  - Tasty-Kiwi/KiwiLM
---

# KiwiLM Playground

Browser-native, streaming inference for the final KiwiLM models:

- **Model X** — hybrid gated-convolution and attention architecture, selected
  for throughput.
- **Model Y direct SFT v2** — modern Transformer baseline with the best
  in-domain validation and greedy-adherence results.
- **Model Y CPT → SFT v2** — SimpleStories continued pretraining followed by
  instruction tuning, selected for broader focused-sampling behavior.

These are approximately 5.4M-parameter educational research models with a
256-token context window. They can repeat, lose entities, ignore constraints,
and produce inconsistent stories. Do not use them for factual, safety-critical,
production, or child-facing applications.

The Space runs entirely in the visitor's browser using ONNX Runtime Web. Model
files are downloaded lazily from this private Space; prompts and generations
are not sent to an inference server. WebAssembly is used as the portable
baseline, with browser support determining available acceleration.

The deployed ONNX files are generated from the published Safetensors bundles
with `scripts/export_onnx.py`. They are release artifacts and are intentionally
excluded from the main Git repository.
