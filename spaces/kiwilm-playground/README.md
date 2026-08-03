---
title: KiwiLM Playground
emoji: 🥝
colorFrom: green
colorTo: yellow
sdk: gradio
app_file: app.py
python_version: "3.11"
pinned: false
license: mit
models:
  - Tasty-Kiwi/KiwiLM-X
  - Tasty-Kiwi/KiwiLM
---

# KiwiLM Playground

Interactive, streaming inference for the final KiwiLM models:

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

The Space mounts the private model repositories read-only at `/models/x` and
`/models/y`. It does not require a runtime Hugging Face access token.
