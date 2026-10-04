---
title: KiwiLM Playground
emoji: 🥝
colorFrom: green
colorTo: yellow
sdk: static
app_file: index.html
license: mit
models:
  - Tasty-Kiwi/KiwiLM-2
  - Tasty-Kiwi/KiwiLM-X
  - Tasty-Kiwi/KiwiLM
---

# KiwiLM 2 browser-local playground

Reuses the original KiwiLM playground UI with four selectable models:

- **KiwiLM 2 Dense**: 64.25M parameters, 32K byte-BPE, 512-token context,
  trained on 1B tokens; cached ONNX decoding.
- **KiwiLM 1 Model X Direct SFT v2**: hybrid convolution/attention, about 5.39M parameters.
- **KiwiLM 1 Model Y Direct SFT v2**: Transformer, about 5.37M parameters.
- **KiwiLM 1 Model Y CPT → SFT v2**: SimpleStories CPT then instruction tuning.

The historical models use their original 8K tokenizer, 256-token context and
uncached full-window ONNX path. Their existing approximately 22 MB bundles are
reused without re-exporting or changing weights. `legacy-manifest.json` records
SHA256/size checks from the original pinned Space commit. Auto uses WebAssembly
for them; explicit WebGPU is experimental. Only the selected model downloads,
and changing models releases the previous inference session. Prompts are retained
when customized; stock examples adapt to the chosen model's base/SFT format.

No inference server: prompts and completions stay in the visitor's browser.
The host receives ordinary asset downloads, not prompt content.

One FP32 ONNX graph derived from the BF16 Safetensors release handles prefill and
cached decoding. GQA KV state, convolution history and n-gram predecessors are
preserved. At the 512-token boundary it clears state and re-prefills the newest
window, matching the package's rollover policy. Browser model assets are ignored
by source Git, and are explicitly included in the static deployment only.

First use of KiwiLM 2 downloads approximately 323 MB plus the tokenizer/runtime; several
hundred MB of additional memory may be required. Desktop browsers recommended.
WebGPU is experimental with a checked WebAssembly fallback. A startup cached/full
logit check must pass before generation. File SHA256/size checks guard the browser
asset cache; best-effort persistent caching depends on browser storage quota.
Inference runs in a Web Worker so the controls remain responsive.

Legacy downloads are independently checksum-verified and checked for finite
8192-wide logits. KiwiLM 2's stronger PyTorch/cached/rollover gates do not imply
new PyTorch-reference parity evaluation of the historical checkpoints.

Recommended controls: temperature 0.8, top-k 40, seed 42, 160 new tokens.
Browser RNG differs from PyTorch: equal seeds do not promise cross-runtime text
parity. The base model can repeat, drift and invent facts; it is not a chatbot.
See the [model card](https://huggingface.co/Tasty-Kiwi/KiwiLM-2).

## Build/export

```bash
uv sync --locked --extra browser
uv run --locked --extra browser python scripts/export_browser_onnx.py \
  --bundle artifacts/huggingface/KiwiLM-2-bf16 \
  --revision 53d6abcb4d11c8a98936c19d55d2b741a99d7602 \
  --output artifacts/browser/kiwilm2-cached-fp32
cd spaces/kiwilm-playground
npm ci
npm run check
npm test
npm run build
```

Stage the verified exporter output in `public/models/kiwilm2/` before building:

```bash
mkdir -p public/models/kiwilm2
cp ../../artifacts/browser/kiwilm2-cached-fp32/* public/models/kiwilm2/
npm run build
cd ../..
uv run --locked --extra browser python scripts/publish_browser_playground.py --publish
```

Without `--publish`, the helper makes no remote changes. It verifies artifact
hashes, accepts only built assets, pins the Space's parent commit, and refuses a
non-static Space. It updates only `dist/` plus this README; the existing X/Y models
and all prior Space commits remain recoverable. Deployment verifies all nine
preserved legacy files exist and match available Hub size/LFS metadata before
committing the selector update. No source tags, paid hardware,
or model-repository mutations are necessary.
