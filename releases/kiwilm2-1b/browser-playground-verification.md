# Browser-local playground — 2026-10-02

Live Space: https://huggingface.co/spaces/Tasty-Kiwi/KiwiLM-Playground

Initial KiwiLM 2 static Space commit: `af1e14d0c37938cfcb2f433cf669064b6f392379`.
The model repository remains unchanged at
`53d6abcb4d11c8a98936c19d55d2b741a99d7602`, publicly visible by the owner's decision.
No inference server, paid hardware, training, source Git commit/tag, or source
push was used. Prior Space history and unused X/Y artifacts were preserved.

## Artifact provenance

- BF16 source Safetensors SHA256:
  `88a775621789bd2de37850b0403f57442827c9552a78936ade8f405b930b2908`.
- Browser `model.onnx`: **322,790,522 bytes**, FP32 execution/storage derived from
  the BF16-rounded source, not a new full-precision training checkpoint.
- ONNX SHA256:
  `df9f02da5c4462f573f23fbe083a616ef6b1f675c8983c90ab6ac6d0ffaffdff`.
- Tokenizer SHA256:
  `4bcfc2d969a7a8c2285b364d709917d14e17a141e281fed9d7770db00329acf3`.
- A single graph handles dynamic prefill and incremental decoding with compact
  GQA KV caches, convolution history, RoPE positions and client-side n-gram IDs.
  Rollover clears state and re-prefills the most recent 512-token window.
- Native ONNX Runtime checks against the BF16-derived PyTorch model passed at
  lengths 1, 31, 512, eight incremental steps and a rollover window. Maximum
  absolute logit difference: **0.0000290871**, tolerance 0.002.

## Actual browser evidence

Chrome on the development machine, live static Space, default prompt
`Once upon a time, a fox found a box.`, seed 42, temperature 0.8, top-k 40:

- WebGPU: **42.1 and 42.5 tokens/s**, two 32-token requests, identical samples.
- WebAssembly: **61.9 tokens/s**, 32-token request, same sample as WebGPU.
- Both providers passed browser-side file integrity, Unicode tokenizer parity,
  PyTorch-reference logits, cached/full parity and 512-token rollover checks
  before generation. These gates are built into startup, not inferred solely
  from the Space's RUNNING status.
- Stop interrupted a 256-token request and restored editable controls.
- No browser warning/error logs were observed during those tests.
- Cache reuse was exercised by switching providers after the initial download.
- After the final cancellation-status refinement was deployed, the default
  160-token WebGPU request completed at **41.6 tokens/s**, with no warning/error
  logs. Proof image: `artifacts/browser/playground-verified.jpg` (local, ignored).

These short local measurements include prompt prefill, exclude download/startup,
and are **not** cross-device performance guarantees or a formal throughput study.
WebAssembly was faster on this machine; other browsers/hardware may differ.
Seed reproducibility is within the browser RNG, not a promise of PyTorch sampled
text parity. Model limitations (looping, semantic drift and false claims) remain.

## Repository checks

- **294 Python tests passed** with the locked browser extra installed.
- Five Node tests cover precise n-gram hashing, causal masks, rollover, stable
  seeded sampling, and input restrictions.
- Ruff, ESLint, locked dependency validation, wheel build, `git diff --check`,
  and the static production build passed.
- npm audit: zero reported vulnerabilities after development-tool patches.
- Native ONNX export uses the explicit legacy TorchScript exporter and emits
  two deprecation warnings; browser and native inference checks still pass.

No mobile, Safari, Firefox, or Windows browser acceptance is claimed.

## KiwiLM 1 selector restoration

Updated static Space commit: `74ab71e68cf052d7d94278068c07b5f316e0dfd2`.
The selector now offers KiwiLM 2 Dense alongside KiwiLM 1 Model X Direct SFT v2,
Model Y Direct SFT v2, and Model Y CPT → SFT v2.

The nine historical ONNX/tokenizer files were downloaded from original Space
revision `ba2457fb3a8b73450afd0785362337aaf32cf5fc` and checksum-verified. They
were reused in place, not replaced or re-exported. Their SHA256 hashes and sizes
are pinned in `spaces/kiwilm-playground/legacy-manifest.json`. Deployment refuses
missing historical files or differing remote size/LFS checksums.

Each selected model downloads lazily, uses its own tokenizer and context length,
and releases the previous inference session. KiwiLM 1 retains its original
8K-vocabulary, 256-token, uncached path; Auto selects compatible WebAssembly.
KiwiLM 2 retains its 32K-vocabulary, 512-token cached path and startup parity
checks. Changing models updates stock prompt templates while preserving custom
prompts. Core Python legacy architectures remain on the `legacy` branch.

Live Chrome tests at temperature 0.8, top-k 40, seed 42 with the historical
instruction-format story prompt:

- Model X: completed at EOS after 128 tokens, **38.3 tokens/s**, WebAssembly.
- Model Y Direct: completed 32 tokens, **53.3 tokens/s**, WebAssembly.
- Model Y CPT → SFT: completed 32 tokens, **54.0 tokens/s**, WebAssembly.
- Switching from all three historical sessions back to KiwiLM 2 passed its
  tokenizer/PyTorch/cached/rollover startup gates and completed 32 tokens at
  **45.4 tokens/s**, WebGPU.

All three passed file-integrity and finite-logit/shape startup checks and produced
distinct continuations. These are functional checks, not new PyTorch-reference
parity claims for historical models or formal performance benchmarks.
Custom prompt preservation was verified across KiwiLM 2 → Model X → KiwiLM 2.
Proof image: `artifacts/browser/playground-all-models.jpg` (local, ignored).

Updated checks: **296 Python tests** and **seven Node tests** passed; Ruff,
ESLint, production build and `git diff --check` passed. The two existing ONNX
export deprecation warnings remain. No paid hardware or training was started.
