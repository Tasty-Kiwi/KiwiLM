# Local release verification — 2026-10-02

This record describes the **original FP32 draft**. New exports default to BF16;
its separate build/results are recorded in `verification-bf16.md`. The original
FP32 draft and training checkpoint are preserved unchanged.

Status: **local draft; not uploaded**. Target `Tasty-Kiwi/KiwiLM-2` must be
**private**. Source commit/tag approval and live private-Hub fresh-download
verification remain pending. No repository, tag, commit, or training run was created.

Prepared bundle: `artifacts/huggingface/KiwiLM-2` (Git-ignored), including FP32
Safetensors, config, tokenizer, metadata, manifest, MIT card/license, package wheel,
and example. Its `release.json` contains per-file checksums and parity results.

- Original checkpoint checksum unchanged before/after export:
  `cdbea1da1f7ad6b9cc27f609cdbe35673e597b7741895905384a21e4060eb173`.
- Safetensors checksum:
  `24c8b3e7b71875e25c33f5c44b68b16967b96e44e561113e7c0aaa751d154c31`.
- Safetensors size: **322,555,664 bytes**, including duplicate serialized tied head.
- Included wheel checksum:
  `9de2fccb28ec20b6d0490aacbe363c214c2eb94eeb82a3152068d53ab6038e2a`.
- Wheel modules and `py.typed` match current source byte-for-byte.
- FP32 CPU source/export logits are bitwise equal at lengths 1, 31, and 512.
- Cached direct/rollover parity passes (maximum direct difference 0.000014782;
  rollover zero; tolerances 0.002).
- Four sampled source/export texts match: seeds 42/46, cache off/auto, 32 new tokens,
  temperature 0.8, top-k 40.
- An installed-wheel CLI generated the expected seed-42 text from a temporary
  directory with no prepared data or source checkout on the import path. Existing
  third-party dependencies were reused; this is **not** a fresh dependency-resolution
  test or live Hub-download proof.
- Hugging Face `ModelCard` successfully parsed the MIT/language/dataset/task/library
  metadata locally. Live Hub metadata validation remains part of publication.
- **277 tests passed**, Ruff passed, locked dependency validation passed, wheel
  build succeeded, and `git diff --check` passed.
- Private-only publication tests verify `private=True`, refusal of unfrozen drafts,
  no upload when visibility is public, and an inference/release-only allowlist.
  Running the helper without `--publish` made no remote calls.

The original `.pt` remains separate, retaining optimizer/RNG/data-generator state.
Inference export is not a training-resume substitute. Rebuild a reviewed release
after committing/tagging the source; drafts intentionally cannot be published by
the helper. See [release runbook](../../docs/huggingface-release.md).
