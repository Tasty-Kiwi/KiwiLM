# KiwiLM 2 freeze and cleanup — 2026-10-05

Scope: Phase 1 / M0–M1 of [V3_PLAN.md](../V3_PLAN.md). V3 itself is not built.

## Immutable identities

| Reference | Identity |
| --- | --- |
| Local annotated source tag | `kiwilm2-v2-1b` |
| Reviewed pre-cleanup source commit | `643aa81fe1895b69dc7f4cf03d55cf7751fa5895` |
| Original Dense 1B checkpoint SHA256 | `cdbea1da1f7ad6b9cc27f609cdbe35673e597b7741895905384a21e4060eb173` |
| Training data fingerprint | `8b9cf7d8fb5497e5b250e2ef4bc2eba19ed892bbaee8d1bc4688f12595ce5d5c` |
| Published model revision | `53d6abcb4d11c8a98936c19d55d2b741a99d7602` |
| Published BF16 Safetensors SHA256 | `88a775621789bd2de37850b0403f57442827c9552a78936ade8f405b930b2908` |

The tag is a reviewed V2 source reference before cleanup, not a claim that all
training/export jobs originally ran from that Git commit. The original bundle
records tooling HEAD `8526ad8` and a dirty worktree; its inference wheel checksum
is an independent code-artifact identity. Historical receipts and the uploaded
`release.json` are **not rewritten** to suggest a retroactively frozen build.
The tag is local only. No cleanup commit, Git push or Hub/Drive write was made.

The original checkpoint remains at
`runs/colab/tpu-v6e1-final-1b-muon-resume1/latest.pt`, step **61,036**, with exactly
**1,000,000,000 tokens** in its training state and its optimizer/scheduler/RNG/data
state intact. The local BF16 bundle remains at
`artifacts/huggingface/KiwiLM-2-bf16/`. Both hashes were checked before and after
regression verification. Existing Drive backups were not accessed or changed.

The public model/card and immutable release revision were checked through the
Hub's read-only metadata API. No weights, publication metadata, permissions or
playground deployment were changed.

## Offline regressions

Captured **before** moving workflow code:

- Six tiny V2 configurations: Dense, minimal/gated Slim, H7/S3, H6/S4, gated H6/S4.
  Fixed initialization, state-dict key/shape hashes, FP32 logits and greedy
  rollover continuations live in `tests/fixtures/kiwilm2_frozen.json`.
- The original 1B checkpoint and BF16 bundle: selected logits at lengths 1, 31,
  512 and their next decode/rollover, plus an eight-token greedy continuation,
  live in `releases/kiwilm2-1b/regression-reference.json`.

Run explicit artifact checks without training data or downloads:

```bash
uv run --locked python scripts/verify_kiwilm2_reference.py --checkpoint runs/colab/tpu-v6e1-final-1b-muon-resume1/latest.pt --bundle artifacts/huggingface/KiwiLM-2-bf16
```

The same one-line command works in PowerShell. Checks refuse different weight
hashes, validate bundle/data provenance, verify embedding tying, compare the
entire cached/direct logit vector including rollover, and require identical
cached/uncached greedy IDs. These are inference regressions, not fresh TPU
restart qualification or a new training evaluation.

On this CPU FP32 runtime, frozen selected logits match **exactly** for both
artifacts (maximum error 0). Cached/direct shape-dependent roundoff is at most
`1.90735e-5` for the original checkpoint and `2.38419e-5` for the BF16-derived
model. The cache check permits `atol=1e-4, rtol=1e-5`; frozen slices use tighter
`atol=2e-5, rtol=1e-5`. Numerical identity across arbitrary hardware/library
versions is not claimed.

`pytest` always runs tiny frozen-schema/output/checkpoint tests. Real artifact
checks run when local defaults exist; otherwise they explicitly skip. Set
`KIWILM_V2_CHECKPOINT` and `KIWILM_V2_BUNDLE` to other local paths. An explicitly
configured missing path fails rather than skips. No test downloads weights.

## Cleanup boundaries

- Moved **20** historical orchestration/experiment scripts to `archive/kiwilm2/scripts/`.
  Active `scripts/` now contains eight evaluation/export/release/rendering tools
  plus the new offline regression verifier.
- Preserved the three complete workflow runbooks in `archive/kiwilm2/docs/`;
  maintained documentation links lead to completed results and archival recipes.
- Kept all **124 tracked comparison evidence files**, model cards/receipts,
  all architecture diagrams, and all four browser playground model choices.
- No model math, state-dict keys, config dispatch, tokenizer, checkpoint format,
  causal training logic or verified latest/previous recovery behavior changed.
- Left approximately 9.7 GiB of runs, 584 MiB of data and 1.4 GiB of release/browser
  artifacts untouched, as well as `.venv`. This is active-surface cleanup, not
  destructive disk reclamation.

The next milestone is M2 notebook infrastructure. See
[the explicit infrastructure boundaries](development.md). The working V2 causal
trainer stays separate from future masked diffusion.

## Verification completed

- **306 Python tests passed**, including both real-weight regressions, historical
  checkpoint/config variants, training/resume and fake-CLI archive recovery tests.
- **Seven browser-runtime tests passed**; ESLint and static production build passed.
- All **eleven archived shell launchers** passed `bash -n`, without executing them.
- Ruff, locked dependency validation, wheel build and `git diff --check` passed.
- Local links in eleven maintained/archived documentation files resolve.
- All 124 tracked comparison files were checked **byte-for-byte against HEAD**.
- Original checkpoint and BF16 weight checksums remain unchanged.

The two existing TorchScript ONNX-export deprecation warnings remain. No new
GPU/TPU, browser-interactive, cloud-recovery or interrupted-training experiment
was run. The playground was built/tested locally, not redeployed.
