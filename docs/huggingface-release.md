# KiwiLM 2 Hugging Face release

Published: [Tasty-Kiwi/KiwiLM-2](https://huggingface.co/Tasty-Kiwi/KiwiLM-2),
model revision `53d6abcb4d11c8a98936c19d55d2b741a99d7602`.
The owner made it public after the verified private upload. Public availability
was rechecked on 2026-10-05. No new publication is needed for repository cleanup.
Weights and code are MIT. This is a custom PyTorch base model, not a Transformers
`AutoModel` or hosted-inference integration.

The original private upload receipt and bundled `release.json` remain historical
records, including their pending source-freeze fields. The subsequent local
source reference is documented in [the V2 freeze record](kiwilm2-freeze.md);
it does not retroactively change the uploaded artifact's provenance.
The private-only publication helper below creates a **new** repository and
refuses existing ones. Do not rerun it against the now-public release.

## Local preparation (no uploads or training)

The release uses the exact evaluated Dense 1B `latest.pt`, SHA256
`cdbea1da1f7ad6b9cc27f609cdbe35673e597b7741895905384a21e4060eb173`.
Preserve that original optimizer/training checkpoint in its run directory and
Drive backup. Do not replace it with the inference-only export.

```bash
uv build --wheel --out-dir artifacts/huggingface/wheels-bf16
uv run --locked python scripts/prepare_kiwilm2_release.py \
  --wheel artifacts/huggingface/wheels-bf16/kiwilm-0.1.0-py3-none-any.whl \
  --validation-data data/smollm-architecture
```

Default BF16 draft output: `artifacts/huggingface/KiwiLM-2-bf16`, ignored by Git. Existing
outputs are never overwritten. Use a new `--output-dir` for each reviewed build.
The builder verifies exact rounded-reference/bundle FP32 CPU logits at lengths 1, 31 and 512,
cached direct/rollover parity, and generated-text parity across two seeds and
cache modes. It writes `release.json` with checksums and pending release gates.
The original checkpoint is checked before/after and remains unchanged.

The bundle contains five inference files (`model.safetensors`, `config.json`,
`tokenizer.json`, `metadata.json`, `manifest.json`), plus the model card, MIT
license, package wheel, example, and release verification record. No optimizer
state, training data, authentication tokens, or original `.pt` checkpoint is
uploaded. **BF16 is the default storage format**, with `--dtype fp32` available
on both the builder and generic exporter. FP32 originals are preserved separately.
The package defaults to FP32 execution independently of storage format; native
BF16 is an explicit loader option where supported.

BF16 rounding means original FP32 logits/text need not match exactly. Export
checks compare against independently BF16-rounded weights instead, while recording
the differences from the original model. `--validation-data` enables paired
32-batch FP32 CPU loss checks (seed 143, batch 2, context 512), with a maximum
allowed increase of 0.01. Use `--validation-batches` to expand that audit. This
small rounding check is not a replacement for the published 200-batch evaluation.

## Independent tokenizer/data provenance

The generic exporter keeps strict dataset-fingerprint matching by default.
When using an earlier preparation with the same frozen tokenizer, provide the
original 1B job explicitly:

```bash
uv run --locked kiwilm export-safetensors \
  --checkpoint runs/colab/tpu-v6e1-final-1b-muon-resume1/latest.pt \
  --tokenizer-from data/smollm-architecture \
  --checkpoint-provenance runs/colab/tpu-v6e1-final-1b-muon-resume1/tpu-job.json \
  --variant dense-tpu-v6e1-muon-0.01-1b \
  --output-dir artifacts/huggingface/inference-only
```

The original job's fingerprint must match the checkpoint; its tokenizer SHA must
match both prepared metadata and the actual tokenizer bytes. The earlier
prepared-data fingerprint is retained separately in export provenance. There is
no blind `--allow-data-mismatch` export bypass.

## Freeze reviewed release source

The V3 cleanup freezes the reviewed pre-cleanup implementation at local tag
`kiwilm2-v2-1b` (`643aa81`). This tag includes the committed export tooling;
the original bundle was built earlier and retains its own wheel checksum and
dirty-worktree provenance. Cleanup edits themselves are not committed or pushed.
For any new release build, use a reviewed **clean checkout of the source tag**.
The builder refuses a dirty checkout or a tag that does not point to HEAD.
Push source/tag or upload a new bundle only with separate authorization.

After source freeze, rebuild the wheel and prepare a **new** output:

```bash
uv build --wheel --out-dir artifacts/huggingface/wheels-bf16
uv run --locked python scripts/prepare_kiwilm2_release.py \
  --source-tag kiwilm2-v2-1b \
  --wheel artifacts/huggingface/wheels-bf16/kiwilm-0.1.0-py3-none-any.whl \
  --output-dir artifacts/huggingface/KiwiLM-2-frozen
```

The tag must already exist, point to HEAD, and the worktree must be clean. The
script does not create tags or commits. Normal publication refuses unfrozen drafts.

## Private publication (separate explicit action)

Do not execute until the owner authorizes the upload and source is frozen:

```bash
hf auth login
uv run --locked python scripts/publish_kiwilm2_release.py \
  --bundle artifacts/huggingface/KiwiLM-2-frozen --publish
```

Without `--publish` the helper performs no remote operations. It creates a new
repository with `private=True`, checks visibility before and after uploading, and uploads
only the release-file allowlist. It never makes a repository public. It refuses
an existing repo rather than silently accepting unknown visibility or altering
it; an interrupted first publication needs an explicitly reviewed retry.
Private fresh-download verification requires an authenticated account with access.

With explicit owner authorization, an unfrozen bundle can instead be staged as a
**private draft**. This does not freeze source, create a tag, or change the pending
provenance fields in `release.json`:

```bash
uv run --locked python scripts/publish_kiwilm2_release.py \
  --bundle artifacts/huggingface/KiwiLM-2-bf16 --publish --stage-private-draft
```

The ordinary frozen-release gate remains in effect without that explicit flag.

## Fresh-download gate

Use the full model-repository commit SHA returned by upload, a new local folder,
and a clean package environment; do not use a cached copy as proof of downloading:

```bash
hf download Tasty-Kiwi/KiwiLM-2 --revision FULL_HF_COMMIT_SHA --local-dir downloaded-KiwiLM-2
python -m pip install ./downloaded-KiwiLM-2/kiwilm-0.1.0-py3-none-any.whl
kiwilm generate --checkpoint downloaded-KiwiLM-2 --prompt "Once upon a time, a fox found a box." --max-new-tokens 32 --temperature 0.8 --top-k 40 --seed 42 --cache auto --device cpu
```

The directory loader validates all inference checksums and metadata. Compare
this output with seed 42/cache auto in `release.json` using FP32 CPU and the same
Torch version; cross-device samples need not be identical. Repeat the builder's
rounded-reference logit parity against the preserved source checkpoint converted
to the declared storage dtype for final acceptance. Original FP32 logits and
sampled text are not required to be identical after BF16 rounding.
`examples/generate_from_hub.py` and `kiwilm.hub.load_pretrained` require a full Hub
commit SHA and download only inference files, not datasets or remote code.

Local export/unit/install tests are not a substitute for live private-Hub upload
and fresh-download proof. That gate stays pending until publication is approved.

References: [Hugging Face uploading](https://huggingface.co/docs/hub/models-uploading),
[model cards](https://huggingface.co/docs/hub/model-cards), and
[revision-pinned downloads](https://huggingface.co/docs/huggingface_hub/guides/download).
