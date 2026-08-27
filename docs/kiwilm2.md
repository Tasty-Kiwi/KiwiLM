# KiwiLM 2 experiment runbook

KiwiLM 2 is a controlled architecture experiment, not a continuation of the
earlier 5M-parameter TinyStories ranking. Both variants use this frozen stack:

```text
32K token embedding + hashed bigram/trigram embeddings
  -> GQA -> Conv31 -> Conv63
  -> GQA -> Conv31 -> Conv63
  -> GQA -> Conv31 -> Conv63
  -> GQA
  -> final RMSNorm -> tied LM head
```

Every mixer is followed by a second pre-RMSNorm residual branch. `kiwilm2`
uses a 1536-wide SwiGLU on that branch. `kiwilm2_slim` uses a width-preserving
learned diagonal -> orthonormal FWHT -> SiLU -> learned diagonal -> FWHT mixer.
Slim is intentionally not depth- or parameter-matched.

The shared defaults are context 512, width 512, 8 query heads, 2 KV heads,
cached RoPE on GQA only, six causal depthwise kernels `31/63/31/63/31/63`, no
dropout, tied embeddings, and 16,384 buckets for each n-gram order. The
n-gram hash at position `t` uses only tokens at or before `t`.

## Frozen default accounting

| Variant | Total params | Dense/non-embedding | Token embedding | N-gram tables | Estimated FLOPs/token at 512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| KiwiLM 2 | 64,252,416 | 31,091,200 | 16,384,000 | 16,777,216 | 99,117,056 |
| KiwiLM 2 Slim | 40,679,936 | 7,518,720 | 16,384,000 | 16,777,216 | 52,115,456 |

Both variants use 1,048,576 bytes of fp16 KV cache per sequence at context
512. FLOPs are a static estimate with multiply-add counted as two operations;
hashing, lookups, norms, and activations are omitted. Reproduce the report:

```bash
uv run kiwilm profile-kiwilm2 --architecture kiwilm2
uv run kiwilm profile-kiwilm2 --architecture kiwilm2_slim
```

## Prepare the controlled corpus

The data recipe streams only the `fineweb-edu-dedup` and `cosmopedia-v2`
configs of `HuggingFaceTB/smollm-corpus`. It resolves `main` to an immutable
dataset SHA, reserves a disjoint prefix from both sources for validation, and
mixes the remaining streams with a seeded 70/30 schedule. Python-Edu is not
loaded. Packed split sizes are exact token counts, including a deterministic
truncation of the last document.

Train the 32K byte-level BPE once on the smoke source, then reuse it for every
larger subset:

```bash
uv run kiwilm prepare-smollm \
  --profile smoke \
  --output-dir data/smollm-smoke

uv run kiwilm prepare-smollm \
  --profile architecture \
  --tokenizer-from data/smollm-smoke \
  --output-dir data/smollm-architecture

uv run kiwilm prepare-smollm \
  --profile final-500m \
  --tokenizer-from data/smollm-smoke \
  --output-dir data/smollm-final-500m
```

Profiles contain 50M, 250M, 500M, or 1B training tokens plus 2M validation
tokens. Use `--train-tokens 25000000` for the lower 25M smoke boundary. Larger
profiles reject a missing `--tokenizer-from` to prevent accidental tokenizer
drift.

## Run the comparisons

The phase runner reconstructs the prepared data independently with the same
seed for each candidate. Token budget, batches, data order, schedule,
tokenizer, and context are therefore identical. It writes checkpoints,
JSONL metrics, samples, per-run profiles, and one `experiment.json` manifest.
The manifest also records a post-training health batch with every mixer/MLP
activation RMS and gradient norm plus n-gram hash occupancy and table gradients.

```bash
uv run python scripts/run_kiwilm2_experiment.py \
  --phase smoke \
  --data-dir data/smollm-smoke \
  --output-dir runs/kiwilm2-smoke \
  --device cuda \
  --precision bf16

uv run python scripts/run_kiwilm2_experiment.py \
  --phase architecture \
  --data-dir data/smollm-architecture \
  --output-dir runs/kiwilm2-architecture \
  --device cuda \
  --precision bf16
```

The smoke phase is an implementation/stability gate only. Do not select the
winner from it. Metrics include validation loss/perplexity, valid and model
tokens/second, padding fraction, current accelerator memory, and peak memory in
the final summary. Cached decoding is tested against uncached decoding across
the context rollover boundary.

After both AdamW baselines, run the optional KiwiLM 2-only Muon sweep:

```bash
uv run python scripts/run_kiwilm2_experiment.py \
  --phase smoke \
  --data-dir data/smollm-smoke \
  --output-dir runs/kiwilm2-smoke-muon \
  --device cuda \
  --precision bf16 \
  --muon-lrs 0.01 0.02 0.04
```

Muon receives untied dense 2D `Linear.weight` matrices. AdamW receives token
and n-gram embeddings, tied LM-head storage, norms, biases, Hadamard diagonals,
and all depthwise convolution kernels. Only promote Muon if its validation
curve reaches a target loss in fewer tokens than AdamW.

## Colab launchers

The Colab launchers require the `colab` CLI, but they do not require prepared
local data. Authenticate the CLI before launching a billable GPU session:

```bash
uv tool install google-colab-cli
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
colab sessions
```

Run both AdamW smoke candidates sequentially on fresh T4 sessions:

```bash
scripts/run_colab_kiwilm2_smoke.sh
```

Run the controlled 250M-token pair:

```bash
scripts/run_colab_kiwilm2_architecture.sh
```

Run the dense KiwiLM 2 Muon sweep at `0.01 / 0.02 / 0.04`:

```bash
KIWILM2_PHASE=smoke scripts/run_colab_kiwilm2_muon_sweep.sh
```

For one customized job, use the generic launcher:

```bash
KIWILM2_PHASE=smoke \
KIWILM2_VARIANT=kiwilm2 \
KIWILM2_OPTIMIZER=adamw \
COLAB_GPU=L4 \
scripts/run_colab_kiwilm2.sh
```

Useful overrides are `KIWILM2_BATCH_SIZE`, `KIWILM2_GRAD_ACCUM_STEPS`,
`KIWILM2_PRECISION`, `COLAB_TIMEOUT_SECONDS`, `COLAB_SESSION_NAME`, and
`KIWILM_RESULT_DIR`. T4 and fp16 are the defaults.

After allocating the VM, the launcher opens the interactive Google Drive mount
flow. Complete that prompt once for each fresh Colab session. The remote worker
then streams FineWeb-Edu and Cosmopedia directly in the VM, resolves the dataset
revision, prepares exact packed splits, and validates the fingerprint, source
mix, Python-Edu exclusion, 32K tokenizer, and token budget before training.

By default, persistent files live below `My Drive/KiwiLM2`:

```text
KiwiLM2/
  tokenizer/                 shared frozen tokenizer bundle
  data/<phase>-<tokens>-seed<seed>/  complete prepared-data cache
  checkpoints/<run-key>/     latest.pt, best.pt, metrics, job, summary
```

The first candidate prepares and caches its phase data; later candidates copy
the same cache to fast VM-local storage. Checkpoint folders include a hash of
all resume-locked settings, preventing an incompatible automatic resume. While
training, each atomically completed local checkpoint is mirrored to Drive by a
background worker. Relaunching the same job automatically restores `latest.pt`,
`best.pt`, and `metrics.jsonl` from that folder.

Set `KIWILM2_DRIVE_ROOT=/content/drive/MyDrive/SomeFolder` to use another Drive
folder. Set `KIWILM2_DRIVE_BACKUPS=0` to skip mounting and persistence; data is
still prepared in the VM, but separate candidates will each prepare it again
and interrupted sessions will only have the launcher's emergency download.

Every job builds and uploads the current wheel, prepares or restores data,
trains one candidate, records the model profile and block/n-gram health report,
downloads checksummed artifact chunks, and stops the Colab session through an
exit trap. A successful local result contains `latest.pt`, `best.pt`,
`metrics.jsonl`, `summary.json`, and the exact `job.json`.

To resume a recovered or previous run with its original locked configuration:

```bash
KIWILM2_PHASE=smoke \
KIWILM2_VARIANT=kiwilm2 \
KIWILM2_RESUME_FROM=runs/colab/kiwilm2-smoke/kiwilm2-adamw/latest.pt \
scripts/run_colab_kiwilm2.sh
```

An explicit `KIWILM2_RESUME_FROM` takes precedence over the automatic Drive
resume. When `best.pt` and `metrics.jsonl` are beside that checkpoint, the
launcher uploads them too. If execution is interrupted before artifact packing,
the exit trap attempts to download `emergency-latest.pt`, `emergency-best.pt`,
and `emergency-metrics.jsonl` before releasing the VM; the last completed
periodic checkpoint should also remain in Drive.

## External evaluation

Prepare TinyStories and SimpleStories with the frozen SmolLM tokenizer, then
opt into cross-dataset evaluation explicitly:

```bash
uv run kiwilm prepare \
  --output-dir data/tinystories-kiwilm2-eval \
  --tokenizer-from data/smollm-smoke \
  --vocab-size 32000

uv run kiwilm prepare-simplestories \
  --output-dir data/simplestories-kiwilm2-eval \
  --tokenizer-from data/smollm-smoke \
  --vocab-size 32000

uv run kiwilm evaluate \
  --data-dir data/tinystories-kiwilm2-eval \
  --checkpoint runs/kiwilm2-architecture/kiwilm2-adamw/best.pt \
  --allow-data-mismatch
```

Run the existing counterfactual retrieval suite with the checkpoint's own
SmolLM prepared-data fingerprint:

```bash
uv run python scripts/evaluate_context_retrieval.py \
  --data-dir data/smollm-architecture \
  --context-length 512 \
  --checkpoint runs/kiwilm2-architecture/kiwilm2-adamw/best.pt \
  --checkpoint runs/kiwilm2-architecture/kiwilm2-slim-adamw/best.pt \
  --label KiwiLM-2 \
  --label KiwiLM-2-Slim \
  --output-dir runs/kiwilm2-architecture/retrieval
```

General side-by-side generation remains available through `kiwilm compare`.
Keep mHC and dynamic routing out of these configs so all 2.0 checkpoints retain
the fixed experimental meaning.
