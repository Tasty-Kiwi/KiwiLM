# KiwiLM 3 — M4 denoising prototype

M4 implements a real MASK tokenizer, variable-noise corruption, masked-token
cross-entropy, fixed validation and a bounded local CPU trainer with state
recovery. It does **not** implement iterative sampling, a diffusion ELBO,
production GPU/TPU training or automated Drive backups. There is no pretrained
V3 release. Training remains user-initiated.

## Tokenizer decision

V3 uses `MaskBPETokenizer`, a separate BPE artifact containing a real special
`[MASK]` token. Conversion appends it rather than replacing a content token:
all original IDs, BPE merges and normal-text encoding remain unchanged. A 32K
base becomes **32,001 entries, MASK ID 32000**. No V2 tokenizer, checkpoint,
packed corpus or published bundle is edited. The new JSON has its own identity.

The user permitted replacing IDs; appending avoids throwing away a learned
subword and allows safe reuse of prepared data. Training verifies that removing
only the new MASK registration reproduces the entire original tokenizer JSON,
not merely a matching vocabulary size. A different tokenizer is rejected.

Create a separate artifact (Bash and PowerShell use the same command):

```bash
uv run --locked python scripts/prepare_kiwilm3_tokenizer.py --source path/to/original-tokenizer.json --output artifacts/kiwilm3/tokenizer.json
```

Use the actual hashed tokenizer file referenced by the prepared dataset's
`metadata.json`. Existing destinations are refused. Reuse the saved V3 artifact
on resume; do not rebuild or overwrite it in place. Its reported
`tokenizer_sha256` is the SHA256 of canonical compact JSON, not the checksum of
the pretty-printed file. The prepared-data fingerprint separately identifies
the original corpus and tokenizer. Raw text containing `[MASK]` is rejected;
corruption inserts the ID directly. Decode removes MASK with special-token
skipping or displays it when `skip_special_tokens=False`.

M3's default `KiwiLM3Config(vocab_size=32000)` remains backward compatible.
For denoising, explicitly use `vocab_size=tokenizer.vocab_size`. The appended
tied embedding/head row adds 512 parameters at default width; M3's published
static profile remains the original 32K prototype profile.

## Frozen objective policy

For each sequence independently:

1. Draw `t ~ Uniform(0.01, 1)`. With probability 0.1, instead set `t=1`.
2. At each eligible position, draw a Bernoulli mask with probability `t`.
3. Replace selected IDs with MASK and pass only corrupted IDs, the attention
   mask and `t` into the encoder.
4. Predict the original token at selected positions.

Padding, UNK, BOS and EOS are protected by default. An explicit boolean
`protected_mask` additionally preserves prompt slots. An attention mask can
exclude other positions, but cannot turn padding into content. Protected and
unselected positions are not loss targets. At `t=1`, all **eligible content**
is masked; boundary tokens may remain. Explicit `t=0` is allowed for diagnostic
checks but is not sampled during default training. No 80/10/10 replacements or
forced-one-mask adjustment are used.

Loss is:

`sum(CE(logits[position, :mask_id], original_id)) / number_of_masked_positions`

The sum/count cover the entire batch, not a mean of per-sequence means. Each
masked token has weight one. Longer/high-noise sequences therefore contribute
more targets; this is an intentional reconstruction baseline, **not** an unbiased
variational diffusion objective. The appended MASK output is excluded from
clean prediction support. Ignored positions have zero logit gradient. Empty
mask batches return differentiable zero and skip the optimizer, including
AdamW weight decay; metrics show `null` loss/accuracy rather than fabricated
measurements.

This distinction matters: the continuous-time variational masked-diffusion
objective is schedule-weighted cross-entropy, not automatically equivalent to
this normalization and endpoint mixture. See
[Shi et al., Simplified and Generalized Masked Diffusion for Discrete Data](https://arxiv.org/abs/2406.04329).
Do not report `exp(masked loss)` as V2-comparable autoregressive perplexity.

## Package API

```python
import torch
from kiwilm.data import PreparedTokenData
from kiwilm.v3 import KiwiLM3Config, build_encoder
from kiwilm.v3.tokenizer import MaskBPETokenizer
from kiwilm.v3.trainer import DenoisingTrainConfig, DenoisingTrainer

data = PreparedTokenData("data/smollm-smoke")
tokenizer = MaskBPETokenizer.load("artifacts/kiwilm3/tokenizer.json")
torch.manual_seed(42)
model = build_encoder(KiwiLM3Config(
    vocab_size=tokenizer.vocab_size, num_blocks=12,
    d_model=16, num_heads=2, context_length=16, swiglu_dim=48,
    noise_embedding_dim=16, dropout=0.1,
))
trainer = DenoisingTrainer(model, tokenizer, data, DenoisingTrainConfig(max_steps=20))
# Explicit user action, not an import-side effect:
# print(trainer.train_step())
# print(trainer.evaluate(batches=4))
```

`corrupt_tokens` and `denoising_forward` also expose boolean prompt protection
and explicit per-sequence levels for diagnostics and future infilling. The
prepared-data adapter uses clean input windows; its next-token labels are
discarded, never used or shifted into the reconstruction objective.

The optimizer-step prototype deliberately requires **CPU FP32**, constant-LR
AdamW, no gradient accumulation and at most 1,000 step attempts. It is not an
accelerator throughput test or long-run scheduler. Lower-level model/loss code
uses tensor-device-aware operations, but GPU/TPU correctness/performance are
unqualified. Before longer training, add and verify the accelerator adapter,
token budget/schedule, accumulation, metric persistence and Drive transport.

Validation uses fixed windows and RNG seeds at noise 0.15, 0.5, 0.9 and 1.0.
It aggregates CE sums and correct predictions by masked-token counts, restoring
the model's prior train/eval mode. It does not advance either training RNG.

## Notebook and local recovery

Open [kiwilm3-denoising.ipynb](../notebooks/kiwilm3-denoising.ipynb). It is a new
thin CPU companion; the three M2 recovery notebooks are unchanged. All installs,
demo preparation, training, validation, saves and resumes are opt-in. Follow
the setup/preparation/preflight cells before enabling any step. Never use it
as a TPU launcher. Its synthetic repeated sentence is a correctness fixture,
not an evaluation corpus or evidence of model quality.

`v3/checkpoints.py` writes separate exclusive local state files containing:
model, optimizer, completed attempts/updates/tokens, data and noise generator
states, global Python/torch RNG and a resume contract. The contract locks model,
masking/objective, training settings, V3 tokenizer identity, original data
fingerprint, package-code digest and CPU runtime. Missing state, a changed
contract, incomplete/failed steps or file checksum mismatches are refused.

```python
import hashlib
from kiwilm.v3.checkpoints import save_training_state, load_training_state

path = save_training_state(trainer, "runs/m4/step4.pt")
original_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
# Reconstruct an identical trainer on the new process, then:
# load_training_state(new_trainer, path, expected_sha256=original_sha256)
```

Save the checksum receipt separately; a checksum recomputed from an untrusted
download is not an independent integrity check. Use a new filename per save.
These files are **not** the M3 inference artifacts, V2 checkpoints or the
production latest/previous Drive store. No upload, retention or Drive recovery
is claimed here. An interrupted/failed step must restore the last verified
state, not silently continue with consumed RNGs. NumPy/XLA RNG is not used by
this CPU prototype's stochastic training path.

## Verification and remaining gates

```bash
uv run --locked --extra browser --extra notebooks pytest -q
uv run --locked python scripts/build_kiwilm3_denoising_notebook.py
```

Tests cover tokenizer conversion/JSON/control-token behavior, mask frequencies
and endpoints, padding/prompt isolation, exact loss normalization/gradients,
no clean-target input channel, empty-mask skips, fixed validation, stable toy
learning and exact fresh-process continuation with dropout/data/noise restored.
These checks establish the local M4 prototype, not general generation quality.

The notebook's default-disabled cells execute in a local Jupyter kernel. Its
executed copy is inspected for errors/skip outputs; **visual HTML preview remains
unverified** because the browser rejected local-file access. To inspect manually,
open the notebook in Jupyter/Colab or run:

```bash
uv run --locked --extra notebooks jupyter nbconvert --execute --to html notebooks/kiwilm3-denoising.ipynb
```

Keep action flags false for this safety check. No training/cloud session was
launched outside bounded automated toy tests. GPU/TPU/Drive acceptance remains
pending. M5 adds iterative reveal/infilling; corpus-scale selection stays in
later matched experiments.

### Local verification record — 2026-10-05

The complete locked environment passes **456 pytest tests**, including 54 M4
cases, with two existing ONNX deprecation warnings. All seven playground tests,
Ruff/format checks, `git diff --check`, offline dependency-lock validation and
wheel build pass. The notebook's seven default-disabled code cells execute
with zero errors and only skipped-action outputs. Visual inspection and live
Colab/Drive qualification remain unverified. V2 source/artifacts and the frozen
branch are unchanged; no Git commit/push or Hub update was performed.
