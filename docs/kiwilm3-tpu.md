# KiwiLM 3 — TPU training and verified continuation

The [default-off notebook](../notebooks/kiwilm3-tpu.ipynb) now drives a separate
V3 trainer with token-based warmup/cosine decay, accumulation and BF16 AMP.
It **prepares** single-TPU qualification; live Colab/Drive recovery and B/C
corpus results are still pending. No session or training run is allocated by
this implementation. A/D remain dropped; all four candidates use dense SwiGLU.

## Runtime and objective

Manually select one TPU device in Colab. Setup reuses the isolated Python 3.12
worker with pinned torch/XLA **2.9.0**, rather than replacing notebook-kernel
torch. XLA requires BF16 and an actual TPU, without CPU fallback. CUDA
FP32/BF16 and CPU FP32 are explicit alternatives; FP16 is not implemented in
this new trainer. CUDA BF16 support is checked. Multi-device training is not
supported.

BF16 autocast covers forward/loss; master parameters and AdamW moments remain
FP32, and backward is outside autocast. BF16 does not require loss scaling.
This follows the matched [PyTorch/XLA AMP guidance](https://docs.pytorch.org/xla/release/r2.9/perf/amp.html).
Reuse `Runtime` only for backend synchronization/autocast/measurements, never
V2's model, objective, training state or parameter routing.

The objective remains M4's **unweighted variable-noise masked reconstruction
CE**, excluding the real MASK output class. It is not AR perplexity, an ELBO
or a complete diffusion likelihood. Corruption and clean-target validation
run on CPU with explicit generators. Fixed-shape CE uses ignored labels;
there is no dynamic masked-target gather on XLA. Accumulated gradients use
the **total selected-token count across microbatches**, not an average of
per-microbatch means. An entirely empty mask skips AdamW/weight decay.

The budget counts **non-padding input tokens**, not supervised tokens.
Masked-token counts and optimizer updates are separately recorded. The last
attempt uses a fixed-shape valid-token prefix to hit the exact budget; discarded
positions are excluded from attention, convolution and loss. Warmup/cosine uses
the attempt's end-token position. Tensor LR and loss-denominator inputs avoid
inserting changing scalar constants into XLA graphs. Fixed-noise validation
uses independent generators at 15/50/90/100% noise.

## User-run two-VM qualification

1. Build one reviewed wheel with `uv build --wheel`. Put it once in
   `MyDrive/KiwiLM3/packages/`. Reuse those exact bytes on both VMs. Large
   datasets stay in Drive and are copied VM-locally, not uploaded per run.
2. Open the notebook on a manually selected single-TPU runtime. Keep
   `QUALIFICATION=True`, `CANDIDATE='hybrid-12'`, and the tiny 4096-token
   controls. Enable mount/setup/demo preparation/preflight deliberately.
   Inspect backend identity, finite logits and counters before training.
3. Set `MODE='fresh'`, `STOP_AFTER_STEP=8`, `START_TRAINING=True`. This is a
   **synthetic width-16/context-16 infrastructure test**, not a corpus smoke.
   Confirm the paused result and verified Drive receipt at step 8, plus
   `previous.json` at step 4. Preserve the log, job identity, token count,
   learning rate and Linux boot ID. Never terminate before the commit succeeds.
4. Terminate VM 1 yourself. On a genuinely new TPU VM, use the same wheel,
   CONFIG, candidate, run name and deterministic demo. Change only
   `MODE='resume'`, `REQUIRE_NEW_VM=True`, `STOP_AFTER_STEP=None`. A distinct
   boot ID is required; changing a local directory is not a new VM.
5. Verify restore starts at the recorded step/token position before any
   update, then finishes at exactly 4096 input tokens. Verify contiguous
   training steps, finite loss/gradients, schedule continuity, backend identity,
   final checkpoint and compilation/fallback counters. Preserve both logs
   and receipts for review. Do not claim success from a finite loss alone.

The qualification retains actual 12/16 block counts and 31/63 kernels but
uses a tiny vocabulary/model. It does **not** establish full-width TPU memory,
speed, corpus quality or bitwise equivalence on TPU. The CPU test performs
exact uninterrupted-versus-fresh-process continuation; that remains local
evidence only. No automatically computed live-qualification flag is asserted.

## Persistence contract

V3 uses its own `kiwilm3-accelerator-training-v1` payload, independent of V2
and M4 local state. It contains model/tied-head values, FP32 AdamW moments and
step counters, token schedule/progress, Python/NumPy/torch/backend RNG,
explicit data/noise generators, exact metrics-prefix digest, and source VM ID.
XLA RNG capture/restore uses the matched
[official XLA APIs](https://github.com/pytorch/xla/blob/v2.9.0/torch_xla/core/xla_model.py).
Resume locks model/objective/masking, data fingerprint, tokenizer checksum,
reviewed package code, runtime/precision/hardware, seeds, budget and controls.
VM-local paths and pause target are not experiment identity fields.

The namespace is `KiwiLM3/checkpoints/v3-{run}-{candidate}-{job-digest}`.
The generic verified transport copies only bytes; **V3 validates its own
payload**. Step zero is committed before the first update. Periodic publication
blocks until checkpoint/metrics/job bytes pass read-back and SHA256 checks;
the latest pointer is committed last. Only the latest and previous distinct
committed steps are retained. Same-step publication does not discard the
previous distinct checkpoint. Failed staging directories can remain; there is
no broad cleanup of Drive or unrelated files.

Drive publication failures halt training. The newest VM-local `latest.pt`
survives a failed upload while the previous committed Drive head stays intact.
On an abrupt VM loss, uncommitted work since the last successful checkpoint
is not guaranteed recoverable. A mounted/available Drive root is mandatory
for accelerator training; no implicit local-only fallback. Keep one writer
per namespace; there is no distributed multi-writer Drive lease.

Fresh execution refuses existing local output or backup progress. Resume
requires an empty local output, verified manifest/checkpoint/job/tokenizer and
metrics prefix. No automatic fresh fallback, silent metrics truncation,
budget extension or relaxed source-version check occurs. If the latest bytes
are corrupt, previous bytes may be restored, but training behind a newer
committed head is refused. Preserve the originals and make an explicit recovery
decision; do not manually reset pointers to bypass the guard.

## Later matched B/C smoke — not launched

After live recovery passes, review and freeze these proposed controls for
`hybrid-12`, `attention-12`, `hybrid-16`, `attention-16`:

```python
QUALIFICATION = False
STOP_AFTER_STEP = None
RUN_NAME = 'bc-smoke-50m'
CONFIG.update(max_tokens=50_000_000, warmup_tokens=1_000_000,
              batch_size=8, grad_accum_steps=4,
              checkpoint_interval=500, eval_interval=500, eval_batches=20)
```

All candidates retain width/context 512, 8-head bidirectional attention/RoPE,
SwiGLU width 2048 and dropout 0.1. Hybrid repeats attention/BiConv31/BiConv63;
the other family uses attention only. Keep the same original prepared SmolLM
subset, converted MASK tokenizer, seeds, LR schedule, batches, evaluation
controls, hardware and precision. Prepare data separately with generic tools;
restore only a complete verified cache and validate the tokenizer's original
packed-ID compatibility. No data download is hidden in the trainer.

Candidate names/job digests create distinct backups. Use a new local `RUN_DIR`
per candidate; do not automatically run all four. Full-width preflight allocates
real weights and forwards before the first optimizer step. Record compile/
fallback counters, throughput/memory and numerical health; the tiny test cannot
substitute for this gate. AdamW defaults are a reviewable baseline, not a tuned
recommendation or an automatic 1B plan.

New accelerator state/metrics are **not** old M5 CPU suite reports. Existing
CPU comparison/inference loaders refuse this training format. Dedicated
post-run health/sampler evaluation, inference export and provenance-validated
B/C comparison integration remain the next implementation step; do not rename
`latest.pt` to `encoder.pt` or weaken old suite checks. No fabricated comparison
artifacts or architecture winner are generated before actual runs exist.

## Local verification

```bash
uv run --locked --extra browser --extra notebooks pytest -q
uv run --locked ruff check src scripts archive tests
uv run --locked --extra notebooks python scripts/build_kiwilm3_tpu_notebook.py
uv run --locked --extra notebooks jupyter nbconvert --execute --to html notebooks/kiwilm3-tpu.ipynb
```

Keep flags false when executing notebook defaults. Tests exercise CE/M4 parity,
accumulation, exact budgets, empty/nonfinite masks, schedule/validation isolation,
state integrity, strict job matching, Drive retention/failure and exact
fresh-process CPU continuation including dropout. No Colab sessions or
corpus/cloud training are started. Execution/HTML rendering checks are separate
from visual/manual Colab review.

### Verification record — 2026-10-05

**523 pytest tests pass** (26 new accelerator cases), with two existing ONNX
deprecation warnings. Seven playground tests, Ruff/format checks, locked
dependency validation, wheel build and `git diff --check` pass. All five
default-disabled notebook code cells execute with zero errors and only skip
messages; those outputs are retained. HTML rendering succeeds. Visual notebook
review remains manual because local-file browser access was previously rejected.
Real TPU execution, full-width performance/memory and fresh-Colab-VM Drive
continuation remain unverified. V2 source/formats, old notebooks and the frozen
V2 branch remain unchanged. No corpus/cloud training, commit, push, Hub update
or actual Drive mutation occurred.
