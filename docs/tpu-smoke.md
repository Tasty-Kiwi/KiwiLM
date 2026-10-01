# Experimental TPU hardware smoke

**Corrected v6e-1 smoke passed:** exactly 50M tokens, 67.7k weighted steady
tokens/s, loss 4.292, 50/50 healthy batches, and aligned TPU/CPU portable reload.
See the [completed v6e-1 report](../examples/comparisons/kiwilm2-tpu-v6e1-50m-smoke/analysis.md).
Artifacts were downloaded and the session terminated. Fresh-VM continuation,
on-XLA cache parity and a matched idle GPU benchmark remain required before 1B.

**Previous v5e-1 smoke is not a valid Dense control:** throughput was 47.9k
tokens/s, but XLA device transfer broke the embedding/head weight tie. The
checkpoint therefore represents an untied model, and normal tied loading gives
incorrect validation/generation. See the
[50M analysis and reproducible audit](../examples/comparisons/kiwilm2-tpu-50m-smoke/analysis.md).
Do not resume those weights as canonical Dense. The corrected worker now
re-ties weights after device transfer, before optimizer construction; rejects
unequal checkpoint matrices; and checks ordinary CPU reconstruction against
the training device. The completed **v6e-1 50M smoke** started fresh;
it is not a continuation or promotion of the v5e-1 result.

The historical untied 200-step probe sustained 53.1k tokens/s on XLA BF16,
with two compiled graphs and no recorded CPU fallbacks. The launcher now runs
the full **50,000,000-token smoke**. The
[initial attempt report](../examples/comparisons/kiwilm2-tpu-hardware-smoke/analysis.md)
describes the earlier bootstrap failures, before the successful probe.

This is a separate single-chip Dense/Muon 0.01 hardware probe, not a change to
the ongoing GPU run or approval for a 1B run. Google has
[introduced v5e-1 into Colab's free tier](https://github.com/googlecolab/colabtools/issues/5566),
but availability varies and CLI allocations can consume compute units.

## Prepare, then explicitly start

From the repository root, with the frozen `data/smollm-smoke` present:

```bash
bash scripts/run_colab_kiwilm2_tpu_smoke.sh setup
```

Setup validates the frozen local data and any resume checkpoint before
allocation, installs the isolated environment, runs an XLA matrix-multiply
preflight, and prepares inputs. **It never starts training.** Omitting the
argument also means `setup`; calling the Python bootstrap without an action
only runs preflight. Setup mounts Drive interactively in your own terminal.
After setup, the TPU remains allocated and **can consume compute units while idle**.
Start training explicitly, with the same environment settings:

```bash
bash scripts/run_colab_kiwilm2_tpu_smoke.sh train
```

During training, JSON progress rows stream to the terminal as the worker emits
them (step 1, then every 10 updates). A heartbeat appears every 30 seconds
without a progress event, including compilation and evaluation pauses. Full
stdout/stderr and verbose XLA diagnostics remain in the append-only `worker.log`;
they are not replayed wholesale at completion. Worker failures print the last
40 log lines and still return an error. Ctrl+C still stops the allocated VM;
the first periodic checkpoint remains at step 500.

Or release the default session without training:

```bash
colab stop -s kiwilm2-tpu-v6e1-muon-smoke-50m-tied-cached
```

The launcher defaults to `v6e1`, in the separate
`kiwilm2-tpu-v6e1-muon-smoke-50m-tied-cached` session. Set `COLAB_TPU=v5e1` to test
v5e-1; other values are rejected and no automatic substitution occurs.
Results go to `runs/colab/tpu-v6e1-muon-smoke-50m-tied-cached`, preserving all previous
probe and smoke artifacts. Local ownership and matching local/remote job hashes
are required by `train`; a missing or changed lock refuses training without
stopping an unverified session. For repeats, set new
`KIWILM_RESULT_DIR` and `COLAB_SESSION_NAME` values, not the GPU session's name.

## Avoid repeated data uploads

Setup first checks a fingerprint-qualified Drive directory:
`/content/drive/MyDrive/KiwiLM2/data/tpu-smoke-<data-fingerprint>`.
A cache hit copies the exact tokenizer and packed splits to VM-local storage
and validates sizes, metadata and SHA-256 checksums. It does not regenerate
data or train from Drive. Existing mismatched, corrupt or incomplete caches fail
setup without being overwritten; choose a new cache directory or inspect the
existing one manually. This does not change any checkpoint backups.

To reuse the already prepared GPU smoke cache, when it matches your local data:

```bash
export KIWILM2_TPU_DRIVE_CACHE=/content/drive/MyDrive/KiwiLM2/data/smoke-50000000-seed42
bash scripts/run_colab_kiwilm2_tpu_smoke.sh setup
```

On a cache miss, setup gzip-compresses the exact prepared files, uploads
checksummed **4MiB chunks with three parallel workers**, and publishes the
manifest last. Each upload has bounded retries. It then verifies the restored
files and publishes a new Drive cache, writing `.complete` only after a verified
copy. Later sessions can avoid the laptop-to-VM data upload entirely. Actual
compression savings and transfer speed depend on the data and connection;
the new transfer path has local tests, not a measured live speedup yet.

Set `KIWILM2_UPLOAD_WORKERS=1` for a fragile connection (allowed range 1–4).
Set `KIWILM2_USE_DRIVE=0` for compressed parallel uploads only, with no Drive
mount or cache. Large single-file uploads remain avoided because they can fail
at the Colab proxy. A missing/disconnected mount is an error, not a cache miss.

The worker uses standalone Python 3.12 in an isolated environment, with matching
Torch/PyTorch-XLA 2.9.0
and the TPU runtime, following the
[versioned installation instructions](https://github.com/pytorch/xla/blob/v2.9.0/README.md).
The normal Windows CUDA environment and lockfile are unchanged. TPU uses
[BF16 autocast without gradient scaling](https://docs.pytorch.org/xla/master/perf/amp.html).

## Frozen controls and scope

- Dense backbone unchanged; context 512, seed 42, batch 8, accumulation 4.
- Embedding/head parameter identity is restored after device transfer. The
  parameter count stays 64,252,416; the shared head is excluded from Muon.
- Muon 0.01 plus auxiliary AdamW 0.0003; original clipping, decay and betas.
- Fresh initialization, frozen tokenizer/data, 50M token LR schedule with 1M
  warmup. The run stops at exactly 50M tokens: 3,052 updates at the default
  settings, with 12,416 valid targets in the masked final update.
- First 20 updates excluded from steady throughput but included in wall time.
- Fifty fixed validation batches every 500 updates and at completion, seed 43.
- Atomic checkpoints and downloadable 4MiB artifact chunks every 500 updates
  and at completion. Throughput excludes checkpoint/validation time; session
  wall time records that overhead and CPU diagnostics, excluding final artifact
  packaging, environment setup and transfers. Partial final updates are excluded from
  steady throughput.
- The first saved checkpoint is reloaded into the model, optimizer and data
  generator on the training device, with exact pre/post-reload logits and
  tied identity checked, followed by further training. This checks
  same-process reload; it does not establish VM-restart equivalence.
- At the first checkpoint, five fixed batches compare training-device loss
  against the saved ordinary CPU FP32 reconstruction. At completion, all fifty
  batches are compared. Absolute loss difference and first-batch relative-logit
  RMS must each be at most 0.02; otherwise the worker stops with an error.
  Checkpoint artifacts are packaged before checking, preserving failure evidence.
- After completion, the portable weights receive a CPU FP32 50-batch health
  audit (25 batches per seed 141/142), direct and rollover cache-parity checks,
  and a short generation sample. CPU diagnostic evidence is labeled explicitly;
  on-TPU cached generation remains unverified.

The worker allows two hours; the launcher caps preflight at five minutes, each
input-preparation call at fifteen minutes, and smoke execution at 125 minutes
(uploads/downloads excluded). At the short
probe's steady rate, 50M training alone would take about 16 minutes; compilation,
validation, checkpoint packaging and CPU diagnostics add overhead. On failure or
training exit, the launcher attempts periodic artifact/checkpoint recovery and
stops only its verified session. Failed setup stops only the session it just
created; successful setup deliberately leaves it available for `train`. If the
network prevents cleanup, verify and manually stop that named TPU session.

## Resume

To resume a downloaded full-smoke checkpoint in a new VM:

```bash
export KIWILM2_RESUME_FROM="runs/colab/<interrupted-tied-smoke>/latest.pt"
export KIWILM_RESULT_DIR=runs/colab/tpu-v6e1-muon-smoke-50m-tied-resumed
export COLAB_SESSION_NAME=kiwilm2-tpu-v6e1-muon-smoke-50m-tied-resumed
bash scripts/run_colab_kiwilm2_tpu_smoke.sh setup
bash scripts/run_colab_kiwilm2_tpu_smoke.sh train
```

Replace the placeholder with an interrupted, corrected tied-smoke checkpoint;
a completed 50M checkpoint has no remaining training budget. Resume uploads the
checkpoint in verified parallel chunks and requires matching model,
data fingerprint, backend and training configuration. The corrected smoke uses
the `single-device-smoke-v2-tied` contract: all old v1 checkpoints are rejected,
and unequal matrices are rejected before allocation. Start the corrected smoke
fresh. The setup job locks the uploaded checkpoint's SHA-256; a data-cache hit
needs only the checkpoint upload. Local logs newer than the saved step are
truncated on same-directory module resume. Checkpoints stay on the VM until
downloaded; this launcher does
not mirror checkpoints to Drive, so recovery cannot survive a lost VM before
downloading. Drive caching here is for frozen data only.

## Interpretation

`summary.json` records synchronized tokens/s, first-step latency, wall time,
loss/PPL, device memory, package versions, configurations, data/tokenizer hashes,
and XLA counters before/after warmup. `worker.log` and `xla-metrics.txt` expose
compilations and CPU fallbacks. `latest.pt` is CPU-portable and includes optimizer
and data-generator state. The probe refuses ordinary GPU training checkpoints;
same-backend smoke checkpoints can resume with the module's `--resume` option.

The experimental `TensorMuon` keeps changing learning rates and Adam correction
steps as device tensors to avoid new Python scalar graph constants each update.
Its math is tested against the unchanged reference optimizer on CPU. Native
attention and depthwise-convolution lowering still need real TPU verification.
See [XLA troubleshooting](https://docs.pytorch.org/xla/master/learn/troubleshoot.html).

Require finite losses/nonzero finite gradients, stable compilation counts after
warmup, no unexpected CPU fallback, memory headroom, and better synchronized
throughput/end-to-end time than a matched idle GPU control. The historical
~17.5k tok/s T4 log is contextual only: the training loop and precision differ.

Matched CUDA control, in PowerShell:

```powershell
uv run --locked python -m kiwilm.tpu_smoke `
  --device cuda --precision fp16 `
  --data-dir "data\smollm-smoke" `
  --output-dir "runs\cuda-muon-hardware-smoke-50m" `
  --warmup-steps 20 --eval-batches 50
```

Compare identical fingerprints, steps, batch/accumulation, optimizer and schedule.
Label BF16-versus-FP16 explicitly; a BF16-capable GPU can also run BF16 to isolate
numerical differences. CPU support exists for tiny tests, not speed claims.

The full smoke now provides convergence, periodic validation/checkpoints,
same-process training-device checkpoint reload, and CPU health/cache checks.
Check a real VM restart and on-TPU cache parity separately before production.
Estimate a 1B run only after including compile, evaluation, checkpoint and
restart overhead—not just extrapolating peak throughput. No 1B job starts here.
