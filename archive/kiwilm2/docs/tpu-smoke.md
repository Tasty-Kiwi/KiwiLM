# Experimental TPU hardware smoke

**Corrected v6e-1 smoke passed:** exactly 50M tokens, 67.7k weighted steady
tokens/s, loss 4.292, 50/50 healthy batches, and aligned TPU/CPU portable reload.
See the [completed v6e-1 report](../../../examples/comparisons/kiwilm2-tpu-v6e1-50m-smoke/analysis.md).
Artifacts were downloaded and the session terminated. Fresh-VM continuation,
on-XLA cache parity and a matched idle GPU benchmark remain required before 1B.
The repeated cached v6e-1 smoke also completed at exactly 50M tokens; its model
weights and recorded loss/gradient trajectory match the previous run exactly.
Drive checkpoint publication and a two-VM continuation qualification are now
implemented and locally tested. The live reference leg committed steps 20/40;
allocation/authorization failures prevented the resumed leg. **Live fresh-VM
restart equivalence remains unverified.** The user chose to stop qualification
and proceed with the separate [1B launcher](tpu-1b.md); it is not gated on that
unfinished test, and does not represent a completed qualification.

**Previous v5e-1 smoke is not a valid Dense control:** throughput was 47.9k
tokens/s, but XLA device transfer broke the embedding/head weight tie. The
checkpoint therefore represents an untied model, and normal tied loading gives
incorrect validation/generation. See the
[50M analysis and reproducible audit](../../../examples/comparisons/kiwilm2-tpu-50m-smoke/analysis.md).
Do not resume those weights as canonical Dense. The corrected worker now
re-ties weights after device transfer, before optimizer construction; rejects
unequal checkpoint matrices; and checks ordinary CPU reconstruction against
the training device. The completed **v6e-1 50M smoke** started fresh;
it is not a continuation or promotion of the v5e-1 result.

The historical untied 200-step probe sustained 53.1k tokens/s on XLA BF16,
with two compiled graphs and no recorded CPU fallbacks. The launcher now runs
the full **50,000,000-token smoke**. The
[initial attempt report](../../../examples/comparisons/kiwilm2-tpu-hardware-smoke/analysis.md)
describes the earlier bootstrap failures, before the successful probe.

This is a separate single-chip Dense/Muon 0.01 hardware probe, not a change to
the ongoing GPU run or approval for a 1B run. Google has
[introduced v5e-1 into Colab's free tier](https://github.com/googlecolab/colabtools/issues/5566),
but availability varies and CLI allocations can consume compute units.

## Prepare, then explicitly start

From the repository root, with the frozen `data/smollm-smoke` present:

```bash
bash archive/kiwilm2/scripts/run_colab_kiwilm2_tpu_smoke.sh setup
```

Setup validates the frozen local data and any resume checkpoint before
allocation, installs the isolated environment, runs an XLA matrix-multiply
preflight, and prepares inputs. **It never starts training.** Omitting the
argument also means `setup`; calling the Python bootstrap without an action
only runs preflight. Setup mounts Drive interactively in your own terminal.
After setup, the TPU remains allocated and **can consume compute units while idle**.
Start training explicitly, with the same environment settings:

```bash
bash archive/kiwilm2/scripts/run_colab_kiwilm2_tpu_smoke.sh train
```

During training, JSON progress rows stream to the terminal as the worker emits
them (step 1, then every 10 updates). A heartbeat appears every 30 seconds
without a progress event, including compilation and evaluation pauses. Full
stdout/stderr and verbose XLA diagnostics remain in the append-only `worker.log`;
they are not replayed wholesale at completion. Worker failures print the last
40 log lines and still return an error. Ctrl+C still stops the allocated VM;
the first periodic checkpoint remains at step 500.
Drive-enabled jobs now synchronously mirror every checkpoint before continuing.
A `drive_checkpoint_committed` progress row acknowledges the verified recovery
point; terminal disconnection is not that acknowledgement.

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
existing one manually. Data-cache and checkpoint directories remain separate.

To reuse the already prepared GPU smoke cache, when it matches your local data:

```bash
export KIWILM2_TPU_DRIVE_CACHE=/content/drive/MyDrive/KiwiLM2/data/smoke-50000000-seed42
bash archive/kiwilm2/scripts/run_colab_kiwilm2_tpu_smoke.sh setup
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
  and at completion. Drive-enabled jobs publish verified backups at those same
  boundaries and stop if publication repeatedly fails. Throughput excludes
  checkpoint/validation/Drive time; session
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

## Checkpoint persistence and recovery

Drive is enabled by default. New full-smoke jobs use a separate namespace:

```text
/content/drive/MyDrive/KiwiLM2/checkpoints/tpu-v6e1-muon-smoke-50m-tied-<fingerprint-prefix>
```

`latest.json` points to the newest verified generation; `previous.json` retains
the previous distinct optimizer step. Each generation contains `latest.pt`, the
metrics sidecar, the frozen job, and a checksummed manifest. Copying and read-back
verification complete before the latest pointer is atomically published. Only
known superseded committed generations are pruned; an interrupted uncommitted
copy does not replace either recovery pointer. Failed attempts may leave orphan
generation directories for manual inspection, not automatically broad cleanup.

The checkpoint includes weights, Muon/AdamW state, step/token/LR schedule state,
the data generator, CPU/process RNG, scaler state, and XLA RNG state. The XLA
state uses the [versioned 2.9 RNG APIs](https://github.com/pytorch/xla/blob/v2.9.0/torch_xla/core/xla_model.py).
Backups are synchronous, not best-effort background copies: a disconnected mount
or failed publication gets bounded retries, then **training stops**. The local
checkpoint and the previous committed Drive recovery point are retained.

This is not a zero-loss guarantee. A sudden VM loss can discard updates since
the last committed checkpoint (up to the 500-update interval by default, about
8.192M tokens). Drive mount write/read verification is not a guarantee against
cloud-side data loss. Setting `KIWILM2_USE_DRIVE=0` explicitly disables both Drive
data caching and checkpoint protection; do not use it for a long run.

To restart an interrupted smoke on a fresh VM **directly from Drive**, choose a
new local/session name, and set the original locked backup directory for both
restore and publication:

```bash
export KIWILM2_TPU_DRIVE_RESUME="/content/drive/MyDrive/KiwiLM2/checkpoints/<original-namespace>"
export KIWILM2_TPU_DRIVE_BACKUP="$KIWILM2_TPU_DRIVE_RESUME"
export KIWILM_RESULT_DIR=runs/colab/tpu-v6e1-smoke-drive-resumed
export COLAB_SESSION_NAME=kiwilm2-tpu-v6e1-smoke-drive-resumed
bash archive/kiwilm2/scripts/run_colab_kiwilm2_tpu_smoke.sh setup
bash archive/kiwilm2/scripts/run_colab_kiwilm2_tpu_smoke.sh train
```

Replace the placeholder with the actual namespace printed during the original
run. Setup verifies the committed checkpoint locally, locks its exact SHA-256,
and copies the metrics sidecar. Missing/corrupt required checkpoints never fall
back to random initialization. A checksum-invalid latest generation can fall
back to the verified previous generation, but continuing behind a newer pointer
requires a new backup namespace; stale local resumes cannot overwrite newer
committed progress. Contract/data/model mismatches are rejected before updates.
The smoke scheduler cannot be repurposed as a 1B schedule by resuming its weights.

Alternatively, resume a downloaded checkpoint:

To resume a downloaded full-smoke checkpoint in a new VM:

```bash
export KIWILM2_RESUME_FROM="runs/colab/<interrupted-tied-smoke>/latest.pt"
export KIWILM_RESULT_DIR=runs/colab/tpu-v6e1-muon-smoke-50m-tied-resumed
export COLAB_SESSION_NAME=kiwilm2-tpu-v6e1-muon-smoke-50m-tied-resumed
bash archive/kiwilm2/scripts/run_colab_kiwilm2_tpu_smoke.sh setup
bash archive/kiwilm2/scripts/run_colab_kiwilm2_tpu_smoke.sh train
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
downloaded **only when Drive has been explicitly disabled**. A local checkpoint
cannot overwrite newer progress in an existing Drive namespace; choose a new
backup directory for a fork. Old completed runs were not retroactively backed
up by this change.

## Two-VM continuation qualification

The completed 50M checkpoint has no remaining schedule budget. To exercise real
optimizer updates after restart without altering that baseline, use a separate
short split run with the same model, data, seed, BF16, batch 8/accumulation 4,
Muon 0.01, and **50M learning-rate schedule**. It is a recovery test, not a quality
smoke or a 1B launch. Both phases use five fixed validation batches and checkpoint
every 20 updates to keep the test bounded.

VM A runs 40 updates uninterrupted, committing steps 20 and 40 to Drive. Its
launcher downloads results and stops that VM. Run each command yourself:

```bash
unset KIWILM2_RESUME_FROM
bash archive/kiwilm2/scripts/run_colab_kiwilm2_tpu_continuation.sh reference setup
bash archive/kiwilm2/scripts/run_colab_kiwilm2_tpu_continuation.sh reference train
```

After it completes, VM B restores the **step-20 checkpoint from Drive**, runs
updates 21–40, and compares its final state with VM A's committed step-40 state:

```bash
bash archive/kiwilm2/scripts/run_colab_kiwilm2_tpu_continuation.sh resume setup
bash archive/kiwilm2/scripts/run_colab_kiwilm2_tpu_continuation.sh resume train
```

The wrapper isolates session, result, and Drive names from ordinary smoke
settings. Default namespaces end in `continuation-restart-v1-reference` and
`continuation-restart-v1-resume`. Results are under
`runs/colab/tpu-v6e1-continuation-restart-v1-{reference,resume}`. For a repeat,
set `KIWILM2_TPU_TEST_ID=restart-v2` for **all four commands**, preserving old
evidence and backups. `COLAB_TPU` must remain the same for both phases.

Pass requires distinct recorded VM identities and exact equality of model
tensors, optimizer state, configs/fingerprint, data-generator state, process RNG,
XLA RNG, scaler state, and final step/token count. The expected endpoint is step
40 / 655,360 tokens; the resumed phase must start at step 20 / 327,680 tokens.
`summary.json.continuation_test.passed` records the decision; mismatch returns
an error and preserves the checkpoint. A CPU fresh-process test verifies the
same comparison mechanism locally, **not TPU restart equivalence**. Neither
`setup` command trains; it leaves a billable idle VM until `train` or `colab stop`.

No 1B trainer or run is enabled here. Qualify real Drive restoration, persistence
overhead, and fresh-VM state equality before adopting this path for a long run.

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
