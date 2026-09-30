# Experimental TPU hardware smoke

**Qualification blocked after the completed 50M run:** throughput was 47.9k
tokens/s, but XLA device transfer broke the embedding/head weight tie. The
checkpoint therefore represents an untied model, and normal tied loading gives
incorrect validation/generation. See the
[50M analysis and reproducible audit](../examples/comparisons/kiwilm2-tpu-50m-smoke/analysis.md).
Do not resume these weights as canonical Dense or launch a larger run with the
current worker. Weight tying must be re-established after device transfer,
before optimizer construction, and checked through portable reload. No worker
fix or new training was performed as part of this analysis.

The completed 200-step hardware probe sustained 53.1k tokens/s on XLA BF16,
with two compiled graphs and no recorded CPU fallbacks. The launcher now runs
the full **50,000,000-token smoke**. The
[initial attempt report](../examples/comparisons/kiwilm2-tpu-hardware-smoke/analysis.md)
describes the earlier bootstrap failures, before the successful probe.

This is a separate single-chip Dense/Muon 0.01 hardware probe, not a change to
the ongoing GPU run or approval for a 1B run. Google has
[introduced v5e-1 into Colab's free tier](https://github.com/googlecolab/colabtools/issues/5566),
but availability varies and CLI allocations can consume compute units.

## Launch

From the repository root, with the frozen `data/smollm-smoke` present:

```bash
bash scripts/run_colab_kiwilm2_tpu_smoke.sh
```

The launcher allocates only `v5e1`, in a separate
`kiwilm2-tpu-v5e1-muon-smoke-50m` session. It verifies a real XLA matrix multiplication
before uploading the validated local smoke data in checksummed 4MiB chunks;
large single-file uploads can fail
at the Colab proxy. It never mounts/writes Drive or accesses 500M checkpoints.
Results go to `runs/colab/tpu-v5e1-muon-smoke-50m`, preserving the downloaded short
probe. For repeats, set new
`KIWILM_RESULT_DIR` and `COLAB_SESSION_NAME` values, not the GPU session's name.

The worker uses standalone Python 3.12 in an isolated environment, with matching
Torch/PyTorch-XLA 2.9.0
and the TPU runtime, following the
[versioned installation instructions](https://github.com/pytorch/xla/blob/v2.9.0/README.md).
The normal Windows CUDA environment and lockfile are unchanged. TPU uses
[BF16 autocast without gradient scaling](https://docs.pytorch.org/xla/master/perf/amp.html).

## Frozen controls and scope

- Dense backbone unchanged; context 512, seed 42, batch 8, accumulation 4.
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
  generator on the training device, followed by further training. This checks
  same-process reload; it does not establish VM-restart equivalence.
- After completion, the portable weights receive a CPU FP32 50-batch health
  audit (25 batches per seed 141/142), direct and rollover cache-parity checks,
  and a short generation sample. CPU diagnostic evidence is labeled explicitly;
  on-TPU cached generation remains unverified.

The worker allows two hours; the launcher caps preflight/setup at five minutes
and smoke execution at 125 minutes (uploads/downloads excluded). At the short
probe's steady rate, 50M training alone would take about 16 minutes; compilation,
validation, checkpoint packaging and CPU diagnostics add overhead. On failure or
exit, the launcher attempts periodic artifact/checkpoint recovery and stops only its own session. If the
network prevents cleanup, verify and manually stop that named TPU session.

## Resume

To resume a downloaded full-smoke checkpoint in a new VM:

```bash
KIWILM2_RESUME_FROM=runs/colab/tpu-v5e1-muon-smoke-50m/latest.pt \
KIWILM_RESULT_DIR=runs/colab/tpu-v5e1-muon-smoke-50m-resumed \
COLAB_SESSION_NAME=kiwilm2-tpu-v5e1-muon-smoke-50m-resumed \
bash scripts/run_colab_kiwilm2_tpu_smoke.sh
```

Resume uploads the checkpoint in verified chunks and requires matching model,
data fingerprint, backend and training configuration. The earlier 200-step
checkpoint has a different validation configuration; start the full smoke
fresh. Local logs newer than the saved step are truncated on same-directory
module resume. Checkpoints stay on the VM until downloaded; this launcher does
not mirror to Drive, so recovery cannot survive a lost VM before downloading.

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
