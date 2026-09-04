# Experimental TPU hardware smoke

Current evidence: [initial attempt report](../examples/comparisons/kiwilm2-tpu-hardware-smoke/analysis.md).
The launcher is implemented; real TPU training remains unverified.

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
`kiwilm2-tpu-v5e1-muon-smoke` session. It verifies a real XLA matrix multiplication
before uploading the validated local smoke data in checksummed 4MiB chunks;
large single-file uploads can fail
at the Colab proxy. It never mounts/writes Drive or accesses 500M checkpoints.
Results go to `runs/colab/tpu-v5e1-muon-smoke`. For repeats, set new
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
  warmup. The first probe stops after 200 updates / 3,276,800 tokens.
- First 20 updates excluded from steady throughput but included in wall time.
- Five fixed validation batches for inexpensive compatibility checking, not
  architecture selection. BF16 is not an exact continuation of the FP16 run.

The worker allows 15 minutes for the probe; the launcher caps preflight/setup
at five minutes and probe execution at 15 minutes (uploads/downloads excluded). On failure or
exit, the launcher attempts log recovery and stops only its own session. If the
network prevents cleanup, verify and manually stop that named TPU session.

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
  --output-dir "runs\cuda-muon-hardware-smoke" `
  --steps 200 --warmup-steps 20
```

Compare identical fingerprints, steps, batch/accumulation, optimizer and schedule.
Label BF16-versus-FP16 explicitly; a BF16-capable GPU can also run BF16 to isolate
numerical differences. CPU support exists for tiny tests, not speed claims.

Only after passing this probe should TPU advance to a full 50M smoke with 50
validation batches, generation/cached-parity checks, and verified on-TPU resume.
The short probe explicitly does **not** measure cached-generation parity.
Estimate a 1B run only after including compile, evaluation, checkpoint and
restart overhead—not just extrapolating peak throughput. No 1B job starts here.
