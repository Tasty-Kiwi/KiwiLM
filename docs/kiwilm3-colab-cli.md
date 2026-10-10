# KiwiLM 3 — script-driven Colab NVIDIA / TPU workflow

The user-run CLI is the primary control surface. Following repeated TPU
allocation timeouts, **NVIDIA CUDA BF16 is the current path**. TPU scripts and
the [notebook](../notebooks/kiwilm3-tpu.ipynb)
remains an unchanged backup. No notebook upload/browser connection is needed
for the CLI path, except optional manual Drive mounting. **This still uses
Colab's backend**; it cannot guarantee a connection if Colab/API/network/quota
is unavailable. Changing hardware is not a fix for a backend/network outage.
No successful live GPU/TPU execution is claimed here.

The launcher reuses the actual V3 denoising trainer and verified Drive
transport, not the archived V2 training algorithm. A/D remain dropped; B/C
are hybrid/attention-only at 12/16 blocks, dense SwiGLU throughout. Read the
[objective/runtime/recovery contract](kiwilm3-tpu.md) before corpus training.

## Stages and safety

- No arguments or `plan`: print instructions, **no allocation or training**.
- `run [setup options]`: explicitly allocate/setup, preflight, submit training,
  watch, verify/download artifacts, and stop on completion/pause. One command,
  same safety checks; interactive Drive authorization still requires you.
- `resume-run --from-state ORIGINAL --state-dir NEW`: the same combined flow
  on a new VM, reusing the exact original reviewed wheel/bootstrap and controls.
- `setup`: freeze wheel/bootstrap/controls locally, allocate one explicitly
  named GPU or TPU, authorize/mount Drive, install the pinned isolated worker, and
  prepare VM-local inputs. **No optimizer steps.**
- `preflight`: separately check the real model/backend and lock its identity.
- `train`: explicitly submit a **detached** VM supervisor. Returns without
  waiting for training completion. A local intent and exclusive VM claim
  refuse duplicate submissions, including an ambiguous socket timeout.
- `status`, `logs`, `watch`: inspect stages, PID liveness, last metrics and log
  tails. A heartbeat appears during quiet compilation. These do not restart
  training or stop a VM on a connection failure/Ctrl-C.
- `diagnose`: collect control-plane status and CLI history **without kernel
  execution**, useful when notebook/worker connections hang.
- `collect`: after the worker exits, hash-verified chunk download/reassembly
  of native checkpoint, metrics, job, summary, profiles and logs. Interrupted
  downloads can retry without repeating verified parts. Completed downloads
  refuse overwrite. Collected training state is not inference `encoder.pt`.
- `stop`: verify ownership, then release the session explicitly.
- `watch --stop-when-done`: collect results, then stop after completion/pause
  or a reported worker failure. If collection/connection fails, the VM stays
  available for recovery; check billing and stop it yourself when safe.

Combined workflows halt at the first error. Preflight failure attempts to stop
only the owned VM, before any training submission. After submission starts,
a timeout/Ctrl-C never automatically stops or retries the worker. Reattach with
`watch --state-dir ... --stop-when-done`, not `run` again. Collection failure
also keeps the VM available for recovery. Existing state directories refuse
overwrite; `run` is not a duplicate-launch/retry command. Staged actions remain
available for debugging. No arguments still only print a plan.

Setup failures collect CLI history and attempt to stop only the session just
created. An allocation timeout is **ambiguous**: inspect the website or
`colab sessions` for an orphan before retrying. Never infer that a timeout
means allocation did not happen. A training-submission timeout similarly
does not justify another launch or killing a possibly healthy worker.

Detached execution avoids an intentional dependency on the local terminal or
notebook websocket. It does **not** prevent Colab idle/resource reclamation,
VM termination, or loss of uncommitted work. Drive latest/previous commits
remain the persistence source. Keep one writer per experiment namespace.

## NVIDIA first: BF16 recovery qualification

The NVIDIA entry point defaults to **L4 / CUDA BF16** and the separate local
state directory `runs/colab/kiwilm3-gpu`. Select `--gpu A100` or `--gpu H100`
explicitly if appropriate for your account. These are supported requests, not
availability guarantees. Native BF16 is required; there is no emulated BF16,
FP16, CPU, TPU or alternative-GPU fallback. T4 is intentionally excluded from
this workflow. Explicit `--precision fp32` remains available for debugging,
not as an automatic fallback. Parameters and AdamW moments stay FP32; BF16 is
autocast compute. The worker uses isolated Python 3.12 and pinned torch 2.13.0
without torch/XLA or `PJRT_DEVICE=TPU`.

Check `colab sessions` and the website for an orphaned assignment from a timed
out TPU allocation **before** creating a GPU session. Stop only a confirmed
owned old assignment. Preserve failed TPU state directories for diagnosis;
the GPU launcher uses a separate directory and never rewrites those locks.

Run each stage yourself from the repository root, proceeding only on success:

```bash
bash scripts/run_colab_kiwilm3_gpu.sh setup --gpu L4 --precision bf16
bash scripts/run_colab_kiwilm3_gpu.sh preflight
bash scripts/run_colab_kiwilm3_gpu.sh train
bash scripts/run_colab_kiwilm3_gpu.sh watch --stop-when-done
```

Or run those stages with one explicit command (choose a new state directory):

```bash
bash scripts/run_colab_kiwilm3_gpu.sh run --gpu L4 --precision bf16 --state-dir runs/colab/v3-gpu-qualification-new
```

Setup allocates billable hardware and mounts Drive interactively, but does not
train. The default qualification is tiny deterministic synthetic data,
hybrid-12 width/context 16, 4096 tokens, seed 42, warmup 512, batch 2/accumulation
2, checkpoint/validation every 4 steps, **pause at step 8**. This is recovery
testing, not the full-width 50M architecture smoke or a throughput benchmark.
Preflight verifies the requested card family, native BF16, and the actual V3
model. Setup cannot certify driver compatibility; preflight failures halt,
and the VM stays billable until explicitly stopped.

Drive commits the checkpoint and locked native job synchronously at each
checkpoint interval, preserving latest and previous. Monitoring disconnects
do not stop the detached worker; failed backup publication halts training.
After collecting and stopping VM 1, qualify a **new** GPU VM:

```bash
bash scripts/run_colab_kiwilm3_gpu.sh resume --from-state runs/colab/kiwilm3-gpu --state-dir runs/colab/kiwilm3-gpu-resume
bash scripts/run_colab_kiwilm3_gpu.sh preflight --state-dir runs/colab/kiwilm3-gpu-resume
bash scripts/run_colab_kiwilm3_gpu.sh train --state-dir runs/colab/kiwilm3-gpu-resume
bash scripts/run_colab_kiwilm3_gpu.sh watch --state-dir runs/colab/kiwilm3-gpu-resume --stop-when-done
```

Equivalent single command, for a **not-yet-created** resume directory:

```bash
bash scripts/run_colab_kiwilm3_gpu.sh resume-run --from-state runs/colab/kiwilm3-gpu --state-dir runs/colab/kiwilm3-gpu-resume-new
```

Resume reuses exact wheel/bootstrap bytes and the original GPU, precision,
data/tokenizer, RNG seeds, budget and runtime contract. It cannot convert a
TPU run to CUDA, change card families, or rebuild training code. Verify restore
at step 8, completion at exactly 4096 tokens and contiguous metrics before
corpus training. An already completed download refuses overwrite; use explicit
`stop` if recovery/collection needs manual handling.

The first live L4/BF16 two-VM qualification has now passed the local artifact
audit: restore at step 8 on a distinct VM, completion at step 64 / 4096 tokens,
verified final backup receipt and successful shutdown. See the
[recovery evidence and limitations](kiwilm3-gpu-recovery.md). This qualifies
operational continuation for the saved runtime, not other GPU families,
bitwise equivalence or full-width memory/throughput.

Windows uses `uv run --locked python scripts/run_colab_kiwilm3_gpu.py` with
the same actions/options. `status`, `logs`, `diagnose`, `collect`, and `stop`
also work through this entry point. Add `--state-dir` for non-default sessions.

Full-width matched B/C smoke remains opt-in after recovery qualification:

```bash
bash scripts/run_colab_kiwilm3_gpu.sh setup --gpu L4 --precision bf16 --profile smoke --candidate hybrid-12 --run-name bc-smoke-50m --state-dir runs/colab/v3-gpu-hybrid-12-smoke --data-cache /content/drive/MyDrive/KiwiLM3/data/reviewed-smoke
```

Use `run` instead of `setup` to include preflight/train/watch/collect/stop in
the same command. The example cache path must exist and contain your reviewed
prepared 50M data; it is not created automatically. `resume-run` inherits all
controls and hardware from `--from-state`, never from new setup options.

Use the same reviewed wheel, prepared cache, CUDA card family, BF16, seeds and
controls for hybrid/attention-only × 12/16. Setup restores data to the VM,
never uploads a corpus from the host or silently rebuilds different data.
GPU memory/throughput at full width remain unverified; do not infer them from
the tiny qualification. Review explicit matched batch/accumulation changes
if needed; never silently reduce batches for one candidate.

## TPU backup: tiny two-VM recovery qualification

Install the Colab CLI if needed (`uv tool install google-colab-cli`) and use
your existing CLI authentication. Setup's `colab drivemount` authorization is
interactive: run it yourself in a terminal. There is no agent-run auth prompt,
automatic allocation or training in repository tests.

From the repository root, macOS/Linux:

```bash
bash scripts/run_colab_kiwilm3_tpu.sh plan
bash scripts/run_colab_kiwilm3_tpu.sh setup
bash scripts/run_colab_kiwilm3_tpu.sh preflight
bash scripts/run_colab_kiwilm3_tpu.sh train
bash scripts/run_colab_kiwilm3_tpu.sh watch --stop-when-done
```

Default state directory: `runs/colab/kiwilm3-tpu`. Default hardware: `v6e1`;
`setup --tpu v5e1` explicitly selects the other supported backend. Availability
is not guaranteed. The default tiny synthetic hybrid-12 uses width/context
16, BF16, 4096 input tokens, seed 42, warmup 512, batch 2/accumulation 2,
checkpoint/validation interval 4, and pauses at **step 8**. This qualifies
infrastructure, not model quality or full-width performance.

No corpus dataset upload: the tiny deterministic data are generated on the
VM. Only the small reviewed wheel, bootstrap and JSON specification are sent.
The wheel/bootstrap are retained under the state directory; do not change or
delete them before continuation. Setup builds a wheel automatically unless
`--wheel path/to/reviewed.whl` is supplied. Resume never rebuilds it.

Before the first VM is stopped, `watch` collects the summary/worker log/native
checkpoint into `downloads/`. Verify the step-8 Drive commit (previous step 4)
and preserve its receipt. Then prepare a **new** VM with the original artifacts:

```bash
bash scripts/run_colab_kiwilm3_tpu.sh resume --from-state runs/colab/kiwilm3-tpu --state-dir runs/colab/kiwilm3-tpu-resume
bash scripts/run_colab_kiwilm3_tpu.sh preflight --state-dir runs/colab/kiwilm3-tpu-resume
bash scripts/run_colab_kiwilm3_tpu.sh train --state-dir runs/colab/kiwilm3-tpu-resume
bash scripts/run_colab_kiwilm3_tpu.sh watch --state-dir runs/colab/kiwilm3-tpu-resume --stop-when-done
```

`resume` allocates/setup only; training remains separate. It reuses exact
wheel/bootstrap bytes, model/data/noise seeds, budget and schedule. Only
VM-local paths, ownership/session identifiers, mode, new-VM requirement and
pause target change. The original data/tokenizer/native job identities must
match. Native restore requires a distinct Linux boot ID; a new directory or
kernel restart is insufficient. It must restore step 8 and finish at exactly
4096 input tokens with contiguous metrics and an uninterrupted token schedule.
Send both logs/receipts for review; live TPU recovery is not yet qualified.

Windows PowerShell uses the same actions through Python, without Bash:

```powershell
uv run --locked python scripts/run_colab_kiwilm3_tpu.py setup
uv run --locked python scripts/run_colab_kiwilm3_tpu.py preflight
uv run --locked python scripts/run_colab_kiwilm3_tpu.py train
uv run --locked python scripts/run_colab_kiwilm3_tpu.py watch --stop-when-done
```

The same `resume --from-state ... --state-dir ...` arguments apply. Live
Windows CLI/auth and Colab execution remain manual checks, not test evidence.

## Debugging without relaunching

```bash
bash scripts/run_colab_kiwilm3_tpu.sh status
bash scripts/run_colab_kiwilm3_tpu.sh diagnose
bash scripts/run_colab_kiwilm3_tpu.sh logs
bash scripts/run_colab_kiwilm3_tpu.sh watch
bash scripts/run_colab_kiwilm3_tpu.sh collect
bash scripts/run_colab_kiwilm3_tpu.sh stop
```

Add `--state-dir YOUR_SESSION_DIRECTORY` for any non-default session. Keep
`owner.json`, `spec.json`, `launcher.log`, the saved wheel/bootstrap and
downloaded receipts. `spec.json` contains the explicit session name; if
ownership checks or the kernel connection fail, inspect with `colab status -s
SESSION`, `colab log -s SESSION -n 30`, or the website. Release a confirmed owned
session with `colab stop -s SESSION` if the wrapper cannot connect to verify it.
Never stop an unrelated session based on a reused name alone.

`--colab-bin PATH` selects another CLI binary; `--colab-config PATH` selects
isolated CLI state. Supply the same config for every action. `--drive-mount
manual` pauses for a human to mount in the session; `existing` skips the prompt
but **does not bypass** the real mount check. Neither is an unattended auth
workaround. No CUDA/CPU fallback occurs on allocation or TPU preflight failure.

If a launch times out, the local `train-intent.json` and remote
`training.claim` remain. Inspect status/logs, not `train` again. If the worker
has actually died, preserve local/Drive state and use a new-VM `resume`.
Do not remove locks or reset Drive pointers to force a fresh run.

## Later B/C smoke controls

Only after two-VM recovery passes, prepare/review the full-width 50M smoke
controls from [the TPU guide](kiwilm3-tpu.md). For example, **setup only**:

```bash
bash scripts/run_colab_kiwilm3_tpu.sh setup --profile smoke --candidate hybrid-12 --run-name bc-smoke-50m --state-dir runs/colab/v3-hybrid-12-smoke --data-cache /content/drive/MyDrive/KiwiLM3/data/reviewed-smoke
```

This requires an existing complete verified prepared cache; it never silently
downloads or rebuilds a different dataset. The VM restores data locally and
derives the real MASK tokenizer from the cache's original BPE IDs. All four
candidates must use the same reviewed wheel (`--wheel ORIGINAL_STATE/package/
kiwilm-0.1.0-py3-none-any.whl`), data, tokenizer, seeds, controls, TPU and
precision. Candidate/experiment/job-digest namespaces isolate backups. Each
candidate still needs explicit preflight/train/watch actions; there is no
automatic matrix or 1B launch.

Optional `setup --controls controls.json` freezes reviewed overrides to
`AcceleratorTrainConfig`. Invalid controls fail before allocation. Changing
controls for resume is intentionally unsupported. The new training-state
evaluation/export/comparison adapter remains a separate pending milestone;
these checkpoints are not old M5 CPU reports and must not bypass provenance
checks. No architecture winner is inferred from synthetic qualification.

## Verification — 2026-10-10

All 586 pytest tests pass, including 61 CLI/bootstrap cases, with two
existing ONNX deprecation warnings. Ruff lint, changed-file formatting, shell
syntax, locked dependency validation and the wheel build pass. Repository-wide
format checking still reports 69 unrelated pre-existing files; these were not
reformatted. The built wheel
contains the current CLI module and pinned worker requirements. Tests use a
fake Colab control plane and bounded real CPU optimization only; they cover
setup/training isolation, ownership locks, ambiguous allocation/submission,
exact wheel resume, interrupted verified downloads, detached launch claims,
kernel-argument handling and kernel-independent diagnostics. GPU checks cover
L4/A100/H100 selection, CUDA/BF16 defaults, refusal of emulated BF16,
CUDA-specific installation without XLA, inherited TPU environment removal,
wrong-card rejection, hardware/precision resume locks and control conflicts.
Combined workflows test exact stage order, collection before shutdown,
preflight cleanup, invalid intervals before allocation, exact-artifact resume,
and preservation of ambiguous submissions/monitoring/download failures.

Notebooks, V2 model/training source, checkpoint formats and the frozen V2
branch remain unchanged. No Colab session, cloud/corpus training, actual Drive
mutation, commit or push was performed by the agent. The user-run L4/BF16
qualification was reviewed separately as described above. Full-width CUDA/TPU
performance, other runtime recovery and Windows authorization remain live
user-run checks. This change does not claim to diagnose the original notebook
connection hang.

## Local collection recovery — 2026-10-11

The 5M lower-LR diagnostic completed at step 306 and published its final Drive
checkpoint. Collection then failed because the local `runs/colab/` tree was
absent while saving chunk 87. The live session still reported `complete`, with
`worker_alive=false`. This is a local state/download failure, not a training
failure; the tool cannot establish which process moved or deleted the folder.

Keep the entire local state directory until collection finishes. Do not clean,
move or delete it while the launcher is running. The downloader now anchors
absolute paths, checks directory ownership/identity between transfers, refuses
to recreate missing state, and commits chunks atomically without overwriting.
Restoring the original folder allows `collect`/`watch` to reuse verified chunks.
An interrupted local copy no longer leaves a corrupt final-named chunk.

For this incident, the original wheel, bootstrap, spec, input provenance and
preflight/completion receipts were recovered from the still-live VM into a
**separate** state directory. Checksums, session/token, job identity and input
fingerprints were verified; live read-only status succeeded. The original
missing folder was not recreated, no training was restarted, and the VM was
not stopped. Resume collection and stop the owned VM after verified collection:

```bash
bash scripts/run_colab_kiwilm3_lr_diagnostic.sh watch --state-dir runs/colab/v3-gpu-hybrid-12-lr3e4-diagnostic-5m-recovered --stop-when-done
```

The recovered directory has no old chunks, so this command downloads the
archive again. If you can restore the original folder/chunks from Trash, use
that original state directory instead. macOS denied agent access to Trash;
restore it manually through Finder if available. Do not run `run` or
`resume-run` for a completed job. If collection fails again, the VM stays alive
and may remain billable; inspect status and stop explicitly when safe.
