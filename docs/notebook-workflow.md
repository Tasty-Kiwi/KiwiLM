# M2: notebook-first workflow qualification

The notebooks test the control surface before V3's model and objective exist.
They train a labelled **tiny V2 causal recovery probe**, not KiwiLM 3. Its
16-wide, ten-block model, synthetic corpus and 256-token budget cannot measure
model quality or meaningful production TPU throughput. Full V3 training is
deliberately unavailable; the probe refuses budgets above 1M tokens.

## Notebooks and ownership

- [Smoke](../notebooks/kiwilm3-smoke.ipynb): stop at step 4 of a locked 8-step budget.
- [Train/resume](../notebooks/kiwilm3-train.ipynb): finish or resume that same budget.
- [Evaluate/export](../notebooks/kiwilm3-evaluate.ipynb): restore, CPU/FP32 validation,
  block health, cached direct/rollover parity, sample and optional BF16 bundle.

Every action switch is false by default. Running all cells with defaults only
prints configuration and skipped-action messages. There is no `colab new`,
remote launcher, automatic accelerator allocation or automatic paid training.
Select the runtime on Colab's website yourself and terminate it when finished.
Run **one writer per experiment namespace**; concurrent notebook writers are
not qualified.

Notebook cells own parameters and explicit actions. Library modules own math,
metrics, checkpoint state and subprocess handling. The future V3 trainer will
remain separate from V2's causal training; the probe must not become its objective.

## Prepare the reviewed wheel

In the current checkout:

```bash
uv sync --locked --extra notebooks
uv build --wheel
```

Upload `dist/kiwilm-0.1.0-py3-none-any.whl` to
`MyDrive/KiwiLM3/wheels/`. Keep that exact wheel for the entire recovery test;
do not rebuild it after code changes and expect to resume the same experiment.
Its bytes and pinned requirements are checked by setup. Future V3 changes
should not mutate a running experiment's reviewed wheel.

Open the smoke notebook in Colab manually. Choose the backend and set:

| Runtime | DEVICE | PRECISION |
| --- | --- | --- |
| CPU | `cpu` | `fp32` |
| T4 | `cuda` | `fp16` |
| BF16-capable GPU | `cuda` | `bf16` |
| Single-chip TPU | `xla` | `bf16` |

Preflight refuses unsupported BF16, missing CUDA, non-TPU XLA and multi-chip
TPU; it never falls back to CPU. TPU setup pins matched torch/torch_xla 2.9.0,
the previously exercised V2 environment. CPU/GPU workers pin torch 2.13.0.
All workers use an isolated Python 3.12 environment. Core dependency versions
are stored in `notebook_requirements.txt` and checked against the repository lock.
Eager tiny Dense does not need Triton or browser exports, so those are not
installed into the worker. Notebook development extras are not needed there.
These install paths are mocked in tests, **not live-Colab-qualified by M2 yet**.
The matched TPU environment follows the
[PyTorch/XLA 2.9 documentation](https://docs.pytorch.org/xla/release/r2.9/index.html).

## Two-runtime recovery procedure

1. Keep `CONFIG` unchanged: seed 42, noise seed 142, 256 tokens, 64 warmup
   tokens, batch 2, accumulation 1, context 16, checkpoint/evaluation interval 2.
   `STOP_AFTER_STEP=4` pauses at the midpoint without shortening the LR schedule.
2. Set `MOUNT_DRIVE=True`, `SETUP_ENVIRONMENT=True`, `PREPARE_DATA=True`,
   `RUN_PREFLIGHT=True`, `MODE='fresh'`. Keep `START_TRAINING=False` until setup
   and the printed contract identity pass. Mount Drive interactively yourself.
3. Explicitly set `START_TRAINING=True` and run the train cell. Expect step 4,
   128 tokens and a verified checkpoint publication. Record the identity,
   generation and step. Do not use the last train log as proof of a backup.
4. Terminate the runtime yourself. Start a new runtime with the **same backend,
   precision, wheel and configuration**. Repeat mount/setup/prepare/preflight.
   Synthetic data regenerates deterministically on the VM; no large corpus upload.
5. Select `MODE='resume'`, `STOP_AFTER_STEP=None`, then explicitly start the
   train cell. It must restore step 4 before continuing at step 5, and finish
   at step 8 / 256 tokens. LR warmup must not restart and metrics must contain
   each train step 1–8 exactly once.
6. Set `RUN_EVALUATION=True`. Confirm finite weights/health and cached direct
   and rollover parity. Record the report and complete JSONL metrics in the
   notebook outputs. Save the executed notebook as qualification evidence.

Use a new empty `RUN_DIR` for each worker invocation. If the VM is still alive,
change `run-1` to `run-2` before resume; the worker refuses overwriting any local
progress. The same backend's isolated worker environment may be reused when
its wheel/setup identity matches. Installation interruptions can retry the
same setup; a changed or unmanaged environment requires a new directory.

For a separate evaluation runtime, copy the original `CONFIG`, prepare the
same corpus, choose an empty `RUN_DIR`, and enable `RESTORE_FOR_EVALUATION`
before evaluation. Restore uses the original runtime identity; therefore
select the original backend environment even though model evaluation runs on
CPU. A local checkpoint can also be evaluated on CPU without XLA installed
through `evaluate_probe`, which validates its recorded contract and data rather
than creating a new optimizer identity.

## What is saved and refused

Checkpoints contain model and AdamW state, exact completed token count and LR
schedule position, Python/NumPy/torch/CUDA or XLA RNG, FP16 scaler state,
data-sampling RNG, validation RNG, and a reserved independent noise RNG.
The noise RNG is exercised as a recovery probe only; **no diffusion corruption
or loss exists in M2**. Saves happen only at completed optimizer boundaries.

The experiment contract locks configuration, model shape, full training
budget, optimizer, tokenizer checksum, prepared-data fingerprint, installed
Python source digest and runtime versions (CPU thread count included). Paths,
VM identity and the temporary stop boundary do not affect the experiment.
Missing optimizer/schedule/RNG state, inconsistent step/token progress and
missing or noncontiguous committed metrics refuse resume. A mismatch never
silently starts fresh or imports weights as a warm start.

Backups live under `MyDrive/KiwiLM3/checkpoints/<experiment>-<digest>/`.
The reused store copies checkpoint, metrics and contract, verifies all bytes
by readback, then commits `latest.json`. It keeps latest/previous distinct
steps and prunes only known superseded committed generations. Existing V2
Drive namespaces and release weights are never touched.

Publication is synchronous. A Drive outage stops the worker, leaving the last
committed recovery point and local checkpoint. An interrupt can lose up to
the checkpoint interval's uncommitted progress; no partial optimizer update
is saved as a completed step. A corrupt latest checkpoint may be restored from
the verified previous generation for inspection, but training **refuses to
roll back behind an existing newer pointer**. Resolve recovery explicitly;
never delete pointers or silently reset the schedule to continue.

## Verification and phase gate

```bash
uv run --locked --extra notebooks pytest -q tests/test_notebook_workflow.py tests/test_notebooks.py
uv run --locked --extra notebooks python scripts/build_kiwilm3_notebooks.py
uv run --locked --extra browser --extra notebooks pytest -q
```

Local tests compare an uninterrupted tiny CPU run against two **separate Python
processes**, interrupted after validation and resumed through the backup store.
Every numerical checkpoint field and loss/LR/RNG metric must match exactly;
only step timing/throughput differs. Tests include exact partial-token budgets,
state rejection, latest/previous retention, dead-mount failure, parity and BF16
export/load round-trips. Installation tests are mocks, not hardware qualification.

The committed notebooks execute top-to-bottom with safe defaults and retain
the skip/configuration outputs. That is a safety check, **not a live training
notebook run**. M2's fresh-Colab-runtime/real-Drive completion criterion remains
pending until you execute the two-runtime procedure on the intended hardware.
Only then proceed to M3's bidirectional encoder. No production training, Drive
write, Hub publication or compute allocation is performed by implementation.

### Local verification record — 2026-10-05

| Check | Result |
| --- | --- |
| Complete Python suite, browser + notebook extras | 344 passed; two existing ONNX deprecation warnings |
| Focused M2 tests | 38 passed, including separate-process exact continuation |
| Playground runtime tests | 7 passed; playground code unchanged |
| Ruff / `git diff --check` / locked dependency check | Passed |
| Wheel build and packaged worker resources | Passed |
| Three notebook kernels, safe defaults | Executed successfully; saved outputs retained |
| Rendered notebook presentation | Inspected in the browser |
| Actual Colab CPU/GPU/TPU install and fresh-VM Drive recovery | Not run; user-run gate remains open |

Existing locked package versions, V2 source/weights, comparison evidence and
release artifacts were preserved. No training session was allocated, no Drive
or Hub writes were made, and no Git commit or push was performed.
