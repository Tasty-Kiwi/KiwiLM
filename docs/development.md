# KiwiLM 3 development boundaries

Scope: M0/M1 cleanup, M2 workflow, M3 backbone, M4 denoising and M5 generation of
[V3_PLAN.md](../V3_PLAN.md). The separate encoder and bounded CPU reconstruction
trainer and fixed-slot sampler are implemented. B/C experiments are prepared
but on hold after corpus quality failures; architecture selection and the
classifier remain pending. The [V3 accelerator/Drive adapter](kiwilm3-tpu.md)
passed the bounded live L4/BF16 recovery qualification; TPU qualification
remains pending. See the
[5M diagnostic and local isolation results](../examples/comparisons/kiwilm3-lr-diagnostic-5m/analysis.md).
See [M3 contracts](kiwilm3.md), [M4 policy](kiwilm3-denoising.md) and
[M5 B/C workflows](kiwilm3-experiments.md).

## Keep the V2 baseline independent

`models/kiwilm2.py`, `config.py`, `training.py`, `sft.py`, `generation.py` and
`tpu_smoke.py` are causal V2 implementations. `CausalLanguageModel` expresses
next-token logits, not a universal encoder contract. Keep their module paths,
config serialization, state-dict keys and checkpoint format stable. Historical
Slim-v3 names must not be mistaken for the new KiwiLM 3 encoder.

V3 introduces `models/encoder.py`, `models/kiwilm3.py` and independent
`v3/config.py`, profiling, qualification and inference-only weight files.
It exposes contextual hidden states without inheriting causal training.
M4 adds `v3/tokenizer.py`, `masking.py`, `objectives.py`, `trainer.py` and separate
local training-state `checkpoints.py`. M5 adds independent `sampling.py`,
`evaluation.py`, frozen `experiments.py` and opt-in `experiment_runner.py`. Do not
generalize the causal trainer or move all modules into packages merely to match
the proposed roadmap tree. Model math remains in Python modules, not notebooks.

## Shared infrastructure and limits

| Existing module | Reusable contract | V3 caveat |
| --- | --- | --- |
| `data.py`, `tokenizer.py` | Deterministic packed data, revision/fingerprint, frozen tokenizer | Define MASK/protected-token behavior separately |
| `checkpoint.py` | Atomic serialization, config checks, RNG capture | Extend training state for noise RNG/objective; do not reinterpret V2 payloads |
| `tpu_checkpoint.py` | Generic byte-level latest/previous transport, integrity, fail-closed publication | V3 validates its separate payload in `v3/accelerator_checkpoint.py`; do not use V2 payload validation |
| `colab_drive.py`, `colab_artifacts.py` | Validated data restore, atomic transfer, chunk reassembly | Preserve mount checks and recovery refusal; never silently start fresh |
| `optim.py` | Optimizers/parameter partition helpers | Revalidate parameter routing for a new encoder |
| `comparison.py`, `retrieval.py`, `diagnostics.py` | Evaluation/report patterns | Current metrics are causal; diffusion loss is not AR perplexity |
| `safetensors_io.py`, `hub.py` | Manifest-verified inference artifacts | Extend config dispatch only when V3 exists and is tested |

`colab_kiwilm2.py`, `tpu_setup.py`, `tpu_final.py`, `compile_benchmark.py`,
`residual_gate.py` and `slim_v3.py` remain V2-specific compatibility helpers.
They are not the new notebook control surface. Their CLI orchestration moved to
`archive/kiwilm2/`; relevant safety/recovery tests still run.

## M2 workflow and remaining gate

`notebook_setup.py`, `notebook_worker.py` and `notebook_workflow.py` provide
explicit setup/configure/train-or-resume/evaluate cells around Python calls.
The bounded, named M2 V2 recovery probe validates optimizer/token schedule,
Python/NumPy/torch/backend RNG, validation/data sampling and a reserved noise
RNG. The noise stream is exercised solely to test state recovery; it does not
corrupt inputs or implement a diffusion objective. Config, code, tokenizer,
data and runtime identities lock resume, independent of VM-local paths.

The three notebooks call a worker **inside the selected notebook VM**. No
remote CLI provisioning or training math is embedded in cells. Setup uses a
separate Python 3.12 environment with matched torch/XLA on TPU; it does not
replace Colab kernel torch. Verified latest/previous transport is reused with
its own M2 namespace and schema. Source V2 training and serialization remain
unchanged.

The fresh-process CPU equivalence test is local evidence, not a live Colab
acceptance test. Complete the [two-runtime recovery procedure](notebook-workflow.md)
on the intended Colab backend with Drive before closing M2. No long training
run is enabled by this phase. M3 introduces the separate bidirectional encoder;
M4 supplies actual corruption and the reconstruction objective with a bounded
CPU trainer. The separate `v3/accelerator.py` now supplies token scheduling,
accumulation and XLA BF16; `accelerator_checkpoint.py` and
`accelerator_workflow.py` use generic Drive transport with native V3 recovery.
The new notebook calls `v3/accelerator_worker.py`, never the M2 V2 worker.
Live fresh-TPU-VM recovery remains a user-run acceptance gate. M4/M5 CPU state,
inference files and suite identities remain unchanged.

The user-run [V3 Colab CLI](kiwilm3-colab-cli.md) now provides staged
setup/preflight, detached training, status/logs, verified collection and exact
wheel fresh-VM continuation. It calls the same V3 worker and generic transport;
notebooks remain a backup. Default plan/import is non-allocating, setup never
trains, and tests use a fake Colab control plane/local synthetic CPU workers.

Following repeated TPU allocation timeouts, the current entry point is
`scripts/run_colab_kiwilm3_gpu.sh` (PowerShell: the matching `.py`). It shares
the same staged orchestration and V3 Drive transport, defaults to L4/CUDA
BF16, supports explicit A100/H100, and excludes T4. Native BF16 and the
requested card family are checked without CPU/XLA/precision fallback.
New GPU state uses `runs/colab/kiwilm3-gpu`; old TPU locks remain intact.
Resume pins the original GPU/precision/artifacts and cannot migrate a TPU
checkpoint to CUDA. Recovery must be qualified on the intended runtime;
full-width B/C performance remains a live user-run acceptance check.

The user-run L4/BF16 two-VM probe now passes the
[downloaded-artifact recovery audit](kiwilm3-gpu-recovery.md): restore step 8,
finish step 64 / 4096 tokens, exact metric prefix and verified final checkpoint
receipt on a distinct VM. This is operational recovery evidence, not bitwise
equivalence or full-width performance. CLI `run` and `resume-run` now combine
all stages with fail-fast checks and collection before shutdown. They never
retry an ambiguous submission or stop a potentially healthy worker on a
monitoring disconnect. Corpus training still requires user invocation.

## Cleanup invariants

- Preserve all V2 reports, checkpoints, datasets, release receipts and playground choices.
- Do not rewrite historical upload provenance to suggest a retroactive frozen release.
- Freeze a local pre-cleanup source tag; no automatic Git push or Hub update.
- Keep V2 causal code unchanged and test its fixed state schemas/logits/cache rollover.
- Archives may allocate compute when invoked; regression tests must use fake/local resources.
