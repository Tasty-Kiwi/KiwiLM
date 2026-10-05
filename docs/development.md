# KiwiLM 3 development boundaries

Scope: M0/M1 cleanup, M2 workflow and M3 backbone of [V3_PLAN.md](../V3_PLAN.md).
The separate bidirectional encoder is implemented. Diffusion training and the
classifier are not. See [M3 contracts and qualification](kiwilm3.md).

## Keep the V2 baseline independent

`models/kiwilm2.py`, `config.py`, `training.py`, `sft.py`, `generation.py` and
`tpu_smoke.py` are causal V2 implementations. `CausalLanguageModel` expresses
next-token logits, not a universal encoder contract. Keep their module paths,
config serialization, state-dict keys and checkpoint format stable. Historical
Slim-v3 names must not be mistaken for the new KiwiLM 3 encoder.

V3 introduces `models/encoder.py`, `models/kiwilm3.py` and independent
`v3/config.py`, profiling, qualification and inference-only weight files.
It exposes contextual hidden states without inheriting causal training.
M4 will add the `v3/` masking/objective/trainer/sampling/checkpoint surface. Do not
generalize the causal trainer or move all modules into packages merely to match
the proposed roadmap tree. Model math remains in Python modules, not notebooks.

## Shared infrastructure and limits

| Existing module | Reusable contract | V3 caveat |
| --- | --- | --- |
| `data.py`, `tokenizer.py` | Deterministic packed data, revision/fingerprint, frozen tokenizer | Define MASK/protected-token behavior separately |
| `checkpoint.py` | Atomic serialization, config checks, RNG capture | Extend training state for noise RNG/objective; do not reinterpret V2 payloads |
| `tpu_checkpoint.py` | Verified latest/previous generations, integrity, fail-closed publication | Give V3 its own experiment identity/schema; it currently uses V2 serialization validation |
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
M4 supplies actual noise schedules/objective and a V3-specific trainer.

## Cleanup invariants

- Preserve all V2 reports, checkpoints, datasets, release receipts and playground choices.
- Do not rewrite historical upload provenance to suggest a retroactive frozen release.
- Freeze a local pre-cleanup source tag; no automatic Git push or Hub update.
- Keep V2 causal code unchanged and test its fixed state schemas/logits/cache rollover.
- Archives may allocate compute when invoked; regression tests must use fake/local resources.
