# Development boundary before KiwiLM 3

Scope: M0/M1 of [V3_PLAN.md](../V3_PLAN.md). No encoder, diffusion training,
notebook workflow or classifier is implemented here.

## Keep the V2 baseline independent

`models/kiwilm2.py`, `config.py`, `training.py`, `sft.py`, `generation.py` and
`tpu_smoke.py` are causal V2 implementations. `CausalLanguageModel` expresses
next-token logits, not a universal encoder contract. Keep their module paths,
config serialization, state-dict keys and checkpoint format stable. Historical
Slim-v3 names must not be mistaken for the new KiwiLM 3 encoder.

V3 should introduce a separate encoder interface exposing hidden states and a
separate `v3/` masking/objective/trainer/sampling/checkpoint surface. Do not
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

## Next gate, not part of cleanup

M2 will provide explicit setup/configure/train-or-resume/evaluate notebook cells
around tested library calls. It must recover optimizer/LR state, RNG, data
position and noise RNG on a fresh runtime, validate an experiment fingerprint,
and retain the latest/previous publication strategy. No training should begin
until the user explicitly starts it.

## Cleanup invariants

- Preserve all V2 reports, checkpoints, datasets, release receipts and playground choices.
- Do not rewrite historical upload provenance to suggest a retroactive frozen release.
- Freeze a local pre-cleanup source tag; no automatic Git push or Hub update.
- Keep V2 causal code unchanged and test its fixed state schemas/logits/cache rollover.
- Archives may allocate compute when invoked; regression tests must use fake/local resources.
