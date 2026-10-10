# KiwiLM

KiwiLM is a small language-model research codebase. **KiwiLM 2 is complete**;
the next project is a bidirectional encoder with masked diffusion.
The [KiwiLM 3 roadmap](V3_PLAN.md) defines that work. Its
[Phase 3 bidirectional backbone](docs/kiwilm3.md) and
[Phase 4 denoising prototype](docs/kiwilm3-denoising.md) are implemented.
[Phase 5 fixed-slot generation and B/C experiments](docs/kiwilm3-experiments.md)
are available as bounded CPU workflows. A/D are dropped; dense SwiGLU is used
throughout. The new [TPU/BF16 continuation notebook](notebooks/kiwilm3-tpu.ipynb)
prepares token-based scheduling, accumulation and verified V3 Drive recovery.
[L4/BF16 recovery qualification](docs/kiwilm3-gpu-recovery.md) has passed;
TPU qualification and corpus architecture selection remain pending.
No training is started automatically.
The [script-driven NVIDIA BF16 launcher](docs/kiwilm3-colab-cli.md) is the
current Colab path following TPU allocation timeouts; TPU scripts and notebooks
remain backup options.

The first V3 corpus smoke completed but collapsed to unigram-like predictions.
See [local collapse diagnostics and the gate before further runs](docs/kiwilm3-collapse-diagnostics.md).
The user-run 5M-token lower-LR diagnostic also completed without resolving
the failure. Local conditioning/mixer-scale isolation tests found no sufficient
fix. See the [results and next diagnostic gate](examples/comparisons/kiwilm3-lr-diagnostic-5m/analysis.md).
Inspect the recovered checkpoint locally without starting training:

```bash
uv run --locked python scripts/isolate_kiwilm3_collapse.py --run-dir runs/colab/v3-gpu-hybrid-12-lr3e4-diagnostic-5m-recovered/downloads --data-dir data/smollm-smoke --replay-steps 0 --output runs/kiwilm3-diagnostics/checkpoint-counterfactuals.json
```

Use `--replay-steps 32` for the bounded fresh CPU diagnostic, not a cloud smoke
or saved-checkpoint continuation. Remaining B/C corpus runs stay on hold.
The subsequent [balanced readout test](examples/comparisons/kiwilm3-readout-fixed-corpus/analysis.md)
passes with the unchanged tied/full-loss model on four fixed corpus windows.
This is contextual memorization evidence, not a repair or generalization result.
The subsequent [paired context-512 masking test](examples/comparisons/kiwilm3-noise-policy-512/analysis.md)
found no fix from switching to fixed 15% noise. Both short CPU conditions
remained worse than unigram CE with negligible context response. The next
gate is fixed-corpus multi-target memorization at context 512, not another
cloud run or a production head/objective change. See the
[masking diagnostic runbook](docs/kiwilm3-noise-diagnostics.md).

## KiwiLM 2 reference

The Dense model has 64,252,416 parameters, a 32K tokenizer, a 512-token context,
and was trained from scratch for exactly 1B tokens on a 70/30
FineWeb-Edu/Cosmopedia mixture using a single Colab TPU v6e-1.

- [Published BF16 weights and model card](https://huggingface.co/Tasty-Kiwi/KiwiLM-2)
- [Matched 500M/1B evaluation and analysis](examples/comparisons/kiwilm2-final-1b-tpu-muon/analysis.md)
- [Frozen V2 source and checkpoint provenance](docs/kiwilm2-freeze.md)
- [V2 architecture and maintained interfaces](docs/kiwilm2.md)
- [Browser-local playground](https://huggingface.co/spaces/Tasty-Kiwi/KiwiLM-Playground)

Matched FP32 validation: **3.486474 loss / 32.6705 perplexity**.
Four-choice retrieval: **62.5%** on a small templated suite, not a general
reasoning benchmark. Generation remains unreliable: **41/120 severe-loop
samples**, semantic drift, and fabricated explanations. This is an experimental
base model, not a chatbot.

The 1B run resumed its step-13,000 checkpoint on a fresh TPU VM and completed
61,036 updates. That demonstrates practical continuation, **not** bitwise
equivalence to an uninterrupted run. Historical TPU qualification failures and
the invalid untied v5e control remain in the [V2 workflow archive](archive/kiwilm2/README.md).

## Install

Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run --locked kiwilm --help
```

Windows PowerShell uses the same commands. The lock selects official CUDA 13.2
PyTorch wheels on Windows; macOS/Linux use their normal PyPI resolution.
Optional browser export dependencies: `uv sync --locked --extra browser`.

## Generate without training data

Download the published bundle at its immutable model revision:

```bash
uv run --locked hf download Tasty-Kiwi/KiwiLM-2 --revision 53d6abcb4d11c8a98936c19d55d2b741a99d7602 --local-dir artifacts/KiwiLM-2
uv run --locked kiwilm generate --checkpoint artifacts/KiwiLM-2 --prompt "Once upon a time, a fox found a box." --max-new-tokens 160 --temperature 0.8 --top-k 40 --seed 42 --cache auto --stream
```

The same one-line commands work in PowerShell. The package verifies the bundle
manifest; no training dataset, Transformers integration, or hosted inference
server is needed. BF16 is the storage format; portable inference defaults to FP32.
See [dataset-free Python loading](examples/generate_from_hub.py) and the
[release runbook](docs/huggingface-release.md).

The static playground retains KiwiLM 2 Dense and all three original KiwiLM 1
X/Y choices. Downloads are lazy and inference runs in a browser worker.
The Dense ONNX download is approximately 323 MB; X/Y bundles are about 22 MB each.
[Playground source and build instructions](spaces/kiwilm-playground/README.md).

## Local V2 tools

The `kiwilm` CLI retains data preparation, causal training/resume, CPT, SFT,
validation, generation, generation comparison, retrieval, diagnostics/profiling,
and tokenizer/Safetensors export. Use `kiwilm <command> --help`.
Dense, gated Slim v2 and hybrid Slim v3 checkpoint dispatch remain compatible;
"Slim v3" is a **V2 ablation**, not KiwiLM 3.

```bash
uv run --locked kiwilm prepare-smollm --profile smoke --output-dir data/smollm-smoke
uv run --locked kiwilm profile-kiwilm2 --architecture kiwilm2
uv run --locked kiwilm evaluate --data-dir data/smollm-smoke --checkpoint runs/my-v2/latest.pt
```

Prepared-data fingerprints must match for training/evaluation. Use the
bundle's tokenizer, not an unrelated dataset, for generation.

V2 CLI-based Colab launchers, ablation runners and their detailed runbooks moved
to [`archive/kiwilm2/`](archive/kiwilm2/README.md). They remain regression-tested
for historical reproduction, not the starting point for V3. Launchers are
user-run only and can allocate billable hardware.

## Development boundary

V2 model, config dispatch, causal trainer and checkpoint formats stay independent
and unchanged. V3 has a separate encoder/config and will have masking, objective,
trainer, sampler and recovery modules; it must not inherit the causal objective. Data/tokenizer,
metrics and verified latest/previous checkpoint utilities can be reused where
their contracts fit. See [infrastructure boundaries](docs/development.md).

The M2 notebook workflow is now available: [smoke](notebooks/kiwilm3-smoke.ipynb),
[train/resume](notebooks/kiwilm3-train.ipynb) and
[evaluate/export](notebooks/kiwilm3-evaluate.ipynb).
They currently qualify recovery with a **tiny V2 causal probe**, not a V3 model
or diffusion objective. All actions are off by default; no VM is allocated by
the notebooks. [Setup and fresh-runtime recovery guide](docs/notebook-workflow.md).
CPU continuation is regression-tested; live Colab/Drive qualification is still
a user-run gate before declaring M2 complete.

The untrained V3 encoder uses repeated full attention → BiConv31 → BiConv63,
RoPE, dense SwiGLUs and a tied reconstruction head. Its 12/16-block configurations
have 65.12M/81.43M parameters. Run bounded CPU structural checks (no training):

```bash
uv run --locked python scripts/check_kiwilm3_encoder.py --depth 12
uv run --locked python scripts/check_kiwilm3_encoder.py --depth 16
```

[Encoder API, static profiles, serialization and remaining gates](docs/kiwilm3.md).
The local `codex/kiwilm2-frozen` branch preserves the pre-M3 V2/M2 baseline.

The new [M4 CPU notebook](notebooks/kiwilm3-denoising.ipynb) uses a separate real
MASK tokenizer, variable-noise masked reconstruction and fixed validation.
All actions default off; the bounded CPU trainer/state files are not TPU/Drive
production recovery. [Tokenizer conversion and local acceptance checks](docs/kiwilm3-denoising.md).

The [M5 B/C notebook](notebooks/kiwilm3-experiments.ipynb) prepares attention-only
vs hybrid and 12 vs 16 blocks with frozen controls, explicit local continuation,
aligned evaluation and fixed-slot generation/infilling. Training is opt-in;
these bounded CPU workflows are not accelerator-quality evidence.

The [V3 TPU notebook](notebooks/kiwilm3-tpu.ipynb) uses the actual dense denoising
trainer, not M2's V2 probe. All actions default off. First qualify a tiny
two-VM continuation with isolated torch/XLA and latest/previous Drive backups;
then review the proposed matched 50M B/C controls. [Runtime and recovery guide](docs/kiwilm3-tpu.md).

For CLI-driven setup and debugging (no notebook upload required):

```bash
bash scripts/run_colab_kiwilm3_gpu.sh run --gpu L4 --precision bf16 --state-dir runs/colab/v3-gpu-qualification-new
```

`run` explicitly combines setup → preflight → train → watch → collect → stop.
Drive authorization still requires your input. `resume-run --from-state OLD
--state-dir NEW` combines fresh-VM continuation with the same original controls.
Staged commands and a non-allocating `plan` remain available for debugging.
Setup alone allocates billable hardware but never trains. Training is explicitly
detached; monitor/collect/stop with the saved session owner. Default is a tiny
step-8 recovery qualification, not a corpus run. On Windows invoke
`uv run --locked python scripts/run_colab_kiwilm3_gpu.py` with the same actions.
Default L4/CUDA BF16; explicit A100/H100 are also supported. T4 is excluded.
The state directory is `runs/colab/kiwilm3-gpu`; preserve failed TPU session
state separately. Check for orphaned allocations before retrying.
[Fresh-VM resume and diagnostics](docs/kiwilm3-colab-cli.md).
The [live L4/BF16 recovery qualification](docs/kiwilm3-gpu-recovery.md) passed;
the full-width smoke and subsequent lower-LR diagnostic failed the quality gate.
Further corpus training remains on hold pending contextual-learning diagnostics.

```bash
uv sync --locked --extra notebooks
uv build --wheel
```

```text
src/kiwilm/           maintained V2 runtime and shared infrastructure
scripts/             evaluation, release, export and rendering utilities
notebooks/           explicit M2 setup, qualification, resume and evaluation
archive/kiwilm2/      historical experiment runners and Colab CLI workflows
docs/                maintained references, roadmap boundaries and diagrams
examples/comparisons/ preserved research evidence
releases/kiwilm2-1b/  model card, publication receipt and verification records
spaces/              browser-local playground
tests/               runtime, recovery, archived workflow and frozen-output tests
```

Historical pre-V2 Python models remain on the `legacy` branch. Local datasets,
checkpoints, release bundles and `.venv` are ignored but preserved; cleanup does
not delete research artifacts or touch Drive/Hugging Face backups.

## Verify

```bash
uv lock --check
uv run --locked ruff check src scripts archive tests
uv run --locked --extra browser --extra notebooks pytest -q
uv build
```

The suite always checks frozen tiny V2 state schemas and outputs. If the original
1B checkpoint / published BF16 bundle are available locally, it also verifies
them against pre-cleanup FP32 logits, greedy continuations and cached rollover.
Those two artifact tests are explicitly skipped in a fresh checkout without
weights; [the freeze record](docs/kiwilm2-freeze.md) documents explicit paths.
