# KiwiLM 3 — Development Plan

Objective: Conclude KiwiLM 2, simplify the research codebase, and build KiwiLM 3 as a bidirectional encoder trained with masked diffusion. KiwiLM 3 should eventually provide the pretrained backbone for KiwiClassifier 1.

The biggest workflow change is that Colab notebooks become the primary interface for serious training. The underlying training logic remains in tested Python modules, but launching, monitoring, checkpoint recovery, and evaluation happen through notebooks rather than remote CLI orchestration.

No repository changes yet—this is the implementation roadmap.

## Phase 1 — Freeze and clean up KiwiLM 2

Before introducing V3, establish a stable reference point.

| Task                          | Implementation                                                            |
| ----------------------------- | ------------------------------------------------------------------------- |
| Freeze V2                     | Tag the final V2 source revision and preserve the 1B checkpoint           |
| Publish weights               | Complete the Hugging Face release with model card and evaluation results  |
| Update documentation          | Replace outdated TPU qualification notes in the main README               |
| Preserve experiments          | Retain V2 comparison reports and historical checkpoints                   |
| Remove obsolete workflow code | Identify superseded Colab launchers, smoke utilities, and one-off scripts |
| Establish regression tests    | Ensure V2 checkpoints still load and produce identical outputs            |

Do not delete historical experimental evidence. The goal is to make the active development surface smaller, not erase the research history.

The 1B checkpoint should remain loadable from the V3 codebase. That is particularly important if we later want to compare causal and bidirectional architectures using the same evaluation harness.

### Refactor the shared infrastructure

The current repository has several V2-specific assumptions that would complicate V3.

For example:

- `src/kiwilm/models/base.py` defines a `CausalLanguageModel` interface.
- `src/kiwilm/training.py` is fundamentally a next-token training implementation.
- `src/kiwilm/config.py` reconstructs the existing V2 configurations.
- The TPU training and recovery functionality is distributed across V2-specific modules.
- The CLI and remote Colab launchers currently contain substantial workflow orchestration.

I would separate the reusable pieces without substantially rewriting the working V2 implementation.

The main architectural boundary should become:

```
kiwilm
│
├── models/
│   ├── base.py          Shared model abstractions
│   ├── kiwilm2.py       Frozen causal architecture
│   └── kiwilm3.py       New bidirectional architecture
│
├── v3/
│   ├── masking.py       Noise schedules and corruption
│   ├── objectives.py    Diffusion training losses
│   ├── trainer.py       Training and validation
│   ├── sampling.py      Iterative denoising
│   └── checkpoints.py   V3 training-state recovery
│
├── data/                Shared data preparation
├── evaluation/          Shared evaluation utilities
│
└── ...
```

This is a proposed target layout, not a requirement to reorganize every existing file immediately.

In particular, I would keep the frozen V2 model and training code working independently rather than attempting to unify causal and diffusion training under a complicated abstraction.

### Phase 1 completion criteria

V2 can be considered archived when the Hugging Face release is usable, its published results are linked from the README, and its regression tests pass after the cleanup.

## Phase 2 — Replace the training workflow

This should happen before building the full V3 training pipeline.

Your experience with Colab exposed a useful distinction: the PyTorch/XLA training itself worked well, but managing sessions through CLI-based remote orchestration introduced unnecessary complexity.

I would change the responsibility split:

| Component      | Responsibility                                                     |
| -------------- | ------------------------------------------------------------------ |
| Python package | Model, optimizer, training math, metrics, checkpoint serialization |
| Colab notebook | Runtime initialization, configuration, execution, monitoring       |
| Google Drive   | Persistent datasets and recoverable checkpoints                    |
| Hugging Face   | Published model artifacts                                          |
| CLI            | Small local experiments, diagnostics and utilities                 |

The notebook is the control surface, not where the training algorithm lives.

That distinction matters. Moving hundreds of lines of training logic into notebook cells would make testing and reproducibility worse.

### Proposed notebook workflow

1\. Setup

Select TPU • install pinned dependencies • mount Drive

2\. Configure experiment

Model • dataset • optimizer • token budget • experiment ID

3\. Train or resume

Explicitly resume from a verified checkpoint, or start fresh

4\. Evaluate and export

Validation • diagnostics • sample generation • artifacts

I would initially create three notebooks:

| Notebook                 | Purpose                                                             |
| ------------------------ | ------------------------------------------------------------------- |
| `kiwilm3-smoke.ipynb`    | Small CPU/GPU/TPU tests and throughput benchmarks                   |
| `kiwilm3-train.ipynb`    | Full training, live metrics, and resumable execution                |
| `kiwilm3-evaluate.ipynb` | Reconstruction, generation, retrieval, and model-health evaluations |

The training notebook needs reliable recovery from interrupted sessions. That means persisting model weights, optimizer state, learning-rate progress, RNG state, data-sampling position, and the noise-sampling RNG.

Each checkpoint should also contain an experiment configuration fingerprint. A resumed session must refuse to silently restart or continue with incompatible settings.

Keep the verified latest/previous checkpoint strategy from V2. Notebooks will make execution easier to manage, but they do not eliminate Colab disconnections or TPU failures.

Completion criterion: A notebook can train a small model, terminate its runtime, reconnect to a new runtime, and continue without resetting the training schedule or corrupting its metrics.

## Phase 3 — Implement the KiwiLM 3 backbone

I would start with the architecture we previously discussed, deliberately excluding the more experimental components.

Token embeddings + noise conditioning

Bidirectional encoder blocks

Pre-RMSNorm → BiConv / Self-attention → residual

Pre-RMSNorm → SwiGLU → residual

Repeated across the model depth

Final RMSNorm

Token reconstruction head

Vocabulary logits at every position

### Initial architecture specification

| Property             | Proposed baseline                       |
| -------------------- | --------------------------------------- |
| Architecture         | Bidirectional encoder                   |
| Width                | 512                                     |
| Initial depth        | 12 blocks                               |
| Candidate full depth | 16 blocks                               |
| Attention            | Full bidirectional self-attention       |
| Convolution          | Gated bidirectional depthwise Conv31/63 |
| FFN                  | Dense SwiGLU                            |
| Normalization        | Pre-RMSNorm                             |
| Residuals            | Ordinary additive residuals             |
| Vocabulary           | Existing 32K tokenizer                  |
| Context              | 512 tokens initially                    |
| Output               | Tied token-reconstruction head          |
| N-grams              | Removed                                 |
| Hadamard             | Excluded from baseline                  |

The 12-block configuration is for proving the implementation. We can benchmark a 16-block model before deciding which configuration deserves the longer training run.

For a 16-block, 512-wide configuration with 2,048-wide SwiGLUs, a roughly 80M-parameter target is plausible, depending on the attention/convolution schedule.

I would avoid GQA initially as well. Full multi-head attention is a simpler baseline for this encoder. GQA can return later as an efficiency ablation.

### Why exclude Hadamard?

The V2 experiments did not establish that structured Hadamard MLPs provide a clear advantage at matched quality and cost. They also introduced additional residual-behavior questions.

V3 is already changing the causal mask, convolutions, training objective, and inference procedure. Adding Hadamard immediately would confound the experiment.

Once the dense encoder is stable, we can test whether replacing lower-layer SwiGLUs with Hadamard mixers improves compute efficiency without damaging representations.

## Phase 4 — Implement masked diffusion

This is where the architecture becomes more than a BERT-style encoder.

I would use an absorbing-mask discrete diffusion process as the first approach.

The training path is:

```
Original sequence
       ↓
Sample corruption level t
       ↓
Replace selected tokens with [MASK]
       ↓
Bidirectional encoder
       ↓
Predict original tokens at masked positions
       ↓
Masked reconstruction loss
```

A few implementation choices should be explicit from the beginning.

Noise schedule: Sample corruption levels spanning nearly clean to fully masked sequences. Include high-noise training so that unconditional or minimally conditioned generation is actually represented in the objective.

Loss: Use masked-position cross-entropy initially, with an explicitly documented normalization and weighting scheme. For a rigorous diffusion objective, the timestep weighting matters; merely training with randomly varying mask ratios is not sufficient to establish equivalence to a particular diffusion formulation.

Special tokens: Handle padding and protected prompt tokens separately. The model must not learn to reconstruct padding as ordinary content, and masked inputs must never access clean target IDs through any auxiliary feature.

Determinism: Noise sampling must be reproducible and recoverable after a session restart.

I would implement the process in two stages:

1. Establish reliable masked reconstruction and bidirectional context use.
2. Add iterative denoising and generation.

For the first sampler, keep a fixed number of output slots and reveal high-confidence predictions over multiple steps. Later, we can investigate remasking, alternate reveal schedules and output-length prediction.

That keeps the initial implementation focused on verifying that the model learns the intended denoising task.

## Phase 5 — Controlled experiments

Your V2 work already provides a good template: smoke tests, health audits, matched evaluations, and explicit promotion criteria.

I would keep that research discipline.

| Experiment | Comparison                            | Primary question                                         |
| ---------- | ------------------------------------- | -------------------------------------------------------- |
| A          | BERT-style MLM vs diffusion objective | Does variable-noise training help enough to justify it?  |
| B          | Attention-only vs attention + BiConv  | Are convolutions providing useful local representations? |
| C          | 12 vs 16 blocks                       | Does additional depth improve results efficiently?       |
| D          | Dense SwiGLU vs lower-layer Hadamard  | Can we reduce cost without losing quality?               |

Only A and B need to happen before choosing the initial architecture. C and D can wait until the basic model demonstrates meaningful learning.

The original KiwiLM 2 evaluation metrics cannot all be reused unchanged. In particular, autoregressive perplexity is not directly comparable with masked-diffusion reconstruction loss.

Instead, I would maintain separate evaluation categories:

- Representation quality: masked reconstruction accuracy, contextual retrieval, and downstream classification probes.
- Generation quality: repetition, semantic consistency, infilling quality, and generation latency.
- Training efficiency: tokens/s, peak TPU memory, and compute units per training token.
- Model health: gradient distributions, residual amplification, and checkpoint parity.

For the first smoke, use a small token budget rather than immediately reproducing the V2 1B run. Once the encoder and sampler pass their acceptance tests, scale the stronger candidate.

## Phase 6 — Prepare KiwiClassifier 1

I would not implement the classifier during the first KiwiLM 3 smoke, but I would design the encoder interface with it in mind.

The encoder should expose contextual hidden states independently of the diffusion head:

```
hidden_states = model.encode(input_ids, ...)logits = model.reconstruction_head(hidden_states)
```

Later, KiwiClassifier can reuse those representations and attach a separate option-scoring head.

Its initial objective would be to accept an input and an arbitrary candidate set, returning a probability distribution across those candidates.

The classifier should be evaluated on held-out tasks before attempting RLCD or other more complicated training objectives. We would also need to test for candidate-order bias if all options are encoded jointly.

This makes KiwiClassifier a downstream experiment rather than something that dictates every V3 implementation detail.

## Proposed milestone order

| Milestone                    | Deliverable                                                | Promotion requirement                                   |
| ---------------------------- | ---------------------------------------------------------- | ------------------------------------------------------- |
| M0 — V2 finalized            | Published checkpoint, updated documentation, frozen source | Existing checkpoint and evaluation parity               |
| M1 — Cleanup                 | Reduced active code surface, shared infrastructure         | V2 regression suite passes                              |
| M2 — Notebook infrastructure | Reusable Colab training and recovery workflow              | Successful fresh-session resume                         |
| M3 — Encoder prototype       | Bidirectional attention + convolution + SwiGLU             | Forward/backward, bidirectionality, serialization tests |
| M4 — Denoising prototype     | Variable-noise pretraining objective                       | Stable training and correct mask handling               |
| M5 — Generation prototype    | Iterative masked sampler                                   | Reproducible infilling and basic generation             |
| M6 — Architecture selection  | Matched baseline experiments                               | Evidence supporting the selected model                  |
| M7 — KiwiLM 3 training       | Longer TPU run and comprehensive evaluation                | Quality, stability, and cost targets met                |
| M8 — KiwiClassifier 1        | Pretrained encoder with classification head                | Improvement over a randomly initialized classifier      |

### The scope I would freeze for KiwiLM 3.0

Include: bidirectional convolutions, encoder attention, dense SwiGLU, masked diffusion, a simple iterative sampler, and notebook-first training.

Defer: Hadamard, mHC, DeltaNet, Mamba, elaborate diffusion samplers, and RLCD.

There is a useful distinction between the two projects now: KiwiLM 2 tested whether a hybrid causal convolution/attention language model could learn effectively at small scale. KiwiLM 3 should test whether a related bidirectional encoder can learn useful representations while supporting iterative text generation.

That is a sufficiently substantial research question on its own. The codebase cleanup and notebook migration should make it easier to investigate without carrying all the V2 training infrastructure into the new experiment.