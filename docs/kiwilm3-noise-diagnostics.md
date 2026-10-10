# KiwiLM 3 paired masking-policy diagnostic

This bounded local comparison follows the
[balanced readout test](../examples/comparisons/kiwilm3-readout-fixed-corpus/analysis.md).
The native tied/full-loss model can memorize that task; the next investigation
changes the training masking policy only. No cloud allocation or production
model/optimizer/checkpoint change occurs. The completed seed-42 comparison
and its limitations are in the
[results](../examples/comparisons/kiwilm3-noise-policy-512/analysis.md): fixed
15% did not recover context sensitivity or beat aligned unigram CE. No
production change or further cloud smoke is authorized by this result.

## Run locally

```bash
uv run --locked python scripts/diagnose_kiwilm3_noise.py \
  --run-dir runs/colab/v3-gpu-hybrid-12-lr3e4-diagnostic-5m-recovered/downloads \
  --data-dir data/smollm-smoke \
  --steps 64 --seed 42 --threads 4 \
  --output runs/kiwilm3-diagnostics/noise-policy-512-repeat.json
```

PowerShell supports the same arguments on one line with its own local bundle
path. Output paths are exclusive. The script bounds each condition to 128
updates and PyTorch intra-op threads to 8; it accepts completed collected
bundles or verified native generations. Data/tokenizer fingerprints must match.
No saved model weights, optimizer or RNG are restored into either trainer.
`--seed 43` with another output path repeats at another initialization; it does
not alter the fixed data/noise/validation seeds.

## Conditions and controls

- **Variable**: native absorbing-mask policy, noise sampled uniformly over
  [0.01, 1], with an additional 10% full-noise mixture component.
- **Fixed-015**: the same Bernoulli selection uniforms thresholded at 0.15;
  supply 0.15 to noise conditioning for every training window. No random
  replacement, forced mask, support restriction, target reweighting or other
  objective change is introduced.

Both use unchanged hybrid-12, width 512, FFN 2048, tied head, dense full-vocab
selected-token CE, context **512**, dropout 0.1 and AdamW. CPU FP32 replaces
the original CUDA BF16, with batch 2/accumulation 1 rather than batch 8/
accumulation 4. Token horizon and warmup are divided by 16 (312,500/62,500)
to match the original 5M diagnostic's **update-level LR**, including peak
0.0003 and floor 0.00003. This does not match its per-update examples, token
budget, gradient variance or accelerator numerics.

At the default 64 updates, each condition sees **65,536 nonpadding input
tokens** and the two together see 131,072. These are early-learning probes,
not completed 312,500-token runs or another 5M smoke. Neither automatically
advances to a cloud run, full smoke, checkpoint export or architecture selection.

Native `AcceleratorTrainer.train_step` remains responsible for data selection,
loss normalization, gradients, clipping and optimizer updates. A temporary
single-owner diagnostic context intercepts **training corruption only**, then
restores the native function even on failure. Both conditions consume the
native level draws and the same Bernoulli-uniform stream. Fixed noise replays
the draws in a private generator; mismatched RNG consumption causes refusal.
Nested/prepatched policy contexts are rejected. Evaluation calls the native
forced-level masking directly, outside this context.

For each update, record data/mask hashes, generator-end hashes, sampled native
levels, effective levels, eligible/selected counts, LR, losses and gradient
norms. Original step-zero metrics, data order, mask RNG, evaluation windows,
unigram baseline and LR must match across conditions or the run fails.
Training rows retain the native trainer's `objective` label; the enclosing
case's `policy` and each row's `pairing.effective_levels` identify the diagnostic
override. They are not native checkpoint/resume metadata.
Native-variable loss/gradient/progress parity against an unmodified trainer is
regression-tested. No diagnostic state can be resumed or published as a native
checkpoint with misleading masking provenance.

## Evaluation and interpretation

At step zero, every 16 updates and completion, evaluate four fixed held-out
batch-1 windows at 15%, 50%, 90% and full noise, using independent fixed seeds.
All conditions have the same input tokens, masks, targets and supplied levels
at evaluation. Report noise levels separately; aggregate window losses and
accuracies by **selected-target count**, not an unweighted window mean.

The unigram is fitted to the first 1M **training** tokens only, excluding
protected-token counts and adding one-count smoothing. Its CE and accuracy use
the exact same selected held-out targets as the model. The aligned difference
is descriptive evidence, not a calibrated promotion threshold; four windows
are not the original larger validation budget.

Reversal changes only visible content, retaining MASK/control positions,
noise and labels. Record CE response, maximum probability change and argmax
response. Also record target margin against the best other vocabulary item,
pre-final-norm RMS/centered energy over all valid and selected tokens, per-block
gradient/update statistics and the final/first dense MLP gradient ratio.
Do not infer semantic understanding from response to an out-of-distribution
reversal, positional variation or raw RMS alone. Fully masked inputs have no
visible context to reverse and are only a negative control.

Input tokens and updates match, but **selected-target budgets and prediction
difficulty do not**: variable noise masks more targets on average and usually
provides less visible context. Report cumulative selected targets and empty
updates explicitly. This is a mask-policy ablation at matched input/update
budget, not proof that low-noise training is inherently more sample-efficient
or a substitute for the intended diffusion objective. Fixed-policy performance
at high noise must not be hidden by averaging with low-noise results.

Passing or failing this short, single-seed CPU test alone cannot select a
production fix. The original checkpoint and remaining B/C smoke controls stay
unchanged; cloud work requires an explicit user-run action.
