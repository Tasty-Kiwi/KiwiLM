# KiwiLM 3: paired masking policies at context 512

## Outcome

**Fixed 15% masking did not resolve the contextual-learning failure in this
bounded probe. Do not change the production policy or restart the B/C smoke
matrix on this evidence.** Both conditions learned lower full-vocabulary CE
than their shared initialization, but remained worse than an aligned
training-only unigram baseline. At update 64, selected-token predictions were
uniformly ` the` under variable-noise training and `.` under fixed-noise
training. Changing the preferred common token is not a repair.

This is a fresh, single-seed **CPU FP32 early-learning diagnostic**, not a
continuation of the 5M checkpoint, GPU parity test, completed smoke, or proof
that either objective cannot eventually learn. The
[machine-readable records](summary.json) preserve controls, pairing evidence,
all training rows, evaluation trajectories and detailed final probes. The
[runbook](../../../docs/kiwilm3-noise-diagnostics.md) describes reproduction.

## Controls and supervision budget

Both conditions use the native hybrid-12 model: width 512, SwiGLU 2048, four
attention/BiConv31/BiConv63 groups, tied head, dropout 0.1, context 512, and
AdamW. No checkpoint weights, optimizer state or saved RNG state enter either
trainer. Model/data/noise/validation seeds are 42/42/142/242.

CPU batch 2/accumulation 1 replaces the source run's CUDA BF16 batch
8/accumulation 4. The schedule horizon/warmup are divided by 16 to
312,500/62,500 tokens, preserving the original **update-level** LR curve
(peak 0.0003, floor 0.00003), not the original per-update token budget or
gradient variance. Each condition stops after 64 updates, at 65,536 input
tokens, just beyond warmup; neither reaches its schedule horizon.

| Training policy | Input tokens | Eligible tokens | Selected targets | Empty updates |
| --- | ---: | ---: | ---: | ---: |
| Native variable noise | 65,536 | 65,380 | 36,751 | 0 |
| Fixed 15% | 65,536 | 65,380 | 9,711 | 0 |

Variable training samples noise uniformly over [0.01, 1] with a 10% additional
full-noise mixture. Fixed training thresholds the same underlying selection
uniforms at 0.15 and supplies 0.15 to conditioning. It retains absorbing MASK
replacement and full-vocabulary selected-token loss. Variable sees **3.7845x**
as many supervised targets but generally less visible context: input/update
budgets match; target budgets and task difficulty do not.

All 64 updates match on clean-data hashes, native level draws, mask RNG end
state, dropout RNG end state and LR. Initial evaluation, validation-window
hashes and unigram hashes also match. Variable's first 32 updates reproduce
the preceding native isolation control exactly for loss, gradient norm,
selected count, LR and token progress. Evaluation never intercepts or consumes
training corruption RNG.

## Aligned held-out results

Four fixed batch-1 validation windows are evaluated at each noise level, with
identical masks, targets and supplied noise across conditions. CE is weighted
by selected-target count within each level; levels are not averaged together.
The unigram uses only the first 1M training tokens, excludes protected-token
counts and applies add-one smoothing. Lower CE is better.

| Evaluation masking | Selected targets | Unigram CE | Variable CE | Fixed-15% CE |
| --- | ---: | ---: | ---: | ---: |
| 15% | 289 | 7.705601 | 8.037327 | 8.230157 |
| 50% | 1,030 | 7.764378 | 8.083483 | 8.249295 |
| 90% | 1,832 | 7.813006 | 8.125448 | 8.307148 |
| 100% | 2,046 | 7.822665 | 8.131000 | 8.303118 |

At 15% evaluation masking, variable is **0.192830 CE lower** than fixed,
but still 0.331726 above unigram; fixed is 0.524556 above unigram. Accuracy
is 2.42% versus 4.15%, respectively. Fixed's superficially higher accuracy
does not outweigh its worse distributional loss or demonstrate context use:
all 289 fixed predictions are periods, while variable predicts ` the` for all
289. Unigram accuracy on these targets is also 4.15%, but its majority token
is comma, not period; identical aggregate accuracy is coincidental.

The 15%-evaluation CE trajectories at updates 0/16/32/48/64 are:

- Variable: 10.4600 / 9.5024 / 8.7461 / 8.0529 / 8.0373.
- Fixed: 10.4600 / 9.5663 / 8.8687 / 8.2202 / 8.2302.

Both switch between common-token argmaxes over training; neither recovers
diverse selected-token predictions at a measured post-initialization point.
Four windows and one initialization are insufficient for population-level
ranking or long-run policy selection.

## Context response and hidden-state health

At 15% evaluation masking, reversing visible content changes 1,738 of 1,757
visible tokens while preserving MASK/control positions and targets. Neither
condition changes any selected-token argmax. Target-weighted reversed-minus-
original CE is only +0.000001690 (variable) and +0.000002957 (fixed); maximum
absolute probability changes are 0.000007857 and 0.000010114. These are tiny
responses, not evidence of useful contextual reconstruction. Reversal is an
out-of-distribution sensitivity check, not a semantic-understanding test.
At 100% noise, no visible content can be reversed; exactly zero response is
the expected negative control, not a separate failure.

Selected-token centered-energy fractions before final RMSNorm are
1.34e-6–2.66e-6 for variable and 1.38e-6–2.62e-6 for fixed across the four
15%-noise windows. Thus almost all measured masked-state energy is shared
within each window in both conditions. This geometry and weak context response,
not a common-token argmax alone, support the negative finding.

The separate fixed first-window 15%-noise backward probe records:

| Diagnostic at update 64 | Variable | Fixed 15% |
| --- | ---: | ---: |
| First attention post/input residual RMS ratio | 4.222630 | 4.179171 |
| Final block output RMS | 2.946819 | 2.721507 |
| Final block mean token cosine | 0.999950 | 0.999940 |
| Last/first MLP gradient norm | 1.331343 | 1.061163 |

All recorded evaluation activations/logits and backward-probe gradients remain
finite, and all parameter gradient norms in the recorded probes are nonzero.
This excludes a simple disconnected-gradient failure in those probes; it does
not certify every intermediate training tensor. The backward probe has its
own fixed masking seed, so its loss need not equal the first validation row.
A slightly smaller absolute RMS in fixed training is not sufficient to claim
healthier representations.

## Decision and next gate

Keep the native variable-noise objective, tied head and Dense SwiGLU unchanged.
No canonical fix is selected. This test finds no evidence that replacing
variable noise with fixed 15% alone restores learning on new corpus windows;
it does not establish the root cause or rule out longer-training recovery.

Before another corpus run, test full-vocabulary **multi-target memorization at
context 512** on a small fixed set of corpus windows and fixed 15% masks. Keep
the architecture/optimizer unchanged and check target margins, masked-state
centered energy and sensitivity to paired contexts. The earlier successful
four-window test had context 64 and one selected target per window; it does
not establish that the model can solve the longer, multi-target task. Report
memorization separately from unseen masks/windows, and retain hard update and
CPU bounds. That further experiment has **not** been launched here.

## Verification and provenance

- Focused masking-policy tests: **10 passed**. Complete suite: **648 passed,
  1 skipped** (original V2 artifact absent), with two existing ONNX warnings.
- Ruff, changed-Python formatting, `git diff --check`, offline lock validation
  (124 packages), offline wheel build and V3 shell syntax checks passed.
- Tests cover native variable-step parity, shared selection uniforms, scoped
  restoration on exceptions, unchanged evaluation RNG/mode/gradients,
  deterministic pairing/aggregation, bounds and exclusive output paths.
- Original checkpoint, job and metrics checksums still match their collected
  manifest. No production defaults or checkpoints were changed; no cloud
  allocation, export, commit or push occurred.

Source checkpoint SHA256:
`b512e0f30b304f6090fab1caf1da4f6e55fcc2d4d0b8b3edcc6ee8ad86a22f0b`.
Source job identity:
`78a6ce7e67c759872aec469fbeb613d53c03bf599e2b9e582747ce31ffc050c5`.
Raw local result: `runs/kiwilm3-diagnostics/noise-policy-512-seed42.json`
(635,212 bytes), SHA256
`e0e94e0f3c7b1adc8db13913f611032c81c34e35685ddb1f86b8af5c3b76cdf4`.
Data/tokenizer identities and diagnostic implementation hashes are retained
in [summary.json](summary.json); it contains no weights or machine-specific
absolute paths. Intermediate evaluations retain low-noise window details and
per-noise aggregates; final evaluation additionally retains all window/noise
rows and full gradient details. Full intermediate details remain in the raw
local output.
