# KiwiLM 3: balanced fixed-corpus output-head diagnostic

## Finding

**The unchanged tied-head encoder can learn this small contextual task with
the native full-vocabulary masked CE.** Untying the head is not necessary for
this result. Both full-loss cases pass original-context, context-pairing and
erased-context checks. This provides no reason to redesign the production
head based solely on the earlier comma-only argmaxes.

This is a four-window memorization test with one initialization, not proof
that the existing corpus objective trains well, that the old checkpoint is
repaired, or that another long run will succeed. No canonical fix is selected.

## Method and provenance

The [runbook](../../../docs/kiwilm3-readout-diagnostics.md) documents the exact
test and command. [results.json](results.json) preserves all measurements,
training rows, task windows/offsets, controls and source hashes.

- Verified source checkpoint SHA256:
  `b512e0f30b304f6090fab1caf1da4f6e55fcc2d4d0b8b3edcc6ee8ad86a22f0b`.
- Native job identity:
  `78a6ce7e67c759872aec469fbeb613d53c03bf599e2b9e582747ce31ffc050c5`.
- Prepared data fingerprint:
  `66b9899b879a5aba9eabdd4a40a54ab9ede62fdd1070f43be9b4c5b0e0e9714b`.
- V3 tokenizer SHA256:
  `7fc721c3c0dfc77ca5786756b5821d292fad399f62298b46638fc83c8f680a8a`.
- Fixed task SHA256:
  `627135348b127ce8d4c15bc140bdebcb4194da35b7d157b45a1668e67e407bae`.

Four nonoverlapping natural corpus windows come deterministically from the
first 1M training tokens, at offsets 1358, 1153, 1867 and 7138. Each has 64
tokens, with a single MASK at position 32. Targets alternate comma (ID 15)
and ` the` (ID 266), two of each. Neither target ID occurs anywhere in visible
context; controls are excluded. The supplied noise is identical at 1/64.

Fresh runs use the original hybrid-12/512-width/2048-SwiGLU configuration and
real 32,001-token vocabulary, but CPU FP32, batch four, no accumulation and
64-token inputs. Each case repeats these windows for 100 updates: **25,600
input tokens and 400 selected-target exposures**. All four cases together
consume 102,400 repeated input tokens. No validation/generalization set exists
for this task; evaluation is on the same four windows in eval mode.

AdamW uses peak LR 0.0003, floor 0.00003, 10-update warmup, cosine decay,
weight decay 0.01, clip 1 and model seed 42. Dropout remains 0.1. This is a
separate local overfit schedule, not the original GPU training contract.
The native dense selected-token CE is used for full-loss cases; two-target
cases deliberately restrict CE support to IDs 15/266.

Untied cases clone the original embedding weights into a separate head before
optimizer construction, so all four cases have **exactly identical step-zero
evaluation results**, including logits-derived metrics and hidden-state energy.
They consume identical dropout RNG draws under the same seed and schedule.
Untying increases parameters from 65,124,352 to 81,508,864; this is a diagnostic
intervention, not a parameter-matched architectural comparison.

Whole-context exchange swaps opposite-class windows while retaining the MASK
slot/noise and pairing expected labels with the new context. It checks pairing,
not an unseen semantic edit. Erasure replaces all context by identical MASK
inputs at the original low noise, intentionally out of distribution. All rows
then must share a prediction, so balanced two-class accuracy is 50%.

## Results after 100 updates

| Case | Full-vocabulary CE | Two-target CE | Full accuracy | Context swap | Erased context | Full-task pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Tied + full loss | 0.000640 | 0.0000938 | 100% | 100% | 50% | Yes |
| Untied + full loss | 0.000543 | 0.0001053 | 100% | 100% | 50% | Yes |
| Tied + two-target loss | 4.990567 | 0.0000218 | 100% | 100% | 50% | No: full CE |
| Untied + two-target loss | 4.990579 | 0.0000218 | 100% | 100% | 50% | No: full CE |

Full-task acceptance was defined as 100% original/swapped full-vocabulary
accuracy, original full CE below 0.1, and erased full accuracy 50%. Both
restricted-loss cases pass the corresponding two-target criterion, but not
full-vocabulary CE. All reported evaluation logits/hidden states and training
losses/gradient norms are finite. Recorded gradient norms are measured before
clipping at 1, not the clipped values.

The tied/full model reaches 100% eval accuracy by update 20 (CE 0.136434),
then full CE 0.004526 by update 40. Its final minimum target margin against
the strongest other vocabulary item is **9.1883**. The average target-pair
probability mass is approximately **99.883%**.

The tiny numerical difference between tied/untied full-loss CE is not evidence
of a production winner: one seed and four memorized windows cannot resolve
that question. The native tied case already passes without added parameters.

## Why probability and representation measurements matter

Two-target training gets every full argmax right, yet assigns only approximately
**0.69%** probability mass to the target pair on average. Its full-vocabulary
CE remains **4.99**, despite almost-zero two-class CE. Accuracy or argmax
diversity alone would falsely suggest equivalent success. Restricting support
is useful for diagnosis, not a usable replacement for the corpus objective.

Before any fresh training, the verified 5M checkpoint predicts commas on all
four windows: full and two-target accuracy **50%**, full CE **3.248780**,
two-target CE **0.698666**. The empirical balanced target-only unigram has
CE **ln(2) = 0.693147**. This prior is not a smoothed full-vocabulary corpus
baseline; the 50% result is not comparable to its original 4.134% accuracy on
the larger, naturally distributed validation set.

Across the four masked target representations, the checkpoint's centered
energy fraction is **4.44e-7**. Fresh step zero is **0.01350**, rising to
**0.28972** in the successful tied/full case. Thus these contexts become
distinguishable in the fresh task, whereas the trained checkpoint's target
representations are overwhelmingly shared-mode on these inputs.

Importantly, the successful tied/full model has final all-token pre-norm RMS
**3.9738**, not a small residual norm. Its centered energy is **28.56%** of
total energy; the checkpoint's corresponding values are RMS **4.8922** and
**0.00759%**. Absolute residual RMS growth alone is not a sufficient failure
criterion. Useful token/context-dependent variation matters. These are
descriptive measurements, not calibrated corpus promotion thresholds.

## What this does and does not settle

- It rules out an absolute inability of the current native tied/full-loss
  implementation to learn a contextual distinction at real model width/vocab
  under this small CPU task. Loss/gradient regression tests also match the
  native dense CE to the selected-logit reference.
- It does **not** rule out harmful tying interactions under the real corpus
  distribution, higher noise, long contexts or accelerator numerics.
- It does **not** isolate which change from the corpus run enables learning:
  balanced repeated targets, fixed low noise, short context, simpler task,
  training schedule and CPU precision all differ.
- Swapping already trained contexts and erasing all content cannot establish
  semantic generalization. No new checkpoint is saved or promoted.

## Recommended next gate

Keep Dense SwiGLU, native weight tying and full-vocabulary loss unchanged.
The next focused experiment should isolate **mask/noise distribution** before
another architectural change: current variable-noise training versus a fixed
15%-noise diagnostic, with the same real-data order, initialization, 512-token
context and optimizer schedule. Report input-token and selected-target budgets
separately; these objectives otherwise see different amounts of supervision.

Evaluate held-out context response, probability changes/margins, full-vocabulary
CE versus an aligned training-only unigram baseline, and centered hidden-state
energy. Argmax diversity, a lower RMS, or passing this four-window task is not
sufficient to restart the full B/C smoke matrix. A fixed-noise diagnostic is
an investigation of the failure, not a replacement for the intended diffusion
training. No such cloud experiment has been launched or prepared as a claimed
fix in this step.

Follow-up: the [paired masking-policy test at context 512](../kiwilm3-noise-policy-512/analysis.md)
is now complete. Fixed 15% did not restore held-out context sensitivity in
64 fresh CPU updates. Neither policy beat the aligned unigram baseline;
production defaults and the corpus-run hold remain unchanged.

## Verification and safety

- **638 tests passed, 1 skipped** (original V2 artifact absent), with two
  existing ONNX deprecation warnings; focused readout tests: **11 passed**.
- Ruff, changed-Python formatting, `git diff --check`, offline dependency-lock
  validation (124 packages), offline wheel build and retained V3 shell syntax
  checks passed.
- Tests cover no visible-label leakage, balanced inputs, deterministic window
  selection, native loss/gradient parity, shared initial metrics, paired LR,
  parameter-count difference, deterministic replays, RNG/mode/hook preservation,
  bounds, exclusive output and read-only verified-checkpoint handling.
- No cloud allocation, checkpoint continuation, production-default mutation,
  commit or push. Source checkpoint/job/metrics remain intact.

Raw local output is `runs/kiwilm3-diagnostics/readout-fixed-corpus-seed42.json`.
Its checksum and diagnostic implementation hashes are recorded in
[results.json](results.json), which contains no weights or machine-specific
absolute paths.
