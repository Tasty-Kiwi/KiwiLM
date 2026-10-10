# KiwiLM 3 collapse diagnostics

The first hybrid-12 50M-token L4/BF16 run completed and its recovered native
checkpoint passed checksum, budget and state-integrity checks. It did **not**
pass the learning-quality check: fixed validation matches a context-free
unigram model, and generation is dominated by commas and common tokens.
Do not promote this checkpoint or treat finite gradients as sufficient health.

## Repeat the local tests

```bash
uv run --locked python scripts/diagnose_kiwilm3.py --output runs/kiwilm3-diagnostics/overfit.json
uv run --locked pytest -q tests/test_v3_diagnostics.py tests/test_v3_accelerator.py
```

The first command runs three CPU FP32 probes at learning rates 0.001, 0.0003
and 0.0001, for 500 steps each. Each probe uses just four synthetic 16-token
windows, a 294-token vocabulary and a width-32, 12-block hybrid encoder. It
calls the real native `AcceleratorTrainer.train_step`, including variable
noise, selected-token normalization, gradient clipping, AdamW and token-based
warmup/cosine scheduling. No Drive access, cloud allocation, corpus training,
existing checkpoint modification or inference export occurs. Reports refuse
to overwrite existing paths. Exit code zero means the diagnostics completed;
inspect each probe's `passed` field rather than interpreting process completion
as a model-quality pass.

Two protected cues map to two equally frequent target symbols at the same
position. Cues occur on either side of the target; all targets are masked at
evaluation. Swapping cues must swap predictions. Erasing cues leaves identical
inputs and therefore 50% accuracy. The strict synthetic pass requires 100%
original/swapped accuracy, original loss below 0.1, and erased-cue accuracy 50%.
This tests contextual learning, not memorization of target position or copying
visible labels. It is not evidence of full-width or CUDA BF16 trainability.

An explicitly requested full-width local test can reuse only the prepared
corpus tokenizer, while still training **only the four synthetic windows**:

```bash
uv run --locked python scripts/diagnose_kiwilm3.py --overfit-width 512 --tokenizer-data-dir data/smollm-smoke --overfit-steps 100 --learning-rates 0.001 0.0003 0.0001 --threads 4 --output runs/kiwilm3-diagnostics/full-width.json
```

This has the real 32K output vocabulary and 512-wide/2048-FFN hybrid-12
configuration, but context remains 16, batch 4, accumulation 1 and CPU FP32.
Warmup is 640 tokens, not the corpus run's 1M-token warmup. Full-width local
probes refuse more than 300 steps. These are diagnostic controls, not frozen
corpus-run replacements or production LR selection.
Use `--seed 43` with a different output path to repeat the test at another
initialization. The floor LR is always peak LR divided by ten; these tests
compare learning-rate scales, not only a peak-LR change with a fixed floor.

To inspect the recovered corpus checkpoint, add its **generation directory**
(the directory containing `latest.pt`, `job.json`, `metrics.jsonl` and
`manifest.json`) and the matching original prepared data:

```bash
uv run --locked python scripts/diagnose_kiwilm3.py --skip-overfit --run-dir runs/kiwilm3-hybrid-12-smoke-50m/v3-bc-smoke-50m-hybrid-12-f5a4064b13e92aa4/step-00003052-274c9788be004ef4a27947428ba82367 --data-dir data/smollm-smoke --threads 4 --output runs/kiwilm3-diagnostics/checkpoint.json
```

The read-only audit checks file sizes/checksums, native format, identities,
tokenizer, data fingerprint, progress, metrics prefix and finite tied weights.
It performs two fixed batch-1 validation probes at each of 15%, 50%, 90% and
100% noise. Reversing only visible content keeps MASK slots, controls, noise
and targets fixed. At 100% noise there is no visible content to reverse: this
is a negative control, not evidence that context is ignored.

One backward probe at 15% noise records per-parameter gradient norms, mixer
and MLP update RMS, both residual amplifications, whole-block amplification,
noise-conditioning/embedding RMS and mean pairwise token cosine. It leaves
weights, `.grad` buffers and model mode unchanged, and removes hooks even on
failure. A fresh same-config, same-seed CPU initialization provides a scale
reference; it is not a recovered original step-zero checkpoint. A temporary
zero-conditioning hook is an inference counterfactual, not a proposed fix.
The descriptive collapse flags are investigation heuristics, not calibrated
promotion thresholds. This audit does not resume the saved CUDA contract or
establish CPU/CUDA BF16 parity.

## Observed results

Local results are saved in
`runs/kiwilm3-hybrid-12-smoke-50m/collapse-diagnostics.json` and
`collapse-checkpoint-final.json` under the same parent. These are ignored
research artifacts, not additional model checkpoints.

- At LR 0.001, the tiny native-path probe reaches loss **0.01954** and **100%**
  accuracy; swapping cues swaps all four predictions and erasing cues gives
  **50%**. Lower LRs also reach 100% cue accuracy, but not the strict loss
  threshold within 500 steps (loss 1.03696 / 3.68616). This does not select a
  corpus learning rate or establish that lowering LR will fix the real run.
- In two real validation windows, all **2,561** tested masked predictions
  across the four noise levels are commas. Changing 448/440 visible tokens
  at 15% masking changes **zero** argmax predictions; maximum probability
  changes are only about **0.000035 / 0.000042**.
- On the first window at 15% masking, block 0 input RMS is **0.08562** and
  post-attention residual RMS is **10.03370**: **117.19x** amplification.
  Its mixer update alone is **116.99x** input RMS. The comparable fresh
  initialization is **1.45x** amplification, not 117x.
- Block 0 output mean pairwise token cosine is **0.9999917**; the final
  pre-final-norm residual RMS is **71.42525**. The fresh reference is
  **0.80796** cosine after block 0 and **0.10588** final residual RMS.
- All parameter gradients in the bounded partial-noise probe remain finite
  and nonzero. This is representational collapse, not a wholly disconnected
  loss graph. MLP-only amplification checks miss the dominant first mixer.
- Zeroing the learned noise-conditioning output at inference still produces
  only commas. That counterfactual does not repair the model or rule out a
  conditioning contribution during training.

Full-width, real-vocabulary synthetic tests provide a more relevant warning
than the tiny-width pass. After 100 steps, with all other controls matched:

| Model seed | Peak LR | Final masked loss | Accuracy | Strict synthetic pass |
| --- | --- | --- | --- | --- |
| 42 | 0.001 | 2.38793 | 50% | No |
| 42 | 0.0003 | 0.00325 | 100% | Yes |
| 42 | 0.0001 | 0.04503 | 100% | Yes |
| 43 | 0.001 | 0.23879 | 100% | No: loss remains above 0.1 |
| 43 | 0.0003 | 0.00450 | 100% | Yes |

All lower-LR full-width successes also pass swapped-cue and erased-cue
controls. The 0.001 run is seed-sensitive, **not** universally unable to use
context: seed 43 learns the cue mapping, whereas seed 42 remains at chance
within the tested 100-step budget. These results support reducing the LR scale
as the first controlled corpus diagnostic, but do not prove the original
50M run's sole root cause. Raw local records are `full-width-overfit-001.json`,
`full-width-overfit-low-lr.json` and `full-width-overfit-seed43.json` under
`runs/kiwilm3-hybrid-12-smoke-50m/`.

For this original 50M run, the earliest measured collapse is at the first
attention block in the **final checkpoint**; its onset remains unknown there.
The subsequent 5M diagnostic below records earlier training points. These diagnostics
localize the failure but do not prove whether LR, residual scaling, conditioning,
the objective distribution or accelerator numerics initiated it. The original
checkpoint is preserved as the failed baseline.

## Gate before further corpus runs

Keep the remaining B/C 50M runs on hold. The user-run lower-LR 5M diagnostic
and bounded local conditioning/mixer-scale tests have now completed: neither
provides a sufficient fix. See the results below. Require meaningful contextual
improvement over a matched unigram baseline before repeating the full smoke.
Do not silently reinterpret a run, resume it under different settings or alter
the frozen V2 baseline.

## Lower-LR corpus diagnostic (completed; controls retained for reproduction)

From the repository root, explicitly start the combined user-run workflow:

```bash
bash scripts/run_colab_kiwilm3_lr_diagnostic.sh run
```

PowerShell uses the same launcher and controls:

```powershell
uv run --locked python scripts/run_colab_kiwilm3_lr_diagnostic.py run
```

Running either launcher without arguments (or with `plan`) only prints the
configuration. No VM is allocated until you explicitly request `run` or
`setup`. The combined `run` builds/freezes the wheel, allocates an L4, asks you
to authorize Drive, restores prepared data into VM-local storage, preflights,
starts a detached worker, watches, collects verified artifacts, then stops the
owned VM. It does not upload datasets, reuse the old optimizer or resume the
collapsed model. Availability, CUDA BF16 behavior and quality still need the
actual user-run cloud test; local verification is not GPU acceptance.

Controls are saved in `configs/kiwilm3-lr-diagnostic-5m.json`:

- Fresh hybrid-12, width 512, FFN 2048, context 512, dropout 0.1; unchanged
  architecture, noise policy, tokenizer IDs and AdamW implementation.
- Exactly **5,000,000 nonpadding input tokens** (306 updates for these packed
  data); batch 8, accumulation 4. The final update consumes only the remainder.
- Peak LR **0.0003**, floor **0.00003**, linear **1,000,000-token warmup**,
  then cosine decay over the remaining 4M tokens; weight decay 0.01, clip 1.
- Model/data/noise/validation seeds **42/42/142/242**, unchanged from the failed
  smoke. Fixed validation remains 20 batches per noise level at batch 8.
- Validation and collapse probes at **step 0**, every **25** updates, and at
  completion. Verified synchronous Drive commits at step 0, every **50**
  updates, and completion. Only latest/previous generations are retained
  within this new run's namespace; original backups are untouched.

This changes both LR scale (peak and floor) and the cosine horizon relative
to the frozen 50M experiment. It is an early-collapse diagnostic, **not** a
matched 50M or peak-only LR ablation. At 5M, context sensitivity can improve
without generation becoming coherent. Completion alone is not a quality pass.
The original frozen 50M defaults, recovered checkpoint and remaining B/C
experiments are unchanged/on hold.

The default cache is the already used prepared 50M dataset:

```text
/content/drive/MyDrive/KiwiLM2/data/tpu-smoke-66b9899b879a5aba9eabdd4a40a54ab9ede62fdd1070f43be9b4c5b0e0e9714b
```

`--data-cache /content/drive/MyDrive/...` can select a relocated copy, but its
data fingerprint and V3 tokenizer SHA256 must still equal the original smoke:
`66b9899b879a5aba9eabdd4a40a54ab9ede62fdd1070f43be9b4c5b0e0e9714b`
and `7fc721c3c0dfc77ca5786756b5821d292fad399f62298b46638fc83c8f680a8a`.
There is no silent corpus download/rebuild. Backup root remains
`/content/drive/MyDrive/KiwiLM3`; the namespace starts
`checkpoints/v3-lr3e4-diagnostic-5m-hybrid-12-` with a locked job-identity suffix.
Local state/artifacts live in `runs/colab/v3-gpu-hybrid-12-lr3e4-diagnostic-5m`.

### Completed 5M result and local isolation tests

The recovered 5M bundle passed integrity/provenance checks, but fixed probe
predictions are comma-only by step 25 (409,600 input tokens, still in warmup)
and remain so at every sampled point. Final larger-validation CE is 7.691744
at 15% noise. This did not resolve the learning failure.

Four bounded fresh CPU FP32 replays tested conditioning on/off × normal/quarter
mixer updates with the real width, vocabulary, context and native objective.
Each ran 32 updates, with batch 2 instead of 32 effective windows/update and
the same update-level LR as the GPU diagnostic. All four reached comma-only
argmaxes by their first post-initialization measurement at update 8. Combining
both interventions reduces residual growth but does not restore contextual
predictions. Inference-only interventions on the trained checkpoint also fail
to restore them. These are small local diagnostics, not GPU parity, long-run
recovery proof or a production fix; SwiGLU residual branches were not scaled.

The portable [analysis](../examples/comparisons/kiwilm3-lr-diagnostic-5m/analysis.md)
and [records](../examples/comparisons/kiwilm3-lr-diagnostic-5m/summary.json) include
controls, source hashes, trajectories, limitations and the next diagnostic gate.
Repeat locally without cloud allocation:

```bash
uv run --locked python scripts/isolate_kiwilm3_collapse.py --run-dir runs/colab/v3-gpu-hybrid-12-lr3e4-diagnostic-5m-recovered/downloads --data-dir data/smollm-smoke --replay-steps 32 --threads 4 --output runs/kiwilm3-diagnostics/isolation-repeat.json
```

`--replay-steps 0` performs frozen-checkpoint counterfactuals only. The script
refuses existing output paths and more than 64 updates per case. It never
loads the saved optimizer into a trainer or changes model defaults/checkpoints.

### Reading the diagnostic metrics

`metrics.jsonl` adds `v3_collapse_diagnostics` rows without changing standard
`v3_validation` or `v3_train` rows. These read-only eval-mode probes use the
current CUDA model and the training BF16 autocast context, without backward,
optimizer updates, model transfers or consuming training RNGs:

- One fixed batch-1 window at 15% noise records all blocks' input, mixer
  update, post-mixer, MLP update and output RMS; mixer/MLP contribution and
  amplification ratios; output mean pairwise token cosine.
- Two fixed batch-1 windows at 15%, 50%, 90% and 100% noise record masked CE,
  accuracy and argmax counts. Reversing visible content preserves MASK slots,
  controls, targets and noise; report loss/probability/argmax response. Full
  noise has no visible content and is explicitly a negative control.
- A context-free unigram baseline is fitted only to the **first 1M training
  tokens**, excluding protected-token counts, with add-one smoothing. Its
  loss and majority-token accuracy use exactly the same selected probe
  targets. `loss_minus_unigram < 0` favors the model, but two windows alone
  are not a statistically sufficient promotion evaluation.

The probe policy is serialized in the job and therefore resume/Drive-identity
locked. The small probes are deterministic across validation points; they do
not replace the larger fixed validation. Diagnostic and validation overhead
are outside reported training-step throughput. No automatic success/early-stop
threshold is imposed: inspect the trajectories, compare against step zero and
the failed checkpoint, and expand evaluation before repeating the full smoke.
Never claim improvement solely from finite gradients or changed argmaxes.

### Reattach or resume safely

If only the terminal disconnected, inspect the same worker first:

```bash
bash scripts/run_colab_kiwilm3_lr_diagnostic.sh status
bash scripts/run_colab_kiwilm3_lr_diagnostic.sh watch --stop-when-done
```

Do not rerun `run` to reconnect. If the VM really terminated, resume with its
preserved wheel/bootstrap and exact original controls into a **new** local
state directory:

```bash
bash scripts/run_colab_kiwilm3_lr_diagnostic.sh resume-run --from-state runs/colab/v3-gpu-hybrid-12-lr3e4-diagnostic-5m --state-dir runs/colab/v3-gpu-hybrid-12-lr3e4-diagnostic-5m-resume1
```

This is another explicit allocation. Failed monitoring/download does not stop
the VM automatically; check status/billing and stop explicitly when safe. The
verified Drive latest/previous commits remain the recovery source. Keep the
local state directory (including its reviewed wheel/bootstrap) for continuation.
