# KiwiLM 3 — L4 BF16 recovery qualification

Reviewed 2026-10-10. **PASS for operational two-VM continuation** of the tiny
synthetic hybrid-12 probe. This is not a corpus smoke or architecture result.
The agent inspected local user-downloaded evidence; no cloud work was launched.

## Evidence

Source artifacts: `runs/colab/kiwilm3-gpu/downloads/` and
`runs/colab/kiwilm3-gpu-resume/downloads/`. These local artifacts are not
published or added to Git. Both original ownership locks and saved
wheel/bootstrap checksums were verified without rebuilding either package.

| Check | Result |
| --- | --- |
| Runtime | NVIDIA L4, CUDA BF16, Python 3.12.3, torch 2.13.0+cu130 |
| Model | Tiny synthetic hybrid-12; width/context 16, 49,680 parameters |
| VM 1 pause | Step 8, 512 input tokens; verified Drive backup receipt |
| VM 2 restore | Same step-8 checkpoint hash, same job identity, distinct Linux boot ID |
| VM 2 finish | Status complete; step 64, 64 optimizer steps, exactly 4,096 input tokens |
| Metrics | Original prefix byte-for-byte preserved; steps 1–64, no gaps/duplicates |
| Validation | 16 events at steps 4–64; four fixed noise levels per event |
| Token schedule | Every logged LR matches locked warmup/cosine; final LR 0.0001 |
| Numeric checks | Finite losses, positive finite gradient norms, finite model/Adam moments |
| Native state | Completed boundary, progress/schedule/job/VM identity and metrics hash match |
| State contents | FP32 model and Adam moments, step-64 Adam counters, matching tied-head weights, saved data/noise and CUDA RNG state |
| Downloads | All 10 manifest files and the archive part pass size/SHA256 checks |
| Final persistence | Verified step-64 Drive publication receipt matches downloaded checkpoint hash |
| Shutdown | Launcher log confirms resumed Colab session terminated |

Job identity:
`7864354adaf1c2476724b718bc289e9c4a3f1caa7d11de2f3b8ec3f08a4845c2`.

Restored step-8 checkpoint SHA256:
`c0ffaa01a126c81433f967fef6f330894d15335f98f7b23daacfa60dd704c76a`.

Final step-64 checkpoint SHA256:
`e61cb7e64634ea758728dd850bfc64cebc5223bdfeb7479614ff53612185a72b`.

Final training masked CE was 4.6364. Fixed validation masked CE was 4.7952,
4.8199, 4.9210 and 4.9217 at noise levels 0.15, 0.5, 0.9 and 1.0. These tiny
synthetic measurements are not general language quality, autoregressive
perplexity, or a model-selection signal. Peak allocated memory was 18,535,424
bytes; width/context 16 makes this unsuitable for full-width capacity planning.

## Boundaries and next gate

The successful restore validates the saved runtime's operational continuation,
including the native checkpoint restore path. It does not prove bitwise
equality against an uninterrupted CUDA run, long-run reliability, current
Drive contents after collection, pruning behavior, other GPU families, or
full-width 512-token throughput/memory. Drive publication is supported by the
worker's verified commit receipts; no independent live Drive inspection was
performed. The summary field `live_continuation_qualified=false` is an
unconditional worker placeholder, not a failed audit; originals remain intact.

Next: matched 50M B/C experiments with dense SwiGLU, variable-noise masked CE,
hybrid/attention-only × 12/16 blocks. First verify the existing prepared 50M
Drive cache and review full-width controls. Use one card family, precision,
tokenizer/data fingerprint, wheel, seeds and token schedule across candidates.
Do not infer throughput or change batch settings from this tiny probe.

The [single-command launcher](kiwilm3-colab-cli.md) uses `run` for fresh jobs
and `resume-run` for new-VM continuation. Both remain user-invoked; no corpus
training is started automatically.
