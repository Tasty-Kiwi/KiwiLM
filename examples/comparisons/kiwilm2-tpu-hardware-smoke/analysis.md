# KiwiLM 2 TPU hardware smoke — runtime validation blocked

Date: 2026-09-04. **No TPU training throughput or loss result was obtained.**
Do not use this attempt to select TPU for the 1B run.

## Intended probe

Fresh Dense, Muon 0.01, BF16, single Colab v5e1 chip, context 512, batch 8,
accumulation 4, seed 42. Run the first 200 updates / 3,276,800 tokens of a
50M-token schedule with 1M warmup tokens; exclude the first 20 updates from
steady throughput. Five fixed validation batches are for compatibility only.

Prepared smoke fingerprint:
`66b9899b879a5aba9eabdd4a40a54ab9ede62fdd1070f43be9b4c5b0e0e9714b`.

Tokenizer SHA-256:
`4bcfc2d969a7a8c2285b364d709917d14e17a141e281fed9d7770db00329acf3`.

## Observed outcomes

| Attempt | Outcome before training | Follow-up |
| --- | --- | --- |
| Initial | Colab rejected a 100MB file upload with HTTP 500 | Replaced uploads with checksummed 4MiB chunks |
| Chunked | System Python could not run `ensurepip` | Isolated standalone Python 3.12 via uv |
| Python 3.12 | Bootstrap passed different chunk and extraction directories | Corrected the call; real chunk reassembly regression test passes locally |
| Reassembly fixed | XLA import failed: `libpython3.12.so.1.0` not found | Added standalone Python's library directory to the child linker search path |
| Preflight first | Colab allocation API hit a 120-second read timeout | No further allocations; check for abandoned assignments |

The fourth attempt successfully installed Python 3.12.14, Torch 2.9.0+cpu,
torch-xla 2.9.0 and libtpu 0.0.21, and reassembled the validated data. It did
**not** execute the model on TPU. The final linker-path fix remains unverified
on the real TPU because the following allocation timed out.

All four allocated sessions were terminated. The final `colab sessions` check
reported **no active sessions on the server**. The ongoing Windows 500M run
and all Google Drive backups were untouched. Raw local logs remain under
`runs/colab/tpu-v5e1-muon-smoke*`; they are not copied into this portable report.

## What is verified locally

- 167 pytest tests passed, plus Ruff, shell syntax, diff checks, offline lock
  validation, and wheel build.
- Tensor-based Muon matches reference updates under a changing learning rate.
- Split/resumed tiny CPU smoke matches uninterrupted training.
- Experimental smoke checkpoints are rejected by the production trainer.
- Bootstrap command syntax, standalone Python setup, shared-library path,
  chunk reassembly, and single-TPU/synchronization dispatch are tested.

These are CPU/mocked proofs, not TPU compilation, performance, numerical
equivalence, cached-generation parity, or on-TPU resume evidence.

## Next action

Retry the [preflight-first launcher](../../../scripts/run_colab_kiwilm2_tpu_smoke.sh)
when Colab allocation is responsive. It must execute a real TPU matrix
multiplication before transferring the dataset. See the
[hardware-smoke runbook](../../../docs/tpu-smoke.md).

Only after a completed 200-step probe should we commission a matched idle GPU
control and a full 50M smoke. Keep BF16-versus-FP16 and package-version
differences explicit. Compare synchronized execution, compilation stability,
CPU fallbacks, memory, and total time including evaluation/checkpoint/restart
overhead. Cached-generation parity and on-TPU checkpoint-resume checks remain
required before considering a 1B job. **No hardware winner is selected.**
