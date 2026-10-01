# Dense 1B on Colab TPU

This is a **fresh Dense KiwiLM 2 run**, not continuation of the 50M TPU smoke
or the 500M GPU checkpoint. The 1B schedule and TensorMuon checkpoint contract
are different; importing those optimizer states is refused.

The user elected to proceed without finishing the fresh-VM continuation test.
The reference test committed steps 20 and 40 to Drive, but remote allocation/
authorization failures prevented the resumed leg. Full-state recovery has local
CPU subprocess tests; **live fresh-TPU restart equivalence remains unverified**.
The corrected 50M TPU smoke passed. This launcher does not imply that the
remaining live recovery, on-XLA cached decoding, or matched GPU benchmark gates
have passed. No VM or training is started by implementation or local tests.

## Prepare, then start explicitly

From the repository root:

```bash
bash scripts/run_colab_kiwilm2_tpu_1b.sh setup
```

Setup uploads the wheel, the frozen tokenizer, and a small job specification;
**it never trains**. By default it reads `metadata.json` and the tokenizer JSON
from `data/smollm-smoke`; packed smoke files are not required. Override
`KIWILM2_DATA_DIR` with another retained prepared-data metadata/tokenizer directory
if necessary. The source metadata, actual 32K vocabulary, and SHA-256 are checked
before allocation; its immutable dataset revision and tokenizer provenance are
locked into the new job.

After the XLA preflight and interactive Drive mount, a cache miss prepares
**1,000,000,000 train tokens plus 2,000,000 validation tokens inside the VM**.
It streams the pinned FineWeb-Edu/Cosmopedia sources, excludes Python-Edu, and
reuses the tokenizer without training a new one. The new subset is not repeated
passes over the 50M smoke data. Expect first preparation to take substantially
longer than a cache hit. Progress output is retained in
`/content/kiwilm-tpu-prepare.log`; the terminal prints heartbeats while waiting.
Setup allows up to six hours for preparation, plus transfers/installations.

Prepared data is verified, cached in a separate Drive namespace, and copied to
VM-local storage on future sessions. Setup locks the resolved data fingerprint
into both the downloaded `tpu-job.json` and the local/remote ownership hashes.
Keep that local job file: it is the exact specification needed for recovery.
Existing corrupt or mismatched caches fail rather than being overwritten.

Once setup says `NO TRAINING STARTED`, start deliberately:

```bash
bash scripts/run_colab_kiwilm2_tpu_1b.sh train
```

The default session is `kiwilm2-tpu-v6e1-final-1b-muon`; local output is
`runs/colab/tpu-v6e1-final-1b-muon`. This is separate from all smoke/continuation
sessions. Setup leaves an idle **potentially billable** VM until you train or stop
it. Only the verified owned session is stopped on training exit/failure.
No automatic VM reallocation or automatic restart occurs.

## Frozen run settings

- Dense backbone, tied embedding/head, context 512, seed 42, random initialization.
- v6e-1 by default; isolated Python 3.12, Torch/XLA 2.9, XLA BF16 autocast.
- TensorMuon LR 0.01, auxiliary AdamW LR 0.0003, minimum LR 0.00003; existing
  clipping, weight decay, and betas.
- Batch 8, accumulation 4: 16,384 tokens per full optimizer update.
- Exactly 1B tokens: 61,036 updates; 2,560 valid targets in the final masked update.
- Token-based cosine schedule, 20M warmup tokens (the existing final-run 2% rule).
- 200 fixed validation batches every 500 updates and at completion, seed 43.
- Atomic local checkpoint and **synchronous verified Drive publication every
  500 updates** and at completion. Training stops after persistent backup failure.
- First-checkpoint portable reload and final CPU FP32 loss/health/cache checks
  remain enabled. CPU cache checks are not on-XLA decoding validation.
- Worker bounded at 22 hours; the outer execution timeout includes a five-minute
  margin. Colab may reclaim the VM sooner. CLI/network timeouts still require a
  new explicitly requested resume. Environment setup, transfers, and Drive I/O
  are excluded from steady training throughput.

## Recovery and Drive storage

Drive cannot be disabled for the 1B job. The ready `tpu-job.json` records the exact
`drive_backup_dir` and `drive_cache_dir`. The backup namespace includes dataset
revision, tokenizer, tokenizer-source provenance, TPU type, and Muon settings;
the checkpoint manifest additionally locks the full training/model/data contract.

`latest.json` and `previous.json` identify the newest two distinct committed
steps. Generation directories contain full weights, optimizer, data-generator,
CPU/process/XLA RNG, LR/token state, metrics, and job metadata. The latest pointer
is published only after copied-file SHA-256/size read-back checks succeed.
Look for `drive_checkpoint_committed` with `verified: true` in progress output.
Use **one writer per backup namespace**; never run fresh and resume concurrently.

A sudden VM loss can lose up to 500 updates (8.192M tokens) since the last
acknowledged publication. Drive availability or cloud durability is not guaranteed
by mount checks. Reserve roughly 6GB of additional free Drive space for the 2GB
data, retained checkpoints and staging, on top of existing backups. Only known
superseded committed generations are pruned; failed copies can leave orphan
directories for manual inspection. No old smoke, GPU, or other Drive backups
are removed by this launcher.

For the long run, downloadable artifact chunks are packaged at session completion,
not redundantly after every save. Drive remains the primary periodic recovery
source. On failure the launcher attempts VM-local emergency downloads before
stopping its session; those transfers can fail, especially after VM loss.

After confirming the old VM is terminated, resume on a **fresh VM**:

```bash
bash scripts/run_colab_kiwilm2_tpu_1b.sh resume setup
bash scripts/run_colab_kiwilm2_tpu_1b.sh resume train
```

These use new `...-resume1` local/session names and the original default
`runs/colab/tpu-v6e1-final-1b-muon/tpu-job.json`. Missing original provenance,
data cache, or valid committed checkpoint is an error: **never start fresh**.
Budget, schedule, tokenizer, fingerprint and optimizer must match. A changed or
corrupt generation is not silently treated as a new run; inspect any restore or
newer-progress refusal before continuing.

For another recovery, or custom original paths, choose new session/results names
and point to a ready job from the original run or its successful resume:

```bash
export KIWILM2_TPU_JOB_FROM=runs/colab/tpu-v6e1-final-1b-muon/tpu-job.json
export COLAB_SESSION_NAME=kiwilm2-tpu-v6e1-final-1b-muon-resume2
export KIWILM_RESULT_DIR=runs/colab/tpu-v6e1-final-1b-muon-resume2
bash scripts/run_colab_kiwilm2_tpu_1b.sh resume setup
bash scripts/run_colab_kiwilm2_tpu_1b.sh resume train
```

Do not change the inherited cache or checkpoint namespace during recovery.
Clear custom session/result environment variables when switching to a new job.

## Authorization/allocation failures

If the CLI Drive mount disconnects, a manual mount is supported:

```bash
export KIWILM2_TPU_DRIVE_MOUNT=manual
bash scripts/run_colab_kiwilm2_tpu_1b.sh setup
```

Open the allocated session's notebook on Colab's website. In a notebook cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Return to the terminal and press Enter only after it mounts successfully.
Preparation verifies the actual mount; pressing Enter cannot bypass Drive checks.
This option pauses **before** preparing data/training; authentication is performed
by you. Use a new session/result name if an earlier setup already has an owner.

An allocation POST timeout has an ambiguous outcome: the server may have created
a VM without the CLI saving its local session record. **Do not blindly retry.**
Check Colab's website or `colab sessions`, terminate any orphan you own, then
retry setup. This launcher does not automatically retry billable allocation.
