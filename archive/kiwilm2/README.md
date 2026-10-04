# KiwiLM 2 workflow archive

These scripts reproduce concluded V2 experiments. They are preserved and tested,
but removed from the active `scripts/` surface before KiwiLM 3 development.
Run them **from the repository root**, with their new explicit archive paths.
Paths in archived commands and tests were updated; experiment settings and
checkpoint/data/Drive identities were not changed.

## Contents

- `scripts/run_colab_*.sh`: eleven GPU/TPU CLI launchers, sweeps and continuation probes.
- `scripts/colab_kiwilm2_train.py`, `colab_kiwilm2_tpu_smoke.py`: remote workers.
- `scripts/prepare_kiwilm2_colab_job.py`: V2-only job specification builder.
- `scripts/run_kiwilm2_experiment.py`: matched V2 training/candidate runner.
- `scripts/audit_kiwilm2_residual_growth.py`, `evaluate_kiwilm2_residual_gate_smoke.py`:
  residual audit / gated-smoke evaluation.
- `scripts/select_kiwilm2_residual_gate.py`, `select_kiwilm2_slim_v3.py`,
  `validate_kiwilm2_slim_v3_ablation.py`: historical promotion/provenance tools.
- [Detailed V2 experiment runbook](docs/kiwilm2.md),
  [TPU qualification](docs/tpu-smoke.md), [1B run/recovery recipe](docs/tpu-1b.md).

The documents record the state of their experiments at the time, including
unmet qualification gates. The later [final 1B report](../../examples/comparisons/kiwilm2-final-1b-tpu-muon/analysis.md)
is authoritative for the completed run.

## Safety and compatibility

Archival does **not** make launchers harmless: running setup may allocate billable
hardware; GPU launchers may train immediately. Only the user should start them.
TPU setup/train separation, strict resume contracts, and verified latest/previous
Drive publication remain tested with fake CLIs and local temporary directories.
No remote Colab command is used by those tests.

The underlying V2 runtime and recovery modules stay in `src/kiwilm/` to preserve
checkpoint loading and tested infrastructure. No experimental report, dataset,
checkpoint, browser bundle, or Drive backup was deleted by cleanup.

Original script paths can also be recovered from the pre-cleanup V2 source tag.
The new V3 notebook workflow will be implemented separately, not by copying the
orchestration into notebook cells.
