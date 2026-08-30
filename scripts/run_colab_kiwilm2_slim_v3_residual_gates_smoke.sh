#!/usr/bin/env bash
set -euo pipefail

audit_path="${KIWILM2_RESIDUAL_AUDIT:-examples/comparisons/kiwilm2-slim-v3-residual-audit/audit.json}"
if [[ ! -f "${audit_path}" ]]; then
  echo "Residual audit not found at ${audit_path}" >&2
  echo "Run scripts/audit_kiwilm2_residual_growth.py before paid smoke training." >&2
  exit 1
fi

common=(
  KIWILM2_PHASE=smoke
  KIWILM2_VARIANT=kiwilm2_slim_v3
  KIWILM2_UPPER_SWIGLU_BLOCKS=4
  KIWILM2_OPTIMIZER=adamw
  KIWILM2_BATCH_SIZE=8
  KIWILM2_GRAD_ACCUM_STEPS=4
  KIWILM2_MAX_TOKENS=50000000
  KIWILM2_RESIDUAL_AUDIT="${audit_path}"
)

env "${common[@]}" \
  KIWILM2_SWIGLU_RESIDUAL_GATE_INIT=0.25 \
  scripts/run_colab_kiwilm2.sh

env "${common[@]}" \
  KIWILM2_SWIGLU_RESIDUAL_GATE_INIT=0.5 \
  scripts/run_colab_kiwilm2.sh
