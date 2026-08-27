#!/usr/bin/env bash
set -euo pipefail

phase="${KIWILM2_PHASE:-smoke}"
result_root="${KIWILM_RESULT_ROOT:-runs/colab/kiwilm2-${phase}-muon}"

for muon_lr in 0.01 0.02 0.04; do
  KIWILM2_PHASE="${phase}" \
  KIWILM2_VARIANT=kiwilm2 \
  KIWILM2_OPTIMIZER=muon \
  KIWILM2_MUON_LR="${muon_lr}" \
  COLAB_SESSION_NAME="kiwilm2-${phase}-muon-${muon_lr//./-}" \
  KIWILM_RESULT_DIR="${result_root}/muon-${muon_lr}" \
  scripts/run_colab_kiwilm2.sh
done
