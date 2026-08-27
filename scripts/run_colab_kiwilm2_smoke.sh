#!/usr/bin/env bash
set -euo pipefail

KIWILM2_PHASE=smoke \
KIWILM2_VARIANT=kiwilm2 \
KIWILM_RESULT_DIR="${KIWILM_RESULT_ROOT:-runs/colab/kiwilm2-smoke}/kiwilm2-adamw" \
scripts/run_colab_kiwilm2.sh

KIWILM2_PHASE=smoke \
KIWILM2_VARIANT=kiwilm2_slim \
KIWILM_RESULT_DIR="${KIWILM_RESULT_ROOT:-runs/colab/kiwilm2-smoke}/kiwilm2-slim-adamw" \
scripts/run_colab_kiwilm2.sh
