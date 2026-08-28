#!/usr/bin/env bash
set -euo pipefail

KIWILM2_PHASE=architecture \
KIWILM2_VARIANT=kiwilm2 \
KIWILM_RESULT_DIR="${KIWILM_RESULT_ROOT:-runs/colab/kiwilm2-architecture}/kiwilm2-adamw" \
scripts/run_colab_kiwilm2.sh

KIWILM2_PHASE=architecture \
KIWILM2_VARIANT=kiwilm2_slim \
KIWILM2_COMPILE_POLICY=auto \
COLAB_SESSION_NAME="kiwilm2-architecture-kiwilm2-slim-gated-v2-adamw" \
KIWILM_RESULT_DIR="${KIWILM_RESULT_ROOT:-runs/colab/kiwilm2-architecture}/kiwilm2-slim-gated-v2-adamw" \
scripts/run_colab_kiwilm2.sh
