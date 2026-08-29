#!/usr/bin/env bash
set -euo pipefail

result_root="${KIWILM_RESULT_ROOT:-runs/colab/kiwilm2-slim-v3-smoke}"

KIWILM2_PHASE=smoke \
KIWILM2_VARIANT=kiwilm2_slim_v3 \
KIWILM2_UPPER_SWIGLU_BLOCKS=3 \
KIWILM2_COMPILE_POLICY=auto \
COLAB_SESSION_NAME="${COLAB_SESSION_NAME_H7S3:-kiwilm2-smoke-slim-v3-h7-s3-adamw}" \
KIWILM_RESULT_DIR="${result_root}/kiwilm2-slim-v3-h7-s3-adamw" \
scripts/run_colab_kiwilm2.sh

KIWILM2_PHASE=smoke \
KIWILM2_VARIANT=kiwilm2_slim_v3 \
KIWILM2_UPPER_SWIGLU_BLOCKS=4 \
KIWILM2_COMPILE_POLICY=auto \
COLAB_SESSION_NAME="${COLAB_SESSION_NAME_H6S4:-kiwilm2-smoke-slim-v3-h6-s4-adamw}" \
KIWILM_RESULT_DIR="${result_root}/kiwilm2-slim-v3-h6-s4-adamw" \
scripts/run_colab_kiwilm2.sh
