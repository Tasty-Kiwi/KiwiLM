#!/usr/bin/env bash
set -euo pipefail

export COLAB_GPU="${COLAB_GPU:-L4}"
export KIWILM2_PHASE=architecture
export KIWILM2_VARIANT=kiwilm2_slim_v3
export KIWILM2_UPPER_SWIGLU_BLOCKS=4
export KIWILM2_SWIGLU_RESIDUAL_GATE_INIT=0.5
export KIWILM2_OPTIMIZER=adamw
export KIWILM2_BATCH_SIZE=8
export KIWILM2_GRAD_ACCUM_STEPS=4
export KIWILM2_MAX_TOKENS=250000000
export KIWILM2_PRECISION=bf16
export KIWILM2_COMPILE_POLICY=compiled
export KIWILM2_RESIDUAL_AUDIT="${KIWILM2_RESIDUAL_AUDIT:-examples/comparisons/kiwilm2-slim-v3-residual-audit/audit.json}"
export KIWILM2_PROMOTION_OVERRIDE="${KIWILM2_PROMOTION_OVERRIDE:-examples/comparisons/kiwilm2-smoke-slim-v3-residual-gates/manual-promotion.json}"

scripts/run_colab_kiwilm2.sh
