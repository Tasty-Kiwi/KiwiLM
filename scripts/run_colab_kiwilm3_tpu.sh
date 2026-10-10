#!/usr/bin/env bash
# User-run only. No arguments are a plan, never allocation/training.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -eq 0 ]]; then
  set -- plan
fi
exec uv run --locked python scripts/run_colab_kiwilm3_tpu.py "$@"
