#!/usr/bin/env bash
# User-run two-VM recovery qualification, never a 1B launch.
set -euo pipefail
phase="${1:-}"
action="${2:-setup}"
if [[ $# -gt 2 || ( "${phase}" != "reference" && "${phase}" != "resume" ) ||
      ( "${action}" != "setup" && "${action}" != "train" ) ]]; then
  echo "Usage: $0 {reference|resume} [setup|train]" >&2
  exit 1
fi
if [[ "${KIWILM2_USE_DRIVE:-1}" != "1" || -n "${KIWILM2_RESUME_FROM:-}" ]]; then
  echo "Continuation qualification requires Drive and its own step-20 checkpoint." >&2
  exit 1
fi
export KIWILM2_TPU_CONTINUATION_PHASE="${phase}"
export KIWILM2_USE_DRIVE=1
tpu="${COLAB_TPU:-v6e1}"
test_id="${KIWILM2_TPU_TEST_ID:-restart-v1}"
if [[ ! "${test_id}" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "KIWILM2_TPU_TEST_ID must contain only letters, numbers, underscores or hyphens." >&2
  exit 1
fi
export KIWILM_RESULT_DIR="${KIWILM2_TPU_TEST_RESULT_ROOT:-runs/colab}/tpu-${tpu}-continuation-${test_id}-${phase}"
export COLAB_SESSION_NAME="kiwilm2-tpu-${tpu}-continuation-${test_id}-${phase}"
# A repeat gets an isolated Drive namespace as well as new local/session names.
export KIWILM2_TPU_DRIVE_BACKUP="/content/drive/MyDrive/KiwiLM2/checkpoints/tpu-${tpu}-continuation-${test_id}-${phase}"
if [[ "${phase}" == "resume" ]]; then
  export KIWILM2_TPU_DRIVE_RESUME="/content/drive/MyDrive/KiwiLM2/checkpoints/tpu-${tpu}-continuation-${test_id}-reference"
else
  export KIWILM2_TPU_DRIVE_RESUME=""
fi
exec bash "$(dirname "$0")/run_colab_kiwilm2_tpu_smoke.sh" "${action}"
