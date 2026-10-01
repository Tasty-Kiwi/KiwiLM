#!/usr/bin/env bash
# Fresh 1B Dense/TensorMuon job; setup never trains. Resume requires the original job.
set -euo pipefail
mode=fresh
if [[ "${1:-}" == "resume" ]]; then
  mode=resume
  shift
fi
action="${1:-setup}"
if [[ $# -gt 1 || ( "${action}" != "setup" && "${action}" != "train" ) ]]; then
  echo "Usage: $0 [setup|train] or $0 resume [setup|train]" >&2
  exit 1
fi
export COLAB_TPU="${COLAB_TPU:-v6e1}"
export KIWILM2_TPU_PHASE=final-1b
export KIWILM2_TPU_ENTRYPOINT="$0"
suffix=""
if [[ "${mode}" == "resume" ]]; then
  suffix=-resume1
  export KIWILM2_TPU_JOB_FROM="${KIWILM2_TPU_JOB_FROM:-runs/colab/tpu-${COLAB_TPU}-final-1b-muon/tpu-job.json}"
  export KIWILM2_TPU_ENTRY_MODE="resume "
elif [[ -n "${KIWILM2_TPU_JOB_FROM:-}" || -n "${KIWILM2_TPU_DRIVE_RESUME:-}" ]]; then
  echo "Use the explicit resume mode for 1B checkpoint continuation." >&2; exit 1
fi
export COLAB_SESSION_NAME="${COLAB_SESSION_NAME:-kiwilm2-tpu-${COLAB_TPU}-final-1b-muon${suffix}}"
export KIWILM_RESULT_DIR="${KIWILM_RESULT_DIR:-runs/colab/tpu-${COLAB_TPU}-final-1b-muon${suffix}}"
exec bash "$(dirname "$0")/run_colab_kiwilm2_tpu_smoke.sh" "${action}"
