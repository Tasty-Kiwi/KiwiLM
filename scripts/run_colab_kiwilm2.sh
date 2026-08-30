#!/usr/bin/env bash
set -euo pipefail

phase="${KIWILM2_PHASE:-smoke}"
variant="${KIWILM2_VARIANT:-kiwilm2}"
optimizer="${KIWILM2_OPTIMIZER:-adamw}"
muon_lr="${KIWILM2_MUON_LR:-0.02}"
gpu="${COLAB_GPU:-T4}"
colab_bin="${COLAB_BIN:-colab}"
batch_size="${KIWILM2_BATCH_SIZE:-8}"
grad_accum_steps="${KIWILM2_GRAD_ACCUM_STEPS:-4}"
learning_rate="${KIWILM2_LEARNING_RATE:-0.0003}"
min_learning_rate="${KIWILM2_MIN_LEARNING_RATE:-0.00003}"
precision="${KIWILM2_PRECISION:-fp16}"
compile_policy="${KIWILM2_COMPILE_POLICY:-auto}"
upper_swiglu_blocks="${KIWILM2_UPPER_SWIGLU_BLOCKS:-}"
swiglu_residual_gate_init="${KIWILM2_SWIGLU_RESIDUAL_GATE_INIT:-}"
seed="${KIWILM2_SEED:-42}"
timeout_seconds="${COLAB_TIMEOUT_SECONDS:-82800}"
resume_from="${KIWILM2_RESUME_FROM:-}"
drive_backups="${KIWILM2_DRIVE_BACKUPS:-1}"
drive_root="${KIWILM2_DRIVE_ROOT:-/content/drive/MyDrive/KiwiLM2}"
schedule_suffix=""
gate_suffix=""
if [[ "${variant}" == "kiwilm2_slim_v3" ]]; then
  upper_swiglu_blocks="${upper_swiglu_blocks:-4}"
  if [[ "${upper_swiglu_blocks}" != "3" && "${upper_swiglu_blocks}" != "4" ]]; then
    echo "KIWILM2_UPPER_SWIGLU_BLOCKS must be 3 or 4 for Slim v3" >&2
    exit 1
  fi
  schedule_suffix="-h$((10 - upper_swiglu_blocks))-s${upper_swiglu_blocks}"
  if [[ -n "${swiglu_residual_gate_init}" ]]; then
    case "${swiglu_residual_gate_init}" in
      0.25) gate_suffix="-gate025" ;;
      0.5|0.50) gate_suffix="-gate050" ;;
      *)
        echo "KIWILM2_SWIGLU_RESIDUAL_GATE_INIT must be 0.25 or 0.5" >&2
        exit 1
        ;;
    esac
    residual_audit="${KIWILM2_RESIDUAL_AUDIT:-}"
    if [[ -z "${residual_audit}" || ! -f "${residual_audit}" ]]; then
      echo "Gated Slim v3 requires KIWILM2_RESIDUAL_AUDIT" >&2
      exit 1
    fi
    uv run --locked python -c \
      'import json,sys; a=json.load(open(sys.argv[1])); assert a.get("gated_smoke_authorized") is True and a.get("residual_growth_reproduced") is True, "audit does not authorize gated smoke"' \
      "${residual_audit}"
  fi
elif [[ -n "${upper_swiglu_blocks}" ]]; then
  echo "KIWILM2_UPPER_SWIGLU_BLOCKS is valid only for kiwilm2_slim_v3" >&2
  exit 1
elif [[ -n "${swiglu_residual_gate_init}" ]]; then
  echo "KIWILM2_SWIGLU_RESIDUAL_GATE_INIT is valid only for kiwilm2_slim_v3" >&2
  exit 1
fi
session_name="${COLAB_SESSION_NAME:-kiwilm2-${phase}-${variant}${schedule_suffix}${gate_suffix}-${optimizer}}"
result_dir="${KIWILM_RESULT_DIR:-runs/colab/${phase}-${variant}${schedule_suffix}${gate_suffix}-${optimizer}}"
artifact_dir="$(mktemp -d "${TMPDIR:-/tmp}/kiwilm2-colab.XXXXXX")"
job_path="${artifact_dir}/kiwilm2-job.json"
session_started=0
artifacts_downloaded=0

case "${gpu}" in
  T4|L4|G4|A100|H100) ;;
  *)
    echo "COLAB_GPU must be T4, L4, G4, A100, or H100; found '${gpu}'" >&2
    exit 1
    ;;
esac

case "${drive_backups}" in
  0|1) ;;
  *)
    echo "KIWILM2_DRIVE_BACKUPS must be 0 or 1; found '${drive_backups}'" >&2
    exit 1
    ;;
esac

mkdir -p "${result_dir}"

if [[ -n "${resume_from}" && ! -f "${resume_from}" ]]; then
  echo "Resume checkpoint not found at ${resume_from}" >&2
  exit 1
fi

cleanup() {
  if [[ "${session_started}" -eq 1 ]]; then
    if [[ "${artifacts_downloaded}" -eq 0 ]]; then
      echo "Attempting emergency checkpoint recovery before stopping Colab..." >&2
      "${colab_bin}" download -s "${session_name}" \
        /content/kiwilm2-colab/run/latest.pt \
        "${result_dir}/emergency-latest.pt" || true
      "${colab_bin}" download -s "${session_name}" \
        /content/kiwilm2-colab/run/best.pt \
        "${result_dir}/emergency-best.pt" || true
      "${colab_bin}" download -s "${session_name}" \
        /content/kiwilm2-colab/run/metrics.jsonl \
        "${result_dir}/emergency-metrics.jsonl" || true
    fi
    "${colab_bin}" log -s "${session_name}" \
      -o "${result_dir}/session.jsonl" || true
    "${colab_bin}" stop -s "${session_name}" || true
  fi
  rm -rf "${artifact_dir}"
}
trap cleanup EXIT INT TERM

existing_status="$("${colab_bin}" status -s "${session_name}" 2>&1 || true)"
if [[ "${existing_status}" != *"not found"* && "${existing_status}" != *"Not found"* ]]; then
  echo "Colab session '${session_name}' already exists:" >&2
  echo "${existing_status}" >&2
  echo "Stop it or set COLAB_SESSION_NAME to a different name." >&2
  exit 1
fi

job_command=(
  uv run python scripts/prepare_kiwilm2_colab_job.py
  --output "${job_path}"
  --phase "${phase}"
  --architecture "${variant}"
  --optimizer "${optimizer}"
  --muon-lr "${muon_lr}"
  --batch-size "${batch_size}"
  --grad-accum-steps "${grad_accum_steps}"
  --learning-rate "${learning_rate}"
  --min-learning-rate "${min_learning_rate}"
  --precision "${precision}"
  --compile-policy "${compile_policy}"
  --seed "${seed}"
  --drive-root "${drive_root}"
)
if [[ -n "${KIWILM2_MAX_TOKENS:-}" ]]; then
  job_command+=(--max-tokens "${KIWILM2_MAX_TOKENS}")
fi
if [[ -n "${upper_swiglu_blocks}" ]]; then
  job_command+=(--upper-swiglu-blocks "${upper_swiglu_blocks}")
fi
if [[ -n "${swiglu_residual_gate_init}" ]]; then
  job_command+=(--swiglu-residual-gate-init "${swiglu_residual_gate_init}")
fi
if [[ "${KIWILM2_ALLOW_DATA_TOKEN_MISMATCH:-0}" == "1" ]]; then
  job_command+=(--allow-data-token-mismatch)
fi
if [[ "${drive_backups}" == "0" ]]; then
  job_command+=(--no-drive-backups)
fi
"${job_command[@]}"

uv build --wheel --out-dir "${artifact_dir}"
built_wheel="$(find "${artifact_dir}" -maxdepth 1 -name 'kiwilm-*.whl' -print -quit)"
if [[ -z "${built_wheel}" ]]; then
  echo "uv build did not produce a KiwiLM wheel" >&2
  exit 1
fi
wheel_name="$(basename "${built_wheel}")"

"${colab_bin}" new -s "${session_name}" --gpu "${gpu}"
session_started=1
if [[ "${drive_backups}" == "1" ]]; then
  echo "Mount Google Drive in the Colab session when prompted..."
  "${colab_bin}" drivemount -s "${session_name}" /content/drive
fi
"${colab_bin}" upload -s "${session_name}" \
  "${built_wheel}" "/content/${wheel_name}"
"${colab_bin}" upload -s "${session_name}" \
  "${job_path}" /content/kiwilm2-job.json

if [[ -n "${resume_from}" ]]; then
  "${colab_bin}" upload -s "${session_name}" \
    "${resume_from}" /content/resume.pt
  resume_dir="$(dirname "${resume_from}")"
  if [[ -f "${resume_dir}/best.pt" ]]; then
    "${colab_bin}" upload -s "${session_name}" \
      "${resume_dir}/best.pt" /content/resume-best.pt
  fi
  if [[ -f "${resume_dir}/metrics.jsonl" ]]; then
    "${colab_bin}" upload -s "${session_name}" \
      "${resume_dir}/metrics.jsonl" /content/resume-metrics.jsonl
  fi
fi

"${colab_bin}" exec -s "${session_name}" \
  --timeout "${timeout_seconds}" \
  -f scripts/colab_kiwilm2_train.py

manifest_path="${result_dir}/artifact-manifest.json"
"${colab_bin}" download -s "${session_name}" \
  /content/kiwilm2-artifacts/artifact-manifest.json \
  "${manifest_path}"

while IFS= read -r part_name; do
  "${colab_bin}" download -s "${session_name}" \
    "/content/kiwilm2-artifacts/${part_name}" \
    "${result_dir}/${part_name}"
done < <(
  uv run python -c \
    'import json,sys; print(*[part["name"] for part in json.load(open(sys.argv[1]))["parts"]], sep="\n")' \
    "${manifest_path}"
)

uv run python scripts/reassemble_colab_artifacts.py \
  "${manifest_path}" "${result_dir}"
artifacts_downloaded=1
echo "KiwiLM 2 Colab artifacts downloaded to ${result_dir}"
