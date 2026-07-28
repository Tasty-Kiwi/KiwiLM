#!/usr/bin/env bash
set -euo pipefail

session_name="${COLAB_SESSION_NAME:-kiwilm-transformer-750k}"
colab_bin="${COLAB_BIN:-colab}"
data_dir="${KIWILM_DATA_DIR:-data/tinystories-750k}"
artifact_dir="$(mktemp -d "${TMPDIR:-/tmp}/kiwilm-transformer-750k.XXXXXX")"
result_dir="${KIWILM_RESULT_DIR:-runs/transformer-750k-colab}"
drive_backup_root="${KIWILM_DRIVE_BACKUP_ROOT:-/content/drive/MyDrive/KiwiLM/transformer-750k}"
backup_run_id="$(date -u +%Y%m%dT%H%M%SZ)"
drive_backup_dir="${drive_backup_root}/${backup_run_id}"
chunk_dir="${artifact_dir}/data-chunks"
drive_config="${artifact_dir}/drive-backup.json"
session_started=0
preserve_session=0

mkdir -p "${artifact_dir}" "${chunk_dir}"
if [[ -d "${result_dir}" ]] &&
   [[ -n "$(find "${result_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Result directory is not empty: ${result_dir}" >&2
  echo "Move it or set KIWILM_RESULT_DIR to a new directory." >&2
  exit 1
fi
mkdir -p "${result_dir}"

if [[ "${drive_backup_root}" != /content/drive/MyDrive/* ]]; then
  echo "KIWILM_DRIVE_BACKUP_ROOT must be inside /content/drive/MyDrive" >&2
  exit 1
fi

existing_status="$("${colab_bin}" status -s "${session_name}" 2>&1 || true)"
if [[ "${existing_status}" != *"not found"* ]]; then
  echo "Colab session '${session_name}' already exists:" >&2
  echo "${existing_status}" >&2
  echo "Stop it or set COLAB_SESSION_NAME to a different name." >&2
  exit 1
fi

cleanup() {
  if [[ "${session_started}" -eq 1 ]]; then
    "${colab_bin}" log -s "${session_name}" \
      -o "${result_dir}/session.jsonl" || true
    if [[ "${preserve_session}" -eq 1 ]]; then
      echo "[colab] Training or artifact handoff did not complete." >&2
      echo "[colab] Preserving session '${session_name}' for recovery." >&2
      "${colab_bin}" status -s "${session_name}" || true
      "${colab_bin}" url -s "${session_name}" || true
      echo "[colab] Stop it after recovery with:" >&2
      echo "  ${colab_bin} stop -s ${session_name}" >&2
    else
      "${colab_bin}" stop -s "${session_name}" || true
    fi
  fi
}
trap cleanup EXIT INT TERM

download_colab_file() {
  local remote_path="$1"
  local local_path="$2"
  local attempt
  for attempt in 1 2 3; do
    if "${colab_bin}" download -s "${session_name}" \
      "${remote_path}" \
      "${local_path}"; then
      return 0
    fi
    echo "[colab] Download attempt ${attempt}/3 failed for ${remote_path}" >&2
    sleep 2
  done
  return 1
}

create_colab_session() {
  local baseline_assignments
  local current_assignments
  local attempt
  local delay_seconds
  local exit_code
  local new_output

  baseline_assignments="$("${colab_bin}" sessions 2>&1 || true)"
  for attempt in 1 2 3 4 5; do
    echo "[colab] T4 allocation attempt ${attempt}/5..."
    if new_output="$("${colab_bin}" new -s "${session_name}" --gpu T4 2>&1)"; then
      printf '%s\n' "${new_output}"
      session_started=1
      return 0
    else
      exit_code=$?
    fi
    printf '%s\n' "${new_output}" >&2

    current_assignments="$("${colab_bin}" sessions 2>&1 || true)"
    if [[ "${current_assignments}" != "${baseline_assignments}" ]]; then
      echo "[colab] Server assignments changed during a failed allocation." >&2
      echo "[colab] Refusing to retry because that could leak another VM:" >&2
      printf '%s\n' "${current_assignments}" >&2
      return "${exit_code}"
    fi

    if [[ "${new_output}" != *"Service Unavailable"* &&
          "${new_output}" != *"Bad Gateway"* &&
          "${new_output}" != *"Gateway Timeout"* ]]; then
      return "${exit_code}"
    fi
    if [[ "${attempt}" -eq 5 ]]; then
      echo "[colab] T4 allocation remained unavailable after 5 attempts." >&2
      return "${exit_code}"
    fi
    delay_seconds=$((5 * 2 ** (attempt - 1)))
    echo "[colab] Transient allocation failure; retrying in ${delay_seconds}s..." >&2
    sleep "${delay_seconds}"
  done
}

if [[ ! -f "${data_dir}/metadata.json" ]]; then
  echo "Prepared 750k dataset not found at ${data_dir}" >&2
  exit 1
fi
uv run python scripts/validate_model_f_data.py "${data_dir}"

uv build --wheel --out-dir "${artifact_dir}"
built_wheel="$(find "${artifact_dir}" -maxdepth 1 -name 'kiwilm-*.whl' -print -quit)"
if [[ -z "${built_wheel}" ]]; then
  echo "uv build did not produce a KiwiLM wheel" >&2
  exit 1
fi
wheel_name="$(basename "${built_wheel}")"
jq -n \
  --arg backup_dir "${drive_backup_dir}" \
  '{backup_dir: $backup_dir}' > "${drive_config}"

echo "Splitting prepared data into 32 MiB upload chunks..."
for local_path in "${data_dir}"/*; do
  if [[ -f "${local_path}" ]]; then
    artifact_name="$(basename "${local_path}")"
    split -b 33554432 -a 4 \
      "${local_path}" \
      "${chunk_dir}/${artifact_name}.part-"
  fi
done

create_colab_session
"${colab_bin}" upload -s "${session_name}" \
  "${built_wheel}" \
  "/content/${wheel_name}"
"${colab_bin}" upload -s "${session_name}" \
  eval/story-consistency-prompts.json \
  /content/story-consistency-prompts.json
"${colab_bin}" upload -s "${session_name}" \
  "${drive_config}" \
  /content/kiwilm-drive-backup.json

for local_path in "${chunk_dir}"/*; do
  if [[ -f "${local_path}" ]]; then
    artifact_name="$(basename "${local_path}")"
    "${colab_bin}" upload -s "${session_name}" \
      "${local_path}" \
      "/content/${artifact_name}"
  fi
done

echo "[colab] Mounting Google Drive for checkpoint backup."
echo "[colab] Complete the interactive authorization prompt if one appears."
"${colab_bin}" drivemount -s "${session_name}" /content/drive

# After Drive is mounted, preserve the session on failure so partial
# checkpoints can be inspected or recovered before the user stops the VM.
preserve_session=1
"${colab_bin}" exec -s "${session_name}" \
  --timeout 14400 \
  -f scripts/colab_transformer_750k.py

download_colab_file \
  /content/kiwilm-transformer-750k/run/best.pt \
  "${result_dir}/best.pt"
download_colab_file \
  /content/kiwilm-transformer-750k/run/latest.pt \
  "${result_dir}/latest.pt"
download_colab_file \
  /content/kiwilm-transformer-750k/run/metrics.jsonl \
  "${result_dir}/metrics.jsonl"
download_colab_file \
  /content/kiwilm-transformer-750k-summary.json \
  "${result_dir}/summary.json"
download_colab_file \
  /content/kiwilm-transformer-750k-examples.md \
  "${result_dir}/examples.md"
download_colab_file \
  /content/kiwilm-transformer-750k-drive-backup-manifest.json \
  "${result_dir}/drive-backup-manifest.json"

preserve_session=0
echo "Transformer 750k artifacts downloaded to ${result_dir}"
echo "Verified Google Drive backup: ${drive_backup_dir}"
