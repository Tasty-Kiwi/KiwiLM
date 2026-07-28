#!/usr/bin/env bash
set -euo pipefail

session_name="${COLAB_SESSION_NAME:-kiwilm-model-b-2m}"
colab_bin="${COLAB_BIN:-colab}"
tokenizer_data_dir="${KIWILM_TOKENIZER_DATA_DIR:-data/tinystories-550k}"
artifact_dir="$(mktemp -d "${TMPDIR:-/tmp}/kiwilm-model-b-2m.XXXXXX")"
result_dir="${KIWILM_RESULT_DIR:-runs/model-b-2m-colab}"
drive_backup_root="${KIWILM_DRIVE_BACKUP_ROOT:-/content/drive/MyDrive/KiwiLM/model-b-2m}"
backup_run_id="$(date -u +%Y%m%dT%H%M%SZ)"
drive_backup_dir="${drive_backup_root}/${backup_run_id}"
bundle_dir="${artifact_dir}/tokenizer-bundle"
drive_config="${artifact_dir}/drive-backup.json"
session_started=0
preserve_session=0

mkdir -p "${artifact_dir}" "${bundle_dir}" "${result_dir}"

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
      echo "[colab] Artifact handoff did not complete." >&2
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

if [[ ! -f "${tokenizer_data_dir}/metadata.json" ]]; then
  echo "Prepared tokenizer source not found at ${tokenizer_data_dir}" >&2
  exit 1
fi
uv run kiwilm export-tokenizer \
  --data-dir "${tokenizer_data_dir}" \
  --output-dir "${bundle_dir}"

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

for local_path in "${bundle_dir}"/*; do
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

# From this point onward training can publish checkpoints. Preserve the session
# on any later failure; Drive is the primary backup and the CLI archive remains
# a separately verified handoff.
preserve_session=1
"${colab_bin}" exec -s "${session_name}" \
  --timeout 18000 \
  -f scripts/colab_model_b_2m.py

download_colab_file \
  /content/kiwilm-model-b-2m-artifacts.json \
  "${result_dir}/artifact-manifest.json"

while IFS= read -r part_name; do
  download_colab_file \
    "/content/${part_name}" \
    "${result_dir}/${part_name}"
done < <(jq -r '.parts[].name' "${result_dir}/artifact-manifest.json")

uv run python scripts/reassemble_colab_artifacts.py \
  "${result_dir}/artifact-manifest.json" \
  "${result_dir}"

preserve_session=0
echo "Model B 2M artifacts downloaded to ${result_dir}"
echo "Google Drive backup: ${drive_backup_dir}"
