#!/usr/bin/env bash
set -euo pipefail

session_name="${COLAB_SESSION_NAME:-kiwilm-model-b-2m}"
colab_bin="${COLAB_BIN:-colab}"
tokenizer_data_dir="${KIWILM_TOKENIZER_DATA_DIR:-data/tinystories-550k}"
artifact_dir="$(mktemp -d "${TMPDIR:-/tmp}/kiwilm-model-b-2m.XXXXXX")"
result_dir="${KIWILM_RESULT_DIR:-runs/model-b-2m-colab}"
bundle_dir="${artifact_dir}/tokenizer-bundle"
session_started=0
preserve_session=0

mkdir -p "${artifact_dir}" "${bundle_dir}" "${result_dir}"

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

session_started=1
"${colab_bin}" new -s "${session_name}" --gpu T4
"${colab_bin}" upload -s "${session_name}" \
  "${built_wheel}" \
  "/content/${wheel_name}"
"${colab_bin}" upload -s "${session_name}" \
  eval/story-consistency-prompts.json \
  /content/story-consistency-prompts.json

for local_path in "${bundle_dir}"/*; do
  if [[ -f "${local_path}" ]]; then
    artifact_name="$(basename "${local_path}")"
    "${colab_bin}" upload -s "${session_name}" \
      "${local_path}" \
      "/content/${artifact_name}"
  fi
done

"${colab_bin}" exec -s "${session_name}" \
  --timeout 18000 \
  -f scripts/colab_model_b_2m.py

# From this point onward the completed weights exist on the remote VM. Preserve
# the billable session if download or verification fails so they can be
# recovered manually instead of being destroyed by the EXIT trap.
preserve_session=1

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
