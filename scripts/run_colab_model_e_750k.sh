#!/usr/bin/env bash
set -euo pipefail

session_name="${COLAB_SESSION_NAME:-kiwilm-model-e-750k}"
colab_bin="${COLAB_BIN:-colab}"
data_dir="${KIWILM_DATA_DIR:-data/tinystories-750k}"
artifact_dir="$(mktemp -d "${TMPDIR:-/tmp}/kiwilm-model-e-750k.XXXXXX")"
result_dir="${KIWILM_RESULT_DIR:-runs/model-e-750k-colab}"
chunk_dir="${artifact_dir}/data-chunks"
session_started=0

mkdir -p "${artifact_dir}" "${chunk_dir}" "${result_dir}"

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
    "${colab_bin}" stop -s "${session_name}" || true
  fi
}
trap cleanup EXIT INT TERM

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

echo "Splitting prepared data into 32 MiB upload chunks..."
for local_path in "${data_dir}"/*; do
  if [[ -f "${local_path}" ]]; then
    artifact_name="$(basename "${local_path}")"
    split -b 33554432 -a 4 \
      "${local_path}" \
      "${chunk_dir}/${artifact_name}.part-"
  fi
done

session_started=1
"${colab_bin}" new -s "${session_name}" --gpu T4
"${colab_bin}" upload -s "${session_name}" \
  "${built_wheel}" \
  "/content/${wheel_name}"
"${colab_bin}" upload -s "${session_name}" \
  eval/story-consistency-prompts.json \
  /content/story-consistency-prompts.json

for local_path in "${chunk_dir}"/*; do
  if [[ -f "${local_path}" ]]; then
    artifact_name="$(basename "${local_path}")"
    "${colab_bin}" upload -s "${session_name}" \
      "${local_path}" \
      "/content/${artifact_name}"
  fi
done

"${colab_bin}" exec -s "${session_name}" \
  --timeout 14400 \
  -f scripts/colab_model_e_750k.py

"${colab_bin}" download -s "${session_name}" \
  /content/kiwilm-model-e-750k/run/best.pt \
  "${result_dir}/best.pt"
"${colab_bin}" download -s "${session_name}" \
  /content/kiwilm-model-e-750k/run/latest.pt \
  "${result_dir}/latest.pt"
"${colab_bin}" download -s "${session_name}" \
  /content/kiwilm-model-e-750k/run/metrics.jsonl \
  "${result_dir}/metrics.jsonl"
"${colab_bin}" download -s "${session_name}" \
  /content/kiwilm-model-e-750k-summary.json \
  "${result_dir}/summary.json"
"${colab_bin}" download -s "${session_name}" \
  /content/kiwilm-model-e-750k-examples.md \
  "${result_dir}/examples.md"

echo "Model E 750k artifacts downloaded to ${result_dir}"
