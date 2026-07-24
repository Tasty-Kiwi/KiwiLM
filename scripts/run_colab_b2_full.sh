#!/usr/bin/env bash
set -euo pipefail

session_name="kiwilm-b2-full"
colab_bin="${COLAB_BIN:-colab}"
data_dir="${KIWILM_DATA_DIR:-data/tinystories-500k}"
artifact_dir="$(mktemp -d "${TMPDIR:-/tmp}/kiwilm-b2-full-colab.XXXXXX")"
result_dir="${KIWILM_RESULT_DIR:-runs/model-b2-colab}"
chunk_dir="${artifact_dir}/data-chunks"
session_started=0

mkdir -p "${artifact_dir}" "${chunk_dir}" "${result_dir}"

cleanup() {
  if [[ "${session_started}" -eq 1 ]]; then
    "${colab_bin}" log -s "${session_name}" \
      -o "${result_dir}/session.jsonl" || true
    "${colab_bin}" stop -s "${session_name}" || true
  fi
}
trap cleanup EXIT INT TERM

if [[ ! -f "${data_dir}/metadata.json" ]]; then
  echo "Prepared 500k dataset not found at ${data_dir}" >&2
  exit 1
fi
uv run python -c \
  'import sys; from kiwilm.data import PreparedTokenData; PreparedTokenData(sys.argv[1], expected_fingerprint="80e38a5ec88a3d4a543aaadfd6cfcbb8e1fc41e1987c382e9d2f8f7db9d843ad")' \
  "${data_dir}"

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

"${colab_bin}" new -s "${session_name}" --gpu T4
session_started=1
"${colab_bin}" upload -s "${session_name}" \
  "${built_wheel}" \
  "/content/${wheel_name}"

for local_path in "${chunk_dir}"/*; do
  if [[ -f "${local_path}" ]]; then
    artifact_name="$(basename "${local_path}")"
    "${colab_bin}" upload -s "${session_name}" \
      "${local_path}" \
      "/content/${artifact_name}"
  fi
done

"${colab_bin}" exec -s "${session_name}" \
  --timeout 7200 \
  -f scripts/colab_b2_full.py

"${colab_bin}" download -s "${session_name}" \
  /content/kiwilm-b2-full/run/best.pt \
  "${result_dir}/best.pt"
"${colab_bin}" download -s "${session_name}" \
  /content/kiwilm-b2-full/run/latest.pt \
  "${result_dir}/latest.pt"
"${colab_bin}" download -s "${session_name}" \
  /content/kiwilm-b2-full/run/metrics.jsonl \
  "${result_dir}/metrics.jsonl"
"${colab_bin}" download -s "${session_name}" \
  /content/kiwilm-b2-full-summary.json \
  "${result_dir}/summary.json"

echo "Model B2 artifacts downloaded to ${result_dir}"
