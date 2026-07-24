#!/usr/bin/env bash
set -euo pipefail

session_name="kiwilm-b2-smoke"
colab_bin="${COLAB_BIN:-colab}"
artifact_dir="$(mktemp -d "${TMPDIR:-/tmp}/kiwilm-b2-colab.XXXXXX")"
result_dir="runs/colab-b2-smoke"
session_started=0

mkdir -p "${artifact_dir}" "${result_dir}"

cleanup() {
  if [[ "${session_started}" -eq 1 ]]; then
    "${colab_bin}" log -s "${session_name}" \
      -o "${result_dir}/session.jsonl" || true
    "${colab_bin}" stop -s "${session_name}" || true
  fi
}
trap cleanup EXIT INT TERM

uv build --wheel --out-dir "${artifact_dir}"
built_wheel="$(find "${artifact_dir}" -maxdepth 1 -name 'kiwilm-*.whl' -print -quit)"
if [[ -z "${built_wheel}" ]]; then
  echo "uv build did not produce a KiwiLM wheel" >&2
  exit 1
fi
wheel_name="$(basename "${built_wheel}")"

"${colab_bin}" new -s "${session_name}" --gpu T4
session_started=1
"${colab_bin}" upload -s "${session_name}" \
  "${built_wheel}" \
  "/content/${wheel_name}"
"${colab_bin}" exec -s "${session_name}" \
  --timeout 1800 \
  -f scripts/colab_b2_smoke.py
"${colab_bin}" download -s "${session_name}" \
  /content/kiwilm-b2-smoke-summary.json \
  "${result_dir}/summary.json"

echo "Colab smoke evidence: ${result_dir}/summary.json"
