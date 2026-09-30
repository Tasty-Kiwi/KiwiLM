#!/usr/bin/env bash
# Separate experimental TPU allocation: never reuses the ongoing GPU run/Drive key.
set -euo pipefail

data_dir="${KIWILM2_DATA_DIR:-data/smollm-smoke}"
result_dir="${KIWILM_RESULT_DIR:-runs/colab/tpu-v5e1-muon-smoke-50m}"
session="${COLAB_SESSION_NAME:-kiwilm2-tpu-v5e1-muon-smoke-50m}"
resume_from="${KIWILM2_RESUME_FROM:-}"
colab_bin="${COLAB_BIN:-colab}"
if [[ -e "${result_dir}/summary.json" || -e "${result_dir}/latest.pt" ]]; then
  echo "Choose a new KIWILM_RESULT_DIR; an earlier probe already exists." >&2
  exit 1
fi
if [[ -n "${resume_from}" && ! -f "${resume_from}" ]]; then
  echo "Resume checkpoint not found: ${resume_from}" >&2
  exit 1
fi
uv run --locked python -c \
  'import sys; from kiwilm.colab_kiwilm2 import build_colab_job; build_colab_job(sys.argv[1], phase="smoke", architecture="kiwilm2")' \
  "${data_dir}"
status="$("${colab_bin}" status -s "${session}" 2>&1 || true)"
if [[ "${status}" != *"not found"* && "${status}" != *"Not found"* ]]; then
  echo "Refusing to reuse session '${session}': ${status}" >&2
  exit 1
fi
staging="$(mktemp -d "${TMPDIR:-/tmp}/kiwilm-tpu-probe.XXXXXX")"
started=0
artifacts_downloaded=0
mkdir -p "${result_dir}"
download_artifacts() {
  "${colab_bin}" download -s "${session}" /content/kiwilm-tpu-artifacts/artifact-manifest.json \
    "${result_dir}/artifact-manifest.json" || return 1
  uv run --locked python -c \
    'import json,sys; print(*[p["name"] for p in json.load(open(sys.argv[1]))["parts"]], sep="\n")' \
    "${result_dir}/artifact-manifest.json" > "${staging}/parts.txt" || return 1
  while IFS= read -r part; do
    "${colab_bin}" download -s "${session}" "/content/kiwilm-tpu-artifacts/${part}" \
      "${result_dir}/${part}" || return 1
  done < "${staging}/parts.txt"
  uv run --locked python scripts/reassemble_colab_artifacts.py \
    "${result_dir}/artifact-manifest.json" "${result_dir}" || return 1
  artifacts_downloaded=1
}
cleanup() {
  if [[ "${started}" == "1" ]]; then
    if [[ "${artifacts_downloaded}" != "1" ]]; then
      echo "Attempting recovery of the latest periodic TPU checkpoint..." >&2
      download_artifacts || true
    fi
    "${colab_bin}" download -s "${session}" /content/kiwilm-tpu-smoke/worker.log \
      "${result_dir}/worker.log" || true
    "${colab_bin}" log -s "${session}" -o "${result_dir}/session.jsonl" || true
    "${colab_bin}" stop -s "${session}" || true
  fi
  rm -rf "${staging}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
uv build --wheel --out-dir "${staging}"
uv run --locked python -c \
  'import json,sys; from pathlib import Path; from kiwilm.colab_artifacts import create_colab_artifacts; p=Path(sys.argv[1]); m=json.loads((p/"metadata.json").read_text()); names=["metadata.json",m["tokenizer"]["file"],*[s["file"] for s in m["splits"].values()]]; create_colab_artifacts({n:p/n for n in names},Path(sys.argv[2]),chunk_size=4*1024*1024)' \
  "${data_dir}" "${staging}/data"
wheel="$(find "${staging}" -maxdepth 1 -name 'kiwilm-*.whl' -print -quit)"
[[ -n "${wheel}" ]]
if [[ -n "${resume_from}" ]]; then
  uv run --locked python -c \
    'import sys; from pathlib import Path; from kiwilm.colab_artifacts import create_colab_artifacts; create_colab_artifacts({"latest.pt":Path(sys.argv[1])},Path(sys.argv[2]),chunk_size=4*1024*1024)' \
    "${resume_from}" "${staging}/resume"
fi
echo "Allocating v5e1 TPU for the full 50M-token smoke (may consume Colab compute units)."
"${colab_bin}" new -s "${session}" --tpu v5e1
started=1
"${colab_bin}" upload -s "${session}" "${wheel}" "/content/$(basename "${wheel}")"
"${colab_bin}" exec -s "${session}" --timeout 300 -f scripts/colab_kiwilm2_tpu_smoke.py
"${colab_bin}" download -s "${session}" /content/kiwilm-tpu-preflight.json \
  "${result_dir}/preflight.json"
printf '%s\n' 'from pathlib import Path; Path("/content/kiwilm-data-artifacts").mkdir(exist_ok=True)' \
  | "${colab_bin}" exec -s "${session}"
for file in "${staging}/data/"*; do
  "${colab_bin}" upload -s "${session}" "${file}" "/content/kiwilm-data-artifacts/$(basename "${file}")"
done
if [[ -n "${resume_from}" ]]; then
  printf '%s\n' 'from pathlib import Path; Path("/content/kiwilm-tpu-resume").mkdir(exist_ok=True)' \
    | "${colab_bin}" exec -s "${session}"
  for file in "${staging}/resume/"*; do
    "${colab_bin}" upload -s "${session}" "${file}" "/content/kiwilm-tpu-resume/$(basename "${file}")"
  done
fi
"${colab_bin}" exec -s "${session}" --timeout 7500 -f scripts/colab_kiwilm2_tpu_smoke.py
"${colab_bin}" download -s "${session}" /content/kiwilm-tpu-smoke/summary.json \
  "${result_dir}/summary.json"
download_artifacts
