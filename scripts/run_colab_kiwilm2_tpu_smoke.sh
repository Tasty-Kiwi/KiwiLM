#!/usr/bin/env bash
# User-run only. Setup NEVER trains; train is an explicit second command.
set -euo pipefail

action="${1:-setup}"
if [[ $# -gt 1 || ( "${action}" != "setup" && "${action}" != "train" ) ]]; then
  echo "Usage: $0 [setup|train] (default: setup; no training)" >&2
  exit 1
fi
data_dir="${KIWILM2_DATA_DIR:-data/smollm-smoke}"
tpu="${COLAB_TPU:-v6e1}"
case "${tpu}" in
  v5e1|v6e1) ;;
  *) echo "COLAB_TPU must be v5e1 or v6e1." >&2; exit 1 ;;
esac
result_dir="${KIWILM_RESULT_DIR:-runs/colab/tpu-${tpu}-muon-smoke-50m-tied-cached}"
session="${COLAB_SESSION_NAME:-kiwilm2-tpu-${tpu}-muon-smoke-50m-tied-cached}"
resume_from="${KIWILM2_RESUME_FROM:-}"
continuation_phase="${KIWILM2_TPU_CONTINUATION_PHASE:-}"
phase="${KIWILM2_TPU_PHASE:-smoke}"
remote_run=/content/kiwilm-tpu-smoke
prepare_timeout=900
train_timeout=7500
if [[ "${phase}" == "final-1b" ]]; then
  remote_run=/content/kiwilm-tpu-final-1b
  prepare_timeout=21900
  train_timeout=79500
elif [[ "${phase}" != "smoke" ]]; then
  echo "Invalid TPU training phase." >&2; exit 1
fi
use_drive="${KIWILM2_USE_DRIVE:-1}"
workers="${KIWILM2_UPLOAD_WORKERS:-3}"
colab_bin="${COLAB_BIN:-colab}"
if [[ "${use_drive}" != "0" && "${use_drive}" != "1" ]]; then
  echo "KIWILM2_USE_DRIVE must be 0 or 1." >&2; exit 1
fi
if [[ "${phase}" == "final-1b" && ( "${use_drive}" != "1" || -n "${resume_from}" || -n "${continuation_phase}" ) ]]; then
  echo "The 1B run requires Drive and its dedicated job-based resume, not smoke/local checkpoints." >&2
  exit 1
fi
mount_mode="${KIWILM2_TPU_DRIVE_MOUNT:-cli}"
if [[ "${mount_mode}" != "cli" && "${mount_mode}" != "manual" ]]; then
  echo "KIWILM2_TPU_DRIVE_MOUNT must be cli or manual." >&2; exit 1
fi
if [[ ! "${workers}" =~ ^[1-4]$ ]]; then
  echo "KIWILM2_UPLOAD_WORKERS must be 1-4." >&2; exit 1
fi
if [[ -e "${result_dir}/summary.json" || -e "${result_dir}/latest.pt" ]]; then
  echo "Choose a new KIWILM_RESULT_DIR; an earlier run already exists." >&2; exit 1
fi
if [[ "${action}" == "setup" && "${use_drive}" == "1" && ! -t 0 ]]; then
  echo "Drive mounting is interactive. Run setup in your terminal, or set KIWILM2_USE_DRIVE=0." >&2
  exit 1
fi
if [[ "${action}" == "train" && ! -f "${result_dir}/setup-owner.json" ]]; then
  echo "No prepared session here; run setup first with these session/result settings." >&2; exit 1
fi
if [[ "${action}" == "setup" && -e "${result_dir}/setup-owner.json" ]]; then
  echo "This session was already prepared. Use train, or choose new session/result names." >&2; exit 1
fi
staging="$(mktemp -d "${TMPDIR:-/tmp}/kiwilm-tpu-setup.XXXXXX")"
owned=0
prepared=0
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
  if [[ "${owned}" == "1" && "${prepared}" != "1" ]]; then
    if [[ "${action}" == "train" && "${artifacts_downloaded}" != "1" ]]; then
      echo "Attempting recovery of the latest periodic TPU checkpoint..." >&2
      if ! download_artifacts; then
        echo "Chunk recovery unavailable; trying VM-local files. Drive committed generations remain the recovery source." >&2
        for name in latest.pt metrics.jsonl; do
          "${colab_bin}" download -s "${session}" "${remote_run}/${name}" "${result_dir}/${name}" || true
        done
      fi
    fi
    if [[ "${action}" == "train" ]]; then
      "${colab_bin}" download -s "${session}" "${remote_run}/worker.log" \
        "${result_dir}/worker.log" || true
    fi
    "${colab_bin}" log -s "${session}" -o "${result_dir}/session.jsonl" || true
    "${colab_bin}" stop -s "${session}" || true
  fi
  rm -rf "${staging}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "${action}" == "setup" ]]; then
  status="$("${colab_bin}" status -s "${session}" 2>&1 || true)"
  if [[ "${status}" != *"not found"* && "${status}" != *"Not found"* ]]; then
    echo "Refusing to reuse session '${session}': ${status}" >&2; exit 1
  fi
  if [[ "${phase}" == "final-1b" ]]; then
    final_args=(--source-dir "${data_dir}" --output "${result_dir}/tpu-job.json" --tpu "${tpu}"
      --cache-dir "${KIWILM2_TPU_DRIVE_CACHE:-}" --backup-dir "${KIWILM2_TPU_DRIVE_BACKUP:-}")
    if [[ -n "${KIWILM2_TPU_JOB_FROM:-}" ]]; then
      final_args+=(--resume-job "${KIWILM2_TPU_JOB_FROM}")
    fi
    uv run --locked python -m kiwilm.tpu_final "${final_args[@]}"
  else
    uv run --locked python -c \
    'import os,sys; from pathlib import Path; from kiwilm.tpu_setup import write_setup_job; write_setup_job(Path(sys.argv[1]),Path(sys.argv[2]),use_drive=sys.argv[3]=="1",cache_dir=os.environ.get("KIWILM2_TPU_DRIVE_CACHE",""),resume=Path(sys.argv[4]) if sys.argv[4] else None,tpu=sys.argv[5],backup_dir=os.environ.get("KIWILM2_TPU_DRIVE_BACKUP",""),drive_resume_dir=os.environ.get("KIWILM2_TPU_DRIVE_RESUME",""),continuation_phase=sys.argv[6] or None)' \
    "${data_dir}" "${result_dir}/tpu-job.json" "${use_drive}" "${resume_from}" "${tpu}" "${continuation_phase}"
  fi
  uv build --wheel --out-dir "${staging}"
  wheel="$(find "${staging}" -maxdepth 1 -name 'kiwilm-*.whl' -print -quit)"
  [[ -n "${wheel}" ]]
  echo "Allocating ${tpu} for SETUP ONLY (billable; you must later train or stop it)."
  if ! "${colab_bin}" new -s "${session}" --tpu "${tpu}"; then
    echo "Allocation failed or timed out; the server may still have allocated a TPU. Check Colab's website/colab sessions and stop any orphan before retrying." >&2
    exit 1
  fi
  owned=1
  "${colab_bin}" upload -s "${session}" "${wheel}" "/content/$(basename "${wheel}")"
  "${colab_bin}" upload -s "${session}" "${result_dir}/tpu-job.json" /content/kiwilm-tpu-job.json
  if [[ "${phase}" == "final-1b" && -z "${KIWILM2_TPU_JOB_FROM:-}" ]]; then
    printf '%s\n' 'from pathlib import Path; Path("/content/kiwilm-tpu-tokenizer").mkdir(exist_ok=True)' \
      | "${colab_bin}" exec -s "${session}"
    for tokenizer_file in "${result_dir}/tokenizer/"*.json; do
      "${colab_bin}" upload -s "${session}" "${tokenizer_file}" "/content/kiwilm-tpu-tokenizer/$(basename "${tokenizer_file}")"
    done
  fi
  "${colab_bin}" exec -s "${session}" --env KIWILM2_TPU_ACTION=preflight \
    --timeout 300 -f scripts/colab_kiwilm2_tpu_smoke.py
  "${colab_bin}" download -s "${session}" /content/kiwilm-tpu-preflight.json \
    "${result_dir}/preflight.json"
  if [[ "${use_drive}" == "1" ]]; then
    if [[ "${mount_mode}" == "manual" ]]; then
      echo "Mount Drive at /content/drive in this session's Colab notebook, then return here."
      read -r -p "Press Enter after Drive is mounted... "
    else
      "${colab_bin}" drivemount -s "${session}"
    fi
  fi
  "${colab_bin}" exec -s "${session}" --env KIWILM2_TPU_ACTION=prepare \
    --timeout "${prepare_timeout}" -f scripts/colab_kiwilm2_tpu_smoke.py
  "${colab_bin}" download -s "${session}" /content/kiwilm-tpu-setup.json "${result_dir}/setup.json"
  state="$(uv run --locked python -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' \
    "${result_dir}/setup.json")"
  if [[ "${state}" == "needs-data-upload" ]]; then
    uv run --locked python -c \
      'import json,sys; from pathlib import Path; from kiwilm.colab_artifacts import create_colab_artifacts; p=Path(sys.argv[1]); m=json.loads((p/"metadata.json").read_text()); names=["metadata.json",m["tokenizer"]["file"],*[s["file"] for s in m["splits"].values()]]; create_colab_artifacts({n:p/n for n in names},Path(sys.argv[2]),archive_name="data.tar.gz",compression="gzip",chunk_size=4*1024*1024)' \
      "${data_dir}" "${staging}/data"
    printf '%s\n' 'from pathlib import Path; Path("/content/kiwilm-data-artifacts").mkdir(exist_ok=True)' \
      | "${colab_bin}" exec -s "${session}"
    uv run --locked python -m kiwilm.colab_transfer "${staging}/data" /content/kiwilm-data-artifacts \
      --session "${session}" --colab-bin "${colab_bin}" --workers "${workers}"
  elif [[ "${state}" != "ready" && "${state}" != "needs-resume-upload" ]]; then
    echo "Unexpected TPU setup state: ${state}" >&2; exit 1
  fi
  if [[ -n "${resume_from}" ]]; then
    uv run --locked python -c \
      'import sys; from pathlib import Path; from kiwilm.colab_artifacts import create_colab_artifacts; create_colab_artifacts({"latest.pt":Path(sys.argv[1])},Path(sys.argv[2]),chunk_size=4*1024*1024)' \
      "${resume_from}" "${staging}/resume"
    printf '%s\n' 'from pathlib import Path; Path("/content/kiwilm-tpu-resume").mkdir(exist_ok=True)' \
      | "${colab_bin}" exec -s "${session}"
    uv run --locked python -m kiwilm.colab_transfer "${staging}/resume" /content/kiwilm-tpu-resume \
      --session "${session}" --colab-bin "${colab_bin}" --workers "${workers}"
  fi
  "${colab_bin}" exec -s "${session}" --env KIWILM2_TPU_ACTION=prepare \
    --timeout "${prepare_timeout}" -f scripts/colab_kiwilm2_tpu_smoke.py
  "${colab_bin}" download -s "${session}" /content/kiwilm-tpu-setup.json "${result_dir}/setup.json"
  if [[ "${phase}" == "final-1b" ]]; then
    "${colab_bin}" download -s "${session}" /content/kiwilm-tpu-job.json "${result_dir}/tpu-job.json"
  fi
  uv run --locked python -c \
    'import sys; from pathlib import Path; from kiwilm.tpu_setup import setup_owner; p=Path(sys.argv[1]); setup_owner(p,p/"setup.json",session=sys.argv[2],tpu=sys.argv[3],claim=True)' \
    "${result_dir}" "${session}" "${tpu}"
  prepared=1
  echo "Setup complete. NO TRAINING STARTED. This TPU remains billable."
  echo "Start with the same environment settings: bash ${KIWILM2_TPU_ENTRYPOINT:-$0} ${KIWILM2_TPU_ENTRY_MODE:-}train"
  echo "Or release it: ${colab_bin} stop -s ${session}"
  exit 0
fi

# Explicit training owns only a session whose local AND remote setup locks match.
"${colab_bin}" download -s "${session}" /content/kiwilm-tpu-setup.json "${staging}/remote-setup.json"
uv run --locked python -c \
  'import sys; from pathlib import Path; from kiwilm.tpu_setup import setup_owner; setup_owner(Path(sys.argv[1]),Path(sys.argv[2]),session=sys.argv[3],tpu=sys.argv[4])' \
  "${result_dir}" "${staging}/remote-setup.json" "${session}" "${tpu}"
owned=1
"${colab_bin}" exec -s "${session}" --env KIWILM2_TPU_ACTION=train \
  --timeout "${train_timeout}" -f scripts/colab_kiwilm2_tpu_smoke.py
"${colab_bin}" download -s "${session}" "${remote_run}/summary.json" "${result_dir}/summary.json"
download_artifacts
