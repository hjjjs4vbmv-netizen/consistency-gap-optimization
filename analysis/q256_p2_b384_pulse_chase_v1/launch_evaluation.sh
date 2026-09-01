#!/usr/bin/env bash
set -Eeuo pipefail
manifest="${1:?usage: launch_evaluation.sh MANIFEST CACHE_ROOT EVAL_ROOT IMPLEMENTATION_REPO}"
cache_root="${2:?missing cache root}"; eval_root="${3:?missing eval root}"; implementation_repo="${4:?missing implementation repo}"
[[ ! -e "${eval_root}" ]] || { echo "evaluation root already exists" >&2; exit 3; }
mkdir "${eval_root}"
tool_dir="${implementation_repo}/analysis/q256_p2_b384_pulse_chase_v1"
worker() { local gpu="$1" start="$2"; local index; for ((index=start; index<60; index+=2)); do bash "${tool_dir}/run_evaluation_job.sh" "${manifest}" "${index}" "${gpu}" "${cache_root}" "${eval_root}" "${implementation_repo}"; done; }
worker 0 0 >"${eval_root}/gpu0-evaluation-worker.log" 2>&1 & pid0=$!
worker 1 1 >"${eval_root}/gpu1-evaluation-worker.log" 2>&1 & pid1=$!
set +e
wait "${pid0}"; code0=$?
wait "${pid1}"; code1=$?
set -e
(( code0 == 0 && code1 == 0 )) || { echo "evaluation worker failed: gpu0=${code0} gpu1=${code1}" >&2; exit 4; }
runtime_sif="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime_sif"]["path"])' "${manifest}")"
apptainer exec --bind /data:/data --pwd "${implementation_repo}" "${runtime_sif}" python \
  analysis/q256_p2_b384_pulse_chase_v1/audit_evaluation_seals.py \
    --manifest "${manifest}" --receipts "${eval_root}/receipts" \
    --output "${eval_root}/evaluation_seal_audit.json"
echo "[P2 evaluation] ALL_60_SEALED_PASS; numeric unseal is now authorized"
