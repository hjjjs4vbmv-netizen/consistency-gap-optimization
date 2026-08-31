#!/usr/bin/env bash
set -Eeuo pipefail

repo="${1:?usage: launch_formal.sh REPO PROTOCOL DATASET TRANSFER RUNTIME_SIF FORMAL_ROOT}"
protocol="${2:?missing protocol}"; dataset="${3:?missing dataset}"; transfer="${4:?missing transfer}"
runtime_sif="${5:?missing runtime}"; formal_root="${6:?missing formal output root}"
[[ ! -e "${formal_root}" ]] || { echo "formal output root already exists" >&2; exit 3; }
mkdir "${formal_root}"
tool_dir="${repo}/analysis/q256_p2_b384_pulse_chase_v1"
preflight_cotenancy=()
[[ "${P2_ALLOW_COTENANCY:-0}" == 1 ]] && preflight_cotenancy+=(--allow-cotenancy)
apptainer exec --nv --bind /data:/data --pwd "${repo}" "${runtime_sif}" python \
  analysis/q256_p2_b384_pulse_chase_v1/preflight.py \
    --repo "${repo}" --protocol "${protocol}" --dataset "${dataset}" \
    --transfer "${transfer}" --runtime-sif "${runtime_sif}" \
    --output "${formal_root}/preflight.json" "${preflight_cotenancy[@]}"

worker() {
  local gpu="$1"; shift
  local seed
  for seed in "$@"; do
    bash "${tool_dir}/run_seed.sh" "${seed}" "${gpu}" "${repo}" "${protocol}" \
      "${dataset}" "${transfer}" "${runtime_sif}" "${formal_root}" 0
  done
}
worker 0 19 20 21 22 23 >"${formal_root}/gpu0-worker.log" 2>&1 & pid0=$!
worker 1 24 25 26 27 28 >"${formal_root}/gpu1-worker.log" 2>&1 & pid1=$!
set +e
wait "${pid0}"; code0=$?
wait "${pid1}"; code1=$?
set -e
(( code0 == 0 && code1 == 0 )) || { echo "training worker failed: gpu0=${code0} gpu1=${code1}" >&2; exit 4; }
apptainer exec --bind /data:/data --pwd "${repo}" "${runtime_sif}" python \
  analysis/q256_p2_b384_pulse_chase_v1/verify_training_matrix.py \
    --formal-root "${formal_root}" --protocol "${protocol}" \
    --output "${formal_root}/training_integrity_report.json" \
    --compute-cost "${formal_root}/compute_cost.csv"
echo "[P2 formal training] PASS root=${formal_root}"
