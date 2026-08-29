#!/usr/bin/env bash
set -Eeuo pipefail

run_kind="${1:?usage: $0 parity|formal OUTPUT_ROOT INVENTORY PROTOCOL IMPLEMENTATION_COMMIT REPO RUNTIME_ROOTFS RUNTIME_SIF TRAIN_DATASET [PARITY_JSON]}"
output_root="${2:?missing output root}"
inventory="${3:?missing source inventory}"
protocol="${4:?missing protocol}"
implementation_commit="${5:?missing implementation commit}"
repo="${6:?missing repository}"
runtime_rootfs="${7:?missing runtime rootfs}"
runtime_sif="${8:?missing runtime SIF}"
dataset="${9:?missing training dataset}"
parity_json="${10:-}"

case "${run_kind}" in
  parity) branches=(A_to_A B_to_B) ;;
  formal)
    branches=(A_to_B B_to_A)
    [[ -f "${parity_json}" ]] || { echo "formal launch requires parity JSON" >&2; exit 2; }
    python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"]=="PASS" and p["pass_count"]==10 and p["verdict"]=="10/10 COMPUTATIONAL_STATE_MATCH"' "${parity_json}" || { echo "parity gate is not 10/10 PASS" >&2; exit 2; }
    ;;
  *) echo "invalid run kind: ${run_kind}" >&2; exit 2 ;;
esac

[[ -f "${inventory}" && -f "${protocol}" && -d "${repo}/.git" ]] || { echo "missing frozen input" >&2; exit 2; }
[[ -d "${runtime_rootfs}" && -f "${runtime_sif}" && -f "${dataset}" ]] || { echo "missing runtime or dataset" >&2; exit 2; }
[[ "$(git -C "${repo}" rev-parse HEAD)" == "${implementation_commit}" ]] || { echo "repository commit mismatch" >&2; exit 2; }
[[ -z "$(git -C "${repo}" status --porcelain)" ]] || { echo "repository is dirty" >&2; exit 2; }
mkdir "${output_root}"
mkdir "${output_root}/logs"
sif_sha=$(sha256sum "${runtime_sif}" | awk '{print $1}')
[[ "${sif_sha}" == 9d5f2c9e68f1f7dcaa20457bf6e0b6fa46f74a8605edaf5d49fdccf9f6bb62ea ]] || { echo "runtime SIF hash mismatch" >&2; exit 2; }
python -c 'import json,os,sys; p=sys.argv[1]; x={"schema":"ect.q256.runtime-integrity/v1","status":"PASS","runtime_sif":sys.argv[2],"runtime_sif_sha256":sys.argv[3]}; f=open(p,"x"); json.dump(x,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())' "${output_root}/runtime_integrity.json" "${runtime_sif}" "${sif_sha}"

for seed in 14 15 16 17 18; do
  mkdir "${output_root}/seed${seed}"
  for branch in "${branches[@]}"; do
    python "${repo}/analysis/q256_schedule_switch_v1/prepare_run_cell.py" \
      --inventory "${inventory}" --protocol "${protocol}" \
      --implementation-commit "${implementation_commit}" \
      --run-kind "${run_kind}" --branch "${branch}" --seed "${seed}" \
      --output-dir "${output_root}/seed${seed}/${branch}"
  done
done

pids=()
for seed in 14 15 16 17 18; do
  gpu=$((seed - 14))
  (
    for branch in "${branches[@]}"; do
      "${repo}/analysis/q256_schedule_switch_v1/run_training_cell.sh" \
        "${output_root}/seed${seed}/${branch}" "${gpu}" "${repo}" \
        "${runtime_rootfs}" "${runtime_sif}" "${dataset}"
    done
  ) >"${output_root}/logs/seed${seed}.log" 2>&1 &
  pids+=("$!")
  printf '%s\n' "$!" >"${output_root}/logs/seed${seed}.pid"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" != 0 ]]; then
  echo "TRAINING_MATRIX_FAIL_CLOSED kind=${run_kind} root=${output_root}" >&2
  exit 4
fi

count=$(find "${output_root}" -mindepth 3 -maxdepth 3 -name trajectory_completion_receipt.json -type f | wc -l | tr -d ' ')
[[ "${count}" == 10 ]] || { echo "completion receipt count ${count}/10" >&2; exit 5; }
echo "TRAINING_MATRIX_PASS kind=${run_kind} cells=10 root=${output_root}"
