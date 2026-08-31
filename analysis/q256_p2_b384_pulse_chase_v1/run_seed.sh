#!/usr/bin/env bash
set -Eeuo pipefail

seed="${1:?usage: run_seed.sh SEED GPU REPO PROTOCOL DATASET TRANSFER RUNTIME_SIF FORMAL_ROOT [TAPE_AUDIT]}"
gpu="${2:?missing GPU}"; repo="${3:?missing repo}"; protocol="${4:?missing protocol}"
dataset="${5:?missing dataset}"; transfer="${6:?missing transfer}"; runtime_sif="${7:?missing runtime}"
formal_root="${8:?missing formal root}"; tape_audit="${9:-0}"
run_kind="${10:-formal}"
seed_dir="${formal_root}/seeds/seed${seed}"
[[ ! -e "${seed_dir}" ]] || { echo "refuse existing seed cell" >&2; exit 3; }
mkdir -p "${formal_root}/seeds"
mkdir "${seed_dir}"

if (( seed % 2 )); then order=(Early-switch Late-switch); else order=(Late-switch Early-switch); fi
python3 -c 'import json,os,sys; d={"schema":"ect.q256.p2-branch-order/v1","seed":int(sys.argv[2]),"order":sys.argv[3:]}; f=open(sys.argv[1],"x"); json.dump(d,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())' \
  "${seed_dir}/branch_order_receipt.json" "${seed}" "${order[@]}"

tool_dir="${repo}/analysis/q256_p2_b384_pulse_chase_v1"
bash "${tool_dir}/run_source.sh" "${seed}" "${gpu}" "${repo}" "${protocol}" \
  "${dataset}" "${transfer}" "${runtime_sif}" "${seed_dir}/source" "${run_kind}"
for branch in "${order[@]}"; do
  bash "${tool_dir}/run_branch.sh" "${seed}" "${branch}" "${gpu}" "${repo}" \
    "${protocol}" "${dataset}" "${runtime_sif}" \
    "${seed_dir}/source/source_inventory.json" "${seed_dir}/${branch}" "${tape_audit}" "${run_kind}"
done
pair_args=()
[[ "${tape_audit}" == 1 ]] && pair_args+=(--require-full-tape)
apptainer exec --bind /data:/data --pwd "${repo}" "${runtime_sif}" python \
  analysis/q256_p2_b384_pulse_chase_v1/verify_pair.py \
    --seed-dir "${seed_dir}" --output "${seed_dir}/pair_integrity_receipt.json" \
    "${pair_args[@]}"
echo "[P2 seed] PASS seed=${seed} gpu=${gpu}"
