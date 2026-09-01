#!/usr/bin/env bash
set -Eeuo pipefail

seed="${1:?usage: run_branch.sh SEED BRANCH GPU REPO PROTOCOL DATASET RUNTIME_SIF SOURCE_INVENTORY BRANCH_DIR [TAPE_AUDIT]}"
branch="${2:?missing branch}"
gpu="${3:?missing GPU}"
repo="${4:?missing repository}"
protocol="${5:?missing protocol}"
dataset="${6:?missing dataset}"
runtime_sif="${7:?missing runtime SIF}"
source_inventory="${8:?missing source inventory}"
branch_dir="${9:?missing branch output directory}"
tape_audit="${10:-0}"
run_kind="${11:-formal}"
case "${branch}" in Early-switch) pulse=A;; Late-switch) pulse=B;; *) echo "invalid branch" >&2; exit 2;; esac
expected_gpu=1; (( seed <= 23 )) && expected_gpu=0
[[ "${run_kind}" == smoke || "${gpu}" == "${expected_gpu}" ]] || { echo "seed/GPU assignment mismatch" >&2; exit 2; }
[[ ! -e "${branch_dir}" ]] || { echo "refuse existing branch cell" >&2; exit 3; }
repo_git() { (cd "${repo}" && git "$@"); }
implementation_commit="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["implementation_commit"])' "${protocol}")"
[[ "$(repo_git rev-parse HEAD^)" == "${implementation_commit}" && -z "$(repo_git status --porcelain)" ]] || { echo "unclean or unbound implementation" >&2; exit 3; }
gpu_uuid="$(nvidia-smi --id="${gpu}" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
if [[ "${P2_ALLOW_COTENANCY:-0}" != 1 ]] && nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -q "^${gpu_uuid},"; then echo "assigned GPU is occupied" >&2; exit 3; fi

audit_arg=""
train_audit_arg=""
if [[ "${tape_audit}" == 1 ]]; then audit_arg=--matched-randomness-audit; train_audit_arg=--p2-matched-randomness-audit; fi
apptainer exec --bind /data:/data --pwd "${repo}" "${runtime_sif}" python \
  analysis/q256_p2_b384_pulse_chase_v1/prepare_branch_manifest.py \
    --source-inventory "${source_inventory}" --protocol "${protocol}" \
    --implementation-commit "${implementation_commit}" --seed "${seed}" --run-kind "${run_kind}" \
    --branch "${branch}" --gpu-index "${gpu}" --gpu-uuid "${gpu_uuid}" \
    --output-dir "${branch_dir}" ${audit_arg}
manifest="${branch_dir}/formal_run_manifest.json"
source_state="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_state"]["path"])' "${manifest}")"
failure_receipt="${branch_dir}/FAILED_RECEIPT.json"
started="$(date +%s)"
phase="pulse"
on_failure() {
  code=$?
  python3 -c 'import json,os,sys,time; d={"schema":"ect.q256.p2-infrastructure-failure/v1","status":"FAILED","cell":"branch","phase":sys.argv[2],"seed":int(sys.argv[3]),"branch":sys.argv[4],"gpu_index":int(sys.argv[5]),"gpu_uuid":sys.argv[6],"exit_code":int(sys.argv[7]),"ended_unix":int(time.time())}; f=open(sys.argv[1],"x"); json.dump(d,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())' "${failure_receipt}" "${phase}" "${seed}" "${branch}" "${gpu}" "${gpu_uuid}" "${code}" || true
  exit "${code}"
}
trap on_failure ERR

run_phase() {
  local phase_name="$1" end_kimg="$2" arm="$3" resume_state="$4"
  local scale=1.0
  [[ "${arm}" == B ]] && scale=1.1
  local master_port=$((44000 + gpu * 100 + seed + (end_kimg == 640 ? 40 : 0) ))
  env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}" \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_CACHE_DISABLE=1 \
      PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MASTER_ADDR=127.0.0.1 \
      MASTER_PORT="${master_port}" RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
      APPTAINERENV_MASTER_ADDR=127.0.0.1 APPTAINERENV_MASTER_PORT="${master_port}" \
      APPTAINERENV_RANK=0 APPTAINERENV_LOCAL_RANK=0 APPTAINERENV_WORLD_SIZE=1 \
    timeout --signal=TERM --kill-after=30s 12h \
    apptainer exec --nv --bind /data:/data --pwd "${repo}" "${runtime_sif}" \
    python ct_train.py \
      --data="${dataset}" --outdir="${branch_dir}" --nosubdir \
      --cond=False --arch=ddpmpp --precond=ect --batch=128 --batch-gpu=16 \
      --optim=RAdam --lr=0.0001 --dropout=0.2 --augment=0 --xflip=False \
      --mean=-1.1 --std=2.0 --mapping=sigmoid --global-gap-scale=1.0 \
      --factorial-protocol=q256_target_weight_v1 \
      --target-gap-scale="${scale}" --denominator-gap-scale="${scale}" \
      -q 256 -k 8 -b 1 -c 0 --double=10000 --ema_beta=0.9993 \
      --seed="${seed}" --fp16=True --tf32=False --ls=1.0 --enable_amp=True \
      --bench=False --cache=True --workers=1 --metrics=none \
      --duration="0.${end_kimg}" --tick=10 --snap=0 --dump=0 --ckpt=10 \
      --sample_every=26 --eval_every=50 --mid_t=0.821 \
      --adaptive-update-kimg=0.5 --immutable-checkpoint-kimg="${end_kimg}" \
      --p2-pulse-chase-manifest="${manifest}" ${train_audit_arg} \
      --resume="${resume_state}" \
    2>&1 | tee "${branch_dir}/${phase_name}.process.log"
}

run_phase pulse 512 "${pulse}" "${source_state}"
phase="chase"
run_phase chase 640 A "${branch_dir}/training-state-kimg000512.pt"
phase="verify"
apptainer exec --nv --bind /data:/data --pwd "${repo}" "${runtime_sif}" python \
  analysis/q256_p2_b384_pulse_chase_v1/verify_branch.py \
    --run-dir "${branch_dir}" --manifest "${manifest}" \
    --output "${branch_dir}/trajectory_completion_receipt.json"
elapsed=$(( $(date +%s) - started ))
python3 -c 'import json,os,sys; p=json.load(open(sys.argv[1])); p.update(elapsed_seconds=int(sys.argv[2])); f=open(sys.argv[3],"x"); json.dump(p,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())' \
  "${branch_dir}/trajectory_completion_receipt.json" "${elapsed}" "${branch_dir}/compute_cost_receipt.json"
trap - ERR
echo "[P2 branch] PASS seed=${seed} branch=${branch} gpu=${gpu} uuid=${gpu_uuid} elapsed=${elapsed}"
