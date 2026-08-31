#!/usr/bin/env bash
set -Eeuo pipefail

run_dir="${1:?usage: $0 RUN_DIR GPU_ID REPO RUNTIME_ROOTFS RUNTIME_SIF TRAIN_DATASET}"
gpu_id="${2:?usage: $0 RUN_DIR GPU_ID REPO RUNTIME_ROOTFS RUNTIME_SIF TRAIN_DATASET}"
repo="${3:?usage: $0 RUN_DIR GPU_ID REPO RUNTIME_ROOTFS RUNTIME_SIF TRAIN_DATASET}"
runtime_rootfs="${4:?usage: $0 RUN_DIR GPU_ID REPO RUNTIME_ROOTFS RUNTIME_SIF TRAIN_DATASET}"
runtime_sif="${5:?usage: $0 RUN_DIR GPU_ID REPO RUNTIME_ROOTFS RUNTIME_SIF TRAIN_DATASET}"
dataset="${6:?usage: $0 RUN_DIR GPU_ID REPO RUNTIME_ROOTFS RUNTIME_SIF TRAIN_DATASET}"
manifest="${run_dir}/formal_run_manifest.json"
source "$(dirname "${BASH_SOURCE[0]}")/runtime_env.sh" "${runtime_rootfs}"

[[ "${gpu_id}" =~ ^[0-9]+$ ]] || { echo "invalid GPU index" >&2; exit 2; }
[[ -d "${run_dir}" && -f "${manifest}" && -d "${repo}/.git" ]] || { echo "missing run cell or repository" >&2; exit 2; }
[[ -d "${runtime_rootfs}" && -f "${runtime_sif}" && -f "${dataset}" ]] || { echo "missing runtime or training dataset" >&2; exit 2; }
[[ ! -e "${run_dir}/trajectory_completion_receipt.json" ]] || { echo "refuse completed output cell" >&2; exit 3; }

mapfile -t fields < <(python -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["seed"]); print(p["branch"]); print(p["continuation_arm"]); print(p["final_kimg"]); print(p["source_state"]["path"]); print(p["implementation_commit"])' "${manifest}")
[[ "${#fields[@]}" == 6 ]] || { echo "invalid run manifest" >&2; exit 3; }
seed="${fields[0]}"
branch="${fields[1]}"
continuation="${fields[2]}"
final_kimg="${fields[3]}"
source_state="${fields[4]}"
implementation_commit="${fields[5]}"

[[ "$(git -C "${repo}" rev-parse HEAD)" == "${implementation_commit}" ]] || { echo "implementation commit mismatch" >&2; exit 3; }
[[ -z "$(git -C "${repo}" status --porcelain)" ]] || { echo "implementation worktree is dirty" >&2; exit 3; }
[[ -f "${source_state}" ]] || { echo "missing frozen source state" >&2; exit 3; }

case "${continuation}" in
  A) target_scale=1.0; denominator_scale=1.0 ;;
  B) target_scale=1.1; denominator_scale=1.1 ;;
  *) echo "invalid continuation arm" >&2; exit 3 ;;
esac
if [[ "${final_kimg}" == 640 ]]; then
  milestones=640
elif [[ "${final_kimg}" == 1024 ]]; then
  milestones=640,768,896,1024
else
  echo "invalid final budget" >&2
  exit 3
fi
duration=$(python -c 'import sys; print(f"{int(sys.argv[1])/1000:.3f}")' "${final_kimg}")
master_port=$((47000 + gpu_id * 100 + seed))
gpu_uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v index="${gpu_id}" '$1 == index {print $2}')
[[ -n "${gpu_uuid}" ]] || { echo "cannot resolve GPU UUID" >&2; exit 3; }

python -c 'import json,os,subprocess,sys; p=sys.argv[1]; payload={"schema":"ect.q256.schedule-switch-compute-start/v1","status":"START","seed":int(sys.argv[2]),"branch":sys.argv[3],"gpu_index":int(sys.argv[4]),"gpu_uuid":sys.argv[5],"runtime_sif":sys.argv[6],"runtime_sif_sha256":sys.argv[7],"implementation_commit":sys.argv[8]}; f=open(p,"x"); json.dump(payload,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())' \
  "${run_dir}/compute_start_receipt.json" "${seed}" "${branch}" "${gpu_id}" "${gpu_uuid}" "${runtime_sif}" \
  "9d5f2c9e68f1f7dcaa20457bf6e0b6fa46f74a8605edaf5d49fdccf9f6bb62ea" "${implementation_commit}"

echo "[q256-switch] START seed=${seed} branch=${branch} gpu=${gpu_id} uuid=${gpu_uuid} final_kimg=${final_kimg}"
timeout --signal=TERM --kill-after=30s 24h \
  env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_id}" \
      CUDA_CACHE_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      MASTER_ADDR=127.0.0.1 MASTER_PORT="${master_port}" \
      RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
      LD_LIBRARY_PATH="${Q256_RUNTIME_LD_LIBRARY_PATH}" PATH="${Q256_RUNTIME_PATH}" \
    "${Q256_RUNTIME_PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=1 \
      --master_port="${master_port}" "${repo}/ct_train.py" \
      --data="${dataset}" --outdir="${run_dir}" --nosubdir \
      --cond=False --arch=ddpmpp --precond=ect --batch=128 --batch-gpu=16 \
      --optim=RAdam --lr=0.0001 --dropout=0.2 --augment=0 --xflip=False \
      --mean=-1.1 --std=2.0 --mapping=sigmoid --global-gap-scale=1.0 \
      --factorial-protocol=q256_target_weight_v1 \
      --target-gap-scale="${target_scale}" \
      --denominator-gap-scale="${denominator_scale}" \
      -q 256 -k 8 -b 1 -c 0 --double=10000 --ema_beta=0.9993 \
      --seed="${seed}" --fp16=True --tf32=False --ls=1.0 --enable_amp=True \
      --bench=False --cache=True --workers=1 --metrics=none \
      --duration="${duration}" --tick=10 --snap=0 --dump=0 --ckpt=10 \
      --sample_every=26 --eval_every=50 --mid_t=0.821 \
      --adaptive-update-kimg=0.5 \
      --immutable-checkpoint-kimg="${milestones}" \
      --schedule-switch-manifest="${manifest}" --resume="${source_state}"

env CUDA_VISIBLE_DEVICES="${gpu_id}" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    WORLD_SIZE=1 LD_LIBRARY_PATH="${Q256_RUNTIME_LD_LIBRARY_PATH}" \
    PATH="${Q256_RUNTIME_PATH}" \
    "${Q256_RUNTIME_PYTHON}" "${repo}/analysis/q256_schedule_switch_v1/export_milestones.py" \
      --run-dir "${run_dir}" --manifest "${manifest}"

echo "[q256-switch] PASS seed=${seed} branch=${branch} gpu=${gpu_id}"
