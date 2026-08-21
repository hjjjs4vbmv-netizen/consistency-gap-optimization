#!/usr/bin/env bash
set -euo pipefail

seed="${1:?usage: $0 SEED GPU_ID MASTER_PORT}"
gpu_id="${2:?usage: $0 SEED GPU_ID MASTER_PORT}"
master_port="${3:?usage: $0 SEED GPU_ID MASTER_PORT}"

case "${seed}" in
  3|4|5) ;;
  *) echo "unsupported seed: ${seed}" >&2; exit 2 ;;
esac
[[ "${master_port}" =~ ^[0-9]+$ ]] || { echo "invalid master port" >&2; exit 2; }

job_root="${Q256_JOB_ROOT:-/root/q256_target_weight_1024k}"
container_root=/workspace/q256_target_weight_1024k
repo="${job_root}/source/recurrence_of_ect"
container_repo="${container_root}/source/recurrence_of_ect"
source_root="${job_root}/source_states/formal-direct-dcca41b-deterministic-v1"
container_source_root="${container_root}/source_states/formal-direct-dcca41b-deterministic-v1"
run_root="${job_root}/runs/q256-target-weight-1024k"
container_run_root="${container_root}/runs/q256-target-weight-1024k"
sif="${job_root}/runtime/ect-pytorch2401-deterministic.sif"
dataset_host=/mnt/ect_project/datasets/cifar10-32x32.zip
dataset_container=/data/raw/ECT/datasets/cifar10-32x32-canonical-08c9ed1b2b1c.zip
private_shm="/dev/shm/q256-target-weight-seed${seed}"
expected_training_parent=458205192722883df393a8d017c26e6fa46f48f7

[[ -f "${sif}" && -f "${dataset_host}" ]] || { echo "missing runtime or dataset" >&2; exit 2; }
[[ "$(git -C "${repo}" rev-parse HEAD^)" == "${expected_training_parent}" ]] || { echo "wrong training implementation parent" >&2; exit 2; }
[[ -z "$(git -C "${repo}" status --porcelain)" ]] || { echo "source worktree is dirty" >&2; exit 2; }
mkdir -p "${run_root}/seed${seed}" "${private_shm}"
chmod 700 "${private_shm}"

for arm_spec in A:1.0:1.0 B:1.1:1.1 C:1.1:1.0 D:1.0:1.1; do
  IFS=: read -r arm target_scale denominator_scale <<<"${arm_spec}"
  source_dir="${source_root}/seed${seed}/arm${arm}"
  run_dir="${run_root}/seed${seed}/arm${arm}"
  container_run_dir="${container_run_root}/seed${seed}/arm${arm}"

  if [[ ! -e "${run_dir}" ]]; then
    mkdir "${run_dir}"
    cp "${source_dir}/training_options.json" "${run_dir}/"
    cp "${source_dir}/train_summary.csv" "${run_dir}/"
    cp "${source_dir}/factorial_training_telemetry_v1.csv" "${run_dir}/"
    cp "${source_dir}/initial_state_receipt_v1.json" "${run_dir}/"
    cp "${source_dir}/log.txt" "${run_dir}/"
  fi

  for required_file in training_options.json train_summary.csv factorial_training_telemetry_v1.csv initial_state_receipt_v1.json; do
    [[ -f "${run_dir}/${required_file}" ]] || { echo "missing ${run_dir}/${required_file}" >&2; exit 3; }
  done

  if [[ -f "${run_dir}/training-state-latest.pt" ]]; then
    resume_state="${container_run_dir}/training-state-latest.pt"
  else
    [[ -f "${source_dir}/training-state-latest.pt" ]] || { echo "missing formal source state: ${source_dir}" >&2; exit 3; }
    resume_state="${container_source_root}/seed${seed}/arm${arm}/training-state-latest.pt"
  fi

  echo "[q256-1024k] START seed=${seed} arm=${arm} gpu=${gpu_id} resume=${resume_state}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" apptainer exec --nv \
    --bind "${job_root}:${container_root}" \
    --bind "${dataset_host}:${dataset_container}:ro" \
    --bind "${private_shm}:/dev/shm" \
    "${sif}" env \
    ECT_Q256_LAUNCHER_IN_SANDBOX=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_id}" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT="${master_port}" \
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 PYTHONUNBUFFERED=1 \
    python "${container_repo}/ct_train.py" \
    --data="${dataset_container}" --outdir="${container_run_dir}" --nosubdir \
    --cond=False --arch=ddpmpp --precond=ect --batch=128 --batch-gpu=16 \
    --optim=RAdam --lr=0.0001 --dropout=0.2 --augment=0 --xflip=False \
    --mean=-1.1 --std=2.0 --mapping=sigmoid --global-gap-scale=1.0 \
    --factorial-protocol=q256_target_weight_v1 \
    --target-gap-scale="${target_scale}" \
    --denominator-gap-scale="${denominator_scale}" \
    -q 256 -k 8 -b 1 -c 0 --double=10000 --ema_beta=0.9993 \
    --seed="${seed}" --fp16=True --tf32=False --ls=1.0 --enable_amp=True \
    --bench=False --cache=True --workers=1 --metrics=none --duration=1.024 \
    --tick=10 --snap=0 --dump=0 --ckpt=10 --sample_every=26 \
    --eval_every=50 --mid_t=0.821 --adaptive-update-kimg=0.5 \
    --resume="${resume_state}"

  final_state="${container_run_dir}/training-state-latest.pt"
  apptainer exec --nv \
    --bind "${job_root}:${container_root}" \
    --pwd "${container_repo}" \
    "${sif}" python -c \
    "import torch; s=torch.load('${final_state}', map_location='cpu'); assert s['cur_nimg'] == 1024000; assert s['attempted_iteration'] == 8000; print('[q256-1024k] VERIFIED seed=${seed} arm=${arm} kimg=1024 attempts=8000 accepted=%d amp_skips=%d' % (s['successful_optimizer_steps'], s['attempted_iteration'] - s['successful_optimizer_steps']))"
  echo "[q256-1024k] PASS seed=${seed} arm=${arm}"
done

echo "[q256-1024k] WORKER_PASS seed=${seed}"
