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

job_root="${Q256_REPLAY_JOB_ROOT:-/root/q256_target_weight_replay_curve_v1}"
repo="${job_root}/source/recurrence_of_ect"
source_root=/mnt/ect_project/q256_target_weight_1024k/source_states/formal-direct-dcca41b-deterministic-v1
run_root="${job_root}/runs/q256-target-weight-replay-curve-v1"
sif=/mnt/ect_project/q256_target_weight_1024k/runtime/ect-pytorch2401-deterministic.sif
sandbox_root="${job_root}/runtime/sandbox"
runtime_python="${sandbox_root}/usr/bin/python"
dataset_host=/mnt/ect_project/datasets/cifar10-32x32.zip
dataset_runtime=/data/raw/ECT/datasets/cifar10-32x32-canonical-08c9ed1b2b1c.zip
private_shm="/dev/shm/q256-replay-curve-seed${seed}"
approved_base=12f905dff2bf474495abb186c215ca2ea959099e
milestones=384,512,640,768,896,1024

runtime_ld_library_path="${sandbox_root}/usr/local/lib/python3.10/dist-packages/torch/lib"
runtime_ld_library_path+=":${sandbox_root}/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib"
runtime_ld_library_path+=":${sandbox_root}/usr/local/cuda/compat/lib"
runtime_ld_library_path+=":${sandbox_root}/usr/local/nvidia/lib:${sandbox_root}/usr/local/nvidia/lib64"
runtime_ld_library_path+=":${sandbox_root}/lib:${sandbox_root}/lib/x86_64-linux-gnu"
runtime_ld_library_path+=":${sandbox_root}/opt/hpcx/clusterkit/lib:${sandbox_root}/opt/hpcx/hcoll/lib"
runtime_ld_library_path+=":${sandbox_root}/opt/hpcx/nccl_rdma_sharp_plugin/lib:${sandbox_root}/opt/hpcx/ompi/lib"
runtime_ld_library_path+=":${sandbox_root}/opt/hpcx/sharp/lib:${sandbox_root}/opt/hpcx/ucc/lib:${sandbox_root}/opt/hpcx/ucx/lib"
runtime_ld_library_path+=":${sandbox_root}/usr/local/cuda/targets/x86_64-linux/lib:${sandbox_root}/usr/local/lib"
runtime_path="${sandbox_root}/usr/local/lib/python3.10/dist-packages/torch_tensorrt/bin"
runtime_path+=":${sandbox_root}/usr/local/mpi/bin:${sandbox_root}/usr/local/nvidia/bin"
runtime_path+=":${sandbox_root}/usr/local/cuda/bin:${sandbox_root}/usr/local/sbin:${sandbox_root}/usr/local/bin"
runtime_path+=":${sandbox_root}/usr/sbin:${sandbox_root}/usr/bin:${sandbox_root}/sbin:${sandbox_root}/bin"
runtime_path+=":${sandbox_root}/usr/local/ucx/bin:${sandbox_root}/opt/tensorrt/bin"

[[ -f "${sif}" && -x "${runtime_python}" && -f "${dataset_host}" ]] || { echo "missing runtime or dataset" >&2; exit 2; }
git -C "${repo}" merge-base --is-ancestor "${approved_base}" HEAD || { echo "missing PR #76 implementation" >&2; exit 2; }
git -C "${repo}" diff --quiet "${approved_base}"..HEAD -- training/loss.py training/schedules.py training/networks.py training/dataset.py || { echo "training mathematics differs from PR #76" >&2; exit 2; }
[[ -z "$(git -C "${repo}" status --porcelain)" ]] || { echo "source worktree is dirty" >&2; exit 2; }
mkdir -p "${run_root}/seed${seed}" "${private_shm}"
chmod 700 "${private_shm}"
mkdir -p "$(dirname "${dataset_runtime}")"
if [[ -e "${dataset_runtime}" || -L "${dataset_runtime}" ]]; then
  [[ "$(readlink -f "${dataset_runtime}")" == "$(readlink -f "${dataset_host}")" ]] || { echo "canonical dataset path points elsewhere" >&2; exit 2; }
else
  ln -s "${dataset_host}" "${dataset_runtime}"
fi

echo "[q256-replay] WORKER_START seed=${seed} gpu=${gpu_id} commit=$(git -C "${repo}" rev-parse HEAD)"
for arm_spec in A:1.0:1.0 B:1.1:1.1 C:1.1:1.0 D:1.0:1.1; do
  IFS=: read -r arm target_scale denominator_scale <<<"${arm_spec}"
  source_dir="${source_root}/seed${seed}/arm${arm}"
  source_state="${source_dir}/training-state-latest.pt"
  run_dir="${run_root}/seed${seed}/arm${arm}"

  if [[ ! -e "${run_dir}" ]]; then
    mkdir "${run_dir}"
    cp -p "${source_dir}/training_options.json" "${run_dir}/"
    cp -p "${source_dir}/train_summary.csv" "${run_dir}/"
    cp -p "${source_dir}/factorial_training_telemetry_v1.csv" "${run_dir}/"
    cp -p "${source_dir}/initial_state_receipt_v1.json" "${run_dir}/"
    cp -p "${source_dir}/log.txt" "${run_dir}/"
  fi

  for required_file in training_options.json train_summary.csv factorial_training_telemetry_v1.csv initial_state_receipt_v1.json; do
    [[ -f "${run_dir}/${required_file}" ]] || { echo "missing ${run_dir}/${required_file}" >&2; exit 3; }
  done
  [[ -s "${source_state}" ]] || { echo "missing formal source: ${source_state}" >&2; exit 3; }

  if [[ -f "${run_dir}/training-state-latest.pt" ]]; then
    resume_state="${run_dir}/training-state-latest.pt"
  else
    resume_state="${source_state}"
  fi
  case "${resume_state}" in
    "${source_root}"/*|"${run_root}"/*) ;;
    *) echo "refuse out-of-scope resume state: ${resume_state}" >&2; exit 3 ;;
  esac

  echo "[q256-replay] START seed=${seed} arm=${arm} gpu=${gpu_id} resume=${resume_state}"
  env \
    ECT_Q256_LAUNCHER_IN_SANDBOX=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_id}" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    CUDA_CACHE_DISABLE=1 CUDA_MODULE_LOADING=LAZY \
    TORCH_CUDNN_V8_API_ENABLED=1 USE_EXPERIMENTAL_CUDNN_V8_API=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONIOENCODING=utf-8 \
    PYTHONNOUSERSITE=1 LC_ALL=C.UTF-8 \
    LD_LIBRARY_PATH="${runtime_ld_library_path}" PATH="${runtime_path}" \
    MASTER_ADDR=127.0.0.1 MASTER_PORT="${master_port}" \
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 PYTHONUNBUFFERED=1 \
    "${runtime_python}" "${repo}/ct_train.py" \
    --data="${dataset_runtime}" --outdir="${run_dir}" --nosubdir \
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
    --immutable-checkpoint-kimg="${milestones}" \
    --resume="${resume_state}"

  for budget in 384 512 640 768 896 1024; do
    [[ -s "${run_dir}/training-state-kimg$(printf '%06d' "${budget}").pt" ]] || { echo "missing immutable milestone seed=${seed} arm=${arm} budget=${budget}" >&2; exit 4; }
  done
  echo "[q256-replay] PASS seed=${seed} arm=${arm}"
done

env PYTHONNOUSERSITE=1 LD_LIBRARY_PATH="${runtime_ld_library_path}" PATH="${runtime_path}" \
  "${runtime_python}" "${repo}/scripts/export_q256_replay_milestone_snapshots.py" \
  --run-root "${run_root}" --source-root "${source_root}" --seed "${seed}"
echo "[q256-replay] WORKER_PASS seed=${seed}"
