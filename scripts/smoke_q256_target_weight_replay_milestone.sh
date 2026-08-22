#!/usr/bin/env bash
set -euo pipefail

gpu_id="${1:?usage: $0 GPU_ID MASTER_PORT}"
master_port="${2:?usage: $0 GPU_ID MASTER_PORT}"
[[ "${master_port}" =~ ^[0-9]+$ ]] || { echo "invalid master port" >&2; exit 2; }

job_root="${Q256_REPLAY_JOB_ROOT:-/root/q256_target_weight_replay_curve_v1}"
repo="${job_root}/source/recurrence_of_ect"
source_dir=/mnt/ect_project/q256_target_weight_1024k/source_states/formal-direct-dcca41b-deterministic-v1/seed3/armA
source_state="${source_dir}/training-state-latest.pt"
run_dir="${job_root}/runs/q256-target-weight-replay-curve-v1/seed3/armA"
sandbox_root="${job_root}/runtime/sandbox"
runtime_python="${sandbox_root}/usr/bin/python"
dataset_host=/mnt/ect_project/datasets/cifar10-32x32.zip
dataset_runtime=/data/raw/ECT/datasets/cifar10-32x32-canonical-08c9ed1b2b1c.zip
milestone_state="${run_dir}/training-state-kimg000384.pt"

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

[[ -x "${runtime_python}" && -s "${source_state}" && -f "${dataset_host}" ]] || { echo "missing runtime/source/dataset" >&2; exit 2; }
[[ -z "$(git -C "${repo}" status --porcelain)" ]] || { echo "source worktree is dirty" >&2; exit 2; }
[[ ! -e "${run_dir}" ]] || { echo "smoke run directory already exists: ${run_dir}" >&2; exit 3; }
mkdir -p "${run_dir}" "$(dirname "${dataset_runtime}")"
cp -p "${source_dir}/training_options.json" "${run_dir}/"
cp -p "${source_dir}/train_summary.csv" "${run_dir}/"
cp -p "${source_dir}/factorial_training_telemetry_v1.csv" "${run_dir}/"
cp -p "${source_dir}/initial_state_receipt_v1.json" "${run_dir}/"
cp -p "${source_dir}/log.txt" "${run_dir}/"
if [[ -e "${dataset_runtime}" || -L "${dataset_runtime}" ]]; then
  [[ "$(readlink -f "${dataset_runtime}")" == "$(readlink -f "${dataset_host}")" ]] || { echo "canonical dataset path points elsewhere" >&2; exit 2; }
else
  ln -s "${dataset_host}" "${dataset_runtime}"
fi

common_args=(
  --data="${dataset_runtime}" --outdir="${run_dir}" --nosubdir
  --cond=False --arch=ddpmpp --precond=ect --batch=128 --batch-gpu=16
  --optim=RAdam --lr=0.0001 --dropout=0.2 --augment=0 --xflip=False
  --mean=-1.1 --std=2.0 --mapping=sigmoid --global-gap-scale=1.0
  --factorial-protocol=q256_target_weight_v1
  --target-gap-scale=1.0 --denominator-gap-scale=1.0
  -q 256 -k 8 -b 1 -c 0 --double=10000 --ema_beta=0.9993
  --seed=3 --fp16=True --tf32=False --ls=1.0 --enable_amp=True
  --bench=False --cache=True --workers=1 --metrics=none --duration=0.384
  --tick=10 --snap=0 --dump=0 --ckpt=10 --sample_every=26
  --eval_every=50 --mid_t=0.821 --adaptive-update-kimg=0.5
  --immutable-checkpoint-kimg=384
)

run_training() {
  local resume_state="${1:?resume state required}"
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
    "${common_args[@]}" --resume="${resume_state}"
}

echo "[q256-replay-smoke] TRAIN source=${source_state} target=384kimg"
run_training "${source_state}"
[[ -s "${milestone_state}" ]] || { echo "missing smoke milestone" >&2; exit 4; }
env PYTHONNOUSERSITE=1 LD_LIBRARY_PATH="${runtime_ld_library_path}" PATH="${runtime_path}" \
  "${runtime_python}" -c \
  "import torch; s=torch.load('${milestone_state}', map_location='cpu', weights_only=False); assert s['cur_nimg']==384000; assert s['attempted_iteration']==3000; assert s['factorial']['arm']=='A'; assert s['optimizer_state']['state']; assert s['gradscaler_state']; assert s['rank_states']; print('[q256-replay-smoke] RELOAD_PASS')"
echo "[q256-replay-smoke] SAME-BUDGET RESUME source=${milestone_state}"
run_training "${milestone_state}"
echo "[q256-replay-smoke] PASS"
