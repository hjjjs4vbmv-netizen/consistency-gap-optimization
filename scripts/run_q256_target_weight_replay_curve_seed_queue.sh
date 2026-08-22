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
if [[ ! -e "${dataset_runtime}" && ! -L "${dataset_runtime}" ]]; then
  ln -s "${dataset_host}" "${dataset_runtime}" 2>/dev/null || true
fi
[[ "$(readlink -f "${dataset_runtime}")" == "$(readlink -f "${dataset_host}")" ]] || { echo "canonical dataset path points elsewhere" >&2; exit 2; }
dataset_sha256="$(sha256sum "${dataset_host}" | cut -d' ' -f1)"
runtime_sha256="$(awk '$2 == "./runtime/ect-pytorch2401-deterministic.sif" {print $1}' /mnt/ect_project/q256_target_weight_1024k/SHA256SUMS.release.txt)"
[[ "${runtime_sha256}" =~ ^[0-9a-f]{64}$ ]] || { echo "missing runtime hash in release manifest" >&2; exit 2; }

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

  latest_state="${run_dir}/training-state-latest.pt"
  latest_milestone="$(find "${run_dir}" -maxdepth 1 -type f -name 'training-state-kimg*.pt' -print | sort | tail -n 1)"
  resume_state="${source_state}"
  latest_nimg=0
  if [[ -s "${latest_state}" ]]; then
    latest_nimg="$(env PYTHONNOUSERSITE=1 LD_LIBRARY_PATH="${runtime_ld_library_path}" PATH="${runtime_path}" "${runtime_python}" -c "import torch; print(int(torch.load('${latest_state}', map_location='cpu', weights_only=False)['cur_nimg']))")"
    resume_state="${latest_state}"
  fi
  if [[ -n "${latest_milestone}" ]]; then
    milestone_name="$(basename "${latest_milestone}")"
    [[ "${milestone_name}" =~ ^training-state-kimg([0-9]{6})\.pt$ ]] || { echo "invalid milestone filename: ${milestone_name}" >&2; exit 3; }
    milestone_nimg=$((10#${BASH_REMATCH[1]} * 1000))
    if (( milestone_nimg >= latest_nimg )); then
      resume_state="${latest_milestone}"
    fi
  fi
  case "${resume_state}" in
    "${source_root}"/*|"${run_root}"/*) ;;
    *) echo "refuse out-of-scope resume state: ${resume_state}" >&2; exit 3 ;;
  esac

  start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  source_sha256="$(sha256sum "${source_state}" | cut -d' ' -f1)"
  resume_sha256="$(sha256sum "${resume_state}" | cut -d' ' -f1)"
  git_commit="$(git -C "${repo}" rev-parse HEAD)"
  git -C "${repo}" diff --binary > "${run_dir}/git-dirty.diff"
  {
    echo "start_utc=${start_utc}"
    echo "hostname=$(hostname)"
    echo "gpu_id=${gpu_id}"
    nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv,noheader
    echo "git_commit=${git_commit}"
    echo "source_state=${source_state}"
    echo "source_state_sha256=${source_sha256}"
    echo "resume_state=${resume_state}"
    echo "resume_state_sha256=${resume_sha256}"
    echo "dataset_host=${dataset_host}"
    echo "dataset_sha256=${dataset_sha256}"
    echo "runtime_sif=${sif}"
    echo "runtime_sif_sha256=${runtime_sha256}"
  } >> "${run_dir}/replay-environment.log"
  printf '%s\n' \
    "${runtime_python} ${repo}/ct_train.py --data=${dataset_runtime} --outdir=${run_dir} --nosubdir --cond=False --arch=ddpmpp --precond=ect --batch=128 --batch-gpu=16 --optim=RAdam --lr=0.0001 --dropout=0.2 --augment=0 --xflip=False --mean=-1.1 --std=2.0 --mapping=sigmoid --global-gap-scale=1.0 --factorial-protocol=q256_target_weight_v1 --target-gap-scale=${target_scale} --denominator-gap-scale=${denominator_scale} -q 256 -k 8 -b 1 -c 0 --double=10000 --ema_beta=0.9993 --seed=${seed} --fp16=True --tf32=False --ls=1.0 --enable_amp=True --bench=False --cache=True --workers=1 --metrics=none --duration=1.024 --tick=10 --snap=0 --dump=0 --ckpt=10 --sample_every=26 --eval_every=50 --mid_t=0.821 --adaptive-update-kimg=0.5 --immutable-checkpoint-kimg=${milestones} --resume=${resume_state}" \
    >> "${run_dir}/replay-launch-commands.txt"
  printf 'START,%s,%s,%s,%s,%s,%s\n' \
    "${start_utc}" "${seed}" "${arm}" "${gpu_id}" \
    "${resume_state}" "${resume_sha256}" \
    >> "${run_dir}/replay-resume-history.csv"

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
  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'END,%s,%s,%s,%s,%s,%s\n' \
    "${end_utc}" "${seed}" "${arm}" "${gpu_id}" \
    "${resume_state}" "${resume_sha256}" \
    >> "${run_dir}/replay-resume-history.csv"
  echo "[q256-replay] PASS seed=${seed} arm=${arm}"
done

env PYTHONNOUSERSITE=1 LD_LIBRARY_PATH="${runtime_ld_library_path}" PATH="${runtime_path}" \
  "${runtime_python}" "${repo}/scripts/export_q256_replay_milestone_snapshots.py" \
  --run-root "${run_root}" --source-root "${source_root}" --seed "${seed}"
echo "[q256-replay] WORKER_PASS seed=${seed}"
