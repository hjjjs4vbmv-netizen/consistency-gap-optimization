#!/usr/bin/env bash
set -Eeuo pipefail

gpu_index="${1:?usage: $0 GPU_INDEX SEED MASTER_PORT}"
seed="${2:?usage: $0 GPU_INDEX SEED MASTER_PORT}"
master_port="${3:?usage: $0 GPU_INDEX SEED MASTER_PORT}"

repo=/data/temp/ECT001/q256-factorial-clean-dcca41b-v2
runtime=/data/temp/q256-cohort3-runtime/ngc-pytorch-24.01-bundle/rootfs
driver_injection=/data/temp/q256-cohort3-runtime/nvidia-driver-injection-570.211.01
dataset=/mnt/ect_project/datasets/cifar10-32x32.zip
dataset_sha=2d4056e80de1a96fe16f2f58945c6c4710ecd9fc02e3cc7aa5b50513b7cdf389
transfer=/data/raw/ECT/pretrained/edm-cifar10-32x32-uncond-vp.pkl
transfer_sha=4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da
run_root=/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260821/secondary-precision-extension/seed14-18-dcca41b-v2
private_shm="/tmp/ECT001-q256-seed14-18-v2-shm-seed${seed}"
active_arm=preflight

umask 027

write_failure() {
  local exit_code=$?
  local failure_path="${run_root}/seed${seed}-worker-failure.md"
  if [[ -d "${run_root}" && ! -e "${failure_path}" ]]; then
    {
      printf '# q256 seed14-18 worker failure\n\n'
      printf -- '- UTC: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf -- '- Seed: %s\n' "${seed}"
      printf -- '- GPU index: %s\n' "${gpu_index}"
      printf -- '- Active arm: %s\n' "${active_arm}"
      printf -- '- Exit code: %s\n' "${exit_code}"
      printf -- '- Action: worker stopped; no automatic retry.\n'
    } >"${failure_path}"
  fi
  exit "${exit_code}"
}
trap write_failure ERR

case "${seed}" in 14|15|16|17|18) ;; *) echo "unsupported seed: ${seed}" >&2; exit 2 ;; esac
[[ "${gpu_index}" =~ ^[0-4]$ ]] || { echo "invalid GPU index: ${gpu_index}" >&2; exit 2; }
[[ "${master_port}" =~ ^[0-9]+$ ]] || { echo "invalid master port" >&2; exit 2; }
[[ -d "${repo}" && -d "${runtime}" && -d "${driver_injection}" ]] || { echo "missing source or runtime" >&2; exit 2; }
[[ -f "${dataset}" && -f "${transfer}" ]] || { echo "missing immutable asset" >&2; exit 2; }
[[ "$(sha256sum "${dataset}" | awk '{print $1}')" == "${dataset_sha}" ]] || { echo "dataset hash mismatch" >&2; exit 2; }
[[ "$(sha256sum "${transfer}" | awk '{print $1}')" == "${transfer_sha}" ]] || { echo "transfer hash mismatch" >&2; exit 2; }

gpu_line=$(nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader,nounits | awk -F', ' -v wanted="${gpu_index}" '$1 == wanted {print $0}')
[[ "${gpu_line}" == "${gpu_index}, "*", NVIDIA A100-PCIE-40GB, 40960" ]] || { echo "assigned GPU identity mismatch: ${gpu_line}" >&2; exit 2; }
gpu_uuid=$(printf '%s\n' "${gpu_line}" | awk -F', ' '{print $2}')
gpu_processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v wanted="${gpu_uuid}" '$1 == wanted {count++} END {print count+0}')
[[ "${gpu_processes}" == 0 ]] || { echo "assigned GPU is not compute-idle: ${gpu_uuid}" >&2; exit 2; }
if ss -H -ltn "sport = :${master_port}" | grep -q .; then
  echo "master port ${master_port} is already listening" >&2
  exit 2
fi
[[ ! -e "${private_shm}" ]] || { echo "refusing existing private shared-memory path: ${private_shm}" >&2; exit 3; }
mkdir -m 700 "${private_shm}"

echo "[worker] START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) seed=${seed} gpu_index=${gpu_index} gpu_uuid=${gpu_uuid} source=dcca41b19e7c45512b5fbe98776520396a1bf9ac"
echo "[worker] DATASET semantic_exact_to_official_cifar10=true archive_sha256=${dataset_sha} canonical_archive_sha256=08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372 byte_identical_to_canonical=false"

run_arm() {
  local arm=$1
  local target_scale=$2
  local denominator_scale=$3
  local outdir=${run_root}/seed${seed}/arm${arm}
  active_arm=${arm}
  [[ ! -e "${outdir}" ]] || { echo "refusing existing fresh cell: ${outdir}" >&2; return 3; }
  mkdir -p "${run_root}/seed${seed}"
  echo "[worker] ARM_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) seed=${seed} arm=${arm} target=${target_scale} denominator=${denominator_scale} mode=fresh"
  timeout --signal=TERM --kill-after=10s 24h \
    proot -0 -r "${runtime}" \
      -b /dev:/dev -b /proc:/proc -b /sys:/sys \
      -b /data:/data -b /mnt:/mnt -b /tmp:/tmp \
      -b /usr/lib/x86_64-linux-gnu:/host-driver-source \
      -b /usr/bin:/host-driver-bin-source \
      -b "${driver_injection}:/usr/local/nvidia" \
      -b "${private_shm}:/dev/shm" \
      /usr/bin/env -i \
        HOME=/root LC_ALL=C.UTF-8 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 \
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
        PATH=/usr/local/lib/python3.10/dist-packages/torch_tensorrt/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/tensorrt/bin \
        LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
        CUDA_VERSION=12.3.2.001 CUDNN_VERSION=8.9.7.29+cuda12.2 PYTORCH_VERSION=2.2.0a0+81ea7a4 \
        NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_index}" \
        CUDA_CACHE_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        MASTER_ADDR=127.0.0.1 MASTER_PORT="${master_port}" \
        RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
        /usr/bin/python "${repo}/ct_train.py" \
          --data="${dataset}" --outdir="${outdir}" --nosubdir \
          --cond=False --arch=ddpmpp --precond=ect --batch=128 --batch-gpu=16 \
          --optim=RAdam --lr=0.0001 --dropout=0.2 --augment=0 --xflip=False \
          --mean=-1.1 --std=2.0 --mapping=sigmoid --global-gap-scale=1.0 \
          --factorial-protocol=q256_target_weight_v1 \
          --target-gap-scale="${target_scale}" \
          --denominator-gap-scale="${denominator_scale}" \
          -q 256 -k 8 -b 1 -c 0 --double=10000 --ema_beta=0.9993 \
          --seed="${seed}" --fp16=True --tf32=False --ls=1.0 --enable_amp=True \
          --bench=False --cache=True --workers=1 --metrics=none --duration=0.256 \
          --tick=10 --snap=0 --dump=0 --ckpt=10 --sample_every=26 \
          --eval_every=50 --mid_t=0.821 --adaptive-update-kimg=0.5 \
          --transfer="${transfer}"
  echo "[worker] ARM_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) seed=${seed} arm=${arm}"
}

run_arm A 1.0 1.0
run_arm B 1.1 1.1
run_arm C 1.1 1.0
run_arm D 1.0 1.1

active_arm=complete
trap - ERR
echo "[worker] WORKER_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) seed=${seed}"
