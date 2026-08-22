#!/usr/bin/env bash

set -euo pipefail

seed="${1:?usage: $0 SEED ARM BUDGET_KIMG NFE GPU_UUID CHECKPOINT CHECKPOINT_SHA PORT}"
arm="${2:?usage: $0 SEED ARM BUDGET_KIMG NFE GPU_UUID CHECKPOINT CHECKPOINT_SHA PORT}"
budget_kimg="${3:?usage: $0 SEED ARM BUDGET_KIMG NFE GPU_UUID CHECKPOINT CHECKPOINT_SHA PORT}"
nfe="${4:?usage: $0 SEED ARM BUDGET_KIMG NFE GPU_UUID CHECKPOINT CHECKPOINT_SHA PORT}"
gpu_uuid="${5:?usage: $0 SEED ARM BUDGET_KIMG NFE GPU_UUID CHECKPOINT CHECKPOINT_SHA PORT}"
checkpoint="${6:?usage: $0 SEED ARM BUDGET_KIMG NFE GPU_UUID CHECKPOINT CHECKPOINT_SHA PORT}"
checkpoint_sha="${7:?usage: $0 SEED ARM BUDGET_KIMG NFE GPU_UUID CHECKPOINT CHECKPOINT_SHA PORT}"
port="${8:?usage: $0 SEED ARM BUDGET_KIMG NFE GPU_UUID CHECKPOINT CHECKPOINT_SHA PORT}"

case "${seed}" in 14|15|16|17|18) ;; *) echo "unsupported seed: ${seed}" >&2; exit 2 ;; esac
case "${arm}" in A|B|C|D) ;; *) echo "unsupported arm: ${arm}" >&2; exit 2 ;; esac
case "${nfe}" in 1|2) ;; *) echo "unsupported NFE: ${nfe}" >&2; exit 2 ;; esac
[[ "${budget_kimg}" =~ ^[0-9]+$ && "${port}" =~ ^[0-9]+$ ]] || exit 2
[[ "${gpu_uuid}" =~ ^GPU-[A-Za-z0-9-]+$ && "${checkpoint_sha}" =~ ^[0-9a-f]{64}$ ]] || exit 2

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/runtime_env.sh"

repo="${Q256_EVAL_ROOT}/source/evaluator"
source_archive="${Q256_EVAL_ROOT}/source/q256-evaluator-source-d6aba02.tar"
dataset="${Q256_EVAL_DATASET:-${Q256_EVAL_ROOT}/assets/cifar10-32x32-canonical-08c9ed1b2b1c.zip}"
sif="${Q256_EVAL_ROOT}/runtime/ect-pytorch2401-deterministic.sif"
cache="${Q256_EVAL_CACHE:-${Q256_EVAL_ROOT}/assets/cache/shared}"
job_id="seed${seed}-arm${arm}-k${budget_kimg}-nfe${nfe}"
target="${Q256_EVAL_ROOT}/runs/${job_id}"
receipt="${Q256_EVAL_ROOT}/receipts/${job_id}.json"
process_log="${Q256_EVAL_ROOT}/logs/${job_id}.process.log"
lock="${Q256_EVAL_ROOT}/gpu-lock-${gpu_uuid}.lock"
runtime_receipt="${Q256_EVAL_ROOT}/runtime/runtime_integrity.json"
portability_gate="${Q256_EVAL_ROOT}/portability_gate.json"
calibration_args=()
if [[ "${Q256_CALIBRATION_MODE:-0}" == 1 ]]; then
  calibration_args+=(--calibration)
else
  [[ -s "${portability_gate}" ]] || { echo "missing portability gate" >&2; exit 3; }
fi

[[ -x "${Q256_RUNTIME_PYTHON}" && -f "${repo}/ct_eval.py" && -f "${source_archive}" && -f "${dataset}" && -f "${sif}" && -s "${runtime_receipt}" ]] || exit 3
[[ -f "${checkpoint}" && ! -L "${checkpoint}" ]] || exit 3
[[ "$(sha256sum "${checkpoint}" | awk '{print $1}')" == "${checkpoint_sha}" ]] || exit 3
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "37560e2eb50a9a361f9fca899a33778616386a622d5f039f53305d8d492eaed6" ]] || exit 3
tar -df "${source_archive}" -C "${repo}"

if [[ -s "${receipt}" ]]; then
  echo "[q256-stream-eval] SKIP ${job_id}"
  exit 0
fi
[[ ! -e "${target}" && ! -e "${receipt}" ]] || { echo "refuse incomplete existing output: ${job_id}" >&2; exit 4; }

mkdir -p "${cache}/downloads" "$(dirname "${target}")" "$(dirname "${receipt}")" "$(dirname "${process_log}")"
exec 9>"${lock}"
flock -n 9 || { echo "GPU lock held: ${gpu_uuid}" >&2; exit 4; }

gpu_identity=$(nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader,nounits | awk -F', ' -v wanted="${gpu_uuid}" '$1 == wanted {print $0}')
[[ "${gpu_identity}" == *"A100"* ]] || { echo "GPU identity mismatch: ${gpu_identity}" >&2; exit 4; }
gpu_processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v wanted="${gpu_uuid}" '$1 == wanted {count++} END {print count+0}')
[[ "${gpu_processes}" == 0 ]] || { echo "GPU is occupied: ${gpu_uuid}" >&2; exit 4; }

mid_args=()
if [[ "${nfe}" == 2 ]]; then mid_args+=(--mid_t=0.821); fi
started_epoch=$(date +%s)
echo "[q256-stream-eval] START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) job=${job_id} gpu=${gpu_uuid}"

env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_uuid}" \
  PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  LC_ALL=C.UTF-8 LD_LIBRARY_PATH="${Q256_RUNTIME_LD_LIBRARY_PATH}" PATH="${Q256_RUNTIME_PATH}" \
  DNNLIB_CACHE_DIR="${cache}" MASTER_ADDR=127.0.0.1 MASTER_PORT="${port}" \
  RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
  "${Q256_RUNTIME_PYTHON}" -m torch.distributed.run \
    --standalone --nproc_per_node=1 --master_port="${port}" \
    "${repo}/ct_eval.py" --resume "${checkpoint}" \
    --outdir "${target}" --nosubdir \
    --data "${dataset}" --cond=False --arch=ddpmpp --precond=ct \
    --dropout=0.2 --augment=0 --xflip=False --fp16=False \
    --cache=True --workers=3 --eval-batch=512 --metric-generator-batch=128 \
    --nfe="${nfe}" "${mid_args[@]}" \
    --metrics=kid50k_full,fid50k_full --metric-repeats=1 \
    --sample-seeds=0-49999 --seed=20260730 --retain-generated-artifacts \
    --desc="q256-seed14-18-streaming-${job_id}" \
  2>&1 | tee "${process_log}"

finished_epoch=$(date +%s)
elapsed=$((finished_epoch - started_epoch))
"${Q256_SANDBOX_ROOT}/usr/bin/python" "${script_dir}/validate_eval_job.py" \
  --job-dir "${target}" --checkpoint "${checkpoint}" \
  --checkpoint-sha256 "${checkpoint_sha}" --dataset "${dataset}" \
  --repo "${repo}" --source-archive "${source_archive}" \
  --sif "${sif}" --runtime-receipt "${runtime_receipt}" \
  --portability-gate "${portability_gate}" "${calibration_args[@]}" --receipt "${receipt}" \
  --seed "${seed}" --arm "${arm}" --budget-kimg "${budget_kimg}" \
  --nfe "${nfe}" --gpu-uuid "${gpu_uuid}" --elapsed-seconds "${elapsed}"

echo "[q256-stream-eval] PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) job=${job_id} elapsed=${elapsed}"
