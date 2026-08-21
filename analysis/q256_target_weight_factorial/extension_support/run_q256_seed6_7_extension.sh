#!/usr/bin/env bash
set -euo pipefail

gpu_uuid=GPU-ef9edaf6-d661-e143-efd1-154c1ad29f10
master_port=29761
repo=/data/temp/ECT001/q256-factorial-clean-25c3d22
expected_head=dcca41b19e7c45512b5fbe98776520396a1bf9ac
dataset=/data/raw/ECT/datasets/cifar10-32x32-canonical-08c9ed1b2b1c.zip
dataset_sha=08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372
transfer=/data/raw/ECT/pretrained/edm-cifar10-32x32-uncond-vp.pkl
transfer_sha=4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da
sandbox=/data/temp/ect001-pytorch2401-sandbox
apptainer=/usr/bin/apptainer
extension_root=/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819/secondary-precision-extension/seed6-7-dcca41b-v1
private_shm=/tmp/ECT001-q256-seed6-7-extension-shm-gpu1
audit_script=/data/temp/ECT001/audit_q256_seed_extension.py
worker_log=${extension_root}/extension-worker-gpu1.log
lock_path=/data/temp/ECT001-q256-seed6-7-extension-gpu1.lock
active_seed=preflight
active_arm=none

umask 027

write_chain_failure() {
  local exit_code=$?
  if [[ -d "${extension_root}" && ! -e "${extension_root}/chain_failure_report.md" ]]; then
    {
      printf '# q256 seed6/7 extension chain failure\n\n'
      printf -- '- Classification: secondary precision extension, not original preregistration\n'
      printf -- '- UTC: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf -- '- Seed: %s\n' "${active_seed}"
      printf -- '- Arm: %s\n' "${active_arm}"
      printf -- '- Exit code: %s\n' "${exit_code}"
      printf -- '- Action: task chain stopped; no parameter modification and no automatic retry.\n'
    } >"${extension_root}/chain_failure_report.md"
  fi
  exit "${exit_code}"
}
trap write_chain_failure ERR

exec 9>"${lock_path}"
flock -n 9 || { echo 'extension GPU lock is already held' >&2; exit 2; }

[[ -d "${repo}" && -d "${sandbox}" ]] || { echo 'missing source or sandbox' >&2; exit 2; }
[[ -f "${dataset}" && -f "${transfer}" && -f "${audit_script}" ]] || { echo 'missing immutable asset or audit adapter' >&2; exit 2; }
[[ "$(cd "${repo}" && git rev-parse HEAD)" == "${expected_head}" ]] || { echo 'wrong source HEAD' >&2; exit 2; }
[[ -z "$(cd "${repo}" && git status --porcelain)" ]] || { echo 'training source is dirty' >&2; exit 2; }
[[ "$(sha256sum "${dataset}" | awk '{print $1}')" == "${dataset_sha}" ]] || { echo 'dataset hash mismatch' >&2; exit 2; }
[[ "$(sha256sum "${transfer}" | awk '{print $1}')" == "${transfer_sha}" ]] || { echo 'transfer hash mismatch' >&2; exit 2; }
[[ ! -e "${extension_root}" ]] || { echo 'refusing existing extension root' >&2; exit 3; }
[[ ! -e "${private_shm}" ]] || { echo 'refusing existing private shared-memory path' >&2; exit 3; }

gpu_identity=$(nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader,nounits | awk -F', ' -v wanted="${gpu_uuid}" '$1 == wanted {print $0}')
[[ "${gpu_identity}" == "${gpu_uuid}, NVIDIA A100 80GB PCIe, 81920" ]] || { echo "assigned GPU identity mismatch: ${gpu_identity}" >&2; exit 2; }
gpu_line=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' -v wanted="${gpu_uuid}" '$1 == wanted {print $0}')
[[ -n "${gpu_line}" ]] || { echo 'assigned GPU UUID is absent' >&2; exit 2; }
gpu_processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v wanted="${gpu_uuid}" '$1 == wanted {count++} END {print count+0}')
[[ "${gpu_processes}" == 0 ]] || { echo 'assigned GPU is not compute-idle' >&2; exit 2; }
raw_available=$(df --output=avail -B1 /data/raw/ECT/ect_runs | tail -n 1 | tr -d ' ')
[[ "${raw_available}" =~ ^[0-9]+$ && "${raw_available}" -ge 64424509440 ]] || { echo 'durable storage is below the 60 GiB gate' >&2; exit 2; }
if ss -H -ltn "sport = :${master_port}" | grep -q .; then
  echo "master port ${master_port} is already listening" >&2
  exit 2
fi

mkdir -p "$(dirname "${extension_root}")"
mkdir "${extension_root}"
mkdir "${extension_root}/integrity"
mkdir -m 700 "${private_shm}"
[[ "$(stat -c '%U:%a' "${private_shm}")" == 'ECT001:700' ]] || { echo 'unsafe private shared-memory ownership or mode' >&2; exit 2; }

exec >>"${worker_log}" 2>&1
echo "[extension] START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=${gpu_uuid} head=${expected_head} root=${extension_root}"
echo "[extension] SCOPE secondary_precision_extension_not_original_preregistration seeds=6,7 replaces_preregistered=false"
echo "[extension] PREFLIGHT gpu=${gpu_line} identity=${gpu_identity} raw_available=${raw_available} dataset_sha=${dataset_sha} transfer_sha=${transfer_sha}"
sha256sum "$0" "${audit_script}"

run_arm() {
  local seed=$1
  local arm=$2
  local target_scale=$3
  local denominator_scale=$4
  local outdir=${extension_root}/seed${seed}/arm${arm}
  active_seed=${seed}
  active_arm=${arm}
  [[ ! -e "${outdir}" ]] || { echo "refusing existing fresh cell: ${outdir}" >&2; return 3; }
  mkdir -p "${extension_root}/seed${seed}"
  echo "[extension] ARM_START seed=${seed} arm=${arm} target=${target_scale} denominator=${denominator_scale} mode=fresh"
  CUDA_VISIBLE_DEVICES="${gpu_uuid}" "${apptainer}" exec --nv \
    --bind /data/raw:/data/raw --bind /data/temp:/data/temp \
    --bind "${private_shm}:/dev/shm" \
    "${sandbox}" env \
    ECT_Q256_LAUNCHER_IN_SANDBOX=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_uuid}" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT="${master_port}" \
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 PYTHONUNBUFFERED=1 \
    python "${repo}/ct_train.py" \
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
  echo "[extension] ARM_PASS seed=${seed} arm=${arm} mode=fresh"
}

audit_seed() {
  local seed=$1
  active_seed=${seed}
  active_arm=integrity_audit
  echo "[extension] AUDIT_START seed=${seed}"
  CUDA_VISIBLE_DEVICES="${gpu_uuid}" "${apptainer}" exec --nv \
    --bind /data/raw:/data/raw --bind /data/temp:/data/temp \
    --bind "${private_shm}:/dev/shm" \
    "${sandbox}" env \
    Q256_EXTENSION_SOURCE_REPO="${repo}" \
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_uuid}" \
    PYTHONUNBUFFERED=1 \
    python "${audit_script}" audit --root "${extension_root}" --seed "${seed}"
  echo "[extension] AUDIT_PASS seed=${seed}"
}

run_seed() {
  local seed=$1
  run_arm "${seed}" A 1.0 1.0
  run_arm "${seed}" B 1.1 1.1
  run_arm "${seed}" C 1.1 1.0
  run_arm "${seed}" D 1.0 1.1
  audit_seed "${seed}"
}

run_seed 6
run_seed 7

active_seed=6,7
active_arm=extension_report
CUDA_VISIBLE_DEVICES="${gpu_uuid}" "${apptainer}" exec --nv \
  --bind /data/raw:/data/raw --bind /data/temp:/data/temp \
  --bind "${private_shm}:/dev/shm" \
  "${sandbox}" env \
  Q256_EXTENSION_SOURCE_REPO="${repo}" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_uuid}" \
  PYTHONUNBUFFERED=1 \
  python "${audit_script}" report --root "${extension_root}"

active_seed=complete
active_arm=none
trap - ERR
echo "[extension] TRAINING_EXTENSION_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
