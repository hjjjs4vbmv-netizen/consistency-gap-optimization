#!/usr/bin/env bash
set -euo pipefail

bundle=/data/temp/ECT001/q256-seed8-13-v1
extension_root=/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819/secondary-precision-extension/seed8-13-dcca41b-v1
audit_script=${bundle}/audit_q256_seed8_13_extension.py
worker=${bundle}/run_q256_seed8_13_worker.sh
evaluation=${bundle}/run_q256_seed8_13_frozen_evaluation.sh
gpu0=GPU-d79117bb-8d91-4f2e-d7bb-718e347ce859
gpu1=GPU-ef9edaf6-d661-e143-efd1-154c1ad29f10
chain_log=${extension_root}/chain.log
active_phase=preflight

umask 027

write_chain_failure() {
  local exit_code=$?
  if [[ -d "${extension_root}" && ! -e "${extension_root}/chain_failure_report.md" ]]; then
    {
      printf '# q256 seed8-13 extension chain failure\n\n'
      printf -- '- Classification: secondary precision extension, not original preregistration\n'
      printf -- '- UTC: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf -- '- Phase: %s\n' "${active_phase}"
      printf -- '- Exit code: %s\n' "${exit_code}"
      printf -- '- Action: chain stopped; no parameter modification and no automatic retry.\n'
    } >"${extension_root}/chain_failure_report.md"
  fi
  exit "${exit_code}"
}
trap write_chain_failure ERR

[[ ! -e "${extension_root}" ]] || { echo 'refusing existing seed8-13 extension root' >&2; exit 3; }
for path in "${audit_script}" "${worker}" "${evaluation}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || { echo "missing regular bundle file: ${path}" >&2; exit 2; }
done
raw_available=$(df --output=avail -B1 /data/raw/ECT/ect_runs | tail -n 1 | tr -d ' ')
[[ "${raw_available}" =~ ^[0-9]+$ && "${raw_available}" -ge 274877906944 ]] || { echo 'durable storage is below the 256 GiB gate' >&2; exit 2; }
for gpu in "${gpu0}" "${gpu1}"; do
  count=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v wanted="${gpu}" '$1 == wanted {count++} END {print count+0}')
  [[ "${count}" == 0 ]] || { echo "GPU is not compute-idle: ${gpu}" >&2; exit 2; }
done

mkdir -p "$(dirname "${extension_root}")"
mkdir "${extension_root}"
mkdir "${extension_root}/integrity"
exec >>"${chain_log}" 2>&1
echo "[extension-chain] START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) seeds=8,9,10,11,12,13"
echo '[extension-chain] CLASSIFICATION secondary_precision_extension_not_original_preregistration replaces_preregistered=false'
sha256sum "${audit_script}" "${worker}" "${evaluation}" "${bundle}/run_q256_seed8_13_frozen_evaluation.py"

active_phase=parallel_training
bash "${worker}" "${gpu0}" 29781 gpu0 8 9 10 >"${extension_root}/worker-gpu0.log" 2>&1 &
pid0=$!
bash "${worker}" "${gpu1}" 29782 gpu1 11 12 13 >"${extension_root}/worker-gpu1.log" 2>&1 &
pid1=$!
printf '%s\n' "${pid0}" >"${extension_root}/worker-gpu0.pid"
printf '%s\n' "${pid1}" >"${extension_root}/worker-gpu1.pid"

set +e
wait "${pid0}"
rc0=$?
wait "${pid1}"
rc1=$?
set -e
[[ "${rc0}" == 0 && "${rc1}" == 0 ]] || { echo "training worker failure: gpu0=${rc0} gpu1=${rc1}" >&2; exit 4; }

active_phase=combined_training_report
CUDA_VISIBLE_DEVICES="${gpu0}" /usr/bin/apptainer exec --nv \
  --bind /data/raw:/data/raw --bind /data/temp:/data/temp \
  --bind /tmp/ECT001-q256-seed8-13-extension-shm-gpu0:/dev/shm \
  /data/temp/ect001-pytorch2401-sandbox env \
  Q256_EXTENSION_SOURCE_REPO=/data/temp/ECT001/q256-factorial-clean-25c3d22 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu0}" \
  PYTHONUNBUFFERED=1 \
  python "${audit_script}" report --root "${extension_root}"

active_phase=frozen_evaluation
bash "${evaluation}"

active_phase=complete
trap - ERR
echo "[extension-chain] FULL_CHAIN_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
