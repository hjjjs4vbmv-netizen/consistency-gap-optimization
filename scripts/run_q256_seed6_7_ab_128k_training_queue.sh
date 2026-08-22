#!/usr/bin/env bash
set -euo pipefail

seed="${1:?usage: $0 SEED GPU_UUID MASTER_PORT}"
gpu_uuid="${2:?usage: $0 SEED GPU_UUID MASTER_PORT}"
master_port="${3:?usage: $0 SEED GPU_UUID MASTER_PORT}"

case "${seed}" in
  6|7) ;;
  *) echo "unsupported seed: ${seed}" >&2; exit 2 ;;
esac
[[ "${gpu_uuid}" =~ ^GPU-[A-Za-z0-9-]+$ ]] || { echo 'invalid GPU UUID' >&2; exit 2; }
[[ "${master_port}" =~ ^[0-9]+$ ]] || { echo 'invalid master port' >&2; exit 2; }

repo="${Q256_LC_REPO:?Q256_LC_REPO is required}"
expected_commit="${Q256_LC_EXPECTED_TRAINING_COMMIT:?Q256_LC_EXPECTED_TRAINING_COMMIT is required}"
source_root="${Q256_LC_SOURCE_ROOT:-/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819/secondary-precision-extension/seed6-7-dcca41b-v1}"
artifact_root="${Q256_LC_ARTIFACT_ROOT:-/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819/secondary-precision-extension/seed6-7-ab-128k-learning-curve-v1}"
dataset="${Q256_LC_DATASET:-/data/raw/ECT/datasets/cifar10-32x32-canonical-08c9ed1b2b1c.zip}"
dataset_sha=08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372
sandbox="${Q256_LC_SANDBOX:-/data/temp/ect001-pytorch2401-sandbox}"
apptainer="${Q256_LC_APPTAINER:-/usr/bin/apptainer}"
audit_script="${repo}/scripts/audit_q256_seed6_7_ab_128k_checkpoints.py"
private_shm="/dev/shm/ECT001-q256-seed${seed}-ab128k-train"
worker_log="${artifact_root}/logs/training-seed${seed}.log"
lock_path="/data/temp/ECT001-q256-seed${seed}-ab128k-training.lock"
active_arm=preflight
arm_started_epoch=0

umask 027

write_failure() {
  local exit_code=$?
  local timestamp
  timestamp=$(date -u +%Y%m%dT%H%M%SZ)
  mkdir -p "${artifact_root}/failures"
  {
    printf '# q256 seed6/7 A/B 128-kimg training failure\n\n'
    printf -- '- UTC: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf -- '- Seed: %s\n' "${seed}"
    printf -- '- Arm: %s\n' "${active_arm}"
    printf -- '- Exit code: %s\n' "${exit_code}"
    printf -- '- Training commit: %s\n' "${expected_commit}"
    printf -- '- Action: queue stopped; completed immutable checkpoints preserved.\n'
  } >"${artifact_root}/failures/training-seed${seed}-${active_arm}-${timestamp}-${BASHPID}.md"
  exit "${exit_code}"
}
trap write_failure ERR

exec 9>"${lock_path}"
flock -n 9 || { echo "training lock is already held for seed ${seed}" >&2; exit 2; }

[[ -d "${repo}/.git" && -d "${sandbox}" && -x "${apptainer}" ]] || { echo 'missing repo, sandbox, or apptainer' >&2; exit 2; }
[[ -f "${dataset}" && -f "${audit_script}" ]] || { echo 'missing dataset or audit script' >&2; exit 2; }
observed_commit=$(cd "${repo}" && git rev-parse HEAD)
[[ "${observed_commit}" == "${expected_commit}" ]] || { echo "wrong training commit: ${observed_commit}" >&2; exit 2; }
[[ -z "$(cd "${repo}" && git status --porcelain)" ]] || { echo 'training source is dirty' >&2; exit 2; }
[[ "$(sha256sum "${dataset}" | awk '{print $1}')" == "${dataset_sha}" ]] || { echo 'dataset SHA256 mismatch' >&2; exit 2; }
[[ -d "${source_root}" && -d "${artifact_root}/integrity" && -f "${artifact_root}/integrity/source_state_audit.json" ]] || { echo 'missing source/artifact preflight' >&2; exit 2; }

gpu_identity=$(nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader,nounits | awk -F', ' -v wanted="${gpu_uuid}" '$1 == wanted {print $0}')
[[ "${gpu_identity}" == "${gpu_uuid}, NVIDIA A100 80GB PCIe, 81920" ]] || { echo "GPU identity mismatch: ${gpu_identity}" >&2; exit 2; }
gpu_processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v wanted="${gpu_uuid}" '$1 == wanted {count++} END {print count+0}')
[[ "${gpu_processes}" == 0 ]] || { echo "GPU is not idle: ${gpu_uuid}" >&2; exit 2; }
if ss -H -ltn "sport = :${master_port}" | grep -q .; then
  echo "master port is already listening: ${master_port}" >&2
  exit 2
fi
[[ ! -e "${private_shm}" ]] || { echo "private shared-memory path already exists: ${private_shm}" >&2; exit 2; }
mkdir -m 700 "${private_shm}"
[[ "$(stat -c '%U:%a' "${private_shm}")" == 'ECT001:700' ]] || { echo 'unsafe private shared-memory ownership/mode' >&2; exit 2; }

exec >>"${worker_log}" 2>&1
echo "[q256-lc-training] WORKER_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) seed=${seed} gpu=${gpu_uuid} commit=${expected_commit}"

container_python() {
  CUDA_VISIBLE_DEVICES="${gpu_uuid}" "${apptainer}" exec --nv \
    --bind /data/raw:/data/raw --bind /data/temp:/data/temp \
    --bind "${private_shm}:/dev/shm" \
    "${sandbox}" env \
    PYTHONPATH="${repo}" \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    Q256_EXTENSION_SOURCE_REPO="${repo}" \
    python "$@"
}

run_arm() {
  local arm=$1
  local target_scale=$2
  local denominator_scale=$3
  local source_dir="${source_root}/seed${seed}/arm${arm}"
  local run_dir="${artifact_root}/seed${seed}/arm${arm}"
  local source_state="${source_dir}/training-state-latest.pt"
  local source_sha
  local resume_state
  local started_epoch
  local finished_epoch
  local wall_seconds

  active_arm="${arm}"
  [[ -f "${source_state}" ]] || { echo "missing source state: ${source_state}" >&2; return 3; }
  source_sha=$(sha256sum "${source_state}" | awk '{print $1}')
  [[ "${#source_sha}" == 64 ]] || { echo 'invalid source SHA256' >&2; return 3; }

  if [[ ! -e "${run_dir}" ]]; then
    mkdir -p "${artifact_root}/seed${seed}"
    mkdir "${run_dir}"
    cp -p "${source_dir}/training_options.json" "${run_dir}/"
    cp -p "${source_dir}/train_summary.csv" "${run_dir}/"
    cp -p "${source_dir}/factorial_training_telemetry_v1.csv" "${run_dir}/"
    cp -p "${source_dir}/initial_state_receipt_v1.json" "${run_dir}/"
    cp -p "${source_dir}/log.txt" "${run_dir}/"
  fi
  [[ -d "${run_dir}" && ! -L "${run_dir}" ]] || { echo "invalid run directory: ${run_dir}" >&2; return 3; }
  for required in training_options.json train_summary.csv factorial_training_telemetry_v1.csv initial_state_receipt_v1.json log.txt; do
    [[ -f "${run_dir}/${required}" ]] || { echo "missing ${run_dir}/${required}" >&2; return 3; }
  done

  resume_state=$(container_python "${audit_script}" select-resume \
    --source-root "${source_root}" --artifact-root "${artifact_root}" \
    --seed "${seed}" --arm "${arm}" --path-only)
  [[ -n "${resume_state}" && -f "${resume_state}" ]] || { echo 'resume selector returned no regular state' >&2; return 3; }
  echo "[q256-lc-training] ARM_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) seed=${seed} arm=${arm} resume=${resume_state} source_sha=${source_sha}"
  started_epoch=$(date +%s)

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
    --data="${dataset}" --outdir="${run_dir}" --nosubdir \
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
    --budget-checkpoint-interval-kimg=128 \
    --budget-checkpoint-start-kimg=384 \
    --budget-checkpoint-root="${run_dir}/checkpoints" \
    --budget-checkpoint-source-sha256="${source_sha}" \
    --budget-checkpoint-training-commit="${expected_commit}" \
    --resume="${resume_state}"

  finished_epoch=$(date +%s)
  wall_seconds=$((finished_epoch - started_epoch))
  [[ ! -e "${run_dir}/arm_runtime.json" ]] || { echo "refuse existing runtime receipt: ${run_dir}/arm_runtime.json" >&2; return 3; }
  printf '{\n  "schema": "ect.q256.seed6-7-ab-arm-runtime/v1",\n  "status": "PASS",\n  "seed": %s,\n  "arm": "%s",\n  "source_kimg": 256,\n  "final_kimg": 1024,\n  "wall_seconds": %s,\n  "gpu_hours": %.12f,\n  "gpu_uuid": "%s",\n  "training_commit": "%s",\n  "started_epoch": %s,\n  "finished_epoch": %s,\n  "finished_utc": "%s"\n}\n' \
    "${seed}" "${arm}" "${wall_seconds}" "$(awk -v seconds="${wall_seconds}" 'BEGIN {print seconds / 3600}')" \
    "${gpu_uuid}" "${expected_commit}" "${started_epoch}" "${finished_epoch}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${run_dir}/.arm_runtime.json.tmp-${BASHPID}"
  mv "${run_dir}/.arm_runtime.json.tmp-${BASHPID}" "${run_dir}/arm_runtime.json"
  echo "[q256-lc-training] ARM_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) seed=${seed} arm=${arm} wall_seconds=${wall_seconds}"
}

run_arm A 1.0 1.0
run_arm B 1.1 1.1

active_arm=inventory
container_python "${audit_script}" inventory \
  --artifact-root "${artifact_root}" \
  --source-audit "${artifact_root}/integrity/source_state_audit.json" \
  --training-commit "${expected_commit}" --seed "${seed}" \
  --output "${artifact_root}/integrity/seed${seed}_checkpoint_inventory.json"

active_arm=complete
trap - ERR
echo "[q256-lc-training] WORKER_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) seed=${seed}"
