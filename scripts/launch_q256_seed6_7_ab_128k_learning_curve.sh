#!/usr/bin/env bash
set -euo pipefail

repo="${Q256_LC_REPO:?Q256_LC_REPO is required}"
expected_commit="${Q256_LC_EXPECTED_TRAINING_COMMIT:?Q256_LC_EXPECTED_TRAINING_COMMIT is required}"
source_root="${Q256_LC_SOURCE_ROOT:-/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819/secondary-precision-extension/seed6-7-dcca41b-v1}"
artifact_root="${Q256_LC_ARTIFACT_ROOT:-/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819/secondary-precision-extension/seed6-7-ab-128k-learning-curve-v1}"
sandbox="${Q256_LC_SANDBOX:-/data/temp/ect001-pytorch2401-sandbox}"
apptainer="${Q256_LC_APPTAINER:-/usr/bin/apptainer}"
dataset="${Q256_LC_DATASET:-/data/raw/ECT/datasets/cifar10-32x32-canonical-08c9ed1b2b1c.zip}"
dataset_sha=08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372
gpu0=GPU-d79117bb-8d91-4f2e-d7bb-718e347ce859
gpu1=GPU-ef9edaf6-d661-e143-efd1-154c1ad29f10
session6=q256_seed6_ab_128k
session7=q256_seed7_ab_128k
queue_script="${repo}/scripts/run_q256_seed6_7_ab_128k_training_queue.sh"
audit_script="${repo}/scripts/audit_q256_seed6_7_ab_128k_checkpoints.py"

umask 027

[[ -d "${repo}/.git" && -d "${source_root}" && -d "${sandbox}" && -x "${apptainer}" ]] || { echo 'missing repo, source, sandbox, or apptainer' >&2; exit 2; }
[[ -f "${queue_script}" && -f "${audit_script}" && -f "${dataset}" ]] || { echo 'missing queue, audit, or dataset' >&2; exit 2; }
[[ "$(cd "${repo}" && git rev-parse HEAD)" == "${expected_commit}" ]] || { echo 'wrong source commit' >&2; exit 2; }
[[ -z "$(cd "${repo}" && git status --porcelain)" ]] || { echo 'source worktree is dirty' >&2; exit 2; }
[[ "$(sha256sum "${dataset}" | awk '{print $1}')" == "${dataset_sha}" ]] || { echo 'dataset SHA256 mismatch' >&2; exit 2; }
[[ ! -e "${artifact_root}" ]] || { echo "refuse existing artifact root: ${artifact_root}" >&2; exit 3; }
[[ ! -e /dev/shm/ECT001-q256-seed6-ab128k-train && ! -e /dev/shm/ECT001-q256-seed7-ab128k-train ]] || { echo 'private training shm path already exists' >&2; exit 3; }
tmux has-session -t "${session6}" 2>/dev/null && { echo "tmux exists: ${session6}" >&2; exit 3; }
tmux has-session -t "${session7}" 2>/dev/null && { echo "tmux exists: ${session7}" >&2; exit 3; }

for gpu in "${gpu0}" "${gpu1}"; do
  identity=$(nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader,nounits | awk -F', ' -v wanted="${gpu}" '$1 == wanted {print $0}')
  [[ "${identity}" == "${gpu}, NVIDIA A100 80GB PCIe, 81920" ]] || { echo "GPU identity mismatch: ${identity}" >&2; exit 2; }
  processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v wanted="${gpu}" '$1 == wanted {count++} END {print count+0}')
  [[ "${processes}" == 0 ]] || { echo "GPU is not idle: ${gpu}" >&2; exit 2; }
done

raw_available=$(df --output=avail -B1 /data/raw/ECT/ect_runs | tail -n 1 | tr -d ' ')
shm_available=$(df --output=avail -B1 /dev/shm | tail -n 1 | tr -d ' ')
[[ "${raw_available}" =~ ^[0-9]+$ && "${raw_available}" -ge 85899345920 ]] || { echo 'durable storage is below the 80-GiB launch gate' >&2; exit 2; }
[[ "${shm_available}" =~ ^[0-9]+$ && "${shm_available}" -ge 68719476736 ]] || { echo 'shared memory is below the 64-GiB launch gate' >&2; exit 2; }

mkdir -p "$(dirname "${artifact_root}")"
mkdir "${artifact_root}"
for directory in integrity logs failures evaluation reports; do
  mkdir "${artifact_root}/${directory}"
done

"${apptainer}" exec --nv \
  --bind /data/raw:/data/raw --bind /data/temp:/data/temp \
  "${sandbox}" env PYTHONPATH="${repo}" \
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  python "${audit_script}" source --source-root "${source_root}" \
  --output "${artifact_root}/integrity/source_state_audit.json" \
  >"${artifact_root}/logs/source-state-audit.log" 2>&1

printf '{\n  "schema": "ect.q256.seed6-7-ab-128k-learning-curve-contract/v1",\n  "status": "AUTHORIZED",\n  "extension_classification": "secondary_precision_extension_not_original_preregistration",\n  "replaces_preregistered_seed": false,\n  "training_commit": "%s",\n  "training_numerical_base_commit": "458205192722883df393a8d017c26e6fa46f48f7",\n  "source_root": "%s",\n  "artifact_root": "%s",\n  "seeds": [6, 7],\n  "arms": ["A", "B"],\n  "budgets_kimg": [384, 512, 640, 768, 896, 1024],\n  "new_checkpoint_count": 24,\n  "evaluation_job_count": 24,\n  "evaluation_nfe": 1,\n  "evaluation_sample_count": 50000,\n  "evaluation_sample_seed_range": "0-49999",\n  "metric_seed": 20260730,\n  "metrics": ["kid50k_full", "fid50k_full"],\n  "gpu_assignment": {"seed6": "%s", "seed7": "%s"},\n  "created_utc": "%s"\n}\n' \
  "${expected_commit}" "${source_root}" "${artifact_root}" "${gpu0}" "${gpu1}" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${artifact_root}/experiment_contract.json"

common_env="Q256_LC_REPO=${repo} Q256_LC_EXPECTED_TRAINING_COMMIT=${expected_commit} Q256_LC_SOURCE_ROOT=${source_root} Q256_LC_ARTIFACT_ROOT=${artifact_root} Q256_LC_SANDBOX=${sandbox} Q256_LC_APPTAINER=${apptainer} Q256_LC_DATASET=${dataset}"
tmux new-session -d -s "${session6}" "env ${common_env} bash ${queue_script} 6 ${gpu0} 33660"
tmux new-session -d -s "${session7}" "env ${common_env} bash ${queue_script} 7 ${gpu1} 33770"

sleep 2
tmux has-session -t "${session6}" 2>/dev/null || { echo "seed6 tmux failed to stay alive" >&2; exit 4; }
tmux has-session -t "${session7}" 2>/dev/null || { echo "seed7 tmux failed to stay alive" >&2; exit 4; }

echo "[q256-lc-launch] PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) root=${artifact_root} seed6_gpu=${gpu0} seed7_gpu=${gpu1}"
