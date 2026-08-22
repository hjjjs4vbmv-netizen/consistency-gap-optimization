#!/usr/bin/env bash
set -euo pipefail

tool_repo="${Q256_LC_TOOL_REPO:?Q256_LC_TOOL_REPO is required}"
expected_tool_commit="${Q256_LC_EXPECTED_TOOL_COMMIT:?Q256_LC_EXPECTED_TOOL_COMMIT is required}"
artifact_root="${Q256_LC_ARTIFACT_ROOT:-/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819/secondary-precision-extension/seed6-7-ab-128k-learning-curve-v1}"
evaluator_repo="${Q256_LC_EVALUATOR_REPO:-/data/temp/ECT001/q256-factorial-eval-feature-reuse-v6}"
expected_evaluator_head=9d06ccc72545d4189af1b86de7f629f9c09d3f73
formal_adapter="${Q256_LC_FORMAL_ADAPTER:-/data/temp/ECT001/run_q256_direct_frozen_evaluation_v6.py}"
formal_adapter_sha=7e687c7664fdd204153f658539393c6ef6dc7e4fb1c62d54d37414433f13b67f
cache_source="${Q256_LC_CACHE_SOURCE:-/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819/secondary-precision-extension/seed6-7-dcca41b-v1/frozen-evaluation/run-primary-first-v1/evaluator_cache}"
baseline_eval_root="${Q256_LC_BASELINE_EVAL_ROOT:-/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819/secondary-precision-extension/seed6-7-dcca41b-v1/frozen-evaluation/run-primary-first-v1}"
dataset=/data/raw/ECT/datasets/cifar10-32x32-canonical-08c9ed1b2b1c.zip
sandbox="${Q256_LC_SANDBOX:-/data/temp/ect001-pytorch2401-sandbox}"
apptainer="${Q256_LC_APPTAINER:-/usr/bin/apptainer}"
gpu_uuid=GPU-d79117bb-8d91-4f2e-d7bb-718e347ce859
private_shm=/dev/shm/ECT001-q256-seed6-7-ab128k-eval-v1
matrix_dir="${artifact_root}/evaluation/matrix-binding-v1"
durable_root="${artifact_root}/evaluation/run-primary-nfe1-v1"
adapter="${tool_repo}/scripts/run_q256_seed6_7_ab_128k_frozen_evaluation.py"
compactor="${tool_repo}/scripts/compact_q256_seed6_7_ab_128k_evaluation.py"
collector="${tool_repo}/scripts/collect_q256_seed6_7_ab_128k_learning_curve.py"
worker_log="${artifact_root}/evaluation/run-primary-nfe1-v1.log"
lock_path=/data/temp/ECT001-q256-seed6-7-ab128k-evaluation.lock
active_phase=preflight

umask 027

write_failure() {
  local exit_code=$?
  local timestamp
  timestamp=$(date -u +%Y%m%dT%H%M%SZ)
  mkdir -p "${artifact_root}/failures"
  {
    printf '# q256 seed6/7 A/B 128-kimg evaluation failure\n\n'
    printf -- '- UTC: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf -- '- Phase: %s\n' "${active_phase}"
    printf -- '- Exit code: %s\n' "${exit_code}"
    printf -- '- Tool commit: %s\n' "${expected_tool_commit}"
    printf -- '- Action: evaluation stopped; completed PASS receipts preserved.\n'
  } >"${artifact_root}/failures/evaluation-${active_phase}-${timestamp}-${BASHPID}.md"
  exit "${exit_code}"
}
trap write_failure ERR

exec 9>"${lock_path}"
flock -n 9 || { echo 'evaluation lock is already held' >&2; exit 2; }

[[ -d "${tool_repo}/.git" && -d "${evaluator_repo}/.git" && -d "${sandbox}" ]] || { echo 'missing tool/evaluator repo or sandbox' >&2; exit 2; }
[[ -f "${adapter}" && -f "${compactor}" && -f "${collector}" && -f "${formal_adapter}" && -d "${cache_source}" && -d "${baseline_eval_root}" && -f "${dataset}" ]] || { echo 'missing evaluation input' >&2; exit 2; }
[[ "$(cd "${tool_repo}" && git rev-parse HEAD)" == "${expected_tool_commit}" ]] || { echo 'wrong tool commit' >&2; exit 2; }
[[ -z "$(cd "${tool_repo}" && git status --porcelain)" ]] || { echo 'tool repo is dirty' >&2; exit 2; }
[[ "$(cd "${evaluator_repo}" && git rev-parse HEAD)" == "${expected_evaluator_head}" ]] || { echo 'wrong evaluator commit' >&2; exit 2; }
[[ -z "$(cd "${evaluator_repo}" && git status --porcelain)" ]] || { echo 'evaluator repo is dirty' >&2; exit 2; }
[[ "$(sha256sum "${formal_adapter}" | awk '{print $1}')" == "${formal_adapter_sha}" ]] || { echo 'formal numerical adapter hash mismatch' >&2; exit 2; }
for seed in 6 7; do
  receipt="${artifact_root}/integrity/seed${seed}_checkpoint_inventory.json"
  [[ -f "${receipt}" ]] || { echo "missing checkpoint inventory: ${receipt}" >&2; exit 3; }
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="PASS" and d["checkpoint_count"]==12' "${receipt}"
done
[[ ! -e "${matrix_dir}" && ! -e "${durable_root}" && ! -e "${private_shm}" ]] || { echo 'refuse existing evaluation output path' >&2; exit 3; }
gpu_processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v wanted="${gpu_uuid}" '$1 == wanted {count++} END {print count+0}')
[[ "${gpu_processes}" == 0 ]] || { echo 'evaluation GPU0 is not idle' >&2; exit 2; }
shm_available=$(df --output=avail -B1 /dev/shm | tail -n 1 | tr -d ' ')
[[ "${shm_available}" =~ ^[0-9]+$ && "${shm_available}" -ge 34359738368 ]] || { echo 'shared memory is below the 32-GiB evaluation gate' >&2; exit 2; }

mkdir -m 700 "${private_shm}"
exec >>"${worker_log}" 2>&1
echo "[q256-lc-evaluation] START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=${gpu_uuid} tool_commit=${expected_tool_commit}"

active_phase=bind
CUDA_VISIBLE_DEVICES="${gpu_uuid}" "${apptainer}" exec --nv \
  --bind /data/raw:/data/raw --bind /data/temp:/data/temp \
  --bind "${private_shm}:/dev/shm" \
  "${sandbox}" env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_uuid}" \
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  python "${adapter}" --repo "${evaluator_repo}" --artifact-root "${artifact_root}" \
  --matrix-dir "${matrix_dir}" --outdir /dev/shm/run-primary \
  --gpu "${gpu_uuid}" --formal-adapter "${formal_adapter}" \
  --cache-source "${cache_source}" --bind-only

active_phase=evaluate
CUDA_VISIBLE_DEVICES="${gpu_uuid}" "${apptainer}" exec --nv \
  --bind /data/raw:/data/raw --bind /data/temp:/data/temp \
  --bind "${private_shm}:/dev/shm" \
  "${sandbox}" env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_uuid}" \
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  python "${adapter}" --repo "${evaluator_repo}" --artifact-root "${artifact_root}" \
  --matrix-dir "${matrix_dir}" --outdir /dev/shm/run-primary \
  --gpu "${gpu_uuid}" --formal-adapter "${formal_adapter}" \
  --cache-source "${cache_source}" --reuse-bound-matrix

active_phase=compact
"${apptainer}" exec --bind /data/raw:/data/raw --bind /data/temp:/data/temp \
  --bind "${private_shm}:/dev/shm" \
  "${sandbox}" env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  python "${compactor}" --ephemeral-root /dev/shm/run-primary \
  --durable-root "${durable_root}" --tool-commit "${expected_tool_commit}" \
  --delete-ephemeral-on-pass
rmdir "${private_shm}"

active_phase=collect
python3 "${collector}" --artifact-root "${artifact_root}" \
  --baseline-eval-root "${baseline_eval_root}"

active_phase=complete
trap - ERR
echo "[q256-lc-evaluation] PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) jobs=24"
