#!/usr/bin/env bash
set -euo pipefail

gpu_uuid=GPU-ef9edaf6-d661-e143-efd1-154c1ad29f10
repo=/data/temp/ECT001/q256-factorial-eval-feature-reuse-v6
training_head=dcca41b19e7c45512b5fbe98776520396a1bf9ac
extension_root=/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819/secondary-precision-extension/seed6-7-dcca41b-v1
evaluation_parent=${extension_root}/frozen-evaluation
matrix_dir=${evaluation_parent}/matrix-binding-v1
outdir=${evaluation_parent}/run-primary-first-v1
log=${evaluation_parent}/run-primary-first-v1.log
adapter=/data/temp/ECT001/run_q256_seed6_7_frozen_evaluation.py
formal_adapter=/data/temp/ECT001/run_q256_direct_frozen_evaluation_v6.py
sandbox=/data/temp/ect001-pytorch2401-sandbox
private_shm=/tmp/ECT001-q256-seed6-7-extension-shm-gpu1

umask 027
[[ -d "${repo}" && -d "${extension_root}" && -d "${sandbox}" ]] || { echo 'missing evaluator, extension root, or sandbox' >&2; exit 2; }
[[ -f "${adapter}" && -f "${formal_adapter}" ]] || { echo 'missing extension or frozen numerical adapter' >&2; exit 2; }
[[ "$(cd "${repo}" && git symbolic-ref --quiet --short HEAD)" == 'experiment/q256-target-weight-factorial' ]] || { echo 'wrong evaluator branch' >&2; exit 2; }
[[ -z "$(cd "${repo}" && git status --porcelain --untracked-files=all)" ]] || { echo 'evaluator source is dirty' >&2; exit 2; }
(cd "${repo}" && git merge-base --is-ancestor "${training_head}" HEAD) || { echo 'evaluator is not based on the frozen training commit' >&2; exit 2; }
[[ -d "${private_shm}" && ! -L "${private_shm}" && "$(stat -c '%U:%a' "${private_shm}")" == 'ECT001:700' ]] || { echo 'invalid private shared-memory directory' >&2; exit 2; }
[[ ! -e "${evaluation_parent}" ]] || { echo 'refusing an existing extension evaluation root' >&2; exit 3; }
gpu_processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v wanted="${gpu_uuid}" '$1 == wanted {count++} END {print count+0}')
[[ "${gpu_processes}" == 0 ]] || { echo 'extension GPU1 is not compute-idle' >&2; exit 2; }

mkdir -m 0750 "${evaluation_parent}"
exec >>"${log}" 2>&1
echo "[extension-evaluation] START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=${gpu_uuid} evaluator_head=$(cd "${repo}" && git rev-parse HEAD) training_head=${training_head}"

CUDA_VISIBLE_DEVICES="${gpu_uuid}" /usr/bin/apptainer exec --nv \
  --bind /data/raw:/data/raw --bind /data/temp:/data/temp \
  --bind "${private_shm}:/dev/shm" \
  "${sandbox}" env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_uuid}" \
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  python "${adapter}" \
  --repo "${repo}" \
  --extension-root "${extension_root}" \
  --matrix-dir "${matrix_dir}" \
  --outdir "${outdir}" \
  --gpu "${gpu_uuid}" \
  --formal-adapter "${formal_adapter}" \
  --bind-only

CUDA_VISIBLE_DEVICES="${gpu_uuid}" /usr/bin/apptainer exec --nv \
  --bind /data/raw:/data/raw --bind /data/temp:/data/temp \
  --bind "${private_shm}:/dev/shm" \
  "${sandbox}" env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_uuid}" \
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  python "${adapter}" \
  --repo "${repo}" \
  --extension-root "${extension_root}" \
  --matrix-dir "${matrix_dir}" \
  --outdir "${outdir}" \
  --gpu "${gpu_uuid}" \
  --formal-adapter "${formal_adapter}" \
  --reuse-bound-matrix

echo "[extension-evaluation] PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
