#!/usr/bin/env bash
set -euo pipefail

gpu_uuid=GPU-d79117bb-8d91-4f2e-d7bb-718e347ce859
repo=/data/temp/ECT001/q256-factorial-clean-25c3d22
expected_head=dcca41b19e7c45512b5fbe98776520396a1bf9ac
formal_root=/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819/formal/formal-direct-dcca41b-deterministic-v1
evaluation_parent=/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819/evaluation
matrix_dir=${evaluation_parent}/matrix-binding-direct-dcca41b-v1
outdir=${evaluation_parent}/frozen-eval-dcca41b-primary-first-v1
log=${evaluation_parent}/formal-frozen-eval-dcca41b-primary-first-v1.log
adapter=/data/temp/ECT001/run_q256_direct_frozen_evaluation.py
adapter_sha=c10e20e779a6cf20895bce4ebb3fede5809f96a88faa6045747d0d57b3953ff8
sandbox=/data/temp/ect001-pytorch2401-sandbox
private_shm=/tmp/ECT001-q256-dcca41b-shm-gpu0

umask 027
[[ -d "${repo}" && -d "${formal_root}" && -d "${sandbox}" ]] || { echo 'missing frozen source, formal root, or sandbox' >&2; exit 2; }
[[ -f "${adapter}" && "$(sha256sum "${adapter}" | awk '{print $1}')" == "${adapter_sha}" ]] || { echo 'evaluation adapter hash mismatch' >&2; exit 2; }
[[ "$(cd "${repo}" && git rev-parse HEAD)" == "${expected_head}" ]] || { echo 'wrong evaluator source HEAD' >&2; exit 2; }
[[ -z "$(cd "${repo}" && git status --porcelain)" ]] || { echo 'evaluator source is dirty' >&2; exit 2; }
[[ -d "${private_shm}" && ! -L "${private_shm}" && "$(stat -c '%U:%a' "${private_shm}")" == 'ECT001:700' ]] || { echo 'invalid private shared-memory directory' >&2; exit 2; }
[[ ! -e "${matrix_dir}" && ! -e "${outdir}" ]] || { echo 'refusing an existing matrix binding or evaluation root' >&2; exit 3; }
gpu_processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v wanted="${gpu_uuid}" '$1 == wanted {count++} END {print count+0}')
[[ "${gpu_processes}" == 0 ]] || { echo 'formal GPU0 is not compute-idle' >&2; exit 2; }

mkdir -p "${evaluation_parent}"
exec >>"${log}" 2>&1
echo "[formal-evaluation] START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=${gpu_uuid} head=${expected_head} adapter_sha=${adapter_sha}"

CUDA_VISIBLE_DEVICES="${gpu_uuid}" /usr/bin/apptainer exec --nv \
  --bind /data/raw:/data/raw --bind /data/temp:/data/temp \
  --bind "${private_shm}:/dev/shm" \
  "${sandbox}" env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_uuid}" \
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  python "${adapter}" \
  --repo "${repo}" \
  --formal-root "${formal_root}" \
  --matrix-dir "${matrix_dir}" \
  --outdir "${outdir}" \
  --gpu "${gpu_uuid}"

echo "[formal-evaluation] PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
