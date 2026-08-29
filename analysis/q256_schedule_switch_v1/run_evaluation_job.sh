#!/usr/bin/env bash
set -Eeuo pipefail

manifest="${1:?usage: $0 MANIFEST JOB_INDEX GPU_ID EVALUATOR_REPO RUNTIME_ROOTFS RUNTIME_SIF CACHE_ROOT EVAL_ROOT IMPLEMENTATION_REPO}"
job_index="${2:?missing job index}"
gpu_id="${3:?missing GPU index}"
evaluator_repo="${4:?missing evaluator repository}"
runtime_rootfs="${5:?missing runtime rootfs}"
runtime_sif="${6:?missing runtime SIF}"
cache_root="${7:?missing evaluator cache}"
eval_root="${8:?missing evaluation output root}"
implementation_repo="${9:?missing implementation repository}"
source "$(dirname "${BASH_SOURCE[0]}")/runtime_env.sh" "${runtime_rootfs}"

[[ "${job_index}" =~ ^[0-9]+$ && "${gpu_id}" =~ ^[0-9]+$ ]] || { echo "invalid numeric argument" >&2; exit 2; }
[[ -f "${manifest}" && -d "${runtime_rootfs}" && -f "${runtime_sif}" && -d "${cache_root}/downloads" ]] || { echo "missing frozen evaluator input" >&2; exit 2; }
[[ "$(git -C "${evaluator_repo}" rev-parse HEAD)" == d6aba02fb88e9db0993623895eb2228ed717d810 ]] || { echo "evaluator commit mismatch" >&2; exit 2; }
[[ -z "$(git -C "${evaluator_repo}" status --porcelain)" ]] || { echo "evaluator source is dirty" >&2; exit 2; }

mapfile -t fields < <(python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["metrics_executed"] is False and p["job_count"]==80; j=p["jobs"][int(sys.argv[2])]; [print(j[k]) for k in ("seed","branch","budget_kimg","nfe","checkpoint_path","checkpoint_sha256","dataset_path","mid_t")]' "${manifest}" "${job_index}")
[[ "${#fields[@]}" == 8 ]] || { echo "invalid frozen job" >&2; exit 3; }
seed="${fields[0]}"
branch="${fields[1]}"
budget="${fields[2]}"
nfe="${fields[3]}"
checkpoint="${fields[4]}"
checkpoint_sha="${fields[5]}"
dataset="${fields[6]}"
mid_t="${fields[7]}"
job_id="seed${seed}-${branch}-kimg${budget}-nfe${nfe}"
target="${eval_root}/jobs/${job_id}"
receipt="${eval_root}/receipts/${job_id}.json"
process_log="${eval_root}/logs/${job_id}.process.log"
job_cache="${eval_root}/job_caches/${job_id}"
[[ -f "${checkpoint}" && ! -L "${checkpoint}" ]] || { echo "missing checkpoint" >&2; exit 3; }
[[ "$(sha256sum "${checkpoint}" | awk '{print $1}')" == "${checkpoint_sha}" ]] || { echo "checkpoint hash mismatch" >&2; exit 3; }
[[ ! -e "${target}" && ! -e "${receipt}" ]] || { echo "refuse existing evaluation output" >&2; exit 4; }
mkdir -p "${eval_root}/jobs" "${eval_root}/receipts" "${eval_root}/logs" "${job_cache}"
cp -a "${cache_root}/." "${job_cache}/"

gpu_uuid=$(nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader | awk -F', ' -v index="${gpu_id}" '$1 == index && $3 ~ /A100/ {print $2}')
[[ -n "${gpu_uuid}" ]] || { echo "cannot resolve assigned A100 UUID" >&2; exit 4; }
master_port=$((52000 + gpu_id * 100 + job_index))
mid_args=()
if [[ "${nfe}" == 2 ]]; then
  [[ "${mid_t}" == 0.821 ]] || { echo "NFE2 mid_t mismatch" >&2; exit 4; }
  mid_args+=(--mid_t=0.821)
else
  [[ -z "${mid_t}" ]] || { echo "NFE1 must not have mid_t" >&2; exit 4; }
fi

started=$(date +%s)
echo "[q256-switch-eval] START job=${job_id} gpu=${gpu_id} uuid=${gpu_uuid}"
timeout --signal=TERM --kill-after=30s 6h \
  env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_id}" \
      PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
      DNNLIB_CACHE_DIR="${job_cache}" MASTER_ADDR=127.0.0.1 \
      MASTER_PORT="${master_port}" RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
      LD_LIBRARY_PATH="${Q256_RUNTIME_LD_LIBRARY_PATH}" PATH="${Q256_RUNTIME_PATH}" \
    "${Q256_RUNTIME_PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=1 \
      --master_port="${master_port}" "${evaluator_repo}/ct_eval.py" \
      --resume "${checkpoint}" --outdir "${target}" --nosubdir \
      --data "${dataset}" --cond=False --arch=ddpmpp --precond=ct \
      --dropout=0.2 --augment=0 --xflip=False --fp16=False \
      --cache=True --workers=1 --eval-batch=512 --metric-generator-batch=128 \
      --nfe="${nfe}" "${mid_args[@]}" \
      --metrics=kid50k_full,fid50k_full --metric-repeats=1 \
      --sample-seeds=0-49999 --seed=20260730 --retain-generated-artifacts \
      --desc="q256-schedule-switch-${job_id}" \
  2>&1 | tee "${process_log}"
elapsed=$(( $(date +%s) - started ))

python "${implementation_repo}/analysis/q256_schedule_switch_v1/validate_evaluation_job.py" \
  --evaluation-manifest "${manifest}" --job-index "${job_index}" \
  --job-dir "${target}" --receipt "${receipt}" \
  --evaluator-repo "${evaluator_repo}" --runtime-sif "${runtime_sif}" \
  --runtime-receipt "${eval_root}/runtime_integrity.json" \
  --gpu-index "${gpu_id}" --gpu-uuid "${gpu_uuid}" \
  --elapsed-seconds "${elapsed}"
echo "[q256-switch-eval] PASS job=${job_id} elapsed=${elapsed}"
