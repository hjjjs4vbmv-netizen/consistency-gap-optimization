#!/usr/bin/env bash
set -Eeuo pipefail

manifest="${1:?usage: run_evaluation_job.sh MANIFEST JOB_INDEX GPU CACHE_ROOT EVAL_ROOT IMPLEMENTATION_REPO}"
job_index="${2:?missing job index}"; gpu="${3:?missing GPU}"; cache_root="${4:?missing cache root}"
eval_root="${5:?missing eval root}"; implementation_repo="${6:?missing implementation repo}"
mapfile -t fields < <(python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); j=p["jobs"][int(sys.argv[2])]; print(p["evaluator"]["repo"]); print(p["runtime_sif"]["path"]); print(p["dataset"]["path"]); [print(j[k] if j[k] is not None else "") for k in ("seed","branch","budget_kimg","nfe","mid_t","checkpoint_path","checkpoint_sha256")]' "${manifest}" "${job_index}")
[[ "${#fields[@]}" == 10 ]] || { echo "invalid evaluation manifest" >&2; exit 2; }
evaluator_repo="${fields[0]}"; runtime_sif="${fields[1]}"; dataset="${fields[2]}"
seed="${fields[3]}"; branch="${fields[4]}"; budget="${fields[5]}"; nfe="${fields[6]}"
mid_t="${fields[7]}"; checkpoint="${fields[8]}"; checkpoint_sha="${fields[9]}"
job_id="job$(printf '%03d' "${job_index}")-seed${seed}-${branch}-kimg${budget}-nfe${nfe}"
job_dir="${eval_root}/jobs/${job_id}"; job_cache="${eval_root}/job_caches/${job_id}"
receipt="${eval_root}/receipts/${job_id}.json"; log="${eval_root}/logs/${job_id}.process.log"
[[ ! -e "${job_dir}" && ! -e "${job_cache}" && ! -e "${receipt}" ]] || { echo "refuse existing evaluation job" >&2; exit 3; }
[[ "$(sha256sum "${checkpoint}" | awk '{print $1}')" == "${checkpoint_sha}" ]] || { echo "checkpoint hash mismatch" >&2; exit 3; }
gpu_uuid="$(nvidia-smi --id="${gpu}" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
if [[ "${P2_ALLOW_COTENANCY:-0}" != 1 ]] && nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -q "^${gpu_uuid},"; then echo "evaluation GPU occupied" >&2; exit 3; fi
mkdir -p "${eval_root}/jobs" "${eval_root}/job_caches" "${eval_root}/receipts" "${eval_root}/logs"
mkdir "${job_cache}"
cp -a "${cache_root}/." "${job_cache}/"
failure="${eval_root}/receipts/${job_id}.FAILED.json"
on_failure() {
  code=$?
  python3 -c 'import json,os,sys,time; d={"schema":"ect.q256.p2-evaluation-failure/v1","status":"FAILED","job_index":int(sys.argv[2]),"gpu_index":int(sys.argv[3]),"gpu_uuid":sys.argv[4],"exit_code":int(sys.argv[5]),"ended_unix":int(time.time())}; f=open(sys.argv[1],"x"); json.dump(d,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())' "${failure}" "${job_index}" "${gpu}" "${gpu_uuid}" "${code}" || true
  exit "${code}"
}
trap on_failure ERR
mid_arg=""; [[ "${nfe}" == 2 ]] && mid_arg=--mid_t=0.821
master_port=$((51000 + gpu * 100 + job_index))
started="$(date +%s)"
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    DNNLIB_CACHE_DIR="${job_cache}" MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="${master_port}" RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
  timeout --signal=TERM --kill-after=30s 6h \
  apptainer exec --nv --bind /data:/data --pwd "${evaluator_repo}" "${runtime_sif}" \
  python ct_eval.py --resume "${checkpoint}" --outdir "${job_dir}" --nosubdir \
    --data "${dataset}" --cond=False --arch=ddpmpp --precond=ct \
    --dropout=0.2 --augment=0 --xflip=False --fp16=False \
    --cache=True --workers=3 --eval-batch=512 --metric-generator-batch=128 \
    --nfe="${nfe}" ${mid_arg} --metrics=kid50k_full,fid50k_full \
    --metric-repeats=1 --sample-seeds=0-49999 --seed=20260730 \
    --retain-generated-artifacts --desc="q256-p2-${job_id}" \
  >"${log}" 2>&1
elapsed=$(( $(date +%s) - started ))
apptainer exec --bind /data:/data --pwd "${implementation_repo}" "${runtime_sif}" python \
  analysis/q256_p2_b384_pulse_chase_v1/seal_evaluation_job.py \
    --manifest "${manifest}" --job-index "${job_index}" --job-dir "${job_dir}" \
    --job-cache "${job_cache}" --receipt "${receipt}" --gpu-index "${gpu}" \
    --gpu-uuid "${gpu_uuid}" --elapsed-seconds "${elapsed}"
trap - ERR
echo "[P2 evaluation] SEALED_PASS job=${job_index} gpu=${gpu} feature-only receipt=${receipt}"
