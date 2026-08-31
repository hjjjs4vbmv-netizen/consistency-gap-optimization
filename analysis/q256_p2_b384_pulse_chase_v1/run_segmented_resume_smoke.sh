#!/usr/bin/env bash
set -Eeuo pipefail
gpu="${1:?usage: run_segmented_resume_smoke.sh GPU REPO DATASET TRANSFER RUNTIME_SIF OUTPUT_ROOT}"
repo="${2:?missing repo}"; dataset="${3:?missing dataset}"; transfer="${4:?missing transfer}"
runtime_sif="${5:?missing runtime}"; output="${6:?missing output root}"
[[ ! -e "${output}" ]] || { echo "smoke output already exists" >&2; exit 3; }
mkdir "${output}"; mkdir "${output}/uninterrupted" "${output}/segmented"
common=(--data="${dataset}" --cond=False --arch=ddpmpp --precond=ect --batch=128 --batch-gpu=16
  --optim=RAdam --lr=0.0001 --dropout=0.2 --augment=0 --xflip=False
  --mean=-1.1 --std=2.0 --mapping=sigmoid --global-gap-scale=1.0
  --factorial-protocol=q256_target_weight_v1 --target-gap-scale=1.1 --denominator-gap-scale=1.1
  -q 256 -k 8 -b 1 -c 0 --double=10000 --ema_beta=0.9993 --seed=17
  --fp16=True --tf32=False --ls=1.0 --enable_amp=True --bench=False --cache=True
  --workers=1 --metrics=none --duration=0.004 --tick=10 --snap=0 --dump=0 --ckpt=10
  --sample_every=26 --eval_every=50 --mid_t=0.821 --adaptive-update-kimg=0.5)
run_train() {
  local port="$1"; shift
  env CUDA_VISIBLE_DEVICES="${gpu}" CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_CACHE_DISABLE=1 \
      PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MASTER_ADDR=127.0.0.1 MASTER_PORT="${port}" \
      RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
    apptainer exec --nv --bind /data:/data --pwd "${repo}" "${runtime_sif}" \
    python ct_train.py "${common[@]}" "$@"
}
run_train 46117 --outdir="${output}/uninterrupted" --nosubdir --transfer="${transfer}" \
  >"${output}/uninterrupted.process.log" 2>&1
run_train 46217 --outdir="${output}/segmented" --nosubdir --transfer="${transfer}" \
  --stop-after-attempts=16 >"${output}/segmented.part1.process.log" 2>&1
run_train 46317 --outdir="${output}/segmented" --nosubdir \
  --resume="${output}/segmented/training-state-latest.pt" \
  >"${output}/segmented.part2.process.log" 2>&1
apptainer exec --bind /data:/data --pwd "${repo}" "${runtime_sif}" python \
  analysis/q256_p2_b384_pulse_chase_v1/compare_computational_states.py \
    --uninterrupted "${output}/uninterrupted/training-state-latest.pt" \
    --segmented "${output}/segmented/training-state-latest.pt" \
    --output "${output}/segmented_resume_parity_report.json"
