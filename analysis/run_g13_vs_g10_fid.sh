#!/usr/bin/env bash
# Clean g=1.3 vs g=1.0 FID/KID comparison at the exact states where the
# non-scalar residual was measured (gap_lr_matched_q128_s3_v1, arm_a g=1.0 /
# arm_b g=1.3, both global_sigmoid, lr fixed 1e-4). 4 evals, sequential on GPU0.
# Per HANDOFF_20260804.md: one task per GPU, background via setsid nohup.
set -u
PYTHON=/data/raw/ECT/recurrence_of_ect/.venv/bin/python
REPO=/data/raw/ECT/recurrence_of_ect
DATA=/data/raw/ECT/datasets/cifar10-32x32.zip
BASE=/data/raw/ECT/ect_runs/gap_lr_matched_q128_s3_v1
OUT=/data/raw/ECT/ect_runs/g13_vs_g10_fid_0809
mkdir -p "$OUT/logs"

eval_one() {
  local arm="$1" nfe="$2"
  local tag="${arm}_nfe${nfe}"
  env CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29777 \
    $PYTHON $REPO/ct_eval.py \
    --resume="$BASE/$arm/network-snapshot-latest.pkl" \
    --outdir="$OUT/$tag" --data="$DATA" \
    --nfe=$nfe --mid_t=0.821 --metrics=fid5k_full,kid5k_full --seed=3 \
    > "$OUT/logs/$tag.log" 2>&1
  echo "done $tag rc=$?"
}

echo "START $(date +%s)"
eval_one arm_a_g1_0_lr_fixed_s3 1
eval_one arm_a_g1_0_lr_fixed_s3 2
eval_one arm_b_g1_3_lr_fixed_s3 1
eval_one arm_b_g1_3_lr_fixed_s3 2
echo "ALL_G13_VS_G10_EVAL_DONE $(date +%s)"
