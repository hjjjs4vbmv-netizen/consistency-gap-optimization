#!/usr/bin/env bash
# E7 — optimizer-state dependence: run the stateful RAdam audit at multiple
# arm_a checkpoints (different n_K / optimizer maturity) and record how the
# update distortion (R_opt, R_grad, h_actual, a_K*) evolves with optimizer state.
# Per HANDOFF_20260804.md: one task per GPU, background via setsid nohup.
set -u
PYTHON=/data/raw/ECT/recurrence_of_ect/.venv/bin/python
REPO=/data/raw/ECT/recurrence_of_ect
DATA=/data/raw/ECT/datasets/cifar10-32x32.zip
BASE=/data/raw/ECT/ect_runs/gap_lr_matched_q128_s3_v1
# loss_fn checkpoint must have schedule.name == 'sigmoid' (audit gate); use g_screen g1_0.
CKPT=/data/raw/ECT/ect_runs/g_screen/g1_0/network-snapshot-latest.pkl
OUT=/data/raw/ECT/ect_runs/e7_optimizer_state_0809
mkdir -p "$OUT/logs"

run_audit() {
  local tick="$1"
  local ts="$BASE/arm_a_g1_0_lr_fixed_s3/training-state-00000${tick}.pt"
  env CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29831 \
    $PYTHON $REPO/analysis/radam_stateful_update_audit.py \
    --training-state="$ts" --checkpoint="$CKPT" --data="$DATA" \
    --batch-size=64 --seed=20260808 --amp --support-atol=0.0 \
    --out="$OUT/tick${tick}" > "$OUT/logs/tick${tick}.log" 2>&1
  echo "done tick${tick} rc=$?"
}

echo "START $(date +%s)"
run_audit 6
run_audit 7
run_audit 8
echo "ALL_E7_DONE $(date +%s)"
