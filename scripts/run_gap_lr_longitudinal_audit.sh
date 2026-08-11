#!/usr/bin/env bash
set -euo pipefail

# Role D: four same-trajectory current-gap counterfactuals from formal Arm A.
# This script is read-only with respect to the frozen training run.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ECT_PYTHON:-$REPO/.venv/bin/python}"
EXPERIMENT_ROOT="${ECT_EXPERIMENT_ROOT:?set ECT_EXPERIMENT_ROOT to the formal experiment directory}"
DATA="${ECT_DATA:?set ECT_DATA to the canonical CIFAR-10 archive}"
TRANSFER="${ECT_TRANSFER:?set ECT_TRANSFER to the pretrained EDM checkpoint}"
LAUNCHER_LOG="${ECT_LAUNCHER_LOG:-${EXPERIMENT_ROOT}.launcher.log}"
OUT_ROOT="${ROLE_D_OUT:?set ROLE_D_OUT to a new external result directory}"
GPU="${ROLE_D_GPU:-0}"
SEED=20260810
ARM_A="$EXPERIMENT_ROOT/arm_a_g1_0_lr_fixed_s3"

test -x "$PYTHON"
test -d "$EXPERIMENT_ROOT"
test -f "$DATA"
test -f "$TRANSFER"
test -f "$LAUNCHER_LOG"
if [ -e "$OUT_ROOT" ]; then
    echo "REFUSING: Role D output already exists: $OUT_ROOT" >&2
    exit 3
fi
mkdir -p "$OUT_ROOT"

"$PYTHON" "$REPO/scripts/prepare_gap_lr_downstream.py" \
    --experiment-root "$EXPERIMENT_ROOT" \
    --data "$DATA" \
    --transfer "$TRANSFER" \
    --launcher-log "$LAUNCHER_LOG" \
    --scope role-d \
    --out "$OUT_ROOT/artifact_manifest.json"

run_point() {
    requested_k="$1"
    actual_k="$2"
    state_id="$3"
    point_out="$OUT_ROOT/k${requested_k}"
    mkdir -p "$point_out"
    env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
        "$REPO/analysis/radam_stateful_update_audit.py" \
        --training-state "$ARM_A/training-state-${state_id}.pt" \
        --checkpoint "$ARM_A/network-snapshot-${state_id}.pkl" \
        --data "$DATA" \
        --batch-size 128 \
        --batch-gpu 16 \
        --seed "$SEED" \
        --state-kimg "$actual_k" \
        --device cuda \
        --amp \
        --lr 1e-4 \
        --betas 0.9,0.999 \
        --eps 1e-8 \
        --out "$point_out" \
        > "$point_out/runner.log" 2>&1
}

run_point 32 32.128 000001
run_point 64 64.128 000002
run_point 128 128.128 000004
run_point 256 256.000 000008

echo "ROLE_D_LONGITUDINAL_COMPLETE"
