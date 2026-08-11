#!/usr/bin/env bash
set -euo pipefail

# Role E: one frozen NFE=1 FID/KID-5k cell for each formal A/B/C EMA.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ECT_PYTHON:-$REPO/.venv/bin/python}"
EXPERIMENT_ROOT="${ECT_EXPERIMENT_ROOT:?set ECT_EXPERIMENT_ROOT to the formal experiment directory}"
DATA="${ECT_DATA:?set ECT_DATA to the canonical CIFAR-10 archive}"
TRANSFER="${ECT_TRANSFER:?set ECT_TRANSFER to the pretrained EDM checkpoint}"
OUT_ROOT="${ROLE_E_OUT:?set ROLE_E_OUT to a new external result directory}"
GPU="${ROLE_E_GPU:-0}"
EVALUATOR_SEED=20260730
SAMPLE_SEEDS=0-4999

test -x "$PYTHON"
test -d "$EXPERIMENT_ROOT"
test -f "$DATA"
test -f "$TRANSFER"
if [ -e "$OUT_ROOT" ]; then
    echo "REFUSING: Role E output already exists: $OUT_ROOT" >&2
    exit 3
fi
mkdir -p "$OUT_ROOT"

"$PYTHON" "$REPO/scripts/prepare_gap_lr_downstream.py" \
    --experiment-root "$EXPERIMENT_ROOT" \
    --data "$DATA" \
    --transfer "$TRANSFER" \
    --scope role-e \
    --out "$OUT_ROOT/artifact_manifest.json"

evaluate_arm() {
    arm="$1"
    run_id="$2"
    run_dir="$EXPERIMENT_ROOT/$run_id"
    checkpoint="$run_dir/network-snapshot-000008.pkl"
    cell="$OUT_ROOT/$arm/nfe1"
    mkdir -p "$cell"
    checkpoint_sha256="$(sha256sum "$checkpoint" | awk '{print $1}')"
    data_sha256="$(sha256sum "$DATA" | awk '{print $1}')"
    command=(
        "$PYTHON" "$REPO/ct_eval.py"
        --data="$DATA"
        --outdir="$cell"
        --nosubdir
        --cond=False
        --arch=ddpmpp
        --precond=ct
        --dropout=0.2
        --augment=0
        --fp16=False
        --seed="$EVALUATOR_SEED"
        --resume="$checkpoint"
        --nfe=1
        --metrics=fid5k_full,kid5k_full
        --metric-repeats=1
        --sample-seeds="$SAMPLE_SEEDS"
    )
    {
        printf 'experiment_id=gap_lr_matched_q128_s3_v1\n'
        printf 'arm=%s\nrun_id=%s\n' "$arm" "$run_id"
        printf 'nfe=1\nmid_t=[]\nprecision=fp32\n'
        printf 'evaluator_seed=%s\nsample_seeds=%s\n' "$EVALUATOR_SEED" "$SAMPLE_SEEDS"
        printf 'checkpoint=%s\ncheckpoint_sha256=%s\n' "$checkpoint" "$checkpoint_sha256"
        printf 'data=%s\ndata_sha256=%s\n' "$DATA" "$data_sha256"
        printf 'source_commit=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
        printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'exact_command='
        printf '%q ' "${command[@]}"
        printf '\n'
    } > "$cell/experiment_meta.env"
    set +e
    env CUDA_VISIBLE_DEVICES="$GPU" "${command[@]}" \
        > "$cell/runner.log" 2>&1
    status="$?"
    set -e
    printf 'exit_code=%s\nfinished_utc=%s\n' \
        "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >> "$cell/experiment_meta.env"
    [ "$status" -eq 0 ] || return "$status"
    for metric in fid5k_full kid5k_full; do
        result="$cell/metric-${metric}.jsonl"
        test -s "$result"
        [ "$(wc -l < "$result")" -eq 1 ]
    done
}

evaluate_arm A arm_a_g1_0_lr_fixed_s3
evaluate_arm B arm_b_g1_3_lr_fixed_s3
evaluate_arm C arm_c_g1_3_lr_matched_s3

echo "ROLE_E_NFE1_QUALITY_COMPLETE"
