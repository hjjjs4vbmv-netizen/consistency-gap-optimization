#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ECT_PYTHON:-$REPO/.venv/bin/python}"
DATA="${ECT_DATA:?set ECT_DATA to the CIFAR-10 training archive}"
TRANSFER="${ECT_TRANSFER:?set ECT_TRANSFER to the pretrained EDM checkpoint}"
RUN_ROOT="${ECT_RUN_ROOT:?set ECT_RUN_ROOT to the external run directory}"
REPLICATION_RECEIPT="${GAP_LR_SEED_REPLICATION_RECEIPT:?set GAP_LR_SEED_REPLICATION_RECEIPT}"
SOURCE_AUDIT_RECEIPT="${GAP_LR_SOURCE_AUDIT_RECEIPT:?set GAP_LR_SOURCE_AUDIT_RECEIPT}"
MATRIX="$REPO/configs/gap_lr_matched_seed_replication_matrix.json"
VALIDATOR="$REPO/scripts/validate_gap_lr_seed_replication_receipt.py"
RUN_VERIFIER="$REPO/scripts/verify_gap_lr_seed_replication_run.py"
GROUP_VERIFIER="$REPO/scripts/verify_gap_lr_seed_replication_group.py"
OUT="$RUN_ROOT/gap_lr_matched_q128_s45_replication_v1"
GPU="${FORMAL_GPU:-1}"
BASE_LR=0.0001

cd "$REPO"

test -x "$PYTHON"
test -f "$DATA"
test -f "$TRANSFER"
test -f "$REPLICATION_RECEIPT"
test -f "$SOURCE_AUDIT_RECEIPT"
test -f "$MATRIX"
test -f "$VALIDATOR"
test -f "$RUN_VERIFIER"
test -f "$GROUP_VERIFIER"

if [ -e "$OUT" ]; then
    echo "REFUSING: formal replication output already exists: $OUT" >&2
    exit 3
fi

C0_STAR="$("$PYTHON" "$VALIDATOR" \
    --receipt "$REPLICATION_RECEIPT" \
    --source-audit-receipt "$SOURCE_AUDIT_RECEIPT" \
    --matrix "$MATRIX" \
    --repo "$REPO" \
    --data "$DATA" \
    --transfer "$TRANSFER")"

C_LR="$("$PYTHON" -c '
import math, sys
c = float(sys.argv[1])
if not math.isfinite(c) or c <= 0:
    raise SystemExit("validated c0_star must be finite and positive")
print(format(c * 1.0e-4, ".17g"))
' "$C0_STAR")"

PROTOCOL_COMMIT="$(git rev-parse HEAD)"
TRAINING_CODE_COMMIT="$("$PYTHON" -c '
import json, sys
print(json.load(open(sys.argv[1]))["source"]["training_code_commit"])
' "$REPLICATION_RECEIPT")"
REPLICATION_RECEIPT_SHA256="$(sha256sum "$REPLICATION_RECEIPT" | awk '{print $1}')"
SOURCE_AUDIT_RECEIPT_SHA256="$(sha256sum "$SOURCE_AUDIT_RECEIPT" | awk '{print $1}')"
MATRIX_SHA256="$(sha256sum "$MATRIX" | awk '{print $1}')"
DATA_SHA256="$(sha256sum "$DATA" | awk '{print $1}')"
TRANSFER_SHA256="$(sha256sum "$TRANSFER" | awk '{print $1}')"

mkdir -p "$OUT/logs" "$OUT/integrity_receipts"

{
    echo "experiment_id=gap_lr_matched_q128_s45_replication_v1"
    echo "source_experiment_id=gap_lr_matched_q128_s3_v1"
    echo "protocol_commit=$PROTOCOL_COMMIT"
    echo "training_code_commit=$TRAINING_CODE_COMMIT"
    echo "replication_receipt=$REPLICATION_RECEIPT"
    echo "replication_receipt_sha256=$REPLICATION_RECEIPT_SHA256"
    echo "source_audit_receipt=$SOURCE_AUDIT_RECEIPT"
    echo "source_audit_receipt_sha256=$SOURCE_AUDIT_RECEIPT_SHA256"
    echo "matrix_sha256=$MATRIX_SHA256"
    echo "dataset_sha256=$DATA_SHA256"
    echo "transfer_sha256=$TRANSFER_SHA256"
    echo "new_seeds=4,5"
    echo "existing_formal_seed=3"
    echo "arm_order=A,B,C"
    echo "gpu=$GPU"
    echo "base_lr=$BASE_LR"
    echo "c0_star=$C0_STAR"
    echo "arm_c_lr=$C_LR"
    echo "automatic_retry=false"
    date -u '+launch_utc=%Y-%m-%dT%H:%M:%SZ'
} > "$OUT/launch_provenance.txt"

nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
    --format=csv,noheader > "$OUT/hardware.txt"
"$PYTHON" -c '
import platform, sys, torch
print("python=" + platform.python_version())
print("torch=" + torch.__version__)
print("cuda=" + str(torch.version.cuda))
print("cudnn=" + str(torch.backends.cudnn.version()))
' > "$OUT/software.txt"

run_arm() {
    seed="$1"
    arm="$2"
    gap="$3"
    lr="$4"
    port="$5"
    run_id="$6"
    run_dir="$OUT/$run_id"

    echo "START seed=$seed arm=$arm gap=$gap lr=$lr gpu=$GPU port=$port"
    env CUDA_VISIBLE_DEVICES="$GPU" MASTER_PORT="$port" \
        "$PYTHON" "$REPO/ct_train.py" \
        --outdir="$run_dir" \
        --data="$DATA" \
        --cond=False \
        --arch=ddpmpp \
        --precond=ect \
        --batch=128 \
        --batch-gpu=16 \
        --optim=RAdam \
        --lr="$lr" \
        --dropout=0.2 \
        --augment=0 \
        -q 128 \
        -k 8 \
        -b 1 \
        -c 0 \
        --double=10000 \
        --ema_beta=0.9993 \
        --fp16=True \
        --enable_amp=True \
        --metrics=none \
        --nosubdir \
        --tick=32 \
        --snap=1 \
        --dump=1 \
        --ckpt=1 \
        --sample_every=9999 \
        --seed="$seed" \
        --transfer="$TRANSFER" \
        --mapping=global_sigmoid \
        --global-gap-scale="$gap" \
        --duration=0.256 \
        > "$OUT/logs/seed${seed}_${arm}.log" 2>&1

    printf '%s\n' "$PROTOCOL_COMMIT" > "$run_dir/protocol_commit.txt"
    printf '%s\n' "$TRAINING_CODE_COMMIT" > "$run_dir/training_code_commit.txt"
    printf '%s\n' "$SOURCE_AUDIT_RECEIPT_SHA256" > "$run_dir/source_audit_receipt_sha256.txt"

    "$PYTHON" "$RUN_VERIFIER" \
        --run-dir "$run_dir" \
        --arm "$arm" \
        --seed "$seed" \
        --protocol-commit "$PROTOCOL_COMMIT" \
        --training-code-commit "$TRAINING_CODE_COMMIT" \
        --source-audit-receipt-sha256 "$SOURCE_AUDIT_RECEIPT_SHA256" \
        --output "$OUT/integrity_receipts/seed${seed}_${arm}.integrity.json" \
        > "$OUT/logs/seed${seed}_${arm}.verification.log" 2>&1
    echo "DONE seed=$seed arm=$arm integrity=passed"
}

run_seed_4() {
    run_arm 4 A 1.0 "$BASE_LR" 29841 arm_a_g1_0_lr_fixed_s4
    run_arm 4 B 1.3 "$BASE_LR" 29842 arm_b_g1_3_lr_fixed_s4
    run_arm 4 C 1.3 "$C_LR" 29843 arm_c_g1_3_lr_matched_s4
    "$PYTHON" "$GROUP_VERIFIER" \
        --experiment-root "$OUT" \
        --completed-seeds 4 \
        --output "$OUT/integrity_receipts/seed4_group.integrity.json" \
        > "$OUT/logs/seed4_group.verification.log" 2>&1
    echo "SEED 4 GROUP COMPLETE"
}

run_seed_5() {
    run_arm 5 A 1.0 "$BASE_LR" 29851 arm_a_g1_0_lr_fixed_s5
    run_arm 5 B 1.3 "$BASE_LR" 29852 arm_b_g1_3_lr_fixed_s5
    run_arm 5 C 1.3 "$C_LR" 29853 arm_c_g1_3_lr_matched_s5
    "$PYTHON" "$GROUP_VERIFIER" \
        --experiment-root "$OUT" \
        --completed-seeds 4,5 \
        --output "$OUT/integrity_receipts/replication_group.integrity.json" \
        > "$OUT/logs/replication_group.verification.log" 2>&1
    echo "ALL FORMAL SEED REPLICATION RUNS COMPLETE"
}

run_seed_4
run_seed_5
