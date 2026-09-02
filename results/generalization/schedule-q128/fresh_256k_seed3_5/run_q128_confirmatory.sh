#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/mnt/ect_project/src/recurrence_of_ect
DATA=/mnt/ect_project/datasets/cifar10-32x32.zip
TRANSFER=/mnt/ect_project/pretrained/edm-cifar10-32x32-uncond-vp.pkl

EXPECTED_DATA_SHA=9818e4b801a52eac437485bc8a69e40b54e9ae9c5d1427467343c91de868f1b3
EXPECTED_TRANSFER_SHA=4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da

cd "$REPO"

test -s "$DATA" || {
  echo "Missing dataset: $DATA"
  exit 1
}

test -s "$TRANSFER" || {
  echo "Missing transfer checkpoint: $TRANSFER"
  exit 1
}

DATA_SHA=$(sha256sum "$DATA" | awk '{print $1}')
TRANSFER_SHA=$(sha256sum "$TRANSFER" | awk '{print $1}')

test "$DATA_SHA" = "$EXPECTED_DATA_SHA" || {
  echo "Dataset SHA mismatch: $DATA_SHA"
  exit 1
}

test "$TRANSFER_SHA" = "$EXPECTED_TRANSFER_SHA" || {
  echo "Transfer SHA mismatch: $TRANSFER_SHA"
  exit 1
}

GIT_COMMIT=$(git rev-parse HEAD)
GIT_SHORT=${GIT_COMMIT:0:8}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

RUN_ROOT="/root/ect-runs/q128-fresh-confirmatory-${GIT_SHORT}-${STAMP}"
LOG_ROOT="/mnt/ect_project/logs/role_e_q128_confirmatory/${GIT_SHORT}-${STAMP}"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

cat > "$LOG_ROOT/matrix_metadata.env" <<EOF
experiment_id=q128-fresh-fixed-vs-global110-v1
git_commit=$GIT_COMMIT
dataset_path=$DATA
dataset_sha256=$DATA_SHA
dataset_byte_identical_to_q256_formal=false
transfer_checkpoint=$TRANSFER
transfer_checkpoint_sha256=$TRANSFER_SHA
q=128
target_kimg=256
seeds=3,4,5
methods=fixed,global110
execution=sequential_single_gpu
run_root=$RUN_ROOT
EOF

run_cell() {
  local seed="$1"
  local method="$2"
  local schedule="$3"
  local scale="$4"

  local run_id="q128-seed${seed}-${method}-256k"
  local outdir="$RUN_ROOT/$run_id"
  local logfile="$LOG_ROOT/${run_id}.log"

  if [[ -e "$outdir" ]]; then
    echo "Refusing to overwrite: $outdir"
    exit 1
  fi

  echo "===== START $run_id =====" | tee "$logfile"

  torchrun --standalone --nproc_per_node=1 ct_train.py \
    --outdir="$outdir" \
    --data="$DATA" \
    --duration=0.256 \
    --batch=128 \
    --batch-gpu=16 \
    --optim=RAdam \
    --lr=0.0001 \
    --dropout=0.2 \
    --augment=0 \
    --schedule="$schedule" \
    --global-gap-scale="$scale" \
    -q 128 \
    -k 8 \
    -b 1 \
    -c 0 \
    --double=10000 \
    --fp16=True \
    --enable_amp=True \
    --tf32=False \
    --metrics=none \
    --seed="$seed" \
    --transfer="$TRANSFER" \
    --tick=1 \
    --snap=64 \
    --dump=0 \
    --ckpt=64 \
    --sample_every=10000 \
    --eval_every=10000 \
    --nosubdir \
    2>&1 | tee -a "$logfile"

  test -s "$outdir/network-snapshot-latest.pkl"
  test -s "$outdir/training-state-latest.pt"
  test -s "$outdir/train_summary.csv"

  grep -q "Exiting..." "$logfile"

  sha256sum \
    "$outdir/network-snapshot-latest.pkl" \
    "$outdir/training-state-latest.pt" \
    "$outdir/training_options.json" \
    > "$outdir/final_sha256.txt"

  echo "===== PASS $run_id =====" | tee -a "$logfile"
}

for seed in 3 4 5; do
  run_cell "$seed" fixed sigmoid 1.0
  run_cell "$seed" global110 global_sigmoid 1.10
done

echo "Q128_CONFIRMATORY_OVERALL_EXIT=0" |
  tee "$LOG_ROOT/overall_status.txt"

echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
