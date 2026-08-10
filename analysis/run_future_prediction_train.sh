#!/usr/bin/env bash
# Future-prediction experiment: train 4 new seeds (0,1,2,4) x 2 gaps (g=1.0,1.3),
# 256 kimg, with early checkpoints (tick=32, ckpt=1, snap=1). Reuses seed 3 arm
# states for the 5th seed. Sequential on GPU0. Per HANDOFF_20260804.md.
set -u
PYTHON=/data/raw/ECT/recurrence_of_ect/.venv/bin/python
REPO=/data/raw/ECT/recurrence_of_ect
DATA=/data/raw/ECT/datasets/cifar10-32x32.zip
TR=/data/raw/ECT/pretrained/edm-cifar10-32x32-uncond-vp.pkl
OUT=/data/raw/ECT/ect_runs/future_pred_0809
mkdir -p "$OUT/logs"

train_one() {
  local seed="$1" g="$2" port="$3"
  local tag="seed${seed}_g$(echo $g | tr . _)"
  env CUDA_VISIBLE_DEVICES=0 MASTER_PORT="$port" \
    $PYTHON $REPO/ct_train.py \
    --outdir="$OUT/$tag" --data="$DATA" --cond=False --arch=ddpmpp --precond=ect \
    --batch=128 --batch-gpu=16 --optim=RAdam --lr=0.0001 --dropout=0.2 --augment=0 \
    -q 128 -k 8 -b 1 -c 0 --double=10000 --ema_beta=0.9993 --fp16=True --enable_amp=True \
    --metrics=none --nosubdir --tick=32 --ckpt=1 --snap=1 --sample_every=26 --seed="$seed" \
    --transfer="$TR" --mapping=global_sigmoid --global-gap-scale="$g" --duration=0.256 \
    > "$OUT/logs/$tag.log" 2>&1
  echo "done $tag rc=$?"
}

echo "START $(date +%s)"
# 4 new seeds x 2 gaps, sequential on GPU0
for seed in 0 1 2 4; do
  train_one $seed 1.0 29921
  train_one $seed 1.3 29922
done
echo "ALL_FUTURE_PRED_TRAIN_DONE $(date +%s)"
