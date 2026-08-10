#!/usr/bin/env bash
# Optimizer-level future prediction: retrain 4 seeds x 2 gaps WITH early
# training-state saving (--dump=1 => state_dump_ticks=1, saves every tick).
# This gives early R_opt (optimizer-state residual) at 32/64k for leave-one-seed-out.
set -u
PYTHON=/data/raw/ECT/recurrence_of_ect/.venv/bin/python
REPO=/data/raw/ECT/recurrence_of_ect
DATA=/data/raw/ECT/datasets/cifar10-32x32.zip
TR=/data/raw/ECT/pretrained/edm-cifar10-32x32-uncond-vp.pkl
OUT=/data/raw/ECT/ect_runs/future_pred_opt_0811
mkdir -p "$OUT/logs"

train_one() {
  local seed="$1" g="$2" port="$3"
  local tag="seed${seed}_g$(echo $g | tr . _)"
  env CUDA_VISIBLE_DEVICES=0 MASTER_PORT="$port" \
    $PYTHON $REPO/ct_train.py \
    --outdir="$OUT/$tag" --data="$DATA" --cond=False --arch=ddpmpp --precond=ect \
    --batch=128 --batch-gpu=16 --optim=RAdam --lr=0.0001 --dropout=0.2 --augment=0 \
    -q 128 -k 8 -b 1 -c 0 --double=10000 --ema_beta=0.9993 --fp16=True --enable_amp=True \
    --metrics=none --nosubdir --tick=32 --ckpt=1 --snap=1 --dump=1 --sample_every=26 --seed="$seed" \
    --transfer="$TR" --mapping=global_sigmoid --global-gap-scale="$g" --duration=0.256 \
    > "$OUT/logs/$tag.log" 2>&1
  echo "done $tag rc=$?"
}

echo "START $(date +%s)"
for seed in 0 1 2 4; do
  train_one $seed 1.0 29931
  train_one $seed 1.3 29932
done
echo "ALL_FUTURE_PRED_OPT_TRAIN_DONE $(date +%s)"
