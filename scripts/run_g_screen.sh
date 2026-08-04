#!/usr/bin/env bash
set -u
export PYTHON=/data/raw/ECT/recurrence_of_ect/.venv/bin/python
REPO=/data/raw/ECT/recurrence_of_ect
DATA=/data/raw/ECT/datasets/cifar10-32x32.zip
TR=/data/raw/ECT/pretrained/edm-cifar10-32x32-uncond-vp.pkl
OUT=/data/raw/ECT/ect_runs/g_screen
mkdir -p "$OUT/logs"
run_one() {
  local g="$1" dev="$2" port="$3"
  local map=sigmoid
  [ "$g" != "1.0" ] && map=global_sigmoid
  local tag="g$(echo $g | tr . _)"
  env CUDA_VISIBLE_DEVICES="$dev" MASTER_PORT="$port" $PYTHON $REPO/ct_train.py     --outdir="$OUT/$tag" --data="$DATA" --cond=False --arch=ddpmpp --precond=ect     --batch=128 --batch-gpu=16 --optim=RAdam --lr=0.0001 --dropout=0.2 --augment=0     -q 128 -k 8 -b 1 -c 0 --double=10000 --ema_beta=0.9993 --fp16=True --enable_amp=True     --metrics=none --nosubdir --tick=10 --ckpt=10 --sample_every=26 --seed=3     --transfer="$TR" --mapping="$map" --global-gap-scale="$g" --duration=0.256     > "$OUT/logs/$tag.log" 2>&1 &
  echo "started $tag pid=$! on dev=$dev"
}
# 3 批, 每批 2 个任务(每 GPU 一个), 批内并行
run_one 0.9  0 29521
run_one 1.0  1 29522
wait
echo '--- batch1 done (g0.9, g1.0) ---'
run_one 1.05 0 29523
run_one 1.1  1 29524
wait
echo '--- batch2 done (g1.05, g1.1) ---'
run_one 1.2  0 29525
run_one 1.3  1 29526
wait
echo ALL_DONE
