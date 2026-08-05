#!/usr/bin/env bash
set -u
export PYTHON=/data/raw/ECT/recurrence_of_ect/.venv/bin/python
REPO=/data/raw/ECT/recurrence_of_ect
DATA=/data/raw/ECT/datasets/cifar10-32x32.zip
CKPT=/data/raw/ECT/ect_runs/g_screen
OUT=/data/raw/ECT/ect_runs/g_screen_eval
mkdir -p "$OUT/logs"
eval_one() {
  local g="$1" nfe="$2" dev="$3" port="$4"
  local tag="g$(echo $g | tr . _)_nfe$nfe"
  env CUDA_VISIBLE_DEVICES="$dev" MASTER_PORT="$port" $PYTHON $REPO/ct_eval.py     --resume="$CKPT/g$(echo $g | tr . _)/network-snapshot-latest.pkl"     --outdir="$OUT/$tag" --data="$DATA"     --nfe=$nfe --mid_t=0.821 --metrics=fid5k_full,kid5k_full --seed=3     > "$OUT/logs/$tag.log" 2>&1 &
  echo "started $tag (g=$g nfe=$nfe) dev=$dev pid=$!"
}
# 每 GPU 一个任务: 每 ckpt 先 nfe=1 后 nfe=2
eval_one 0.9  1 0 29601
eval_one 1.0  1 1 29602
wait
echo '--- batch eval1 done ---'
eval_one 0.9  2 0 29603
eval_one 1.0  2 1 29604
wait
echo '--- batch eval2 done ---'
eval_one 1.05 1 0 29605
eval_one 1.1  1 1 29606
wait
echo '--- batch eval3 done ---'
eval_one 1.05 2 0 29607
eval_one 1.1  2 1 29608
wait
echo '--- batch eval4 done ---'
eval_one 1.2  1 0 29609
eval_one 1.3  1 1 29610
wait
echo '--- batch eval5 done ---'
eval_one 1.2  2 0 29611
eval_one 1.3  2 1 29612
wait
echo ALL_EVAL_DONE
