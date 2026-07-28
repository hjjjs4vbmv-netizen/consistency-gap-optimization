#!/usr/bin/env bash
# =============================================================================
# Role C — Confirmatory fixed-vs-global (g=1.10) training commands.
# -----------------------------------------------------------------------------
# Frozen training code : branch role-c/confirmatory-gap-g110
#                        commit 3a0d603da97dd93ddbb6c7ce49e4a7351d54bb43  (PR #23 merged into main)
# Env                  : conda env `myconda` (python 3.13.5, torch 2.8.0+cu128, A100)
# Dataset              : /mnt/ect_project/datasets/cifar10-32x32.zip
# Pretrained (transfer): /mnt/ect_project/pretrained/edm-cifar10-32x32-uncond-vp.pkl
#
# Method definitions (ONLY these two are compared):
#   Fixed         : --mapping=sigmoid            --global-gap-scale=1.0
#                   (official ECT sigmoid; global_gap_scale==1.0 short-circuits
#                    to bitwise parity with the official formula)
#   Global-only   : --mapping=global_sigmoid      --global-gap-scale=1.10
#                   (same official sigmoid gap scaled by a single fixed g=1.10)
#
# INVARIANT: fixed and global differ ONLY by {mapping, global_gap_scale, outdir}.
#            No local controller is enabled in either arm.
# =============================================================================
set -Eeuo pipefail

export ECT_BRANCH="role-c/confirmatory-gap-g110"
export ECT_COMMIT="3a0d603da97dd93ddbb6c7ce49e4a7351d54bb43"
DATA="/mnt/ect_project/datasets/cifar10-32x32.zip"
TRANSFER="/mnt/ect_project/pretrained/edm-cifar10-32x32-uncond-vp.pkl"
PYTHON="${PYTHON:-python}"

# ---- shared parameters (fixed == global except mapping + global-gap-scale) ----
COMMON=(
  --data="$DATA"
  --cond=False --arch=ddpmpp --precond=ect
  --batch=128 --batch-gpu=16 --optim=RAdam --lr=0.0001 --dropout=0.2 --augment=0
  -q 256 -k 8 -b 1 -c 0 --double=10000 --ema_beta=0.9993
  --fp16=True --enable_amp=True --metrics=none
  --transfer="$TRANSFER" --nosubdir
)

# =============================================================================
# 1) SMOKE TESTS  (seed 3, 2 kimg)   -- run BEFORE the formal run
#    --duration=0.002 (=2 kimg), --tick=1, --ckpt=1 so a checkpoint is saved.
# =============================================================================
SMOKE_DURATION=0.002      # 2 kimg
SMOKE_RUN_ARGS=(--tick=1 --snap=0 --dump=0 --ckpt=1 --seed=3 --duration=$SMOKE_DURATION)

# Smoke A — fixed sigmoid
$PYTHON ct_train.py "${COMMON[@]}" --mapping=sigmoid       --global-gap-scale=1.0  \
  "${SMOKE_RUN_ARGS[@]}" --outdir=/root/ect_runs/smoke/seed3_fixed

# Smoke B — global-only g=1.10
$PYTHON ct_train.py "${COMMON[@]}" --mapping=global_sigmoid --global-gap-scale=1.10 \
  "${SMOKE_RUN_ARGS[@]}" --outdir=/root/ect_runs/smoke/seed3_global110

# =============================================================================
# 2) FORMAL PAIRED RUN  (256 kimg)
#    --duration=0.256 (=256 kimg), --tick=10, --ckpt=10, --sample_every=26
#    Today: ONLY seed 3 is launched.  Seeds 4 & 5 are written but NOT started.
# =============================================================================
FORMAL_DURATION=0.256     # 256 kimg
FORMAL_RUN_ARGS=(--tick=10 --snap=0 --dump=0 --ckpt=10 --sample_every=26 --duration=$FORMAL_DURATION)

# ---- seed 3 (STARTED TODAY) ----
$PYTHON ct_train.py "${COMMON[@]}" --mapping=sigmoid       --global-gap-scale=1.0  --seed=3 \
  "${FORMAL_RUN_ARGS[@]}" --outdir=/root/ect_runs/confirmatory_256k/seed3_fixed

$PYTHON ct_train.py "${COMMON[@]}" --mapping=global_sigmoid --global-gap-scale=1.10 --seed=3 \
  "${FORMAL_RUN_ARGS[@]}" --outdir=/root/ect_runs/confirmatory_256k/seed3_global110

# ---- seed 4 (NOT started today) ----
$PYTHON ct_train.py "${COMMON[@]}" --mapping=sigmoid       --global-gap-scale=1.0  --seed=4 \
  "${FORMAL_RUN_ARGS[@]}" --outdir=/root/ect_runs/confirmatory_256k/seed4_fixed
$PYTHON ct_train.py "${COMMON[@]}" --mapping=global_sigmoid --global-gap-scale=1.10 --seed=4 \
  "${FORMAL_RUN_ARGS[@]}" --outdir=/root/ect_runs/confirmatory_256k/seed4_global110

# ---- seed 5 (NOT started today) ----
$PYTHON ct_train.py "${COMMON[@]}" --mapping=sigmoid       --global-gap-scale=1.0  --seed=5 \
  "${FORMAL_RUN_ARGS[@]}" --outdir=/root/ect_runs/confirmatory_256k/seed5_fixed
$PYTHON ct_train.py "${COMMON[@]}" --mapping=global_sigmoid --global-gap-scale=1.10 --seed=5 \
  "${FORMAL_RUN_ARGS[@]}" --outdir=/root/ect_runs/confirmatory_256k/seed5_global110

# =============================================================================
# 3) RESUME  (after interruption; writes into the SAME outdir, no overwrite)
#    --resume replaces --transfer; --global-gap-scale is re-stated for safety.
#    The training-state also carries the schedule/gap state; verify g==1.10.
# =============================================================================
# COMMON_RESUME = COMMON without --transfer
COMMON_RESUME=(
  --data="$DATA"
  --cond=False --arch=ddpmpp --precond=ect
  --batch=128 --batch-gpu=16 --optim=RAdam --lr=0.0001 --dropout=0.2 --augment=0
  -q 256 -k 8 -b 1 -c 0 --double=10000 --ema_beta=0.9993
  --fp16=True --enable_amp=True --metrics=none --nosubdir
)

# Example: resume seed3 fixed / global110 (into the same outdir)
#$PYTHON ct_train.py "${COMMON_RESUME[@]}" --mapping=sigmoid       --global-gap-scale=1.0  --seed=3 \
#  "${FORMAL_RUN_ARGS[@]}" --resume=/root/ect_runs/confirmatory_256k/seed3_fixed/training-state-latest.pt \
#  --outdir=/root/ect_runs/confirmatory_256k/seed3_fixed
#$PYTHON ct_train.py "${COMMON_RESUME[@]}" --mapping=global_sigmoid --global-gap-scale=1.10 --seed=3 \
#  "${FORMAL_RUN_ARGS[@]}" --resume=/root/ect_runs/confirmatory_256k/seed3_global110/training-state-latest.pt \
#  --outdir=/root/ect_runs/confirmatory_256k/seed3_global110
