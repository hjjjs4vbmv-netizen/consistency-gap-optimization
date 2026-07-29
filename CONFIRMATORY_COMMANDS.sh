#!/usr/bin/env bash
# =============================================================================
# Role C — Confirmatory fixed-vs-global (g=1.10) training commands.
# -----------------------------------------------------------------------------
# Provenance:
#   training_code_sha : 3a0d603da97dd93ddbb6c7ce49e4a7351d54bb43  (seed-3 was
#                       trained on this commit; recorded in commit_sha.txt)
#   pr_head_sha       : 79143c685e5588948972c17457b1c51c7a77bb49  (this PR head;
#                       only adds docs + the resume fix, not a training baseline)
# Branch              : role-c/confirmatory-gap-g110
# Env                 : conda env `myconda` (python 3.13.5, torch 2.8.0+cu128, A100)
# Dataset             : /mnt/ect_project/datasets/cifar10-32x32.zip
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
export ECT_COMMIT="3a0d603da97dd93ddbb6c7ce49e4a7351d54bb43"  # training_code_sha
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

# COMMON_RESUME = COMMON without --transfer (resume replaces transfer)
COMMON_RESUME=(
  --data="$DATA"
  --cond=False --arch=ddpmpp --precond=ect
  --batch=128 --batch-gpu=16 --optim=RAdam --lr=0.0001 --dropout=0.2 --augment=0
  -q 256 -k 8 -b 1 -c 0 --double=10000 --ema_beta=0.9993
  --fp16=True --enable_amp=True --metrics=none --nosubdir
)

# =============================================================================
# Usage:
#   MODE=smoke bash CONFIRMATORY_COMMANDS.sh                    # 2-kimg smoke (seed 3)
#   MODE=formal RUN_SEEDS=3 bash CONFIRMATORY_COMMANDS.sh        # formal seed 3 (default)
#   MODE=formal RUN_SEEDS="4 5" bash CONFIRMATORY_COMMANDS.sh    # formal seeds 4+5
#
# Defaults: MODE=formal, RUN_SEEDS=3.
# Running `bash CONFIRMATORY_COMMANDS.sh` with no env vars will ONLY run seed 3.
# Seeds 4 & 5 are never started unless explicitly requested via RUN_SEEDS.
# =============================================================================
MODE="${MODE:-formal}"
RUN_SEEDS="${RUN_SEEDS:-3}"

FORMAL_DURATION=0.256     # 256 kimg
FORMAL_RUN_ARGS=(--tick=10 --snap=0 --dump=0 --ckpt=10 --sample_every=26 --duration=$FORMAL_DURATION)

# =============================================================================
# 1) SMOKE TESTS  (seed 3, 2 kimg)   -- run BEFORE the formal run
#    --duration=0.002 (=2 kimg), --tick=1, --ckpt=1 so a checkpoint is saved.
# =============================================================================
if [ "$MODE" = "smoke" ]; then
  SMOKE_DURATION=0.002      # 2 kimg
  SMOKE_RUN_ARGS=(--tick=1 --snap=0 --dump=0 --ckpt=1 --seed=3 --duration=$SMOKE_DURATION)

  # Smoke A — fixed sigmoid
  $PYTHON ct_train.py "${COMMON[@]}" --mapping=sigmoid       --global-gap-scale=1.0  \
    "${SMOKE_RUN_ARGS[@]}" --outdir=/root/ect_runs/smoke/seed3_fixed

  # Smoke B — global-only g=1.10
  $PYTHON ct_train.py "${COMMON[@]}" --mapping=global_sigmoid --global-gap-scale=1.10 \
    "${SMOKE_RUN_ARGS[@]}" --outdir=/root/ect_runs/smoke/seed3_global110

  exit 0
fi

# =============================================================================
# 2) FORMAL PAIRED RUN  (256 kimg)
#    --duration=0.256 (=256 kimg), --tick=10, --ckpt=10, --sample_every=26
#    Default: ONLY seed 3.  To run seeds 4 & 5: RUN_SEEDS="4 5" ...
# =============================================================================
for seed in $RUN_SEEDS; do
  $PYTHON ct_train.py "${COMMON[@]}" --mapping=sigmoid       --global-gap-scale=1.0  --seed="$seed" \
    "${FORMAL_RUN_ARGS[@]}" --outdir="/root/ect_runs/confirmatory_256k/seed${seed}_fixed"

  $PYTHON ct_train.py "${COMMON[@]}" --mapping=global_sigmoid --global-gap-scale=1.10 --seed="$seed" \
    "${FORMAL_RUN_ARGS[@]}" --outdir="/root/ect_runs/confirmatory_256k/seed${seed}_global110"
done

# =============================================================================
# 3) RESUME  (after interruption)
#    Two modes — both examples are COMMENTED OUT.  Uncomment and edit as needed.
#
#    a) Verification resume (writes to a NEW outdir, source untouched):
#       Use this to test that resume works correctly without risking the
#       authoritative run handed off to Role D.
#
#    b) Actual interruption resume (writes to the SAME outdir):
#       Use this only when the original run was interrupted and must continue
#       in-place.  Do NOT use this mode for testing.
#
#    --resume replaces --transfer; --global-gap-scale is re-stated for safety.
#    The training-state also carries the schedule/gap state; verify g==1.10
#    after resume.
# =============================================================================

# --- a) Verification resume (new outdir, source untouched) ---
#$PYTHON ct_train.py "${COMMON_RESUME[@]}" --mapping=global_sigmoid --global-gap-scale=1.10 --seed=3 \
#  "${FORMAL_RUN_ARGS[@]}" \
#  --resume=/root/ect_runs/confirmatory_256k/seed3_global110/training-state-latest.pt \
#  --outdir=/root/ect_runs/resume_checks/seed3_global110

# --- b) Actual interruption resume (same outdir, continues the run) ---
#$PYTHON ct_train.py "${COMMON_RESUME[@]}" --mapping=global_sigmoid --global-gap-scale=1.10 --seed=3 \
#  "${FORMAL_RUN_ARGS[@]}" \
#  --resume=/root/ect_runs/confirmatory_256k/seed3_global110/training-state-latest.pt \
#  --outdir=/root/ect_runs/confirmatory_256k/seed3_global110
