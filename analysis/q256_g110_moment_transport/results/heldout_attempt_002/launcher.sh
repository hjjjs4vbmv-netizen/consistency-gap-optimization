#!/bin/bash
set -u
umask 0027

cd /data/raw/ECT/worktrees/q256_g110_moment_transport || exit 90

test ! -e /data/raw/ECT/ect_runs/q256_g110_moment_transport/preflight/heldout-attempt-002/stdout.log || exit 91
test ! -e /data/raw/ECT/ect_runs/q256_g110_moment_transport/preflight/heldout-attempt-002/stderr.log || exit 92
date -u +%Y-%m-%dT%H:%M:%SZ > /data/raw/ECT/ect_runs/q256_g110_moment_transport/preflight/heldout-attempt-002/started_utc.txt

CUDA_VISIBLE_DEVICES=1 \
PYTHONPYCACHEPREFIX=/data/raw/ECT/ect_runs/q256_g110_moment_transport/preflight/heldout-attempt-002/pycache \
apptainer exec --nv \
  --bind /data/raw/ECT:/data/raw/ECT:rw \
  --pwd /data/raw/ECT/worktrees/q256_g110_moment_transport \
  /data/temp/ect001-pytorch2401-sandbox \
  /usr/bin/python3 scripts/numpy_pickle_compat_exec.py \
  analysis/q256_moment_transport_preflight.py \
  --receipt-root /home/ECT001/q256-factorial-results-0033dbb/combined_receipts \
  --inputs analysis/q256_g110_moment_transport/preflight_inputs.json \
  --out /data/raw/ECT/ect_runs/q256_g110_moment_transport/preflight/heldout-attempt-002 \
  --device cuda \
  --amp \
  > /data/raw/ECT/ect_runs/q256_g110_moment_transport/preflight/heldout-attempt-002/stdout.log \
  2> /data/raw/ECT/ect_runs/q256_g110_moment_transport/preflight/heldout-attempt-002/stderr.log
PREFLIGHT_RC=$?

printf '%s\n' "$PREFLIGHT_RC" > /data/raw/ECT/ect_runs/q256_g110_moment_transport/preflight/heldout-attempt-002/exit_code.txt
date -u +%Y-%m-%dT%H:%M:%SZ > /data/raw/ECT/ect_runs/q256_g110_moment_transport/preflight/heldout-attempt-002/finished_utc.txt
exit "$PREFLIGHT_RC"
