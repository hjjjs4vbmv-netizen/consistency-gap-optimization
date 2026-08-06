#!/usr/bin/env bash
set -euo pipefail

# These commands validate and collect the portable retrospective package.
# The original 1024-kimg evaluation command is not available: its source
# manifest and host-local inputs were never delivered to this repository.

python scripts/validate_q128_1024k_retrospective_package.py \
  --package results/q128_1024k_retrospective

# Exact historical collector/evaluator commands: NOT_RECORDED.
# Do not treat this script as a command to reproduce metrics from checkpoints;
# that requires the source checkpoints, receipts, environment, and manifest
# identified as unavailable in source_run_manifest.json.
