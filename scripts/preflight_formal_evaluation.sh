#!/usr/bin/env bash
# Validate a formal staged-evaluation launch without starting a metric job.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/preflight_formal_evaluation.sh \
    --frozen-matrix PATH --runtime-manifest PATH --data PATH --outdir PATH

The command fails closed if the repository is dirty, the output directory is
non-empty, a runtime binding drifts from its frozen matrix, an input SHA or
integrity receipt is invalid, or the formal promotion policy is incomplete.
It performs only a dry run; it never starts metric generation.
EOF
}

fail() {
  printf '[preflight_formal_evaluation] ERROR: %s\n' "$*" >&2
  exit 1
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FROZEN_MATRIX=""
RUNTIME_MANIFEST=""
DATA=""
OUTDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --frozen-matrix) FROZEN_MATRIX="${2:-}"; shift 2 ;;
    --runtime-manifest) RUNTIME_MANIFEST="${2:-}"; shift 2 ;;
    --data) DATA="${2:-}"; shift 2 ;;
    --outdir) OUTDIR="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[[ -n "$FROZEN_MATRIX" && -n "$RUNTIME_MANIFEST" && -n "$DATA" && -n "$OUTDIR" ]] || {
  usage >&2
  fail "--frozen-matrix, --runtime-manifest, --data, and --outdir are required"
}

cd "$ROOT_DIR"
[[ -z "$(git status --porcelain)" ]] || \
  fail "repository has uncommitted or untracked changes; launch from a committed checkout"
[[ -f "$FROZEN_MATRIX" ]] || fail "frozen matrix not found: $FROZEN_MATRIX"
[[ -f "$RUNTIME_MANIFEST" ]] || fail "runtime manifest not found: $RUNTIME_MANIFEST"
[[ -f "$DATA" ]] || fail "dataset not found: $DATA"
if [[ -e "$OUTDIR" ]] && [[ ! -d "$OUTDIR" ]]; then
  fail "output path exists but is not a directory: $OUTDIR"
fi
if [[ -d "$OUTDIR" ]] && [[ -n "$(find "$OUTDIR" -mindepth 1 -print -quit)" ]]; then
  fail "output directory is non-empty: $OUTDIR"
fi

python scripts/run_staged_evaluation.py \
  --frozen-matrix "$FROZEN_MATRIX" \
  --manifest "$RUNTIME_MANIFEST" \
  --data "$DATA" \
  --outdir "$OUTDIR" \
  --phase formal \
  --dry-run

printf '[preflight_formal_evaluation] PASS: no metric job was started\n'
