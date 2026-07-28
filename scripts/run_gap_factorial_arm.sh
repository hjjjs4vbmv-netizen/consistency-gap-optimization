#!/usr/bin/env bash

# Run one frozen 256 kimg arm of the global/local gap-factorization study.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ECT_ENV_NAME:-ect-exp}"
CONDA_BIN="${ECT_CONDA_BIN:-/root/miniconda3/bin/conda}"
PYTHON_BIN="${ECT_BOOTSTRAP_PYTHON:-/root/miniconda3/bin/python}"
PROJECT_ROOT="${ECT_PROJECT_ROOT:-/root/autodl-tmp/ect_project}"
RUNS_ROOT="${ECT_GAP_RUNS_ROOT:-${PROJECT_ROOT}/runs/gap-factorial-20260726}"
DATA="${ECT_DATA_PATH:-${PROJECT_ROOT}/datasets/cifar10-32x32.zip}"
TRANSFER="${ECT_TRANSFER_PATH:-${PROJECT_ROOT}/pretrained/edm-cifar10-32x32-uncond-vp.pkl}"

ARM=""
SEED=""
GLOBAL_SCALE="1.0"
DURATION="0.256"

fail() {
    printf '[run_gap_factorial_arm] ERROR: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_gap_factorial_arm.sh \
    --arm {fixed|global|local-v1-bridge|local-conservative|combined-conservative|local-aggressive|combined-aggressive} \
    --seed {0|1|2} [--global-scale FLOAT] [--duration MIMG]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arm) ARM="${2:-}"; shift 2 ;;
        --seed) SEED="${2:-}"; shift 2 ;;
        --global-scale) GLOBAL_SCALE="${2:-}"; shift 2 ;;
        --duration) DURATION="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "unknown option: $1" ;;
    esac
done

[[ -n "${ARM}" ]] || fail "--arm is required"
case "${SEED}" in 0|1|2) ;; *) fail "--seed must be 0, 1, or 2" ;; esac
[[ -x "${CONDA_BIN}" ]] || fail "conda executable not found: ${CONDA_BIN}"
[[ -x "${PYTHON_BIN}" ]] || fail "bootstrap Python not found: ${PYTHON_BIN}"
"${PYTHON_BIN}" - "${GLOBAL_SCALE}" "${DURATION}" <<'PY'
import math
import sys
for name, raw in [('global scale', sys.argv[1]), ('duration', sys.argv[2])]:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(f'{name} must be finite and > 0, got {raw}')
PY
GLOBAL_SCALE="$("${PYTHON_BIN}" - "${GLOBAL_SCALE}" <<'PY'
import sys
print(f'{float(sys.argv[1]):.17g}')
PY
)"
DURATION="$("${PYTHON_BIN}" - "${DURATION}" <<'PY'
import sys
print(f'{float(sys.argv[1]):.17g}')
PY
)"

SCHEDULE=""
PROFILE="none"
case "${ARM}" in
    fixed)
        SCHEDULE="sigmoid"
        [[ "${GLOBAL_SCALE}" == "1" ]] || \
            fail "fixed arm requires --global-scale 1"
        ;;
    global)
        SCHEDULE="global_sigmoid"
        ;;
    local-v1-bridge)
        SCHEDULE="local_tbin_v1"
        PROFILE="aggressive"
        [[ "${GLOBAL_SCALE}" == "1" ]] || \
            fail "local-v1-bridge requires --global-scale 1"
        ;;
    local-conservative)
        SCHEDULE="local_tbin_v2"
        PROFILE="conservative"
        [[ "${GLOBAL_SCALE}" == "1" ]] || \
            fail "local-conservative requires --global-scale 1"
        ;;
    combined-conservative)
        SCHEDULE="local_tbin_v3"
        PROFILE="conservative"
        ;;
    local-aggressive)
        SCHEDULE="local_tbin_v2"
        PROFILE="aggressive"
        [[ "${GLOBAL_SCALE}" == "1" ]] || \
            fail "local-aggressive requires --global-scale 1"
        ;;
    combined-aggressive)
        SCHEDULE="local_tbin_v3"
        PROFILE="aggressive"
        ;;
    *) fail "unsupported arm: ${ARM}" ;;
esac

SCALE_SLUG="$("${PYTHON_BIN}" - "${GLOBAL_SCALE}" <<'PY'
import sys
print(f'{float(sys.argv[1]):.4f}'.replace('.', 'p'))
PY
)"
KIMG="$("${PYTHON_BIN}" - "${DURATION}" <<'PY'
import sys
print(int(float(sys.argv[1]) * 1000))
PY
)"
OUTDIR="${RUNS_ROOT}/${ARM}-g${SCALE_SLUG}-seed${SEED}-${KIMG}k"
[[ ! -e "${OUTDIR}" ]] || fail "refusing to overwrite existing output: ${OUTDIR}"
[[ -f "${DATA}" ]] || fail "dataset not found: ${DATA}"
[[ -f "${TRANSFER}" ]] || fail "transfer checkpoint not found: ${TRANSFER}"
mkdir -p "${OUTDIR}"

LOCAL_ARGS=()
if [[ "${PROFILE}" == "conservative" ]]; then
    LOCAL_ARGS+=(
        --local-tbin-warmup-updates=64
        --local-tbin-gain=0.25
        --local-tbin-min-scale=0.85
        --local-tbin-max-scale=1.25
    )
elif [[ "${PROFILE}" == "aggressive" ]]; then
    LOCAL_ARGS+=(
        --local-tbin-warmup-updates=32
        --local-tbin-gain=0.5
        --local-tbin-min-scale=0.75
        --local-tbin-max-scale=1.5
    )
fi

CMD=(
    "${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}"
    python "${ROOT_DIR}/ct_train.py"
    --data="${DATA}"
    --outdir="${OUTDIR}"
    --nosubdir
    --cond=False
    --arch=ddpmpp
    --precond=ect
    --batch=128
    --batch-gpu=16
    --optim=RAdam
    --lr=0.0001
    --dropout=0.2
    --augment=0
    --mapping="${SCHEDULE}"
    --global-gap-scale="${GLOBAL_SCALE}"
    -q 256
    -k 8
    -b 1
    -c 0
    --double=10000
    --ema_beta=0.9993
    --seed="${SEED}"
    --fp16=True
    --enable_amp=True
    --metrics=none
    --duration="${DURATION}"
    --tick=10
    --snap=0
    --dump=0
    --ckpt=10
    --sample_every=26
    --adaptive-update-kimg=0.5
    --transfer="${TRANSFER}"
    "${LOCAL_ARGS[@]}"
)

{
    printf 'arm=%s\n' "${ARM}"
    printf 'schedule=%s\n' "${SCHEDULE}"
    printf 'local_profile=%s\n' "${PROFILE}"
    printf 'global_gap_scale=%s\n' "${GLOBAL_SCALE}"
    printf 'seed=%s\n' "${SEED}"
    printf 'duration_mimg=%s\n' "${DURATION}"
    printf 'data_sha256=%s\n' "$(sha256sum "${DATA}" | awk '{print $1}')"
    printf 'transfer_sha256=%s\n' "$(sha256sum "${TRANSFER}" | awk '{print $1}')"
    printf 'source_sha256=%s\n' "$(
        for source_file in \
            "${ROOT_DIR}/ct_train.py" \
            "${ROOT_DIR}/training/loss.py" \
            "${ROOT_DIR}/training/schedules.py" \
            "${ROOT_DIR}/training/ct_training_loop.py" \
            "${ROOT_DIR}/training/networks.py" \
            "${ROOT_DIR}/training/dataset.py" \
            "${ROOT_DIR}/scripts/run_gap_factorial_arm.sh"
        do
            sha256sum "${source_file}" | awk '{print $1}'
        done | sha256sum | awk '{print $1}'
    )"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'exact_command='
    printf '%q ' "${CMD[@]}"
    printf '\n'
} > "${OUTDIR}/experiment_meta.env"

printf '[run_gap_factorial_arm] output=%s\n' "${OUTDIR}"
printf '[run_gap_factorial_arm] command='
printf '%q ' "${CMD[@]}"
printf '\n'

set +e
"${CMD[@]}" 2>&1 | tee "${OUTDIR}/runner.log"
STATUS="${PIPESTATUS[0]}"
set -e
printf 'exit_code=%s\nfinished_utc=%s\n' \
    "${STATUS}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUTDIR}/experiment_meta.env"
[[ "${STATUS}" -eq 0 ]] || exit "${STATUS}"

[[ -s "${OUTDIR}/train_summary.csv" ]] || fail "missing train_summary.csv"
[[ -s "${OUTDIR}/network-snapshot-latest.pkl" ]] || fail "missing final snapshot"
[[ -s "${OUTDIR}/training-state-latest.pt" ]] || fail "missing final training state"
sha256sum "${OUTDIR}/network-snapshot-latest.pkl" > "${OUTDIR}/checkpoint.sha256"
VALIDATION_ARGS=(
    --run-dir "${OUTDIR}" \
    --expected-kimg "${KIMG}" \
    --expected-schedule "${SCHEDULE}"
)
case "${SCHEDULE}" in
    local_tbin_v1|local_tbin_v2|local_tbin_v3)
        VALIDATION_ARGS+=(--require-controller-active)
        ;;
esac
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/verify_gap_factorial_arm.py" \
    "${VALIDATION_ARGS[@]}"
tail -n 1 "${OUTDIR}/train_summary.csv"
