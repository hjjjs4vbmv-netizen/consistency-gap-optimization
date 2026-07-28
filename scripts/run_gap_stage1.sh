#!/usr/bin/env bash

# Run and evaluate the frozen seed-0 global gap response curve sequentially.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${ECT_PROJECT_ROOT:-/root/autodl-tmp/ect_project}"
RUNS_ROOT="${ECT_GAP_RUNS_ROOT:-${PROJECT_ROOT}/runs/gap-factorial-20260726}"
EVAL_ROOT="${ECT_GAP_EVAL_ROOT:-${PROJECT_ROOT}/runs/gap-factorial-20260726-eval}"
PYTHON_BIN="${ECT_BOOTSTRAP_PYTHON:-/root/miniconda3/bin/python}"
MIN_FREE_KB="${ECT_MIN_FREE_KB:-8388608}"
TRAIN_TIMEOUT="${ECT_TRAIN_TIMEOUT:-4200}"
EVAL_TIMEOUT="${ECT_EVAL_TIMEOUT:-1800}"

mkdir -p "${RUNS_ROOT}" "${EVAL_ROOT}"

free_kb() {
    df -Pk "${PROJECT_ROOT}" | awk 'NR == 2 {print $4}'
}

require_disk_headroom() {
    local available
    available="$(free_kb)"
    if (( available < MIN_FREE_KB )); then
        printf '[run_gap_stage1] ERROR: only %s KiB free; require %s KiB\n' \
            "${available}" "${MIN_FREE_KB}" >&2
        exit 1
    fi
}

scale_slug() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import sys
print(f'{float(sys.argv[1]):.4f}'.replace('.', 'p'))
PY
}

run_one() {
    local arm="$1"
    local scale="$2"
    local slug run_dir label
    slug="$(scale_slug "${scale}")"
    label="${arm}-g${slug}-seed0-256k"
    run_dir="${RUNS_ROOT}/${label}"

    require_disk_headroom
    if [[ -f "${run_dir}/validation.json" ]] &&
       grep -q '"status": "passed"' "${run_dir}/validation.json"; then
        printf '[run_gap_stage1] verified training exists; skipping %s\n' "${label}"
    elif [[ -e "${run_dir}" ]]; then
        printf '[run_gap_stage1] ERROR: incomplete existing training: %s\n' \
            "${run_dir}" >&2
        exit 1
    else
        printf '[run_gap_stage1] training %s started_utc=%s\n' \
            "${label}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        timeout --signal=TERM --kill-after=15s "${TRAIN_TIMEOUT}" \
            bash "${ROOT_DIR}/scripts/run_gap_factorial_arm.sh" \
            --arm "${arm}" --seed 0 --global-scale "${scale}" --duration 0.256
    fi

    require_disk_headroom
    printf '[run_gap_stage1] evaluating %s started_utc=%s\n' \
        "${label}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    timeout --signal=TERM --kill-after=15s "${EVAL_TIMEOUT}" \
        env ECT_GAP_EVAL_ROOT="${EVAL_ROOT}" \
        bash "${ROOT_DIR}/scripts/evaluate_gap_factorial_arm.sh" \
        --run-dir="${run_dir}" --label="${label}"
    printf '[run_gap_stage1] completed %s finished_utc=%s free_kb=%s\n' \
        "${label}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(free_kb)"
}

run_one fixed 1.0
run_one global 0.97
run_one global 1.032
run_one global 1.06
run_one global 1.10

printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${RUNS_ROOT}/stage1.complete"
printf '[run_gap_stage1] all five response-curve cells completed\n'
