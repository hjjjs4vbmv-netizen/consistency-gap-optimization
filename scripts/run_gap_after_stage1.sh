#!/usr/bin/env bash

# Watch the stage-1 screen, select and freeze g*, run the 16-cell stage 2, and
# create the final factorial summary.  Safe to restart: completed outputs are
# skipped only after validation; incomplete outputs stop the pipeline.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${ECT_PROJECT_ROOT:-/root/autodl-tmp/ect_project}"
RUNS_ROOT="${ECT_GAP_RUNS_ROOT:-${PROJECT_ROOT}/runs/gap-factorial-20260726}"
EVAL_ROOT="${ECT_GAP_EVAL_ROOT:-${PROJECT_ROOT}/runs/gap-factorial-20260726-eval}"
ANALYSIS_ROOT="${ECT_GAP_ANALYSIS_ROOT:-${PROJECT_ROOT}/runs/gap-factorial-20260726-analysis}"
PYTHON_BIN="${ECT_BOOTSTRAP_PYTHON:-/root/miniconda3/bin/python}"
STAGE1_SCREEN="${ECT_STAGE1_SCREEN_NAME:-ect-gap-stage1}"
WAIT_INTERVAL="${ECT_STAGE1_POLL_INTERVAL:-30}"
WAIT_TIMEOUT="${ECT_STAGE1_WAIT_TIMEOUT:-43200}"
SELECT_TIMEOUT="${ECT_SELECT_TIMEOUT:-600}"
STAGE2_TIMEOUT="${ECT_STAGE2_WALL_TIMEOUT:-108000}"
SUMMARY_TIMEOUT="${ECT_SUMMARY_TIMEOUT:-600}"

fail() {
    printf '[run_gap_after_stage1] ERROR: %s\n' "$*" >&2
    exit 1
}

for integer_setting in \
    "ECT_STAGE1_POLL_INTERVAL:${WAIT_INTERVAL}" \
    "ECT_STAGE1_WAIT_TIMEOUT:${WAIT_TIMEOUT}" \
    "ECT_SELECT_TIMEOUT:${SELECT_TIMEOUT}" \
    "ECT_STAGE2_WALL_TIMEOUT:${STAGE2_TIMEOUT}" \
    "ECT_SUMMARY_TIMEOUT:${SUMMARY_TIMEOUT}"
do
    setting_name="${integer_setting%%:*}"
    setting_value="${integer_setting#*:}"
    [[ "${setting_value}" =~ ^[1-9][0-9]*$ ]] ||
        fail "${setting_name} must be a positive integer"
done
[[ "${STAGE1_SCREEN}" =~ ^[A-Za-z0-9_.-]+$ ]] ||
    fail "invalid screen session name: ${STAGE1_SCREEN}"
[[ -x "${PYTHON_BIN}" ]] || fail "bootstrap Python not found: ${PYTHON_BIN}"
[[ -x "$(command -v screen || true)" ]] || fail "screen is required"
[[ -x "$(command -v timeout || true)" ]] || fail "GNU timeout is required"
[[ -x "$(command -v flock || true)" ]] || fail "flock is required"

mkdir -p "${RUNS_ROOT}" "${EVAL_ROOT}" "${ANALYSIS_ROOT}"
exec 8>>"${ANALYSIS_ROOT}/after_stage1.lock"
flock -n 8 || fail "another after-stage1 watcher is already running"

record_failure() {
    local status="$?"
    if (( status != 0 )); then
        printf 'failed_utc=%s exit_code=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" \
            >> "${ANALYSIS_ROOT}/orchestration.failures.log"
    fi
}
trap record_failure EXIT

screen_is_alive() {
    screen -ls 2>/dev/null |
        awk -v target="${STAGE1_SCREEN}" '
            {
                entry=$1
                sub(/^[0-9]+\./, "", entry)
                if (entry == target && $0 ~ /\((Detached|Attached)\)/) {
                    found=1
                }
            }
            END {exit(found ? 0 : 1)}
        '
}

run_with_timeout() {
    local time_limit="$1"
    local context="$2"
    shift 2
    local status
    set +e
    timeout --signal=TERM --kill-after=15s "${time_limit}" "$@"
    status="$?"
    set -e
    (( status == 0 )) || fail "${context} failed with exit code ${status}"
}

selection_is_valid() {
    local selected_file="${ANALYSIS_ROOT}/selected_g.txt"
    local selection_json="${ANALYSIS_ROOT}/selection.json"
    [[ -s "${selected_file}" && -s "${selection_json}" ]] || return 1
    "${PYTHON_BIN}" - "${selected_file}" "${selection_json}" <<'PY'
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import sys

selected_path, json_path = map(Path, sys.argv[1:])
lines = [line.strip() for line in selected_path.read_text(
    encoding="utf-8"
).splitlines() if line.strip()]
if len(lines) != 1:
    raise SystemExit("selected_g.txt must have exactly one non-empty line")
try:
    selected = Decimal(lines[0])
except InvalidOperation as exc:
    raise SystemExit(f"invalid selected_g.txt: {exc}")
allowed = {Decimal(text) for text in ("0.97", "1.032", "1.06", "1.10")}
if selected not in allowed:
    raise SystemExit(f"unexpected selected global scale: {selected}")
payload = json.loads(json_path.read_text(encoding="utf-8"))
if payload.get("status") != "passed":
    raise SystemExit("selection.json does not record status=passed")
if Decimal(str(payload.get("selected_global_scale"))) != selected:
    raise SystemExit("selection.json numeric scale mismatch")
if Decimal(str(payload.get("selected_global_scale_text"))) != selected:
    raise SystemExit("selection.json text scale mismatch")
PY
}

summary_is_valid() {
    local outdir="${ANALYSIS_ROOT}/final-summary"
    local required
    for required in \
        per_cell_metrics.csv \
        per_seed_effects.csv \
        factorial_summary.csv \
        factorial_summary.json \
        factorial_summary.md
    do
        [[ -s "${outdir}/${required}" ]] || return 1
    done
    "${PYTHON_BIN}" - \
        "${outdir}/factorial_summary.json" \
        "${ANALYSIS_ROOT}/selected_g.txt" <<'PY'
from decimal import Decimal
import json
from pathlib import Path
import sys
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "passed":
    raise SystemExit("factorial_summary.json does not record status=passed")
selected = Decimal(Path(sys.argv[2]).read_text(encoding="utf-8").strip())
if Decimal(str(payload.get("selected_global_scale"))) != selected:
    raise SystemExit("factorial summary was produced for a different g*")
matrix = payload.get("matrix")
if not isinstance(matrix, dict):
    raise SystemExit("factorial summary has no matrix accounting")
expected = {
    "unique_training_cells": 18,
    "evaluated_training_seed_nfe_cells": 36,
    "unique_metric_files_read_once": 72,
}
for key, value in expected.items():
    if matrix.get(key) != value:
        raise SystemExit(
            f"factorial summary matrix mismatch for {key}: {matrix.get(key)!r}"
        )
PY
}

STAGE1_COMPLETE="${RUNS_ROOT}/stage1.complete"
STAGE1_FAILED="${RUNS_ROOT}/stage1.failed"
wait_started="$(date +%s)"
while [[ ! -s "${STAGE1_COMPLETE}" ]]; do
    if [[ -e "${STAGE1_FAILED}" ]]; then
        fail "stage 1 recorded failure in ${STAGE1_FAILED}"
    fi
    if ! screen_is_alive; then
        # Avoid racing the final sentinel write against screen teardown.
        sleep 2
        [[ -s "${STAGE1_COMPLETE}" ]] ||
            fail "stage-1 screen '${STAGE1_SCREEN}' disappeared without stage1.complete"
        break
    fi
    now="$(date +%s)"
    if (( now - wait_started >= WAIT_TIMEOUT )); then
        fail "timed out waiting for stage 1 after ${WAIT_TIMEOUT} seconds"
    fi
    printf '[run_gap_after_stage1] waiting for %s; screen=%s utc=%s\n' \
        "${STAGE1_COMPLETE}" "${STAGE1_SCREEN}" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    sleep "${WAIT_INTERVAL}"
done
printf '[run_gap_after_stage1] stage 1 completion detected\n'

if selection_is_valid; then
    printf '[run_gap_after_stage1] validated frozen g* selection exists; skipping selection\n'
else
    if [[ -e "${ANALYSIS_ROOT}/selected_g.txt" ||
          -e "${ANALYSIS_ROOT}/selection.json" ||
          -e "${ANALYSIS_ROOT}/response_curve.csv" ||
          -e "${ANALYSIS_ROOT}/response_curve.md" ]]; then
        fail "incomplete or invalid pre-existing selection artifacts"
    fi
    [[ -f "${ROOT_DIR}/scripts/select_gap_scale.py" ]] ||
        fail "missing scripts/select_gap_scale.py"
    run_with_timeout "${SELECT_TIMEOUT}" "global scale selection" \
        "${PYTHON_BIN}" "${ROOT_DIR}/scripts/select_gap_scale.py" \
        --runs-root "${RUNS_ROOT}" \
        --eval-root "${EVAL_ROOT}" \
        --outdir "${ANALYSIS_ROOT}"
    selection_is_valid || fail "new global scale selection failed validation"
fi

[[ -f "${ROOT_DIR}/scripts/run_gap_stage2.sh" ]] ||
    fail "missing scripts/run_gap_stage2.sh"
run_with_timeout "${STAGE2_TIMEOUT}" "stage 2" \
    env \
    ECT_PROJECT_ROOT="${PROJECT_ROOT}" \
    ECT_GAP_RUNS_ROOT="${RUNS_ROOT}" \
    ECT_GAP_EVAL_ROOT="${EVAL_ROOT}" \
    ECT_GAP_ANALYSIS_ROOT="${ANALYSIS_ROOT}" \
    ECT_SELECTED_G_FILE="${ANALYSIS_ROOT}/selected_g.txt" \
    bash "${ROOT_DIR}/scripts/run_gap_stage2.sh"
[[ -s "${RUNS_ROOT}/stage2.complete" ]] ||
    fail "stage 2 returned without stage2.complete"

if summary_is_valid; then
    printf '[run_gap_after_stage1] validated final summary exists; skipping summary\n'
else
    if [[ -e "${ANALYSIS_ROOT}/final-summary" ]]; then
        fail "incomplete or invalid pre-existing final-summary directory"
    fi
    [[ -f "${ROOT_DIR}/scripts/summarize_gap_factorial.py" ]] ||
        fail "missing scripts/summarize_gap_factorial.py"
    run_with_timeout "${SUMMARY_TIMEOUT}" "factorial summary" \
        "${PYTHON_BIN}" "${ROOT_DIR}/scripts/summarize_gap_factorial.py" \
        --runs-root "${RUNS_ROOT}" \
        --eval-root "${EVAL_ROOT}" \
        --selection-json "${ANALYSIS_ROOT}/selection.json" \
        --outdir "${ANALYSIS_ROOT}/final-summary"
    summary_is_valid || fail "new factorial summary failed validation"
fi

PIPELINE_COMPLETE="${ANALYSIS_ROOT}/pipeline.complete"
if [[ ! -e "${PIPELINE_COMPLETE}" ]]; then
    (
        set -o noclobber
        {
            printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            printf 'stage1_complete=%s\n' "${STAGE1_COMPLETE}"
            printf 'stage2_complete=%s\n' "${RUNS_ROOT}/stage2.complete"
            printf 'selection_json=%s\n' "${ANALYSIS_ROOT}/selection.json"
            printf 'summary_json=%s\n' \
                "${ANALYSIS_ROOT}/final-summary/factorial_summary.json"
        } > "${PIPELINE_COMPLETE}"
    ) || fail "could not create pipeline completion marker"
fi
printf '[run_gap_after_stage1] full gap-factorial pipeline completed and validated\n'
