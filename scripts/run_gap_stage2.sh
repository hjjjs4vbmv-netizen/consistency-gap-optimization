#!/usr/bin/env bash

# Run and evaluate the 16 confirmation/factorial cells that remain after the
# seed-0 global-gap response curve.  The selected scale is immutable for the
# lifetime of this stage and every pre-existing artifact is validated before it
# is skipped.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${ECT_PROJECT_ROOT:-/root/autodl-tmp/ect_project}"
RUNS_ROOT="${ECT_GAP_RUNS_ROOT:-${PROJECT_ROOT}/runs/gap-factorial-20260726}"
EVAL_ROOT="${ECT_GAP_EVAL_ROOT:-${PROJECT_ROOT}/runs/gap-factorial-20260726-eval}"
ANALYSIS_ROOT="${ECT_GAP_ANALYSIS_ROOT:-${PROJECT_ROOT}/runs/gap-factorial-20260726-analysis}"
SELECTED_G_FILE="${ECT_SELECTED_G_FILE:-${ANALYSIS_ROOT}/selected_g.txt}"
PYTHON_BIN="${ECT_BOOTSTRAP_PYTHON:-/root/miniconda3/bin/python}"
MIN_FREE_KB="${ECT_MIN_FREE_KB:-8388608}"
TRAIN_TIMEOUT="${ECT_TRAIN_TIMEOUT:-4200}"
EVAL_TIMEOUT="${ECT_EVAL_TIMEOUT:-1800}"
MONITOR_INTERVAL="${ECT_DISK_MONITOR_INTERVAL:-15}"

fail() {
    printf '[run_gap_stage2] ERROR: %s\n' "$*" >&2
    exit 1
}

for integer_setting in \
    "ECT_MIN_FREE_KB:${MIN_FREE_KB}" \
    "ECT_TRAIN_TIMEOUT:${TRAIN_TIMEOUT}" \
    "ECT_EVAL_TIMEOUT:${EVAL_TIMEOUT}" \
    "ECT_DISK_MONITOR_INTERVAL:${MONITOR_INTERVAL}"
do
    setting_name="${integer_setting%%:*}"
    setting_value="${integer_setting#*:}"
    [[ "${setting_value}" =~ ^[1-9][0-9]*$ ]] ||
        fail "${setting_name} must be a positive integer"
done

[[ -x "${PYTHON_BIN}" ]] || fail "bootstrap Python not found: ${PYTHON_BIN}"
[[ -x "$(command -v timeout || true)" ]] || fail "GNU timeout is required"
[[ -x "$(command -v flock || true)" ]] || fail "flock is required"
[[ -f "${ROOT_DIR}/scripts/run_gap_factorial_arm.sh" ]] ||
    fail "missing per-arm training runner"
[[ -f "${ROOT_DIR}/scripts/evaluate_gap_factorial_arm.sh" ]] ||
    fail "missing per-arm evaluation runner"
[[ -f "${ROOT_DIR}/scripts/verify_gap_factorial_arm.py" ]] ||
    fail "missing training verifier"
[[ -s "${SELECTED_G_FILE}" ]] ||
    fail "missing frozen selection: ${SELECTED_G_FILE}"

mkdir -p "${RUNS_ROOT}" "${EVAL_ROOT}" "${ANALYSIS_ROOT}"

# Prevent concurrent stage-2 drivers without preventing a safe later resume.
exec 9>>"${RUNS_ROOT}/stage2.lock"
flock -n 9 || fail "another stage-2 driver holds ${RUNS_ROOT}/stage2.lock"

SELECTED_G="$("${PYTHON_BIN}" - "${SELECTED_G_FILE}" <<'PY'
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sys

path = Path(sys.argv[1])
lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
         if line.strip()]
if len(lines) != 1:
    raise SystemExit(f"expected exactly one non-empty line in {path}")
try:
    selected = Decimal(lines[0])
except InvalidOperation as exc:
    raise SystemExit(f"invalid selected scale {lines[0]!r}: {exc}")
allowed = {Decimal(text): text for text in ("0.97", "1.032", "1.06", "1.10")}
if selected not in allowed:
    raise SystemExit(
        f"selected scale {selected} is not one of {tuple(allowed.values())}"
    )
print(allowed[selected])
PY
)" || fail "could not parse ${SELECTED_G_FILE}"
SELECTED_G_SHA256="$(sha256sum "${SELECTED_G_FILE}" | awk '{print $1}')"

SELECTION_JSON="${ANALYSIS_ROOT}/selection.json"
if [[ -e "${SELECTION_JSON}" ]]; then
    "${PYTHON_BIN}" - "${SELECTION_JSON}" "${SELECTED_G}" <<'PY'
from decimal import Decimal
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "passed":
    raise SystemExit("selection.json does not record status=passed")
expected = Decimal(sys.argv[2])
if Decimal(str(payload.get("selected_global_scale"))) != expected:
    raise SystemExit("selection.json numeric scale disagrees with selected_g.txt")
if Decimal(str(payload.get("selected_global_scale_text"))) != expected:
    raise SystemExit("selection.json text scale disagrees with selected_g.txt")
PY
fi

SELECTION_LOCK="${RUNS_ROOT}/stage2_selection.lock"
if [[ -e "${SELECTION_LOCK}" ]]; then
    [[ -f "${SELECTION_LOCK}" ]] || fail "selection lock is not a file"
    LOCKED_G="$(awk -F= '$1 == "selected_g" {print $2}' "${SELECTION_LOCK}")"
    LOCKED_SHA="$(awk -F= '$1 == "selected_g_sha256" {print $2}' "${SELECTION_LOCK}")"
    [[ "${LOCKED_G}" == "${SELECTED_G}" ]] ||
        fail "stage-2 selection is already locked to ${LOCKED_G}"
    [[ "${LOCKED_SHA}" == "${SELECTED_G_SHA256}" ]] ||
        fail "selected_g.txt changed after stage-2 was first initialized"
else
    (
        set -o noclobber
        {
            printf 'selected_g=%s\n' "${SELECTED_G}"
            printf 'selected_g_sha256=%s\n' "${SELECTED_G_SHA256}"
            printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        } > "${SELECTION_LOCK}"
    ) || fail "could not create immutable selection lock"
fi

assert_selection_unchanged() {
    local current_sha
    current_sha="$(sha256sum "${SELECTED_G_FILE}" | awk '{print $1}')"
    [[ "${current_sha}" == "${SELECTED_G_SHA256}" ]] ||
        fail "selected_g.txt changed while stage 2 was running"
}

free_kb() {
    df -Pk "${PROJECT_ROOT}" | awk 'NR == 2 {print $4}'
}

require_disk_headroom() {
    local context="${1:-operation}"
    local available
    available="$(free_kb)"
    [[ "${available}" =~ ^[0-9]+$ ]] ||
        fail "could not determine free disk space before ${context}"
    if (( available < MIN_FREE_KB )); then
        fail "only ${available} KiB free before ${context}; require ${MIN_FREE_KB} KiB"
    fi
}

ACTIVE_TIMEOUT_PID=""
ACTIVE_MONITOR_PID=""
cleanup_active_jobs() {
    if [[ -n "${ACTIVE_MONITOR_PID}" ]] &&
       kill -0 "${ACTIVE_MONITOR_PID}" 2>/dev/null; then
        kill -TERM "${ACTIVE_MONITOR_PID}" 2>/dev/null || true
    fi
    if [[ -n "${ACTIVE_TIMEOUT_PID}" ]] &&
       kill -0 "${ACTIVE_TIMEOUT_PID}" 2>/dev/null; then
        kill -TERM "${ACTIVE_TIMEOUT_PID}" 2>/dev/null || true
    fi
}
trap cleanup_active_jobs EXIT

run_timed_monitored() {
    local time_limit="$1"
    local context="$2"
    shift 2
    local status monitor_status available

    require_disk_headroom "${context}"
    timeout --signal=TERM --kill-after=15s "${time_limit}" "$@" &
    ACTIVE_TIMEOUT_PID="$!"
    (
        while kill -0 "${ACTIVE_TIMEOUT_PID}" 2>/dev/null; do
            available="$(free_kb)"
            if [[ ! "${available}" =~ ^[0-9]+$ ]] ||
               (( available < MIN_FREE_KB )); then
                printf '[run_gap_stage2] ERROR: disk headroom monitor stopped %s; free_kb=%s required_kb=%s\n' \
                    "${context}" "${available:-unknown}" "${MIN_FREE_KB}" >&2
                kill -TERM "${ACTIVE_TIMEOUT_PID}" 2>/dev/null || true
                exit 72
            fi
            sleep "${MONITOR_INTERVAL}"
        done
    ) &
    ACTIVE_MONITOR_PID="$!"

    set +e
    wait "${ACTIVE_TIMEOUT_PID}"
    status="$?"
    set -e
    ACTIVE_TIMEOUT_PID=""

    if kill -0 "${ACTIVE_MONITOR_PID}" 2>/dev/null; then
        kill -TERM "${ACTIVE_MONITOR_PID}" 2>/dev/null || true
    fi
    set +e
    wait "${ACTIVE_MONITOR_PID}"
    monitor_status="$?"
    set -e
    ACTIVE_MONITOR_PID=""

    require_disk_headroom "${context} completion"
    if (( status != 0 )); then
        fail "${context} failed with exit code ${status}"
    fi
    # 0 means the command outlived a final monitor iteration; 143 means this
    # driver stopped the monitor after successful command completion.
    if (( monitor_status != 0 && monitor_status != 143 )); then
        fail "${context} disk monitor failed with exit code ${monitor_status}"
    fi
}

scale_slug() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import sys
print(f"{float(sys.argv[1]):.4f}".replace(".", "p"))
PY
}

verify_training() {
    local run_dir="$1"
    local expected_schedule="$2"
    local controller_required="$3"
    local args=(
        --run-dir "${run_dir}"
        --expected-kimg 256
        --expected-schedule "${expected_schedule}"
    )
    if [[ "${controller_required}" == "1" ]]; then
        args+=(--require-controller-active)
    fi
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/verify_gap_factorial_arm.py" "${args[@]}"
}

validate_eval_nfe() {
    local eval_dir="$1"
    local label="$2"
    local nfe="$3"
    local checkpoint="$4"
    "${PYTHON_BIN}" - "${eval_dir}" "${label}" "${nfe}" "${checkpoint}" <<'PY'
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

eval_dir = Path(sys.argv[1])
label = sys.argv[2]
nfe = int(sys.argv[3])
checkpoint = Path(sys.argv[4])

def fail(message: str) -> None:
    raise SystemExit(f"[run_gap_stage2 eval validator] {message}")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for required in ("experiment_meta.env", "runner.log"):
    path = eval_dir / required
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"missing or empty {path}")

meta: dict[str, str] = {}
for line in (eval_dir / "experiment_meta.env").read_text(
    encoding="utf-8", errors="strict"
).splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        meta[key] = value
if meta.get("exit_code") != "0":
    fail(f"{eval_dir} does not record exit_code=0")
if meta.get("label") != label or meta.get("nfe") != str(nfe):
    fail(f"{eval_dir} metadata identity mismatch")
if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
    fail(f"missing checkpoint {checkpoint}")
actual_checkpoint_sha = sha256(checkpoint)
if meta.get("checkpoint_sha256") != actual_checkpoint_sha:
    fail(f"{eval_dir} checkpoint hash mismatch")

metrics: dict[str, float] = {}
for metric in ("kid5k_full", "fid5k_full"):
    result_path = eval_dir / f"metric-{metric}.jsonl"
    if not result_path.is_file() or result_path.stat().st_size == 0:
        fail(f"missing metric file {result_path}")
    lines = result_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        fail(f"expected exactly one line in {result_path}")
    payload = json.loads(lines[0])
    if payload.get("metric") != metric:
        fail(f"metric identity mismatch in {result_path}")
    try:
        value = float(payload["results"][metric])
    except (KeyError, TypeError, ValueError) as exc:
        fail(f"invalid metric value in {result_path}: {exc}")
    if not math.isfinite(value):
        fail(f"non-finite metric value in {result_path}")
    metrics[metric] = value

report = {
    "status": "passed",
    "label": label,
    "nfe": nfe,
    "checkpoint_sha256": actual_checkpoint_sha,
    "metrics": metrics,
}
validation_path = eval_dir / "validation.json"
encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
if validation_path.exists():
    if validation_path.read_text(encoding="utf-8") != encoded:
        fail(f"existing validation report disagrees: {validation_path}")
else:
    with validation_path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
PY
}

run_one() {
    local arm="$1"
    local seed="$2"
    local scale="$3"
    local expected_schedule="$4"
    local controller_required="$5"
    local slug label run_dir checkpoint nfe eval_dir valid_evals

    assert_selection_unchanged
    slug="$(scale_slug "${scale}")"
    label="${arm}-g${slug}-seed${seed}-256k"
    run_dir="${RUNS_ROOT}/${label}"
    checkpoint="${run_dir}/network-snapshot-latest.pkl"

    require_disk_headroom "training ${label}"
    if [[ -e "${run_dir}" ]]; then
        [[ -d "${run_dir}" ]] || fail "training output is not a directory: ${run_dir}"
        if verify_training "${run_dir}" "${expected_schedule}" "${controller_required}"; then
            printf '[run_gap_stage2] validated training exists; skipping %s\n' "${label}"
        else
            fail "pre-existing training failed validation: ${run_dir}"
        fi
    else
        printf '[run_gap_stage2] training %s started_utc=%s\n' \
            "${label}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        run_timed_monitored "${TRAIN_TIMEOUT}" "training ${label}" \
            bash "${ROOT_DIR}/scripts/run_gap_factorial_arm.sh" \
            --arm "${arm}" --seed "${seed}" \
            --global-scale "${scale}" --duration 0.256
        verify_training "${run_dir}" "${expected_schedule}" "${controller_required}" ||
            fail "new training failed validation: ${run_dir}"
    fi

    valid_evals=0
    for nfe in 1 2; do
        eval_dir="${EVAL_ROOT}/${label}/nfe${nfe}"
        if [[ ! -e "${eval_dir}" ]]; then
            continue
        fi
        [[ -d "${eval_dir}" ]] ||
            fail "evaluation output is not a directory: ${eval_dir}"
        if validate_eval_nfe "${eval_dir}" "${label}" "${nfe}" "${checkpoint}"; then
            valid_evals=$((valid_evals + 1))
        else
            fail "pre-existing evaluation failed validation: ${eval_dir}"
        fi
    done

    if (( valid_evals == 2 )); then
        printf '[run_gap_stage2] validated evaluation exists; skipping %s\n' "${label}"
    else
        printf '[run_gap_stage2] evaluating %s started_utc=%s\n' \
            "${label}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        run_timed_monitored "${EVAL_TIMEOUT}" "evaluation ${label}" \
            env ECT_GAP_EVAL_ROOT="${EVAL_ROOT}" \
            bash "${ROOT_DIR}/scripts/evaluate_gap_factorial_arm.sh" \
            --run-dir="${run_dir}" --label="${label}"
        for nfe in 1 2; do
            eval_dir="${EVAL_ROOT}/${label}/nfe${nfe}"
            validate_eval_nfe "${eval_dir}" "${label}" "${nfe}" "${checkpoint}" ||
                fail "new evaluation failed validation: ${eval_dir}"
        done
    fi

    assert_selection_unchanged
    printf '[run_gap_stage2] completed %s finished_utc=%s free_kb=%s\n' \
        "${label}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(free_kb)"
}

# Exactly 16 unique remaining cells.  Stage 1 already supplies fixed/global
# seed 0, so only seeds 1/2 are repeated for those arms.
CELL_ARMS=(
    fixed fixed
    global global
    local-conservative local-conservative local-conservative
    combined-conservative combined-conservative combined-conservative
    local-aggressive local-aggressive local-aggressive
    combined-aggressive combined-aggressive combined-aggressive
)
CELL_SEEDS=(1 2 1 2 0 1 2 0 1 2 0 1 2 0 1 2)

(( ${#CELL_ARMS[@]} == 16 )) || fail "internal matrix must contain exactly 16 arms"
(( ${#CELL_SEEDS[@]} == 16 )) || fail "internal seed matrix must contain exactly 16 entries"

SEEN_LABELS=$'\n'
UNIQUE_LABEL_COUNT=0
PLAN_LINES="arm	seed	global_scale	label"
for index in "${!CELL_ARMS[@]}"; do
    arm="${CELL_ARMS[${index}]}"
    seed="${CELL_SEEDS[${index}]}"
    case "${arm}" in
        fixed|local-conservative|local-aggressive) scale="1.0" ;;
        global|combined-conservative|combined-aggressive) scale="${SELECTED_G}" ;;
        *) fail "internal unsupported arm: ${arm}" ;;
    esac
    slug="$(scale_slug "${scale}")"
    label="${arm}-g${slug}-seed${seed}-256k"
    [[ "${SEEN_LABELS}" != *$'\n'"${label}"$'\n'* ]] ||
        fail "duplicate internal cell: ${label}"
    SEEN_LABELS+="${label}"$'\n'
    UNIQUE_LABEL_COUNT=$((UNIQUE_LABEL_COUNT + 1))
    PLAN_LINES+=$'\n'"${arm}"$'\t'"${seed}"$'\t'"${scale}"$'\t'"${label}"
done
(( UNIQUE_LABEL_COUNT == 16 )) ||
    fail "internal matrix does not have 16 unique labels"

PLAN_FILE="${RUNS_ROOT}/stage2_plan.tsv"
if [[ -e "${PLAN_FILE}" ]]; then
    [[ -f "${PLAN_FILE}" ]] || fail "stage-2 plan path is not a file"
    [[ "$(<"${PLAN_FILE}")" == "${PLAN_LINES}" ]] ||
        fail "pre-existing stage-2 plan differs from the frozen 16-cell plan"
else
    (
        set -o noclobber
        printf '%s\n' "${PLAN_LINES}" > "${PLAN_FILE}"
    ) || fail "could not create frozen stage-2 plan"
fi

printf '[run_gap_stage2] selected_g=%s selected_sha256=%s cells=16\n' \
    "${SELECTED_G}" "${SELECTED_G_SHA256}"

for index in "${!CELL_ARMS[@]}"; do
    arm="${CELL_ARMS[${index}]}"
    seed="${CELL_SEEDS[${index}]}"
    case "${arm}" in
        fixed)
            scale="1.0"; expected_schedule="sigmoid"; controller_required=0
            ;;
        global)
            scale="${SELECTED_G}"; expected_schedule="global_sigmoid"; controller_required=0
            ;;
        local-conservative|local-aggressive)
            scale="1.0"; expected_schedule="local_tbin_v2"; controller_required=1
            ;;
        combined-conservative|combined-aggressive)
            scale="${SELECTED_G}"; expected_schedule="local_tbin_v3"; controller_required=1
            ;;
        *) fail "internal unsupported arm: ${arm}" ;;
    esac
    run_one "${arm}" "${seed}" "${scale}" \
        "${expected_schedule}" "${controller_required}"
done

assert_selection_unchanged
STAGE2_COMPLETE="${RUNS_ROOT}/stage2.complete"
if [[ ! -e "${STAGE2_COMPLETE}" ]]; then
    (
        set -o noclobber
        {
            printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            printf 'selected_g=%s\n' "${SELECTED_G}"
            printf 'selected_g_sha256=%s\n' "${SELECTED_G_SHA256}"
            printf 'cells=16\n'
        } > "${STAGE2_COMPLETE}"
    ) || fail "could not create stage-2 completion marker"
fi
printf '[run_gap_stage2] all 16 remaining cells completed and validated\n'
