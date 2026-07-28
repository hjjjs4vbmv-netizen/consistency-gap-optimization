#!/usr/bin/env bash

# Evaluate one completed gap-factorization arm with the frozen 5k protocol.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ECT_ENV_NAME:-ect-exp}"
CONDA_BIN="${ECT_CONDA_BIN:-/root/miniconda3/bin/conda}"
PROJECT_ROOT="${ECT_PROJECT_ROOT:-/root/autodl-tmp/ect_project}"
DATA="${ECT_DATA_PATH:-${PROJECT_ROOT}/datasets/cifar10-32x32.zip}"
EVAL_ROOT="${ECT_GAP_EVAL_ROOT:-${PROJECT_ROOT}/runs/gap-factorial-20260726-eval}"

RUN_DIR=""
LABEL=""
for arg in "$@"; do
    case "${arg}" in
        --run-dir=*) RUN_DIR="${arg#*=}" ;;
        --label=*) LABEL="${arg#*=}" ;;
        -h|--help)
            echo "Usage: $0 --run-dir=DIR --label=LABEL"
            exit 0
            ;;
        *) echo "Unknown option: ${arg}" >&2; exit 2 ;;
    esac
done

[[ -d "${RUN_DIR}" ]] || { echo "Missing run dir: ${RUN_DIR}" >&2; exit 1; }
[[ -n "${LABEL}" ]] || { echo "--label is required" >&2; exit 1; }
[[ -x "${CONDA_BIN}" ]] || { echo "Missing conda executable: ${CONDA_BIN}" >&2; exit 1; }
[[ -s "${DATA}" ]] || { echo "Missing dataset: ${DATA}" >&2; exit 1; }
CHECKPOINT="${RUN_DIR}/network-snapshot-latest.pkl"
[[ -s "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 1; }

for NFE in 1 2; do
    OUTDIR="${EVAL_ROOT}/${LABEL}/nfe${NFE}"
    if [[ -d "${OUTDIR}" ]] &&
       grep -qx 'exit_code=0' "${OUTDIR}/experiment_meta.env" 2>/dev/null &&
       [[ -s "${OUTDIR}/metric-kid5k_full.jsonl" ]] &&
       [[ -s "${OUTDIR}/metric-fid5k_full.jsonl" ]] &&
       [[ "$(wc -l < "${OUTDIR}/metric-kid5k_full.jsonl")" -eq 1 ]] &&
       [[ "$(wc -l < "${OUTDIR}/metric-fid5k_full.jsonl")" -eq 1 ]]; then
        printf '[evaluate_gap_factorial_arm] verified completed NFE=%s; skipping %s\n' \
            "${NFE}" "${OUTDIR}"
        continue
    fi
    [[ ! -e "${OUTDIR}" ]] || {
        echo "Refusing to overwrite incomplete evaluation: ${OUTDIR}" >&2
        exit 1
    }
    mkdir -p "${OUTDIR}"
    CMD=(
        "${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}"
        python "${ROOT_DIR}/ct_eval.py"
        --data="${DATA}"
        --outdir="${OUTDIR}"
        --nosubdir
        --cond=False
        --arch=ddpmpp
        --precond=ct
        --dropout=0.2
        --augment=0
        --fp16=False
        --seed=20260722
        --resume="${CHECKPOINT}"
        --nfe="${NFE}"
        --mid_t=0.821
        --metrics=kid5k_full,fid5k_full
        --metric-repeats=1
        --sample-seeds=0-4999
    )
    {
        printf 'label=%s\nnfe=%s\n' "${LABEL}" "${NFE}"
        printf 'checkpoint_sha256=%s\n' \
            "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
        printf 'data_sha256=%s\n' "$(sha256sum "${DATA}" | awk '{print $1}')"
        printf 'evaluation_source_sha256=%s\n' "$(
            for source_file in \
                "${ROOT_DIR}/ct_eval.py" \
                "${ROOT_DIR}/metrics/metric_main.py" \
                "${ROOT_DIR}/metrics/metric_utils.py" \
                "${ROOT_DIR}/scripts/evaluate_gap_factorial_arm.sh"
            do
                sha256sum "${source_file}" | awk '{print $1}'
            done | sha256sum | awk '{print $1}'
        )"
        printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'exact_command='
        printf '%q ' "${CMD[@]}"
        printf '\n'
    } > "${OUTDIR}/experiment_meta.env"
    set +e
    "${CMD[@]}" 2>&1 | tee "${OUTDIR}/runner.log"
    STATUS="${PIPESTATUS[0]}"
    set -e
    printf 'exit_code=%s\nfinished_utc=%s\n' \
        "${STATUS}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUTDIR}/experiment_meta.env"
    [[ "${STATUS}" -eq 0 ]] || exit "${STATUS}"
    for METRIC in kid5k_full fid5k_full; do
        RESULT="${OUTDIR}/metric-${METRIC}.jsonl"
        [[ -s "${RESULT}" ]] || {
            echo "Missing metric output: ${RESULT}" >&2
            exit 1
        }
        [[ "$(wc -l < "${RESULT}")" -eq 1 ]] || {
            echo "Expected exactly one result line: ${RESULT}" >&2
            exit 1
        }
    done
done
