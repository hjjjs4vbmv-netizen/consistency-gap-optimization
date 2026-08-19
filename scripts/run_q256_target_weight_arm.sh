#!/usr/bin/env bash

# Fail-closed entry point for one q=256 target-geometry x loss-weighting cell.
# All scientific and provenance checks live in the versioned Python launcher.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOOTSTRAP_PYTHON="${ECT_BOOTSTRAP_PYTHON:-python3}"
RUNTIME_SANDBOX="${ECT_RUNTIME_SANDBOX:-/data/temp/ect001-pytorch2401-sandbox}"
IN_SANDBOX_VALUE="${ECT_Q256_LAUNCHER_IN_SANDBOX:-}"

if [[ -n "${IN_SANDBOX_VALUE}" && "${IN_SANDBOX_VALUE}" != "1" ]]; then
    printf '[run_q256_target_weight_arm] ERROR: ECT_Q256_LAUNCHER_IN_SANDBOX must be unset or exactly 1\n' >&2
    exit 1
fi

if [[ "${IN_SANDBOX_VALUE}" == "1" ]]; then
    command -v python >/dev/null 2>&1 || {
        printf '[run_q256_target_weight_arm] ERROR: canonical in-sandbox python not found\n' >&2
        exit 1
    }
    python -c \
        'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' || {
        printf '[run_q256_target_weight_arm] ERROR: canonical in-sandbox python is older than 3.10\n' >&2
        exit 1
    }
    exec python "${ROOT_DIR}/scripts/run_q256_target_weight_matrix.py" arm "$@"
fi

command -v "${BOOTSTRAP_PYTHON}" >/dev/null 2>&1 || {
    printf '[run_q256_target_weight_arm] ERROR: bootstrap Python not found: %s\n' \
        "${BOOTSTRAP_PYTHON}" >&2
    exit 1
}

if "${BOOTSTRAP_PYTHON}" -c \
    'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'
then
    exec "${BOOTSTRAP_PYTHON}" \
        "${ROOT_DIR}/scripts/run_q256_target_weight_matrix.py" arm "$@"
fi

command -v apptainer >/dev/null 2>&1 || {
    printf '[run_q256_target_weight_arm] ERROR: host Python is older than 3.10 and apptainer was not found\n' >&2
    exit 1
}

[[ -d "${RUNTIME_SANDBOX}" && ! -L "${RUNTIME_SANDBOX}" ]] || {
    printf '[run_q256_target_weight_arm] ERROR: runtime sandbox is not a real directory: %s\n' \
        "${RUNTIME_SANDBOX}" >&2
    exit 1
}

exec apptainer exec --nv \
    --bind /data/raw:/data/raw \
    --bind /data/temp:/data/temp \
    "${RUNTIME_SANDBOX}" \
    env ECT_Q256_LAUNCHER_IN_SANDBOX=1 \
    python "${ROOT_DIR}/scripts/run_q256_target_weight_matrix.py" arm "$@"
