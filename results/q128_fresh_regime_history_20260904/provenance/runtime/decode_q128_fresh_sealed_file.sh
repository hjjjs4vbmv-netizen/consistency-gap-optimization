#!/usr/bin/env bash
set -Eeuo pipefail
sealed="${1:?usage: $0 SEALED_FILE OUTPUT_FILE}"
output="${2:?usage: $0 SEALED_FILE OUTPUT_FILE}"
root="/root/q128_fresh_regime_history_n8_v1/evaluation"
[[ "${sealed}" == "${root}/"* && "${sealed}" == *.sealed ]] || {
  echo "refuse path outside q128 evaluation root" >&2
  exit 2
}
[[ ! -e "${output}" ]] || { echo "refuse existing output" >&2; exit 2; }
openssl enc -d -aes-256-cbc -pbkdf2 \
  -pass file:"${root}/control/decode.key" -in "${sealed}" -out "${output}"
