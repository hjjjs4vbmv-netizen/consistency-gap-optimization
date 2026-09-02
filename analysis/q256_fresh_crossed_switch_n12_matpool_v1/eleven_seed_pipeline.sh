#!/usr/bin/env bash
set -Eeuo pipefail

protocol="${1:?usage: $0 PROTOCOL PREFLIGHT AUTHORIZATION EVALUATOR_REPO CACHE_ROOT}"
preflight="${2:?missing preflight receipt}"
authorization="${3:?missing eleven-seed authorization}"
evaluator_repo="${4:?missing evaluator repository}"
cache_root="${5:?missing evaluator cache root}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime_manifest="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["assets"]["runtime_manifest"]["path"])' "${protocol}")"
python="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_prefix"]+"/bin/python")' "${runtime_manifest}")"
output_root="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["formal_output_root"])' "${protocol}")"
control_root="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["control_root"])' "${protocol}")"
public_manifest="${control_root}/frozen_242_job_evaluation_manifest_11seed.json"
private_map="${control_root}/sealed_private_evaluation_map_11seed.json"

"${python}" "${script_dir}/integrity.py" \
  --protocol "${protocol}" --output "${output_root}/training_integrity_11seed_report.json" \
  --eleven-seed-authorization "${authorization}"
"${python}" "${script_dir}/evaluation.py" prepare \
  --protocol "${protocol}" --public-manifest "${public_manifest}" \
  --private-map "${private_map}" --eleven-seed-authorization "${authorization}"
"${python}" "${script_dir}/evaluation.py" launch \
  --protocol "${protocol}" --public-manifest "${public_manifest}" \
  --private-map "${private_map}" --eleven-seed-authorization "${authorization}" \
  --evaluator-repo "${evaluator_repo}" --cache-root "${cache_root}"
mkdir "${output_root}/analysis_11seed"
"${python}" "${script_dir}/evaluation.py" decode \
  --protocol "${protocol}" --public-manifest "${public_manifest}" \
  --private-map "${private_map}" --eleven-seed-authorization "${authorization}" \
  --output "${output_root}/analysis_11seed/decoded_results.json"
"${python}" "${script_dir}/statistics.py" \
  --protocol "${protocol}" --eleven-seed-authorization "${authorization}" \
  --decoded-results "${output_root}/analysis_11seed/decoded_results.json" \
  --output-dir "${output_root}/analysis_11seed/statistics"
"${python}" "${script_dir}/finalize.py" \
  --protocol "${protocol}" --preflight "${preflight}" \
  --public-manifest "${public_manifest}" --output-root "${output_root}" \
  --eleven-seed-authorization "${authorization}"
