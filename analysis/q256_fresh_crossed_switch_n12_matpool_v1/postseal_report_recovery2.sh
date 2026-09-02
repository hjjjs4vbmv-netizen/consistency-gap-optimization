#!/usr/bin/env bash
set -Eeuo pipefail

protocol="${1:?usage: $0 PROTOCOL PREFLIGHT ELEVEN_AUTH EVALUATION_AUTH POSTSEAL_AUTH}"
preflight="${2:?missing preflight receipt}"
eleven_authorization="${3:?missing eleven-seed authorization}"
evaluation_authorization="${4:?missing evaluation recovery authorization}"
postseal_authorization="${5:?missing postseal report recovery v2 authorization}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime_manifest="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["assets"]["runtime_manifest"]["path"])' "${protocol}")"
python="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_prefix"]+"/bin/python")' "${runtime_manifest}")"
output_root="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["formal_output_root"])' "${protocol}")"
control_root="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["control_root"])' "${protocol}")"
public_manifest="${control_root}/frozen_242_job_evaluation_manifest_11seed_recovery1.json"
analysis_root="${output_root}/analysis_11seed_recovery1"

"${python}" "${script_dir}/postseal_recovery.py" prepare \
  --protocol "${protocol}" --authorization "${postseal_authorization}"
"${python}" "${script_dir}/statistics.py" \
  --protocol "${protocol}" --eleven-seed-authorization "${eleven_authorization}" \
  --evaluation-recovery-authorization "${evaluation_authorization}" \
  --postseal-report-recovery-authorization "${postseal_authorization}" \
  --decoded-results "${analysis_root}/decoded_results.json" \
  --output-dir "${analysis_root}/statistics"
"${python}" "${script_dir}/finalize.py" \
  --protocol "${protocol}" --preflight "${preflight}" \
  --public-manifest "${public_manifest}" --output-root "${output_root}" \
  --eleven-seed-authorization "${eleven_authorization}" \
  --evaluation-recovery-authorization "${evaluation_authorization}" \
  --postseal-report-recovery-authorization "${postseal_authorization}"
