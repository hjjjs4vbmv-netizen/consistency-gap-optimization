#!/usr/bin/env bash
set -Eeuo pipefail
manifest="${1:?usage: $0 EVALUATION_MANIFEST EVAL_ROOT}"
eval_root="${2:?usage: $0 EVALUATION_MANIFEST EVAL_ROOT}"
python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["job_count"]==272 and p["quality_values_decoded"] is False and all(j["status"] in {"FROZEN_NOT_RUN","SEALED_PASS"} for j in p["jobs"])' "${manifest}"
mkdir -p "${eval_root}/sealed" "${eval_root}/opaque_logs"
echo "Evaluation queue is frozen. Jobs must be launched by opaque_id only; decode is forbidden until all PRIMARY and KEY_SECONDARY receipts are SEALED_PASS."
