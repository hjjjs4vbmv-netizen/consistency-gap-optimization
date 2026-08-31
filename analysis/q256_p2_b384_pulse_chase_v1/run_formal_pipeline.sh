#!/usr/bin/env bash
set -Eeuo pipefail
repo="${1:?usage: run_formal_pipeline.sh REPO PROTOCOL DATASET TRANSFER RUNTIME_SIF CACHE_ROOT KID_REAL_FEATURES FID_REAL_FEATURES EXPERIMENT_ROOT}"
protocol="${2:?missing protocol}"; dataset="${3:?missing dataset}"; transfer="${4:?missing transfer}"
runtime_sif="${5:?missing runtime}"; cache_root="${6:?missing cache root}"
kid_real="${7:?missing KID real features}"; fid_real="${8:?missing FID real features}"
root="${9:?missing experiment root}"
[[ ! -e "${root}" ]] || { echo "experiment root already exists" >&2; exit 3; }
mkdir "${root}"
tool_dir="${repo}/analysis/q256_p2_b384_pulse_chase_v1"
python3 -c 'import os,shlex,sys; f=open(sys.argv[1],"x"); f.write(shlex.join(sys.argv[2:])+"\n"); f.flush(); os.fsync(f.fileno())' \
  "${root}/REPRODUCE_COMMAND.txt" bash "${tool_dir}/run_formal_pipeline.sh" \
  "${repo}" "${protocol}" "${dataset}" "${transfer}" "${runtime_sif}" \
  "${cache_root}" "${kid_real}" "${fid_real}" "${root}"
bash "${tool_dir}/launch_formal.sh" "${repo}" "${protocol}" "${dataset}" \
  "${transfer}" "${runtime_sif}" "${root}/training"
apptainer exec --bind /data:/data --pwd "${repo}" "${runtime_sif}" python \
  analysis/q256_p2_b384_pulse_chase_v1/prepare_evaluation_manifest.py \
    --formal-root "${root}/training" \
    --training-integrity "${root}/training/training_integrity_report.json" \
    --protocol "${protocol}" --dataset "${dataset}" --runtime-sif "${runtime_sif}" \
    --evaluator-repo "${repo}" --kid-real-features "${kid_real}" \
    --fid-real-features "${fid_real}" --output "${root}/evaluation_manifest.json"
bash "${tool_dir}/launch_evaluation.sh" "${root}/evaluation_manifest.json" \
  "${cache_root}" "${root}/evaluation" "${repo}"
apptainer exec --bind /data:/data --pwd "${repo}" "${runtime_sif}" python \
  analysis/q256_p2_b384_pulse_chase_v1/analyze_results.py \
    --manifest "${root}/evaluation_manifest.json" \
    --seal-audit "${root}/evaluation/evaluation_seal_audit.json" \
    --receipts "${root}/evaluation/receipts" --protocol "${protocol}" \
    --output-dir "${root}/results"
apptainer exec --bind /data:/data --pwd "${repo}" "${runtime_sif}" python \
  analysis/q256_p2_b384_pulse_chase_v1/finalize_artifacts.py \
    --root "${root}" --repo "${repo}" --protocol "${protocol}"
echo "[P2 pipeline] COMPLETE root=${root}"
