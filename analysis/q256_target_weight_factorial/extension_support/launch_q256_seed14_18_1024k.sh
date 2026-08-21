#!/usr/bin/env bash
set -euo pipefail

bundle=/data/temp/ECT001/q256-seed14-18-1024k-v1
source_archive=${bundle}/q256-training-source-4582051.tar
source_archive_sha_file=${bundle}/q256-training-source-4582051.tar.sha256
repo=/data/temp/ECT001/q256-factorial-1024k-4582051-seed14-18-v1
worker=${bundle}/run_q256_seed14_18_1024k_worker.sh
evaluation_worker=${bundle}/run_q256_seed14_18_1024k_native_evaluation_worker.sh
evaluation_driver=${bundle}/run_q256_seed14_18_1024k_frozen_evaluation.py
source_training_root=/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260821/secondary-precision-extension/seed14-18-dcca41b-v2
run_root=/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260821/secondary-precision-extension/seed14-18-1024k-from-v2-4582051-v1
dataset=/mnt/ect_project/datasets/cifar10-32x32.zip
canonical_dataset=/data/raw/ECT/datasets/cifar10-32x32-canonical-08c9ed1b2b1c.zip
evaluator_source_root=/data/temp/ECT001/q256-factorial-eval-d6aba02-seed14-18-v1
evaluator_source_archive=/data/temp/ECT001/q256-evaluator-source-d6aba02.tar
cache_template=${source_training_root}/frozen-evaluation-seed14-18-v4-native/seed14/evaluator_cache

umask 027

[[ -f "${source_archive}" && -f "${source_archive_sha_file}" && -f "${worker}" && -f "${evaluation_worker}" && -f "${evaluation_driver}" ]] || { echo "missing launch bundle" >&2; exit 2; }
(cd "${bundle}" && sha256sum -c "$(basename "${source_archive_sha_file}")")
[[ "$(sha256sum "${dataset}" | awk '{print $1}')" == 2d4056e80de1a96fe16f2f58945c6c4710ecd9fc02e3cc7aa5b50513b7cdf389 ]] || { echo "dataset hash mismatch" >&2; exit 2; }
[[ "$(sha256sum "${canonical_dataset}" | awk '{print $1}')" == 08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372 ]] || { echo "canonical dataset hash mismatch" >&2; exit 2; }
[[ -d "${source_training_root}" && -d "${evaluator_source_root}" && -d "${cache_template}" && -f "${evaluator_source_archive}" ]] || { echo "missing source training/evaluator asset" >&2; exit 2; }
[[ "$(sha256sum "${evaluator_source_archive}" | awk '{print $1}')" == 37560e2eb50a9a361f9fca899a33778616386a622d5f039f53305d8d492eaed6 ]] || { echo "evaluator source hash mismatch" >&2; exit 2; }
[[ ! -e "${repo}" ]] || { echo "refusing existing source directory: ${repo}" >&2; exit 3; }
[[ ! -e "${run_root}" ]] || { echo "refusing existing run root: ${run_root}" >&2; exit 3; }

for seed in 14 15 16 17 18; do
  grep -q 'WORKER_PASS' "${source_training_root}/seed${seed}-worker.log" || { echo "source seed${seed} lacks WORKER_PASS" >&2; exit 2; }
  for arm in A B C D; do
    source_dir=${source_training_root}/seed${seed}/arm${arm}
    for required in training-state-latest.pt training_options.json train_summary.csv factorial_training_telemetry_v1.csv initial_state_receipt_v1.json log.txt; do
      [[ -s "${source_dir}/${required}" ]] || { echo "missing source artifact: ${source_dir}/${required}" >&2; exit 2; }
    done
    last=$(tail -n 1 "${source_dir}/train_summary.csv")
    attempted=$(printf '%s\n' "${last}" | awk -F, '{print $1}')
    successful=$(printf '%s\n' "${last}" | awk -F, '{print $2}')
    processed=$(printf '%s\n' "${last}" | awk -F, '{print $4}')
    skipped=$(printf '%s\n' "${last}" | awk -F, '{print $7}')
    [[ "${attempted}" == 2000 && "${successful}" -gt 0 && "${successful}" -le 2000 && "${processed}" == 256.000000 && "${skipped}" == 0 ]] || { echo "invalid source endpoint: seed${seed}/arm${arm}: ${last}" >&2; exit 2; }
  done
done

for gpu_index in 0 1 2 3 4; do
  gpu_line=$(nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader,nounits | awk -F', ' -v wanted="${gpu_index}" '$1 == wanted {print $0}')
  [[ "${gpu_line}" == "${gpu_index}, "*", NVIDIA A100-PCIE-40GB, 40960" ]] || { echo "GPU identity mismatch: ${gpu_line}" >&2; exit 2; }
  gpu_uuid=$(printf '%s\n' "${gpu_line}" | awk -F', ' '{print $2}')
  count=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v wanted="${gpu_uuid}" '$1 == wanted {count++} END {print count+0}')
  [[ "${count}" == 0 ]] || { echo "GPU is not compute-idle: ${gpu_uuid}" >&2; exit 2; }
done

for seed in 14 15 16 17 18; do
  [[ ! -e "/tmp/ECT001-q256-seed14-18-1024k-v1-shm-seed${seed}" ]] || { echo "existing private shm for seed ${seed}" >&2; exit 3; }
  tmux has-session -t "q1024_s${seed}" 2>/dev/null && { echo "existing tmux session q1024_s${seed}" >&2; exit 3; } || true
done

mkdir -p "$(dirname "${repo}")" "$(dirname "${run_root}")"
mkdir "${repo}"
tar -xf "${source_archive}" -C "${repo}"
mkdir "${run_root}"
mkdir "${run_root}/provenance"
mkdir "${run_root}/frozen-evaluation-1024k-v1"
cp "${source_archive_sha_file}" "${run_root}/provenance/"
cp "${worker}" "${run_root}/provenance/"
cp "${evaluation_worker}" "${run_root}/provenance/"
cp "${evaluation_driver}" "${run_root}/provenance/"
cp "$0" "${run_root}/provenance/"
{
  printf 'budget_extension_source_commit=458205192722883df393a8d017c26e6fa46f48f7\n'
  printf 'parent_256k_source_commit=dcca41b19e7c45512b5fbe98776520396a1bf9ac\n'
  printf 'source_archive_sha256=%s\n' "$(sha256sum "${source_archive}" | awk '{print $1}')"
  printf 'resume_source_root=%s\n' "${source_training_root}"
  printf 'resume_contract=strict_budget_only_256_to_1024_kimg\n'
  printf 'target_attempted_iteration=8000\n'
  printf 'dataset_archive_sha256=2d4056e80de1a96fe16f2f58945c6c4710ecd9fc02e3cc7aa5b50513b7cdf389\n'
  printf 'canonical_dataset_archive_sha256=08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372\n'
  printf 'dataset_byte_identical_to_canonical=false\n'
  printf 'dataset_semantic_equivalence=PASS_50000_IMAGES_LABELS_AND_ORDER_AGAINST_OFFICIAL_CIFAR10_SOURCE\n'
  printf 'official_cifar10_source_md5=c58f30108f718f92721af3b95e74349a\n'
  printf 'evaluation_protocol=FP32_50000_samples_seed0_to_49999_metric_seed20260730_KID_then_FID_NFE1_and_NFE2_mid_t_0.821\n'
  printf 'evaluator_source_commit=d6aba02fb88e9db0993623895eb2228ed717d810\n'
  printf 'one_gpu_per_seed=true\n'
  printf 'launch_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"${run_root}/provenance/launch_receipt.txt"
sha256sum "${worker}" "${evaluation_worker}" "${evaluation_driver}" "$0" >"${run_root}/provenance/launcher_files.sha256"

for seed in 14 15 16 17 18; do
  for arm in A B C D; do
    sha256sum "${source_training_root}/seed${seed}/arm${arm}/training-state-latest.pt"
  done
done >"${run_root}/provenance/source_256k_training_states.sha256"

for row in '0 14 30114' '1 15 30115' '2 16 30116' '3 17 30117' '4 18 30118'; do
  read -r gpu_index seed master_port <<<"${row}"
  tmux new-session -d -s "q1024_s${seed}" \
    "bash '${worker}' '${gpu_index}' '${seed}' '${master_port}' > '${run_root}/seed${seed}-worker.log' 2>&1"
  tmux display-message -p -t "q1024_s${seed}" '#{session_name} #{pane_pid}' >"${run_root}/seed${seed}-tmux.txt"
done

sleep 5
for seed in 14 15 16 17 18; do
  tmux has-session -t "q1024_s${seed}" 2>/dev/null || { echo "session died during startup: seed${seed}" >&2; exit 4; }
done

echo "LAUNCH_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) seeds=14,15,16,17,18 one_gpu_per_seed=true target_kimg=1024 evaluation_chained=true"
