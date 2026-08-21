#!/usr/bin/env bash
set -euo pipefail

bundle=/data/temp/ECT001/q256-seed14-18-v2
source_archive=${bundle}/q256-training-source-dcca41b.tar
source_archive_sha_file=${bundle}/q256-training-source-dcca41b.tar.sha256
repo=/data/temp/ECT001/q256-factorial-clean-dcca41b-v2
worker=${bundle}/run_q256_seed14_18_worker.sh
run_root=/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260821/secondary-precision-extension/seed14-18-dcca41b-v2
dataset=/mnt/ect_project/datasets/cifar10-32x32.zip
transfer=/data/raw/ECT/pretrained/edm-cifar10-32x32-uncond-vp.pkl

umask 027

[[ -f "${source_archive}" && -f "${source_archive_sha_file}" && -f "${worker}" ]] || { echo "missing launch bundle" >&2; exit 2; }
(cd "${bundle}" && sha256sum -c "$(basename "${source_archive_sha_file}")")
[[ "$(sha256sum "${dataset}" | awk '{print $1}')" == 2d4056e80de1a96fe16f2f58945c6c4710ecd9fc02e3cc7aa5b50513b7cdf389 ]] || { echo "dataset hash mismatch" >&2; exit 2; }
[[ "$(sha256sum "${transfer}" | awk '{print $1}')" == 4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da ]] || { echo "transfer hash mismatch" >&2; exit 2; }
[[ ! -e "${repo}" ]] || { echo "refusing existing source directory: ${repo}" >&2; exit 3; }
[[ ! -e "${run_root}" ]] || { echo "refusing existing run root: ${run_root}" >&2; exit 3; }

for gpu_index in 0 1 2 3 4; do
  gpu_line=$(nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader,nounits | awk -F', ' -v wanted="${gpu_index}" '$1 == wanted {print $0}')
  [[ "${gpu_line}" == "${gpu_index}, "*", NVIDIA A100-PCIE-40GB, 40960" ]] || { echo "GPU identity mismatch: ${gpu_line}" >&2; exit 2; }
  gpu_uuid=$(printf '%s\n' "${gpu_line}" | awk -F', ' '{print $2}')
  count=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v wanted="${gpu_uuid}" '$1 == wanted {count++} END {print count+0}')
  [[ "${count}" == 0 ]] || { echo "GPU is not compute-idle: ${gpu_uuid}" >&2; exit 2; }
done

for seed in 14 15 16 17 18; do
  [[ ! -e "/tmp/ECT001-q256-seed14-18-v2-shm-seed${seed}" ]] || { echo "existing private shm for seed ${seed}" >&2; exit 3; }
  tmux has-session -t "q256_s${seed}" 2>/dev/null && { echo "existing tmux session q256_s${seed}" >&2; exit 3; } || true
done

mkdir -p "$(dirname "${repo}")" "$(dirname "${run_root}")"
mkdir "${repo}"
tar -xf "${source_archive}" -C "${repo}"
mkdir "${run_root}"
mkdir "${run_root}/provenance"
cp "${source_archive_sha_file}" "${run_root}/provenance/"
cp "${worker}" "${run_root}/provenance/"
cp "$0" "${run_root}/provenance/"
{
  printf 'source_commit=dcca41b19e7c45512b5fbe98776520396a1bf9ac\n'
  printf 'source_archive_sha256=%s\n' "$(sha256sum "${source_archive}" | awk '{print $1}')"
  printf 'dataset_archive_sha256=2d4056e80de1a96fe16f2f58945c6c4710ecd9fc02e3cc7aa5b50513b7cdf389\n'
  printf 'canonical_dataset_archive_sha256=08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372\n'
  printf 'dataset_byte_identical_to_canonical=false\n'
  printf 'dataset_semantic_equivalence=PASS_50000_IMAGES_LABELS_AND_ORDER_AGAINST_OFFICIAL_CIFAR10_SOURCE\n'
  printf 'official_cifar10_source_md5=c58f30108f718f92721af3b95e74349a\n'
  printf 'transfer_sha256=4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da\n'
  printf 'retry_of=seed14-18-dcca41b-v1\n'
  printf 'retry_authorized_by_user=true\n'
  printf 'runtime_fix=PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python\n'
  printf 'launch_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"${run_root}/provenance/launch_receipt.txt"
sha256sum "${worker}" "$0" >"${run_root}/provenance/launcher_files.sha256"

for row in '0 14 30114' '1 15 30115' '2 16 30116' '3 17 30117' '4 18 30118'; do
  read -r gpu_index seed master_port <<<"${row}"
  tmux new-session -d -s "q256_s${seed}" \
    "bash '${worker}' '${gpu_index}' '${seed}' '${master_port}' > '${run_root}/seed${seed}-worker.log' 2>&1"
  tmux display-message -p -t "q256_s${seed}" '#{session_name} #{pane_pid}' >"${run_root}/seed${seed}-tmux.txt"
done

echo "LAUNCH_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) seeds=14,15,16,17,18 one_gpu_per_seed=true"
