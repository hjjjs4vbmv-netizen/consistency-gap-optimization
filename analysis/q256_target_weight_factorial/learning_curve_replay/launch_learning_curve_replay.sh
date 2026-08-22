#!/usr/bin/env bash
set -Eeuo pipefail

bundle=/data/temp/ECT001/q256-learning-curve-replay-bundle-v1
source_archive=${bundle}/q256-learning-curve-replay-source.tar
source_archive_sha_file=${bundle}/q256-learning-curve-replay-source.tar.sha256
replay_commit_file=${bundle}/replay_commit.txt
repo=/data/temp/ECT001/q256-learning-curve-replay-source-v1
worker=${bundle}/run_learning_curve_replay_worker.sh
inventory_builder=${bundle}/build_replay_source_inventory.py
trajectory_verifier=${bundle}/verify_replay_trajectory.py
evaluation_worker=${bundle}/run_learning_curve_native_evaluation_worker.sh
evaluation_driver=${bundle}/run_learning_curve_frozen_evaluation.py
runtime=/data/temp/q256-cohort3-runtime/ngc-pytorch-24.01-bundle/rootfs
driver_injection=/data/temp/q256-cohort3-runtime/nvidia-driver-injection-570.211.01
source_training_root=/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260821/secondary-precision-extension/seed14-18-dcca41b-v2
original_1024_root=/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260821/secondary-precision-extension/seed14-18-1024k-from-v2-4582051-recovery-v2
run_root=/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260821/secondary-precision-extension/seed14-18-256to1024-learning-curve-replay-v1
evaluation_root=${run_root}/formal-evaluation-learning-curves-v1
dataset=/mnt/ect_project/datasets/cifar10-32x32.zip
canonical_dataset=/data/raw/ECT/datasets/cifar10-32x32-canonical-08c9ed1b2b1c.zip
evaluator_source_root=/data/temp/ECT001/q256-factorial-eval-d6aba02-seed14-18-v1
evaluator_source_archive=/data/temp/ECT001/q256-evaluator-source-d6aba02.tar
cache_template=${source_training_root}/frozen-evaluation-seed14-18-v4-native/seed14/evaluator_cache

umask 027

for required in \
  "${source_archive}" "${source_archive_sha_file}" "${replay_commit_file}" \
  "${worker}" "${inventory_builder}" "${trajectory_verifier}" \
  "${evaluation_worker}" "${evaluation_driver}"; do
  [[ -f "${required}" ]] || { echo "missing launch bundle file: ${required}" >&2; exit 2; }
done
for required in \
  "${runtime}" "${driver_injection}" "${source_training_root}" \
  "${original_1024_root}" "${evaluator_source_root}" "${cache_template}"; do
  [[ -d "${required}" ]] || { echo "missing required directory: ${required}" >&2; exit 2; }
done
replay_commit=$(tr -d '\r\n' <"${replay_commit_file}")
[[ "${replay_commit}" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid replay commit" >&2; exit 2; }
(cd "${bundle}" && sha256sum -c "$(basename "${source_archive_sha_file}")")
[[ "$(sha256sum "${dataset}" | awk '{print $1}')" == 2d4056e80de1a96fe16f2f58945c6c4710ecd9fc02e3cc7aa5b50513b7cdf389 ]] || { echo "training dataset hash mismatch" >&2; exit 2; }
[[ "$(sha256sum "${canonical_dataset}" | awk '{print $1}')" == 08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372 ]] || { echo "canonical dataset hash mismatch" >&2; exit 2; }
[[ "$(sha256sum "${evaluator_source_archive}" | awk '{print $1}')" == 37560e2eb50a9a361f9fca899a33778616386a622d5f039f53305d8d492eaed6 ]] || { echo "evaluator source hash mismatch" >&2; exit 2; }
[[ ! -e "${repo}" && ! -L "${repo}" ]] || { echo "refusing existing replay source directory: ${repo}" >&2; exit 3; }
[[ ! -e "${run_root}" && ! -L "${run_root}" ]] || { echo "refusing existing replay root: ${run_root}" >&2; exit 3; }

for gpu_index in 0 1 2 3 4; do
  gpu_line=$(nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader,nounits | awk -F', ' -v wanted="${gpu_index}" '$1 == wanted {print $0}')
  [[ "${gpu_line}" == "${gpu_index}, "*", NVIDIA A100-PCIE-40GB, 40960" ]] || { echo "GPU identity mismatch: ${gpu_line}" >&2; exit 2; }
  gpu_uuid=$(printf '%s\n' "${gpu_line}" | awk -F', ' '{print $2}')
  count=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v wanted="${gpu_uuid}" '$1 == wanted {count++} END {print count+0}')
  [[ "${count}" == 0 ]] || { echo "GPU is not compute-idle: ${gpu_uuid}" >&2; exit 2; }
done
for seed in 14 15 16 17 18; do
  [[ ! -e "/tmp/ECT001-q256-learning-curve-replay-v1-shm-seed${seed}" ]] || { echo "existing replay shm for seed ${seed}" >&2; exit 3; }
  tmux has-session -t "q256lc_s${seed}" 2>/dev/null && { echo "existing tmux session q256lc_s${seed}" >&2; exit 3; } || true
done

mkdir -p "$(dirname "${repo}")" "$(dirname "${run_root}")"
mkdir "${repo}"
tar -xf "${source_archive}" -C "${repo}"
mkdir "${run_root}"
mkdir "${run_root}/provenance"
mkdir "${evaluation_root}"
cp "${source_archive_sha_file}" "${run_root}/provenance/"
cp "${replay_commit_file}" "${run_root}/provenance/"
cp "${worker}" "${inventory_builder}" "${trajectory_verifier}" \
  "${evaluation_worker}" "${evaluation_driver}" "$0" \
  "${run_root}/provenance/"
sha256sum "${run_root}"/provenance/* >"${run_root}/provenance/launcher_files.sha256"

inventory=${run_root}/replay_source_inventory.json
inventory_shm=/tmp/ECT001-q256-learning-curve-inventory-v1-shm
[[ ! -e "${inventory_shm}" ]] || { echo "existing inventory shm" >&2; exit 3; }
mkdir -m 700 "${inventory_shm}"
proot -0 -r "${runtime}" \
  -b /dev:/dev -b /proc:/proc -b /sys:/sys \
  -b /data:/data -b /mnt:/mnt -b /tmp:/tmp \
  -b /usr/lib/x86_64-linux-gnu:/host-driver-source \
  -b /usr/bin:/host-driver-bin-source \
  -b "${driver_injection}:/usr/local/nvidia" \
  -b "${inventory_shm}:/dev/shm" \
  /usr/bin/env -i HOME=/root LC_ALL=C.UTF-8 PYTHONNOUSERSITE=1 \
    PYTHONPATH="${repo}" \
    PATH=/usr/local/bin:/usr/bin:/bin \
    LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    /usr/bin/python "${inventory_builder}" \
      --source-root "${source_training_root}" --out "${inventory}"

inventory_status=$(python3 -c "import json; print(json.load(open('${inventory}'))['status'])")
inventory_pass=$(python3 -c "import json; print(json.load(open('${inventory}'))['pass_count'])")
[[ "${inventory_pass}" -gt 0 ]] || { echo "inventory has no runnable cells" >&2; exit 4; }

cat >"${run_root}/provenance/launch_receipt.json" <<EOF
{
  "schema": "ect.q256.learning-curve-replay-launch/v1",
  "status": "LAUNCHED",
  "launch_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "source_commit": "dcca41b19e7c45512b5fbe98776520396a1bf9ac",
  "replay_commit": "${replay_commit}",
  "inventory_status": "${inventory_status}",
  "inventory_pass_count": ${inventory_pass},
  "milestones_kimg": [384, 512, 640, 768, 896, 1024],
  "seed_gpu_mapping": {"14": 0, "15": 1, "16": 2, "17": 3, "18": 4},
  "arm_order": ["A", "B", "C", "D"],
  "evaluation_order": "all NFE1 before all NFE2 within each seed worker",
  "numerical_training_protocol_modified": false,
  "only_change": "immutable full-state and EMA snapshot persistence at exact milestones"
}
EOF

for row in '0 14 31114' '1 15 31115' '2 16 31116' '3 17 31117' '4 18 31118'; do
  read -r gpu_index seed master_port <<<"${row}"
  tmux new-session -d -s "q256lc_s${seed}" \
    "bash '${worker}' '${gpu_index}' '${seed}' '${master_port}' '${replay_commit}' > '${run_root}/seed${seed}-worker.log' 2>&1"
  tmux display-message -p -t "q256lc_s${seed}" '#{session_name} #{pane_pid}' >"${run_root}/seed${seed}-tmux.txt"
done

sleep 5
for seed in 14 15 16 17 18; do
  tmux has-session -t "q256lc_s${seed}" 2>/dev/null || { echo "replay session died during startup: seed${seed}" >&2; exit 5; }
done

echo "REPLAY_LAUNCH_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) replay_commit=${replay_commit} inventory=${inventory_status}:${inventory_pass}/20 checkpoints=120 evaluation_jobs=240"
