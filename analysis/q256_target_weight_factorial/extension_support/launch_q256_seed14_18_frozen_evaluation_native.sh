#!/usr/bin/env bash
set -Eeuo pipefail

rootfs=/data/temp/q256-cohort3-runtime/ngc-pytorch-24.01-bundle/rootfs
driver_injection=/data/temp/q256-cohort3-runtime/nvidia-driver-injection-570.211.01
source_root=/data/temp/ECT001/q256-factorial-eval-d6aba02-seed14-18-v1
source_archive=/data/temp/ECT001/q256-evaluator-source-d6aba02.tar
source_archive_sha=37560e2eb50a9a361f9fca899a33778616386a622d5f039f53305d8d492eaed6
worker=/data/temp/ECT001/run_q256_seed14_18_frozen_eval_native_worker-v1.sh
python_driver=/data/temp/ECT001/run_q256_seed14_18_frozen_eval-v2.py
shim_dir=/data/temp/ECT001/q256-native-runtime-shims-v1
training_root=/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260821/secondary-precision-extension/seed14-18-dcca41b-v2
prior_root=${training_root}/frozen-evaluation-seed14-18-v2
recovery_root=${training_root}/frozen-evaluation-seed14-18-v3-recovery
evaluation_root=${training_root}/frozen-evaluation-seed14-18-v4-native
cache_template=${prior_root}/seed14/evaluator_cache
dataset=/data/raw/ECT/datasets/cifar10-32x32-canonical-08c9ed1b2b1c.zip
dataset_sha=08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372
detector_sha=f58cb9b6ec323ed63459aa4fb441fe750cfe39fafad6da5cb504a16f19e958f4
reference_sha=8478c50d243e9876a4e82d71feb0a8420036bdc16f7962fba1511d24e4a7cfde

umask 027

for required in "$rootfs" "$driver_injection" "$source_root" "$training_root" "$prior_root" "$recovery_root" "$cache_template" "$shim_dir"; do
  [[ -d "$required" ]] || { echo "required directory missing: $required" >&2; exit 2; }
done
for required in "$source_archive" "$worker" "$python_driver" "$dataset" "$shim_dir/torchrun"; do
  [[ -f "$required" ]] || { echo "required file missing: $required" >&2; exit 2; }
done
[[ "$(sha256sum "$source_archive" | awk '{print $1}')" == "$source_archive_sha" ]] || { echo "source archive hash mismatch" >&2; exit 2; }
[[ "$(sha256sum "$dataset" | awk '{print $1}')" == "$dataset_sha" ]] || { echo "dataset hash mismatch" >&2; exit 2; }
detector=$(find "$cache_template/downloads" -maxdepth 1 -type f -name '*inception-2015-12-05.pt' -print -quit)
reference=$(find "$cache_template/gan-metrics" -maxdepth 1 -type f -name '*.pkl' -print -quit)
[[ "$(sha256sum "$detector" | awk '{print $1}')" == "$detector_sha" ]] || { echo "detector template hash mismatch" >&2; exit 2; }
[[ "$(sha256sum "$reference" | awk '{print $1}')" == "$reference_sha" ]] || { echo "reference template hash mismatch" >&2; exit 2; }
[[ ! -e "$evaluation_root" && ! -L "$evaluation_root" ]] || { echo "refusing existing native evaluation root: $evaluation_root" >&2; exit 3; }

for seed in 14 15 16 17 18; do
  gpu=$((seed - 14))
  session=q256_eval4_s${seed}
  tmux has-session -t "$session" 2>/dev/null && { echo "tmux session exists: $session" >&2; exit 3; }
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F', ' -v wanted="$gpu" '$1 == wanted {print $2}')
  processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v wanted="$uuid" '$1 == wanted {count++} END {print count+0}')
  [[ "$processes" -eq 0 ]] || { echo "GPU $gpu is not compute-idle" >&2; exit 2; }
done

mkdir "$evaluation_root"
mkdir "$evaluation_root/provenance"
cp "$worker" "$evaluation_root/provenance/"
cp "$python_driver" "$evaluation_root/provenance/"
cp "$shim_dir/torchrun" "$evaluation_root/provenance/torchrun-native-rootfs.sh"
cp "$0" "$evaluation_root/provenance/"
sha256sum "$evaluation_root"/provenance/* >"$evaluation_root/provenance/launcher_files.sha256"

cat >"$evaluation_root/provenance/native_launch_receipt.json" <<EOF
{
  "schema": "q256-target-weight-seed14-18-native-evaluation-launch-v1",
  "status": "LAUNCHED",
  "launched_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "continuation_from": ["$prior_root", "$recovery_root"],
  "reason": "proot ptrace serialized MKL-intensive KID while native rootfs dynamic loading passed exact runtime and CUDA probes",
  "runtime_invariants": {"python": "3.10.12", "torch": "2.2.0a0+81ea7a4", "cuda": "12.3", "evaluator_commit": "d6aba02fb88e9db0993623895eb2228ed717d810"},
  "numerical_protocol_modified": false,
  "cache_template": "$cache_template",
  "detector_sha256": "$detector_sha",
  "canonical_reference_stats_sha256": "$reference_sha",
  "seed_gpu_mapping": {"14": 0, "15": 1, "16": 2, "17": 3, "18": 4},
  "job_count": 40
}
EOF

for seed in 14 15 16 17 18; do
  gpu=$((seed - 14))
  base_port=$((37140 + (seed - 14) * 20))
  session=q256_eval4_s${seed}
  seed_output=$evaluation_root/seed${seed}
  worker_log=$evaluation_root/seed${seed}-worker.log
  printf -v command '%q ' \
    "$worker" "$seed" "$gpu" "$rootfs" "$driver_injection" \
    "$source_root" "$source_archive" "$source_archive_sha" \
    "$training_root" "$seed_output" "$dataset" "$base_port" "$python_driver" \
    "$cache_template"
  command+=" >$(printf '%q' "$worker_log") 2>&1"
  tmux new-session -d -s "$session" "$command"
  echo "NATIVE_LAUNCHED seed=$seed gpu=$gpu session=$session"
done

sleep 3
for seed in 14 15 16 17 18; do
  tmux has-session -t q256_eval4_s${seed} 2>/dev/null || { echo "native session died during startup: seed$seed" >&2; exit 4; }
done

echo "NATIVE_MATRIX_LAUNCHED root=$evaluation_root"
