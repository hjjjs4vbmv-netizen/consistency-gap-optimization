#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 14 ]; then
  echo "usage: $0 SEED GPU ROOTFS DRIVER_INJECTION SOURCE_ROOT SOURCE_ARCHIVE SOURCE_ARCHIVE_SHA TRAINING_ROOT OUTPUT_ROOT DATASET BASE_PORT PYTHON_DRIVER CACHE_TEMPLATE REPLAY_COMMIT" >&2
  exit 64
fi

seed=$1
gpu=$2
rootfs=$3
driver_injection=$4
source_root=$5
source_archive=$6
source_archive_sha=$7
training_root=$8
output_root=$9
dataset=${10}
base_port=${11}
python_driver=${12}
cache_template=${13}
replay_commit=${14}
shim_dir=/data/temp/ECT001/q256-native-runtime-shims-v1

expected_gpu=$((seed - 14))
if [ "$gpu" -ne "$expected_gpu" ]; then
  echo "seed/GPU mapping violation: seed=$seed gpu=$gpu expected=$expected_gpu" >&2
  exit 65
fi
for required in "$rootfs" "$driver_injection" "$source_root" "$training_root" "$cache_template" "$shim_dir"; do
  [ -d "$required" ] || { echo "required directory missing: $required" >&2; exit 66; }
done
for required in "$source_archive" "$dataset" "$python_driver" "$shim_dir/torchrun"; do
  [ -f "$required" ] || { echo "required file missing: $required" >&2; exit 67; }
done

library_path="$rootfs/lib/x86_64-linux-gnu:$rootfs/usr/lib/x86_64-linux-gnu:$rootfs/usr/local/lib:$rootfs/usr/local/lib/python3.10/dist-packages/torch/lib:$rootfs/usr/local/cuda/compat/lib:$rootfs/opt/hpcx/ucc/lib:$rootfs/opt/hpcx/ucx/lib:$rootfs/opt/hpcx/sharp/lib:$rootfs/usr/local/mpi/lib:$driver_injection/lib:$driver_injection/lib64"

export Q256_NATIVE_ROOTFS="$rootfs"
export Q256_NATIVE_LIBRARY_PATH="$library_path"
export PYTHONHOME="$rootfs/usr"
export PYTHONPATH="$rootfs/usr/local/lib/python3.10/dist-packages:$rootfs/usr/lib/python3/dist-packages:$source_root"
export PATH="$shim_dir:/usr/bin:/bin"
export LD_LIBRARY_PATH="$library_path"
export HOME=/root
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu"
export CUDA_CACHE_DISABLE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

exec timeout --signal=TERM --kill-after=30s 24h \
  "$rootfs/lib64/ld-linux-x86-64.so.2" \
    --library-path "$library_path" \
    "$rootfs/usr/bin/python" "$python_driver" \
      --seed "$seed" --gpu "$gpu" \
      --source-root "$source_root" \
      --source-archive "$source_archive" \
      --source-archive-sha256 "$source_archive_sha" \
      --training-root "$training_root" \
      --output-root "$output_root" \
      --dataset "$dataset" \
      --base-port "$base_port" \
      --cache-template "$cache_template" \
      --replay-commit "$replay_commit"
