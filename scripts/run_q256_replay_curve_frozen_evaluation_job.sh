#!/usr/bin/env bash
set -euo pipefail

manifest="${1:?usage: $0 MANIFEST_JSON JOB_INDEX GPU_ID MASTER_PORT}"
job_index="${2:?usage: $0 MANIFEST_JSON JOB_INDEX GPU_ID MASTER_PORT}"
gpu_id="${3:?usage: $0 MANIFEST_JSON JOB_INDEX GPU_ID MASTER_PORT}"
master_port="${4:?usage: $0 MANIFEST_JSON JOB_INDEX GPU_ID MASTER_PORT}"
[[ "${Q256_REPLAY_METRICS_ENABLE:-}" == "YES_AFTER_TRAINING_AUDIT" ]] || {
  echo "frozen replay metrics are disabled until training audit passes" >&2
  exit 2
}
[[ "${job_index}" =~ ^[0-9]+$ && "${master_port}" =~ ^[0-9]+$ ]] || { echo "invalid numeric argument" >&2; exit 2; }

job_root="${Q256_REPLAY_JOB_ROOT:-/root/q256_target_weight_replay_curve_v1}"
repo="${job_root}/source/recurrence_of_ect"
sandbox_root="${job_root}/runtime/sandbox"
runtime_python="${sandbox_root}/usr/bin/python"
dataset=/data/raw/ECT/datasets/cifar10-32x32-canonical-08c9ed1b2b1c.zip
eval_root="${Q256_REPLAY_EVAL_ROOT:-${job_root}/evaluation/q256-replay-curve-fidkid50k-v1}"
detector_cache_source="${Q256_DETECTOR_CACHE_SOURCE:?set Q256_DETECTOR_CACHE_SOURCE to the frozen detector download cache}"

runtime_ld_library_path="${sandbox_root}/usr/local/lib/python3.10/dist-packages/torch/lib"
runtime_ld_library_path+=":${sandbox_root}/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib"
runtime_ld_library_path+=":${sandbox_root}/usr/local/cuda/compat/lib"
runtime_ld_library_path+=":${sandbox_root}/usr/local/nvidia/lib:${sandbox_root}/usr/local/nvidia/lib64"
runtime_ld_library_path+=":${sandbox_root}/lib:${sandbox_root}/lib/x86_64-linux-gnu"
runtime_ld_library_path+=":${sandbox_root}/opt/hpcx/clusterkit/lib:${sandbox_root}/opt/hpcx/hcoll/lib"
runtime_ld_library_path+=":${sandbox_root}/opt/hpcx/nccl_rdma_sharp_plugin/lib:${sandbox_root}/opt/hpcx/ompi/lib"
runtime_ld_library_path+=":${sandbox_root}/opt/hpcx/sharp/lib:${sandbox_root}/opt/hpcx/ucc/lib:${sandbox_root}/opt/hpcx/ucx/lib"
runtime_ld_library_path+=":${sandbox_root}/usr/local/cuda/targets/x86_64-linux/lib:${sandbox_root}/usr/local/lib"
runtime_path="${sandbox_root}/usr/local/lib/python3.10/dist-packages/torch_tensorrt/bin"
runtime_path+=":${sandbox_root}/usr/local/mpi/bin:${sandbox_root}/usr/local/nvidia/bin"
runtime_path+=":${sandbox_root}/usr/local/cuda/bin:${sandbox_root}/usr/local/sbin:${sandbox_root}/usr/local/bin"
runtime_path+=":${sandbox_root}/usr/sbin:${sandbox_root}/usr/bin:${sandbox_root}/sbin:${sandbox_root}/bin"
runtime_path+=":${sandbox_root}/usr/local/ucx/bin:${sandbox_root}/opt/tensorrt/bin"

mapfile -t job_fields < <("${runtime_python}" - "${manifest}" "${job_index}" <<'PY'
import json, sys
manifest=json.load(open(sys.argv[1], encoding='utf-8'))
assert manifest['metrics_executed'] is False
job=manifest['jobs'][int(sys.argv[2])]
for key in ('seed','arm','budget_kimg','nfe','checkpoint_path','checkpoint_sha256','phase'):
    print(job[key])
print('' if json.loads(job['mid_t']) == [] else json.loads(job['mid_t'])[0])
PY
)
[[ "${#job_fields[@]}" == 8 ]] || { echo "invalid manifest row" >&2; exit 3; }
seed="${job_fields[0]}"
arm="${job_fields[1]}"
budget="${job_fields[2]}"
nfe="${job_fields[3]}"
checkpoint="${job_fields[4]}"
checkpoint_sha256="${job_fields[5]}"
phase="${job_fields[6]}"
mid_t="${job_fields[7]}"
[[ -s "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; exit 3; }
[[ "$(sha256sum "${checkpoint}" | cut -d' ' -f1)" == "${checkpoint_sha256}" ]] || { echo "checkpoint hash mismatch" >&2; exit 3; }

job_id="seed${seed}-arm${arm}-kimg${budget}-nfe${nfe}"
target="${eval_root}/${phase}/jobs/${job_id}"
job_cache="${eval_root}/${phase}/job_caches/${job_id}"
[[ ! -e "${target}" ]] || { echo "refuse existing evaluation output: ${target}" >&2; exit 4; }
mkdir -p "${target}" "${job_cache}/downloads"
cp -a "${detector_cache_source}/." "${job_cache}/downloads/"
mid_args=()
if [[ -n "${mid_t}" ]]; then
  mid_args+=(--mid_t="${mid_t}")
fi

cd "${repo}"
env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_id}" \
  PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  LD_LIBRARY_PATH="${runtime_ld_library_path}" PATH="${runtime_path}" \
  DNNLIB_CACHE_DIR="${job_cache}" \
  MASTER_ADDR=127.0.0.1 MASTER_PORT="${master_port}" \
  RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
  "${runtime_python}" -m torch.distributed.run \
    --standalone --nproc_per_node=1 --master_port="${master_port}" \
    "${repo}/ct_eval.py" --resume "${checkpoint}" \
    --outdir "${target}" --nosubdir \
    --data "${dataset}" --cond=False --arch=ddpmpp --precond=ct \
    --dropout=0.2 --augment=0 --xflip=False --fp16=False \
    --cache=True --workers=3 --eval-batch=512 --metric-generator-batch=128 \
    --nfe="${nfe}" "${mid_args[@]}" \
    --metrics=kid50k_full,fid50k_full --metric-repeats=1 \
    --sample-seeds=0-49999 --seed=20260730 --retain-generated-artifacts \
    --desc="q256-replay-curve-${job_id}"

kid_sha="$(sha256sum "${target}/generated-features-kid50k_full-repeat00.npy" | cut -d' ' -f1)"
fid_sha="$(sha256sum "${target}/generated-features-fid50k_full-repeat00.npy" | cut -d' ' -f1)"
[[ "${kid_sha}" == "${fid_sha}" ]] || { echo "KID/FID feature mismatch" >&2; exit 5; }
echo "EVALUATION_JOB_PASS ${job_id} feature_sha256=${kid_sha}"
