#!/usr/bin/env bash
set -euo pipefail

seed="${1:?usage: $0 SEED GPU_ID BASE_PORT}"
gpu_id="${2:?usage: $0 SEED GPU_ID BASE_PORT}"
base_port="${3:?usage: $0 SEED GPU_ID BASE_PORT}"

case "${seed}" in
  3|4|5) ;;
  *) echo "unsupported seed: ${seed}" >&2; exit 2 ;;
esac
[[ "${base_port}" =~ ^[0-9]+$ ]] || { echo "invalid base port" >&2; exit 2; }

job_root="${Q256_JOB_ROOT:-/root/q256_target_weight_1024k}"
repo="${job_root}/source/recurrence_of_ect"
run_root="${job_root}/runs/q256-target-weight-1024k"
eval_tag="${Q256_EVAL_TAG:-q256-target-weight-1024k-fidkid50k-v2-shared-features}"
eval_root="${job_root}/evaluation/${eval_tag}"
detector_cache_source="${Q256_DETECTOR_CACHE_SOURCE:-${job_root}/evaluation/q256-target-weight-1024k-fidkid50k/evaluator_cache/downloads}"
sandbox_root="${job_root}/runtime/sandbox"
runtime_python="${sandbox_root}/usr/bin/python"
dataset=/data/raw/ECT/datasets/cifar10-32x32-canonical-08c9ed1b2b1c.zip

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

[[ -x "${runtime_python}" && -f "${dataset}" ]] || { echo "missing runtime or dataset" >&2; exit 2; }
[[ -f "${repo}/scripts/evaluate_checkpoint.sh" ]] || { echo "missing evaluator" >&2; exit 2; }
[[ -z "$(git -C "${repo}" status --porcelain)" ]] || { echo "source worktree is dirty" >&2; exit 2; }

mkdir -p "${eval_root}/jobs" "${eval_root}/logs" "${eval_root}/job_caches"
cd "${repo}"

ordinal=0
for arm in A B C D; do
  checkpoint="${run_root}/seed${seed}/arm${arm}/network-snapshot-latest.pkl"
  [[ -s "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; exit 3; }
  for nfe in 1 2; do
    ordinal=$((ordinal + 1))
    job_id="seed${seed}-arm${arm}-nfe${nfe}"
    target="${eval_root}/jobs/${job_id}"
    log="${eval_root}/logs/${job_id}.log"
    job_cache="${eval_root}/job_caches/${job_id}"
    port=$((base_port + ordinal))
    if [[ -d "${target}" ]]; then
      if grep -q 'Exiting...' "${target}/log.txt" 2>/dev/null \
          && [[ -s "${target}/metric-kid50k_full.jsonl" ]] \
          && [[ -s "${target}/metric-fid50k_full.jsonl" ]] \
          && [[ -s "${target}/generated-samples.npy" ]] \
          && [[ -s "${target}/generated-features-kid50k_full-repeat00.npy" ]] \
          && [[ -s "${target}/generated-features-fid50k_full-repeat00.npy" ]]; then
        echo "[q256-eval] SKIP completed ${job_id}"
        continue
      fi
      echo "[q256-eval] STOP incomplete existing output: ${target}" >&2
      exit 4
    fi

    [[ -d "${detector_cache_source}" ]] || { echo "missing detector cache source" >&2; exit 4; }
    mkdir -p "${job_cache}/downloads"
    cp -a "${detector_cache_source}/." "${job_cache}/downloads/"

    echo "[q256-eval] START ${job_id} gpu=${gpu_id} checkpoint=${checkpoint}"
    mid_args=()
    if [[ "${nfe}" == 2 ]]; then
      mid_args+=(--mid_t=0.821)
    fi
    env \
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_id}" \
      PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
      LD_LIBRARY_PATH="${runtime_ld_library_path}" PATH="${runtime_path}" \
      DNNLIB_CACHE_DIR="${job_cache}" \
      MASTER_ADDR=127.0.0.1 MASTER_PORT="${port}" \
      RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
      "${runtime_python}" -m torch.distributed.run \
        --standalone --nproc_per_node=1 --master_port="${port}" \
        "${repo}/ct_eval.py" --resume "${checkpoint}" \
        --outdir "${target}" --nosubdir \
        --data "${dataset}" --cond=False --arch=ddpmpp --precond=ct \
        --dropout=0.2 --augment=0 --xflip=False --fp16=False \
        --cache=True --workers=3 --eval-batch=512 --metric-generator-batch=128 \
        --nfe="${nfe}" "${mid_args[@]}" \
        --metrics=kid50k_full,fid50k_full --metric-repeats=1 \
        --sample-seeds=0-49999 --seed=20260730 --retain-generated-artifacts \
        --desc="${eval_tag}-${job_id}" \
      2>&1 | tee "${log}"

    grep -q 'Exiting...' "${target}/log.txt"
    for required in \
      metric-kid50k_full.jsonl metric-fid50k_full.jsonl generated-samples.npy \
      generated-features-kid50k_full-repeat00.npy generated-features-fid50k_full-repeat00.npy; do
      [[ -s "${target}/${required}" ]] || { echo "missing ${target}/${required}" >&2; exit 5; }
    done
    kid_feature_sha=$(sha256sum "${target}/generated-features-kid50k_full-repeat00.npy" | cut -d' ' -f1)
    fid_feature_sha=$(sha256sum "${target}/generated-features-fid50k_full-repeat00.npy" | cut -d' ' -f1)
    [[ "${kid_feature_sha}" == "${fid_feature_sha}" ]] || {
      echo "[q256-eval] STOP nonidentical KID/FID features ${job_id}" >&2
      exit 6
    }
    echo "[q256-eval] SHARED_FEATURE_SHA256 ${job_id} ${kid_feature_sha}"
    echo "[q256-eval] PASS ${job_id}"
    tail -n 1 "${target}/metric-kid50k_full.jsonl"
    tail -n 1 "${target}/metric-fid50k_full.jsonl"
  done
done

echo "[q256-eval] WORKER_PASS seed=${seed}"
