#!/usr/bin/env bash
set -euo pipefail

gpu_id="${1:?usage: $0 GPU_ID WORKER_ID}"
worker_id="${2:?usage: $0 GPU_ID WORKER_ID}"

root="${Q128_SERVER_ROOT:-/data/raw/ECT/q128_matched_spacing_v1}"
repo="${Q128_EVALUATOR_REPO:-${root}/evaluator_c8721a0}"
runs="${root}/runs"
eval_root="${root}/evaluation"
sif="${Q128_RUNTIME_SIF:-/data/raw/ECT/ect_runs/q256-target-weight-replay-curve-v1-20260822/runtime/ect-pytorch2401-deterministic.sif}"
dataset="${Q128_DATASET:-/data/raw/ECT/datasets/cifar10-32x32-canonical-08c9ed1b2b1c.zip}"
cache_source="${Q128_CACHE_SOURCE:-/data/raw/ECT/ect_runs/q256-longitudinal-seed6-13-cd-20260823/relay_evaluation/cache/downloads}"
worker_cache="${eval_root}/worker-cache-${worker_id}"
exec_mode="${Q128_EXEC_MODE:-sif}"
sandbox_root="${Q128_RUNTIME_SANDBOX:-/root/q128_runtime/sandbox}"

[[ "${gpu_id}" =~ ^[0-9]+$ ]] || { echo "gpu_id must be a non-negative integer" >&2; exit 2; }
[[ -f "${repo}/ct_eval.py" && -f "${dataset}" ]] || {
  echo "missing evaluator, runtime, or canonical dataset" >&2
  exit 2
}
if [[ "${exec_mode}" == sif ]]; then
  [[ -f "${sif}" ]] || { echo "missing SIF runtime" >&2; exit 2; }
elif [[ "${exec_mode}" == sandbox ]]; then
  [[ -x "${sandbox_root}/usr/bin/python" ]] || { echo "missing sandbox runtime" >&2; exit 2; }
else
  echo "Q128_EXEC_MODE must be sif or sandbox" >&2
  exit 2
fi

mkdir -p "${eval_root}/jobs" "${eval_root}/locks" "${eval_root}/done" \
  "${eval_root}/logs" "${worker_cache}/downloads"
if [[ -d "${cache_source}" ]]; then
  cp -a "${cache_source}/." "${worker_cache}/downloads/"
fi

runtime_ld_library_path="${sandbox_root}/usr/local/lib/python3.10/dist-packages/torch/lib"
runtime_ld_library_path+=":${sandbox_root}/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib"
runtime_ld_library_path+=":${sandbox_root}/usr/local/cuda/compat/lib"
runtime_ld_library_path+=":${sandbox_root}/usr/local/nvidia/lib:${sandbox_root}/usr/local/nvidia/lib64"
runtime_ld_library_path+=":${sandbox_root}/lib:${sandbox_root}/lib/x86_64-linux-gnu"
runtime_ld_library_path+=":${sandbox_root}/opt/hpcx/clusterkit/lib:${sandbox_root}/opt/hpcx/hcoll/lib"
runtime_ld_library_path+=":${sandbox_root}/opt/hpcx/nccl_rdma_sharp_plugin/lib:${sandbox_root}/opt/hpcx/ompi/lib"
runtime_ld_library_path+=":${sandbox_root}/opt/hpcx/sharp/lib:${sandbox_root}/opt/hpcx/ucc/lib:${sandbox_root}/opt/hpcx/ucx/lib"
runtime_ld_library_path+=":${sandbox_root}/usr/local/cuda/targets/x86_64-linux/lib:${sandbox_root}/usr/local/lib"

job_count() {
  find "${eval_root}/done" -maxdepth 1 -type f -name '*.SEALED_PASS' | wc -l
}

gpu_is_idle() {
  ! nvidia-smi -i "${gpu_id}" \
    --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | grep -Eq '[0-9]'
}

while [[ "$(job_count)" -lt 210 ]]; do
  claimed=0
  while IFS= read -r snapshot; do
    run_dir="$(dirname "${snapshot}")"
    arm_dir="$(basename "${run_dir}")"
    seed_dir="$(basename "$(dirname "${run_dir}")")"
    arm="${arm_dir#arm}"
    seed="${seed_dir#seed}"
    base="$(basename "${snapshot}")"
    budget_text="${base#network-snapshot-kimg}"
    budget="${budget_text%.pkl}"
    receipt="${run_dir}/network-snapshot-kimg${budget}.receipt.json"
    [[ -s "${receipt}" ]] || continue
    expected_sha="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["snapshot_sha256"])' "${receipt}")"
    actual_sha="$(sha256sum "${snapshot}" | cut -d' ' -f1)"
    [[ "${actual_sha}" == "${expected_sha}" ]] || continue

    for nfe in 1 2; do
      job_id="seed${seed}-arm${arm}-kimg${budget}-nfe${nfe}"
      done_receipt="${eval_root}/done/${job_id}.SEALED_PASS"
      [[ -s "${done_receipt}" ]] && continue
      if ! gpu_is_idle; then
        continue
      fi
      lock="${eval_root}/locks/${job_id}.lock"
      mkdir "${lock}" 2>/dev/null || continue
      claimed=1
      target="${eval_root}/jobs/${job_id}"
      log="${eval_root}/logs/${job_id}.log"
      if [[ -e "${target}" ]]; then
        echo "existing nonterminal output for ${job_id}" > "${lock}/STOPPED_FOR_AUDIT"
        exit 3
      fi
      mid_t=""
      if [[ "${nfe}" == 2 ]]; then
        mid_t="0.821"
      fi
      port=$((33000 + worker_id * 1000 + 10#${budget} + nfe))
      echo "START ${job_id} worker=${worker_id} gpu=${gpu_id}" > "${lock}/launch.txt"
      if [[ "${exec_mode}" == sif ]]; then
        (
          cd "${repo}"
          env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_id}" \
            DNNLIB_CACHE_DIR="${worker_cache}" PYTHONUNBUFFERED=1 \
            singularity exec --nv --bind /data:/data "${sif}" \
            python -m torch.distributed.run \
            --standalone --nproc_per_node=1 --master_port="${port}" \
            "${repo}/ct_eval.py" --resume "${snapshot}" \
            --outdir "${target}" --nosubdir \
            --data "${dataset}" --cond=False --arch=ddpmpp --precond=ct \
            --dropout=0.2 --augment=0 --xflip=False --fp16=False \
            --cache=True --workers=3 --eval-batch=512 --metric-generator-batch=128 \
            --nfe="${nfe}" ${mid_t:+--mid_t="${mid_t}"} \
            --metrics=kid50k_full,fid50k_full --metric-repeats=1 \
            --sample-seeds=0-49999 --seed=20260730 --retain-generated-artifacts \
              --desc="q128-matched-spacing-${job_id}"
        ) > "${log}" 2>&1
      else
        (
          cd "${repo}"
          env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_id}" \
            DNNLIB_CACHE_DIR="${worker_cache}" PYTHONUNBUFFERED=1 \
            PYTHONNOUSERSITE=1 LD_LIBRARY_PATH="${runtime_ld_library_path}" \
            "${sandbox_root}/usr/bin/python" -m torch.distributed.run \
              --standalone --nproc_per_node=1 --master_port="${port}" \
              "${repo}/ct_eval.py" --resume "${snapshot}" \
              --outdir "${target}" --nosubdir \
              --data "${dataset}" --cond=False --arch=ddpmpp --precond=ct \
              --dropout=0.2 --augment=0 --xflip=False --fp16=False \
              --cache=True --workers=3 --eval-batch=512 --metric-generator-batch=128 \
              --nfe="${nfe}" ${mid_t:+--mid_t="${mid_t}"} \
              --metrics=kid50k_full,fid50k_full --metric-repeats=1 \
              --sample-seeds=0-49999 --seed=20260730 --retain-generated-artifacts \
              --desc="q128-matched-spacing-${job_id}"
        ) > "${log}" 2>&1
      fi

      grep -q 'Exiting...' "${target}/log.txt"
      for required in metric-kid50k_full.jsonl metric-fid50k_full.jsonl \
        generated-samples.npy generated-features-kid50k_full-repeat00.npy \
        generated-features-fid50k_full-repeat00.npy; do
        [[ -s "${target}/${required}" ]] || {
          echo "missing ${required}" > "${lock}/STOPPED_FOR_AUDIT"
          exit 4
        }
      done
      kid_sha="$(sha256sum "${target}/generated-features-kid50k_full-repeat00.npy" | cut -d' ' -f1)"
      fid_sha="$(sha256sum "${target}/generated-features-fid50k_full-repeat00.npy" | cut -d' ' -f1)"
      [[ "${kid_sha}" == "${fid_sha}" ]] || {
        echo "KID/FID feature mismatch" > "${lock}/STOPPED_FOR_AUDIT"
        exit 5
      }
      {
        echo "status=SEALED_PASS"
        echo "job_id=${job_id}"
        echo "checkpoint_sha256=${actual_sha}"
        echo "shared_feature_sha256=${kid_sha}"
        echo "sample_seeds=0-49999"
        echo "metric_seed=20260730"
      } > "${done_receipt}"
      rm -rf "${lock}"
      echo "SEALED_PASS ${job_id} completed=$(job_count)/210"
    done
  done < <(find "${runs}" -type f -name 'network-snapshot-kimg*.pkl' | sort)
  if [[ "${claimed}" == 0 ]]; then
    sleep 15
  fi
done

echo "WORKER_COMPLETE worker=${worker_id}"
