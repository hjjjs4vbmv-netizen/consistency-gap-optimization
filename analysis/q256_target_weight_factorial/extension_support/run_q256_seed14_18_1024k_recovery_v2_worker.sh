#!/usr/bin/env bash
set -Eeuo pipefail

gpu_index="${1:?usage: $0 GPU_INDEX SEED MASTER_PORT}"
seed="${2:?usage: $0 GPU_INDEX SEED MASTER_PORT}"
master_port="${3:?usage: $0 GPU_INDEX SEED MASTER_PORT}"

repo=/data/temp/ECT001/q256-factorial-1024k-4582051-seed14-18-v1
runtime=/data/temp/q256-cohort3-runtime/ngc-pytorch-24.01-bundle/rootfs
driver_injection=/data/temp/q256-cohort3-runtime/nvidia-driver-injection-570.211.01
dataset=/mnt/ect_project/datasets/cifar10-32x32.zip
dataset_sha=2d4056e80de1a96fe16f2f58945c6c4710ecd9fc02e3cc7aa5b50513b7cdf389
source_training_root=/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260821/secondary-precision-extension/seed14-18-dcca41b-v2
failed_root=/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260821/secondary-precision-extension/seed14-18-1024k-from-v2-4582051-v1
run_root=/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260821/secondary-precision-extension/seed14-18-1024k-from-v2-4582051-recovery-v2
evaluation_root=${run_root}/frozen-evaluation-1024k-v1
evaluator_source_root=/data/temp/ECT001/q256-factorial-eval-d6aba02-seed14-18-v1
evaluator_source_archive=/data/temp/ECT001/q256-evaluator-source-d6aba02.tar
evaluator_source_archive_sha=37560e2eb50a9a361f9fca899a33778616386a622d5f039f53305d8d492eaed6
canonical_dataset=/data/raw/ECT/datasets/cifar10-32x32-canonical-08c9ed1b2b1c.zip
cache_template=${source_training_root}/frozen-evaluation-seed14-18-v4-native/seed14/evaluator_cache
evaluation_worker=/data/temp/ECT001/q256-seed14-18-1024k-recovery-v2/run_q256_seed14_18_1024k_native_evaluation_worker.sh
evaluation_driver=/data/temp/ECT001/q256-seed14-18-1024k-recovery-v2/run_q256_seed14_18_1024k_frozen_evaluation.py
private_shm="/tmp/ECT001-q256-seed14-18-1024k-recovery-v2-shm-seed${seed}"
active_arm=preflight

umask 027

write_failure() {
  local exit_code=$?
  local failure_path="${run_root}/seed${seed}-worker-failure.md"
  if [[ -d "${run_root}" && ! -e "${failure_path}" ]]; then
    {
      printf '# q256 seed14-18 1024k recovery-v2 pipeline failure\n\n'
      printf -- '- UTC: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf -- '- Seed: %s\n' "${seed}"
      printf -- '- GPU index: %s\n' "${gpu_index}"
      printf -- '- Active arm: %s\n' "${active_arm}"
      printf -- '- Exit code: %s\n' "${exit_code}"
      printf -- '- Action: pipeline stopped; preserve artifacts and resume only from a validated state in a new recovery root.\n'
    } >"${failure_path}"
  fi
  exit "${exit_code}"
}
trap write_failure ERR

case "${seed}" in 14|15|16|17|18) ;; *) echo "unsupported seed: ${seed}" >&2; exit 2 ;; esac
[[ "${gpu_index}" =~ ^[0-4]$ ]] || { echo "invalid GPU index: ${gpu_index}" >&2; exit 2; }
[[ "${master_port}" =~ ^[0-9]+$ ]] || { echo "invalid master port" >&2; exit 2; }
[[ -d "${repo}" && -d "${runtime}" && -d "${driver_injection}" ]] || { echo "missing source or runtime" >&2; exit 2; }
[[ -d "${source_training_root}" && -d "${failed_root}" && -d "${evaluation_root}" && -d "${evaluator_source_root}" && -d "${cache_template}" ]] || { echo "missing training/evaluation root" >&2; exit 2; }
[[ -f "${dataset}" && -f "${canonical_dataset}" && -f "${evaluator_source_archive}" && -f "${evaluation_worker}" && -f "${evaluation_driver}" ]] || { echo "missing immutable asset or evaluator" >&2; exit 2; }
[[ "$(sha256sum "${dataset}" | awk '{print $1}')" == "${dataset_sha}" ]] || { echo "dataset hash mismatch" >&2; exit 2; }
[[ "$(sha256sum "${canonical_dataset}" | awk '{print $1}')" == "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372" ]] || { echo "canonical dataset hash mismatch" >&2; exit 2; }
[[ "$(sha256sum "${evaluator_source_archive}" | awk '{print $1}')" == "${evaluator_source_archive_sha}" ]] || { echo "evaluator source hash mismatch" >&2; exit 2; }

gpu_line=$(nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader,nounits | awk -F', ' -v wanted="${gpu_index}" '$1 == wanted {print $0}')
[[ "${gpu_line}" == "${gpu_index}, "*", NVIDIA A100-PCIE-40GB, 40960" ]] || { echo "assigned GPU identity mismatch: ${gpu_line}" >&2; exit 2; }
gpu_uuid=$(printf '%s\n' "${gpu_line}" | awk -F', ' '{print $2}')
gpu_processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v wanted="${gpu_uuid}" '$1 == wanted {count++} END {print count+0}')
[[ "${gpu_processes}" == 0 ]] || { echo "assigned GPU is not compute-idle: ${gpu_uuid}" >&2; exit 2; }
if ss -H -ltn "sport = :${master_port}" | grep -q .; then
  echo "master port ${master_port} is already listening" >&2
  exit 2
fi
[[ ! -e "${private_shm}" ]] || { echo "refusing existing private shared-memory path: ${private_shm}" >&2; exit 3; }
mkdir -m 700 "${private_shm}"

echo "[worker] START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) seed=${seed} gpu_index=${gpu_index} gpu_uuid=${gpu_uuid} parent_source=dcca41b19e7c45512b5fbe98776520396a1bf9ac budget_extension_source=458205192722883df393a8d017c26e6fa46f48f7"
echo "[worker] DATASET semantic_exact_to_official_cifar10=true archive_sha256=${dataset_sha} canonical_archive_sha256=08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372 byte_identical_to_canonical=false"

verify_final_state() {
  local outdir=$1
  local arm=$2
  proot -0 -r "${runtime}" \
    -b /dev:/dev -b /proc:/proc -b /sys:/sys -b /data:/data -b /mnt:/mnt -b /tmp:/tmp \
    -b /usr/lib/x86_64-linux-gnu:/host-driver-source -b /usr/bin:/host-driver-bin-source \
    -b "${driver_injection}:/usr/local/nvidia" -b "${private_shm}:/dev/shm" \
    /usr/bin/env -i HOME=/root LC_ALL=C.UTF-8 PYTHONNOUSERSITE=1 PYTHONPATH="${repo}" \
      PATH=/usr/local/bin:/usr/bin:/bin \
      LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
      /usr/bin/python -c "import csv, math, torch; d='${outdir}'; s=torch.load(d+'/training-state-latest.pt', map_location='cpu'); assert s['cur_nimg']==1024000, s['cur_nimg']; assert s['attempted_iteration']==8000, s['attempted_iteration']; r=list(csv.DictReader(open(d+'/train_summary.csv')))[-1]; assert int(r['attempted_iteration'])==8000; assert math.isclose(float(r['processed_kimg']),1024.0); assert math.isfinite(float(r['loss'])); assert int(r['step_skipped'])==0; print('[worker] FINAL_STATE_PASS seed=${seed} arm=${arm} attempts=8000 processed_kimg=1024 successful_optimizer_steps=%d' % s['successful_optimizer_steps'])"
}

adopt_completed_arm_a() {
  local arm=A
  local source_dir=${failed_root}/seed${seed}/armA
  local outdir=${run_root}/seed${seed}/armA
  local source_manifest=${run_root}/provenance/seed${seed}-armA-failed-v1-files.sha256
  local copied_manifest=${run_root}/provenance/seed${seed}-armA-recovery-v2-files.sha256
  active_arm=recoverA
  [[ -d "${source_dir}" && ! -L "${source_dir}" ]] || { echo "missing failed-v1 armA: ${source_dir}" >&2; return 3; }
  [[ ! -e "${outdir}" ]] || { echo "refusing existing recovery armA: ${outdir}" >&2; return 3; }
  if find "${source_dir}" -type l -print -quit | grep -q .; then
    echo "failed-v1 armA contains a symlink: ${source_dir}" >&2
    return 3
  fi
  verify_final_state "${source_dir}" A
  mkdir -p "${run_root}/seed${seed}"
  (cd "${source_dir}" && find . -type f -print0 | sort -z | xargs -0 sha256sum) >"${source_manifest}"
  cp -a "${source_dir}" "${run_root}/seed${seed}/"
  (cd "${outdir}" && find . -type f -print0 | sort -z | xargs -0 sha256sum) >"${copied_manifest}"
  cmp "${source_manifest}" "${copied_manifest}"
  verify_final_state "${outdir}" A
  {
    printf 'status=PASS\nseed=%s\narm=A\ntarget_kimg=1024\nattempted_iteration=8000\n' "${seed}"
    printf 'recovery_kind=hash_identical_adoption_after_post_training_verifier_failure\n'
    printf 'adopted_from=%s\n' "${source_dir}"
    sha256sum "${outdir}/training-state-latest.pt" "${outdir}/network-snapshot-latest.pkl" "${outdir}/train_summary.csv" "${source_manifest}" "${copied_manifest}"
  } >"${run_root}/provenance/seed${seed}-armA-1024k-PASS.txt"
  echo "[worker] ARM_ADOPT_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) seed=${seed} arm=A source=failed-v1 hash_identical=true"
}

run_arm() {
  local arm=$1
  local target_scale=$2
  local denominator_scale=$3
  local outdir=${run_root}/seed${seed}/arm${arm}
  local source_dir=${source_training_root}/seed${seed}/arm${arm}
  local resume_state=${source_dir}/training-state-latest.pt
  active_arm=${arm}
  [[ ! -e "${outdir}" ]] || { echo "refusing existing fresh cell: ${outdir}" >&2; return 3; }
  mkdir -p "${run_root}/seed${seed}"
  [[ -f "${resume_state}" ]] || { echo "missing 256k source state: ${resume_state}" >&2; return 3; }
  mkdir "${outdir}"
  for artifact in training_options.json train_summary.csv factorial_training_telemetry_v1.csv initial_state_receipt_v1.json log.txt; do
    [[ -f "${source_dir}/${artifact}" ]] || { echo "missing source artifact: ${source_dir}/${artifact}" >&2; return 3; }
    cp "${source_dir}/${artifact}" "${outdir}/${artifact}"
  done
  echo "[worker] ARM_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) seed=${seed} arm=${arm} target=${target_scale} denominator=${denominator_scale} mode=strict_budget_only_resume source_kimg=256 target_kimg=1024"
  timeout --signal=TERM --kill-after=10s 24h \
    proot -0 -r "${runtime}" \
      -b /dev:/dev -b /proc:/proc -b /sys:/sys \
      -b /data:/data -b /mnt:/mnt -b /tmp:/tmp \
      -b /usr/lib/x86_64-linux-gnu:/host-driver-source \
      -b /usr/bin:/host-driver-bin-source \
      -b "${driver_injection}:/usr/local/nvidia" \
      -b "${private_shm}:/dev/shm" \
      /usr/bin/env -i \
        HOME=/root LC_ALL=C.UTF-8 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 \
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
        PATH=/usr/local/lib/python3.10/dist-packages/torch_tensorrt/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/tensorrt/bin \
        LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
        CUDA_VERSION=12.3.2.001 CUDNN_VERSION=8.9.7.29+cuda12.2 PYTORCH_VERSION=2.2.0a0+81ea7a4 \
        NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_index}" \
        CUDA_CACHE_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        MASTER_ADDR=127.0.0.1 MASTER_PORT="${master_port}" \
        RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
        /usr/bin/python "${repo}/ct_train.py" \
          --data="${dataset}" --outdir="${outdir}" --nosubdir \
          --cond=False --arch=ddpmpp --precond=ect --batch=128 --batch-gpu=16 \
          --optim=RAdam --lr=0.0001 --dropout=0.2 --augment=0 --xflip=False \
          --mean=-1.1 --std=2.0 --mapping=sigmoid --global-gap-scale=1.0 \
          --factorial-protocol=q256_target_weight_v1 \
          --target-gap-scale="${target_scale}" \
          --denominator-gap-scale="${denominator_scale}" \
          -q 256 -k 8 -b 1 -c 0 --double=10000 --ema_beta=0.9993 \
          --seed="${seed}" --fp16=True --tf32=False --ls=1.0 --enable_amp=True \
          --bench=False --cache=True --workers=1 --metrics=none --duration=1.024 \
          --tick=10 --snap=0 --dump=0 --ckpt=10 --sample_every=26 \
          --eval_every=50 --mid_t=0.821 --adaptive-update-kimg=0.5 \
          --resume="${resume_state}"

  verify_final_state "${outdir}" "${arm}"
  {
    printf 'status=PASS\nseed=%s\narm=%s\ntarget_kimg=1024\nattempted_iteration=8000\n' "${seed}" "${arm}"
    sha256sum "${outdir}/training-state-latest.pt" "${outdir}/network-snapshot-latest.pkl" "${outdir}/train_summary.csv"
  } >"${run_root}/provenance/seed${seed}-arm${arm}-1024k-PASS.txt"
  echo "[worker] ARM_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) seed=${seed} arm=${arm}"
}

adopt_completed_arm_a
run_arm B 1.1 1.1
run_arm C 1.1 1.0
run_arm D 1.0 1.1

printf 'status=TRAINING_WORKER_PASS\nseed=%s\ncompleted_at_utc=%s\n' "${seed}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${run_root}/seed${seed}-TRAINING_WORKER_PASS.txt"
echo "[worker] TRAINING_WORKER_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) seed=${seed}"

active_arm=evaluation
base_port=$((48140 + (seed - 14) * 20))
"${evaluation_worker}" \
  "${seed}" "${gpu_index}" "${runtime}" "${driver_injection}" \
  "${evaluator_source_root}" "${evaluator_source_archive}" "${evaluator_source_archive_sha}" \
  "${run_root}" "${evaluation_root}/seed${seed}" "${canonical_dataset}" \
  "${base_port}" "${evaluation_driver}" "${cache_template}"

active_arm=complete
trap - ERR
printf 'status=PIPELINE_WORKER_PASS\nseed=%s\ncompleted_at_utc=%s\n' "${seed}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${run_root}/seed${seed}-PIPELINE_WORKER_PASS.txt"
echo "[worker] PIPELINE_WORKER_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) seed=${seed}"
