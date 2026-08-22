#!/usr/bin/env bash
set -Eeuo pipefail

gpu_index="${1:?usage: $0 GPU_INDEX SEED MASTER_PORT REPLAY_COMMIT}"
seed="${2:?usage: $0 GPU_INDEX SEED MASTER_PORT REPLAY_COMMIT}"
master_port="${3:?usage: $0 GPU_INDEX SEED MASTER_PORT REPLAY_COMMIT}"
replay_commit="${4:?usage: $0 GPU_INDEX SEED MASTER_PORT REPLAY_COMMIT}"

repo=/data/temp/ECT001/q256-learning-curve-replay-source-v1
runtime=/data/temp/q256-cohort3-runtime/ngc-pytorch-24.01-bundle/rootfs
driver_injection=/data/temp/q256-cohort3-runtime/nvidia-driver-injection-570.211.01
dataset=/mnt/ect_project/datasets/cifar10-32x32.zip
dataset_sha=2d4056e80de1a96fe16f2f58945c6c4710ecd9fc02e3cc7aa5b50513b7cdf389
source_training_root=/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260821/secondary-precision-extension/seed14-18-dcca41b-v2
original_1024_root=/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260821/secondary-precision-extension/seed14-18-1024k-from-v2-4582051-recovery-v2
run_root=/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260821/secondary-precision-extension/seed14-18-256to1024-learning-curve-replay-v1
evaluation_root=${run_root}/formal-evaluation-learning-curves-v1
evaluator_source_root=/data/temp/ECT001/q256-factorial-eval-d6aba02-seed14-18-v1
evaluator_source_archive=/data/temp/ECT001/q256-evaluator-source-d6aba02.tar
evaluator_source_archive_sha=37560e2eb50a9a361f9fca899a33778616386a622d5f039f53305d8d492eaed6
canonical_dataset=/data/raw/ECT/datasets/cifar10-32x32-canonical-08c9ed1b2b1c.zip
cache_template=${source_training_root}/frozen-evaluation-seed14-18-v4-native/seed14/evaluator_cache
bundle=/data/temp/ECT001/q256-learning-curve-replay-bundle-v1
evaluation_worker=${bundle}/run_learning_curve_native_evaluation_worker.sh
evaluation_driver=${bundle}/run_learning_curve_frozen_evaluation.py
trajectory_verifier=${bundle}/verify_replay_trajectory.py
inventory=${run_root}/replay_source_inventory.json
private_shm="/tmp/ECT001-q256-learning-curve-replay-v1-shm-seed${seed}"
active_arm=preflight

umask 027

write_failure() {
  local exit_code=$?
  local failure_path="${run_root}/seed${seed}-worker-failure.md"
  if [[ -d "${run_root}" && ! -e "${failure_path}" ]]; then
    {
      printf '# q256 seed14-18 learning-curve replay pipeline failure\n\n'
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
[[ -d "${source_training_root}" && -d "${original_1024_root}" && -d "${evaluation_root}" && -d "${evaluator_source_root}" && -d "${cache_template}" ]] || { echo "missing training/evaluation root" >&2; exit 2; }
[[ -f "${dataset}" && -f "${canonical_dataset}" && -f "${evaluator_source_archive}" && -f "${evaluation_worker}" && -f "${evaluation_driver}" && -f "${trajectory_verifier}" && -f "${inventory}" ]] || { echo "missing immutable asset, inventory, verifier, or evaluator" >&2; exit 2; }
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

echo "[worker] START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) seed=${seed} gpu_index=${gpu_index} gpu_uuid=${gpu_uuid} parent_source=dcca41b19e7c45512b5fbe98776520396a1bf9ac replay_source=${replay_commit}"
echo "[worker] DATASET semantic_exact_to_official_cifar10=true archive_sha256=${dataset_sha} canonical_archive_sha256=08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372 byte_identical_to_canonical=false"

run_arm() {
  local arm=$1
  local target_scale=$2
  local denominator_scale=$3
  local outdir=${run_root}/seed${seed}/arm${arm}
  local source_dir=${source_training_root}/seed${seed}/arm${arm}
  local resume_state=${source_dir}/training-state-latest.pt
  local original_1024_state=${original_1024_root}/seed${seed}/arm${arm}/training-state-latest.pt
  active_arm=${arm}
  [[ ! -e "${outdir}" ]] || { echo "refusing existing fresh cell: ${outdir}" >&2; return 3; }
  mkdir -p "${run_root}/seed${seed}"
  [[ -f "${resume_state}" && -f "${original_1024_state}" ]] || { echo "missing 256k source or original 1024 state: seed${seed}/arm${arm}" >&2; return 3; }
  python3 -c "import json; p=json.load(open('${inventory}')); c=next(x for x in p['cells'] if x['seed']==${seed} and x['arm']=='${arm}'); assert c['status']=='PASS', c" || { echo "inventory blocks seed${seed}/arm${arm}" >&2; return 3; }
  mkdir "${outdir}"
  for artifact in training_options.json train_summary.csv factorial_training_telemetry_v1.csv initial_state_receipt_v1.json log.txt; do
    [[ -f "${source_dir}/${artifact}" ]] || { echo "missing source artifact: ${source_dir}/${artifact}" >&2; return 3; }
    cp "${source_dir}/${artifact}" "${outdir}/${artifact}"
  done
  python3 -c "import json,os; p='${outdir}/run_config.json'; payload={'schema':'ect.q256.learning-curve-run-config/v1','seed':${seed},'arm':'${arm}','gpu_index':${gpu_index},'source_state':'${resume_state}','source_state_sha256':next(x for x in json.load(open('${inventory}'))['cells'] if x['seed']==${seed} and x['arm']=='${arm}')['files']['training-state-latest.pt']['sha256'],'source_commit':'dcca41b19e7c45512b5fbe98776520396a1bf9ac','replay_commit':'${replay_commit}','target_kimg':1024,'milestones_kimg':[384,512,640,768,896,1024],'target_gap_scale':${target_scale},'denominator_gap_scale':${denominator_scale}}; f=open(p,'x'); json.dump(payload,f,indent=2,sort_keys=True); f.write('\\n'); f.flush(); os.fsync(f.fileno())"
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
          --checkpoint-milestone-kimg=384 \
          --checkpoint-milestone-kimg=512 \
          --checkpoint-milestone-kimg=640 \
          --checkpoint-milestone-kimg=768 \
          --checkpoint-milestone-kimg=896 \
          --checkpoint-milestone-kimg=1024 \
          --resume="${resume_state}"

  cp "${outdir}/log.txt" "${outdir}/training.log"
  proot -0 -r "${runtime}" \
    -b /dev:/dev -b /proc:/proc -b /sys:/sys -b /data:/data -b /mnt:/mnt -b /tmp:/tmp \
    -b /usr/lib/x86_64-linux-gnu:/host-driver-source -b /usr/bin:/host-driver-bin-source \
    -b "${driver_injection}:/usr/local/nvidia" -b "${private_shm}:/dev/shm" \
    /usr/bin/env -i HOME=/root LC_ALL=C.UTF-8 PYTHONNOUSERSITE=1 PYTHONPATH="${repo}" \
      PATH=/usr/local/bin:/usr/bin:/bin \
      LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
      /usr/bin/python "${trajectory_verifier}" \
        --run-dir "${outdir}" \
        --source-state "${resume_state}" \
        --original-1024-state "${original_1024_state}" \
        --seed "${seed}" --arm "${arm}" \
        --source-commit dcca41b19e7c45512b5fbe98776520396a1bf9ac \
        --replay-commit "${replay_commit}"
  echo "[worker] ARM_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) seed=${seed} arm=${arm}"
}

run_arm A 1.0 1.0
run_arm B 1.1 1.1
run_arm C 1.1 1.0
run_arm D 1.0 1.1

printf 'status=TRAINING_WORKER_PASS\nseed=%s\ncompleted_at_utc=%s\n' "${seed}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${run_root}/seed${seed}-TRAINING_WORKER_PASS.txt"
echo "[worker] TRAINING_WORKER_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) seed=${seed}"

active_arm=evaluation
base_port=$((50140 + (seed - 14) * 60))
"${evaluation_worker}" \
  "${seed}" "${gpu_index}" "${runtime}" "${driver_injection}" \
  "${evaluator_source_root}" "${evaluator_source_archive}" "${evaluator_source_archive_sha}" \
  "${run_root}" "${evaluation_root}/seed${seed}" "${canonical_dataset}" \
  "${base_port}" "${evaluation_driver}" "${cache_template}" "${replay_commit}"

active_arm=complete
trap - ERR
printf 'status=PIPELINE_WORKER_PASS\nseed=%s\ncompleted_at_utc=%s\n' "${seed}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${run_root}/seed${seed}-PIPELINE_WORKER_PASS.txt"
echo "[worker] PIPELINE_WORKER_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) seed=${seed}"
