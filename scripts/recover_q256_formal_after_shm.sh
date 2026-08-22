#!/usr/bin/env bash
set -euo pipefail

gpu_uuid="${1:?missing GPU UUID}"
worker_id="${2:?missing worker id}"
master_port="${3:?missing master port}"
private_shm="${4:?missing private shared-memory directory}"
shift 4

case "${gpu_uuid}" in
  GPU-d79117bb-8d91-4f2e-d7bb-718e347ce859|GPU-ef9edaf6-d661-e143-efd1-154c1ad29f10) ;;
  *) echo "unexpected GPU UUID: ${gpu_uuid}" >&2; exit 2 ;;
esac
case "${worker_id}" in gpu0|gpu1) ;; *) echo "unexpected worker id" >&2; exit 2 ;; esac
[[ "${master_port}" =~ ^[0-9]+$ ]] || { echo "invalid master port" >&2; exit 2; }
(( $# > 0 )) || { echo "no recovery cells supplied" >&2; exit 2; }

repo=/data/temp/ECT001/q256-factorial-clean-25c3d22
expected_head=dcca41b19e7c45512b5fbe98776520396a1bf9ac
runs_root=/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819
formal_root=${runs_root}/formal/formal-direct-dcca41b-deterministic-v1
dataset=/data/raw/ECT/datasets/cifar10-32x32-canonical-08c9ed1b2b1c.zip
transfer=/data/raw/ECT/pretrained/edm-cifar10-32x32-uncond-vp.pkl
sandbox=/data/temp/ect001-pytorch2401-sandbox
apptainer=/usr/bin/apptainer
worker_log=${formal_root}/recovery-worker-${worker_id}.log

exec >>"${worker_log}" 2>&1

[[ -d "${repo}" && -d "${sandbox}" ]] || { echo "missing source or sandbox" >&2; exit 2; }
[[ -f "${dataset}" && -f "${transfer}" ]] || { echo "missing immutable asset" >&2; exit 2; }
[[ -d "${private_shm}" && ! -L "${private_shm}" ]] || { echo "invalid private shared-memory directory" >&2; exit 2; }
[[ "$(stat -c '%U:%a' "${private_shm}")" == "ECT001:700" ]] || { echo "unsafe private shared-memory ownership or mode" >&2; exit 2; }
[[ "$(cd "${repo}" && git rev-parse HEAD)" == "${expected_head}" ]] || { echo "wrong source HEAD" >&2; exit 2; }
[[ -z "$(cd "${repo}" && git status --porcelain)" ]] || { echo "source worktree is dirty" >&2; exit 2; }

for cell in "$@"; do
  IFS=: read -r seed arm target_scale denominator_scale mode <<<"${cell}"
  case "${seed}" in 3|4|5) ;; *) echo "invalid seed in ${cell}" >&2; exit 2 ;; esac
  case "${arm}:${target_scale}:${denominator_scale}" in
    A:1.0:1.0|B:1.1:1.1|C:1.1:1.0|D:1.0:1.1) ;;
    *) echo "invalid arm factors in ${cell}" >&2; exit 2 ;;
  esac
  case "${mode}" in resume|fresh) ;; *) echo "invalid mode in ${cell}" >&2; exit 2 ;; esac

  outdir=${formal_root}/seed${seed}/arm${arm}
  launch_source=()
  if [[ "${mode}" == resume ]]; then
    state=${outdir}/training-state-latest.pt
    [[ -d "${outdir}" && -f "${state}" ]] || { echo "missing resume cell or state: ${outdir}" >&2; exit 3; }
    [[ -f "${outdir}/initial_state_receipt_v1.json" ]] || { echo "missing resume receipt: ${outdir}" >&2; exit 3; }
    launch_source=(--resume="${state}")
  else
    [[ ! -e "${outdir}" ]] || { echo "refusing existing fresh cell: ${outdir}" >&2; exit 3; }
    mkdir -p "${formal_root}/seed${seed}"
    launch_source=(--transfer="${transfer}")
  fi

  echo "[formal-recovery] START seed=${seed} arm=${arm} mode=${mode} gpu=${gpu_uuid} head=${expected_head} shm=${private_shm}"
  CUDA_VISIBLE_DEVICES="${gpu_uuid}" "${apptainer}" exec --nv \
    --bind /data/raw:/data/raw --bind /data/temp:/data/temp \
    --bind "${private_shm}:/dev/shm" \
    "${sandbox}" env \
    ECT_Q256_LAUNCHER_IN_SANDBOX=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_uuid}" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT="${master_port}" \
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 PYTHONUNBUFFERED=1 \
    python "${repo}/ct_train.py" \
    --data="${dataset}" --outdir="${outdir}" --nosubdir \
    --cond=False --arch=ddpmpp --precond=ect --batch=128 --batch-gpu=16 \
    --optim=RAdam --lr=0.0001 --dropout=0.2 --augment=0 --xflip=False \
    --mean=-1.1 --std=2.0 --mapping=sigmoid --global-gap-scale=1.0 \
    --factorial-protocol=q256_target_weight_v1 \
    --target-gap-scale="${target_scale}" \
    --denominator-gap-scale="${denominator_scale}" \
    -q 256 -k 8 -b 1 -c 0 --double=10000 --ema_beta=0.9993 \
    --seed="${seed}" --fp16=True --tf32=False --ls=1.0 --enable_amp=True \
    --bench=False --cache=True --workers=1 --metrics=none --duration=0.256 \
    --tick=10 --snap=0 --dump=0 --ckpt=10 --sample_every=26 \
    --eval_every=50 --mid_t=0.821 --adaptive-update-kimg=0.5 \
    "${launch_source[@]}"
  echo "[formal-recovery] PASS seed=${seed} arm=${arm} mode=${mode} gpu=${gpu_uuid}"
done

echo "[formal-recovery] WORKER_PASS ${worker_id}"
