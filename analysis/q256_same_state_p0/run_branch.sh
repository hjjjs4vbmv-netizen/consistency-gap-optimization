#!/usr/bin/env bash
set -euo pipefail

seed="${1:?usage: run_branch.sh SEED ARM GPU END_KIMG MASTER_PORT}"
arm="${2:?usage: run_branch.sh SEED ARM GPU END_KIMG MASTER_PORT}"
gpu="${3:?usage: run_branch.sh SEED ARM GPU END_KIMG MASTER_PORT}"
end_kimg="${4:?usage: run_branch.sh SEED ARM GPU END_KIMG MASTER_PORT}"
master_port="${5:?usage: run_branch.sh SEED ARM GPU END_KIMG MASTER_PORT}"

case "${seed}" in 3|4|5) ;; *) echo "invalid seed" >&2; exit 2;; esac
case "${arm}" in
  A) target=1.0; denominator=1.0 ;;
  B) target=1.1; denominator=1.1 ;;
  C) target=1.1; denominator=1.0 ;;
  D) target=1.0; denominator=1.1 ;;
  *) echo "invalid arm" >&2; exit 2 ;;
esac
if [[ "${arm}" == B ]]; then
  [[ "${end_kimg}" == 512 ]] || { echo "B parity must end at 512" >&2; exit 2; }
  immutable_kimg=448,512
  immutable_attempts=3001,3004,3016,3064,3128,3256,3500,4000
else
  [[ "${end_kimg}" == 448 ]] || { echo "formal A/C/D must end at 448" >&2; exit 2; }
  immutable_kimg=448
  immutable_attempts=3001,3004,3016,3064,3128,3256,3500
fi

repo="${Q256_P0_REPO:-/data/raw/ECT/worktrees/q256-b384-same-state-p0-v1}"
output_root="${Q256_P0_OUTPUT_ROOT:-/data/raw/ECT/ect_runs/q256-b384-same-state-p0-v1}"
source_root="${Q256_P0_SOURCE_ROOT:-/data/raw/ECT/ect_runs/q256-target-weight-replay-curve-v1-20260822/runs/q256-target-weight-replay-curve-v1}"
runtime="${Q256_P0_RUNTIME:-/data/raw/ECT/ect_runs/q256-target-weight-replay-curve-v1-20260822/runtime/ect-pytorch2401-deterministic.sif}"
dataset="${Q256_P0_DATASET:-/data/raw/ECT/datasets/cifar10-32x32-canonical-08c9ed1b2b1c.zip}"
protocol="${repo}/analysis/q256_same_state_p0/protocol.json"
run_dir="${output_root}/runs/seed${seed}/B384_to_${arm}"
source_state="${source_root}/seed${seed}/armB/training-state-kimg000384.pt"

repo_git() {
  (cd "${repo}" && git "$@")
}

declare -A expected_source=(
  [3]=5173a6b1532c3589c8dd1e6095ab3fca4fffd77331c08932688d11df5e7cf7b8
  [4]=724d47531a8ded39af61cd98265efa8dc1dc6ed03e2e080886a243ad9650d210
  [5]=23805fe2eceefed7ed58006f96253d5f5fcfa32887e0833b7af7a5750a2fcb17
)

[[ -s "${source_state}" && -s "${runtime}" && -s "${dataset}" && -s "${protocol}" ]] || { echo "missing frozen input" >&2; exit 3; }
[[ "$(sha256sum "${source_state}" | cut -d' ' -f1)" == "${expected_source[${seed}]}" ]] || { echo "source SHA mismatch" >&2; exit 3; }
[[ "$(sha256sum "${dataset}" | cut -d' ' -f1)" == 08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372 ]] || { echo "dataset SHA mismatch" >&2; exit 3; }
repo_git diff --quiet && repo_git diff --cached --quiet || { echo "repository is dirty" >&2; exit 3; }
protocol_sha="$(sha256sum "${protocol}" | cut -d' ' -f1)"
implementation_commit="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["implementation_commit"])' "${protocol}")"
[[ "$(repo_git rev-parse HEAD^)" == "${implementation_commit}" ]] || { echo "protocol implementation binding mismatch" >&2; exit 3; }

mkdir -p "${output_root}/runs/seed${seed}" "${output_root}/logs"
mkdir "${run_dir}"
gpu_uuid="$(nvidia-smi --id="${gpu}" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
  echo "start_utc=${start_utc}"
  echo "hostname=$(hostname)"
  echo "physical_gpu=${gpu}"
  echo "gpu_uuid=${gpu_uuid}"
  echo "git_commit=$(repo_git rev-parse HEAD)"
  echo "implementation_commit=${implementation_commit}"
  echo "protocol_sha256=${protocol_sha}"
  echo "source_state=${source_state}"
  echo "source_sha256=${expected_source[${seed}]}"
  echo "runtime=${runtime}"
  echo "runtime_sha256=$(sha256sum "${runtime}" | cut -d' ' -f1)"
  echo "dataset=${dataset}"
} > "${run_dir}/launch_environment.txt"

duration="0.${end_kimg}"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}" \
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_CACHE_DISABLE=1 CUDA_MODULE_LOADING=LAZY \
TORCH_CUDNN_V8_API_ENABLED=1 USE_EXPERIMENTAL_CUDNN_V8_API=1 \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONIOENCODING=utf-8 \
PYTHONNOUSERSITE=1 LC_ALL=C.UTF-8 MASTER_ADDR=127.0.0.1 MASTER_PORT="${master_port}" \
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 PYTHONUNBUFFERED=1 \
apptainer exec --nv --bind /data:/data --pwd "${repo}" "${runtime}" python ct_train.py \
  --data="${dataset}" --outdir="${run_dir}" --nosubdir \
  --cond=False --arch=ddpmpp --precond=ect --batch=128 --batch-gpu=16 \
  --optim=RAdam --lr=0.0001 --dropout=0.2 --augment=0 --xflip=False \
  --mean=-1.1 --std=2.0 --mapping=sigmoid --global-gap-scale=1.0 \
  --factorial-protocol=q256_target_weight_v1 \
  --target-gap-scale="${target}" --denominator-gap-scale="${denominator}" \
  -q 256 -k 8 -b 1 -c 0 --double=10000 --ema_beta=0.9993 \
  --seed="${seed}" --fp16=True --tf32=False --ls=1.0 --enable_amp=True \
  --bench=False --cache=True --workers=1 --metrics=none --duration="${duration}" \
  --tick=10 --snap=0 --dump=0 --ckpt=10 --sample_every=26 --eval_every=50 \
  --mid_t=0.821 --adaptive-update-kimg=0.5 \
  --immutable-checkpoint-kimg="${immutable_kimg}" \
  --immutable-checkpoint-attempts="${immutable_attempts}" \
  --q256-b384-same-state-fork \
  --q256-b384-protocol-sha256="${protocol_sha}" --resume="${source_state}" \
  2>&1 | tee "${output_root}/logs/seed${seed}-B384_to_${arm}.log"

[[ "$(sha256sum "${source_state}" | cut -d' ' -f1)" == "${expected_source[${seed}]}" ]] || { echo "source mutated" >&2; exit 4; }
apptainer exec --nv --bind /data:/data --pwd "${repo}" "${runtime}" python \
  analysis/q256_same_state_p0/verify_branch.py \
  --run-dir "${run_dir}" --source-state "${source_state}" \
  --source-sha256 "${expected_source[${seed}]}" --protocol-sha256 "${protocol_sha}" \
  --seed "${seed}" --arm "${arm}" --end-kimg "${end_kimg}"
echo "PASS seed=${seed} branch=B384_to_${arm} gpu_uuid=${gpu_uuid}"
