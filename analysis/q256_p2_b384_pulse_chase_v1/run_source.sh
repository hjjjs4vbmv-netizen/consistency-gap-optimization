#!/usr/bin/env bash
set -Eeuo pipefail

seed="${1:?usage: run_source.sh SEED GPU REPO PROTOCOL DATASET TRANSFER RUNTIME_SIF SOURCE_DIR}"
gpu="${2:?missing GPU}"
repo="${3:?missing repository}"
protocol="${4:?missing protocol}"
dataset="${5:?missing dataset}"
transfer="${6:?missing transfer checkpoint}"
runtime_sif="${7:?missing runtime SIF}"
source_dir="${8:?missing source output directory}"
run_kind="${9:-formal}"

if [[ "${run_kind}" == formal ]]; then
  case "${seed}" in 19|20|21|22|23|24|25|26|27|28) ;; *) echo "invalid formal seed" >&2; exit 2;; esac
else
  [[ "${run_kind}" == smoke && "${seed}" == 18 ]] || { echo "invalid smoke seed" >&2; exit 2; }
fi
expected_gpu=1; (( seed <= 23 )) && expected_gpu=0
[[ "${run_kind}" == smoke || "${gpu}" == "${expected_gpu}" ]] || { echo "seed/GPU assignment mismatch" >&2; exit 2; }
[[ -d "${repo}/.git" && -f "${protocol}" && -f "${dataset}" && -f "${transfer}" && -f "${runtime_sif}" ]] || { echo "missing frozen input" >&2; exit 2; }
[[ ! -e "${source_dir}" ]] || { echo "refuse existing source cell" >&2; exit 3; }

implementation_commit="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["implementation_commit"])' "${protocol}")"
[[ "$(git -C "${repo}" rev-parse HEAD^)" == "${implementation_commit}" ]] || { echo "protocol implementation binding mismatch" >&2; exit 3; }
[[ -z "$(git -C "${repo}" status --porcelain)" ]] || { echo "implementation worktree is dirty" >&2; exit 3; }
[[ "$(sha256sum "${dataset}" | awk '{print $1}')" == 08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372 ]] || { echo "dataset hash mismatch" >&2; exit 3; }
[[ "$(sha256sum "${transfer}" | awk '{print $1}')" == 4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da ]] || { echo "transfer hash mismatch" >&2; exit 3; }
[[ "$(sha256sum "${runtime_sif}" | awk '{print $1}')" == 9d5f2c9e68f1f7dcaa20457bf6e0b6fa46f74a8605edaf5d49fdccf9f6bb62ea ]] || { echo "runtime hash mismatch" >&2; exit 3; }

gpu_uuid="$(nvidia-smi --id="${gpu}" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
[[ -n "${gpu_uuid}" ]] || { echo "cannot resolve GPU UUID" >&2; exit 3; }
if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -q "^${gpu_uuid},"; then
  echo "assigned GPU has another compute process" >&2; exit 3
fi

mkdir -p "$(dirname "${source_dir}")"
mkdir "${source_dir}"
failure_receipt="${source_dir}/FAILED_RECEIPT.json"
started="$(date +%s)"
on_failure() {
  code=$?
  python3 -c 'import json,os,sys,time; p=sys.argv[1]; d={"schema":"ect.q256.p2-infrastructure-failure/v1","status":"FAILED","cell":"source","seed":int(sys.argv[2]),"gpu_index":int(sys.argv[3]),"gpu_uuid":sys.argv[4],"exit_code":int(sys.argv[5]),"ended_unix":int(time.time())}; f=open(p,"x"); json.dump(d,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())' "${failure_receipt}" "${seed}" "${gpu}" "${gpu_uuid}" "${code}" || true
  exit "${code}"
}
trap on_failure ERR

python3 -c 'import json,os,sys,time; d={"schema":"ect.q256.p2-source-start/v1","status":"START","seed":int(sys.argv[2]),"gpu_index":int(sys.argv[3]),"gpu_uuid":sys.argv[4],"protocol_sha256":sys.argv[5],"implementation_commit":sys.argv[6],"execution_commit":sys.argv[7],"started_unix":int(time.time())}; f=open(sys.argv[1],"x"); json.dump(d,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())' \
  "${source_dir}/compute_start_receipt.json" "${seed}" "${gpu}" "${gpu_uuid}" \
  "$(sha256sum "${protocol}" | awk '{print $1}')" "${implementation_commit}" "$(git -C "${repo}" rev-parse HEAD)"

master_port=$((43000 + gpu * 100 + seed))
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_CACHE_DISABLE=1 \
    PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="${master_port}" RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
  timeout --signal=TERM --kill-after=30s 24h \
  apptainer exec --nv --bind /data:/data --pwd "${repo}" "${runtime_sif}" \
  python ct_train.py \
    --data="${dataset}" --outdir="${source_dir}" --nosubdir \
    --cond=False --arch=ddpmpp --precond=ect --batch=128 --batch-gpu=16 \
    --optim=RAdam --lr=0.0001 --dropout=0.2 --augment=0 --xflip=False \
    --mean=-1.1 --std=2.0 --mapping=sigmoid --global-gap-scale=1.0 \
    --factorial-protocol=q256_target_weight_v1 \
    --target-gap-scale=1.1 --denominator-gap-scale=1.1 \
    -q 256 -k 8 -b 1 -c 0 --double=10000 --ema_beta=0.9993 \
    --seed="${seed}" --fp16=True --tf32=False --ls=1.0 --enable_amp=True \
    --bench=False --cache=True --workers=1 --metrics=none --duration=0.384 \
    --tick=10 --snap=0 --dump=0 --ckpt=10 --sample_every=26 --eval_every=50 \
    --mid_t=0.821 --adaptive-update-kimg=0.5 \
    --immutable-checkpoint-kimg=384 --transfer="${transfer}" \
  2>&1 | tee "${source_dir}/process.log"

apptainer exec --nv --bind /data:/data --pwd "${repo}" "${runtime_sif}" python \
  analysis/q256_p2_b384_pulse_chase_v1/build_source_inventory.py \
    --source-state "${source_dir}/training-state-kimg000384.pt" \
    --seed "${seed}" --run-kind "${run_kind}" --protocol "${protocol}" \
    --implementation-commit "${implementation_commit}" --dataset "${dataset}" \
    --transfer "${transfer}" --runtime-sif "${runtime_sif}" \
    --output "${source_dir}/source_inventory.json"

elapsed=$(( $(date +%s) - started ))
python3 -c 'import json,os,sys; d={"schema":"ect.q256.p2-source-completion/v1","status":"PASS","seed":int(sys.argv[2]),"gpu_index":int(sys.argv[3]),"gpu_uuid":sys.argv[4],"elapsed_seconds":int(sys.argv[5]),"source_inventory_sha256":sys.argv[6]}; f=open(sys.argv[1],"x"); json.dump(d,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())' \
  "${source_dir}/source_completion_receipt.json" "${seed}" "${gpu}" "${gpu_uuid}" "${elapsed}" \
  "$(sha256sum "${source_dir}/source_inventory.json" | awk '{print $1}')"
trap - ERR
echo "[P2 source] PASS seed=${seed} gpu=${gpu} uuid=${gpu_uuid} elapsed=${elapsed}"
