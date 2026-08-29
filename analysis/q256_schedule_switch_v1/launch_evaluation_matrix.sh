#!/usr/bin/env bash
set -Eeuo pipefail

manifest="${1:?usage: $0 MANIFEST EVALUATOR_REPO RUNTIME_ROOTFS RUNTIME_SIF CACHE_ROOT EVAL_ROOT IMPLEMENTATION_REPO}"
evaluator_repo="${2:?missing evaluator repository}"
runtime_rootfs="${3:?missing runtime rootfs}"
runtime_sif="${4:?missing runtime SIF}"
cache_root="${5:?missing evaluator cache}"
eval_root="${6:?missing evaluation output root}"
implementation_repo="${7:?missing implementation repository}"

python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["metrics_executed"] is False and p["job_count"]==80 and len(p["jobs"])==80; assert len({(j["seed"],j["branch"],j["budget_kimg"],j["nfe"]) for j in p["jobs"]})==80' "${manifest}" || { echo "invalid frozen 80-job manifest" >&2; exit 2; }
[[ "$(git -C "${evaluator_repo}" rev-parse HEAD)" == d6aba02fb88e9db0993623895eb2228ed717d810 ]] || { echo "evaluator commit mismatch" >&2; exit 2; }
[[ -z "$(git -C "${evaluator_repo}" status --porcelain)" ]] || { echo "evaluator source is dirty" >&2; exit 2; }
mkdir "${eval_root}"
mkdir "${eval_root}/logs"
sif_sha=$(sha256sum "${runtime_sif}" | awk '{print $1}')
[[ "${sif_sha}" == 9d5f2c9e68f1f7dcaa20457bf6e0b6fa46f74a8605edaf5d49fdccf9f6bb62ea ]] || { echo "runtime SIF hash mismatch" >&2; exit 2; }
python -c 'import json,os,sys; p=sys.argv[1]; x={"schema":"ect.q256.runtime-integrity/v1","status":"PASS","runtime_sif":sys.argv[2],"runtime_sif_sha256":sys.argv[3]}; f=open(p,"x"); json.dump(x,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())' "${eval_root}/runtime_integrity.json" "${runtime_sif}" "${sif_sha}"
mapfile -t seeds < <(python -c 'import json,sys; p=json.load(open(sys.argv[1])); print(*sorted({j["seed"] for j in p["jobs"]}),sep="\n")' "${manifest}")
[[ "${#seeds[@]}" == 5 ]] || { echo "evaluation manifest must contain five seeds" >&2; exit 2; }
gpu_count="${Q256_SWITCH_GPU_COUNT:-5}"
[[ "${gpu_count}" =~ ^[1-5]$ ]] || { echo "invalid Q256_SWITCH_GPU_COUNT" >&2; exit 2; }

pids=()
for ((gpu=0; gpu<gpu_count; gpu++)); do
  (
    seed_index=0
    for seed in "${seeds[@]}"; do
      if (( seed_index % gpu_count == gpu )); then
        mapfile -t indices < <(python -c 'import json,sys; p=json.load(open(sys.argv[1])); seed=int(sys.argv[2]); [print(j["job_index"]) for j in p["jobs"] if j["seed"]==seed]' "${manifest}" "${seed}")
        [[ "${#indices[@]}" == 16 ]] || { echo "seed ${seed} does not have 16 jobs" >&2; exit 3; }
        for index in "${indices[@]}"; do
          "${implementation_repo}/analysis/q256_schedule_switch_v1/run_evaluation_job.sh" \
            "${manifest}" "${index}" "${gpu}" "${evaluator_repo}" \
            "${runtime_rootfs}" "${runtime_sif}" "${cache_root}" "${eval_root}" \
            "${implementation_repo}"
        done
      fi
      seed_index=$((seed_index + 1))
    done
  ) >"${eval_root}/logs/gpu${gpu}.queue.log" 2>&1 &
  pids+=("$!")
  printf '%s\n' "$!" >"${eval_root}/logs/gpu${gpu}.pid"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" != 0 ]]; then
  echo "EVALUATION_MATRIX_FAIL_CLOSED root=${eval_root}" >&2
  exit 4
fi
count=$(find "${eval_root}/receipts" -maxdepth 1 -type f -name '*.json' | wc -l | tr -d ' ')
[[ "${count}" == 80 ]] || { echo "evaluation receipt count ${count}/80" >&2; exit 5; }
echo "EVALUATION_MATRIX_PASS jobs=80 root=${eval_root}"
