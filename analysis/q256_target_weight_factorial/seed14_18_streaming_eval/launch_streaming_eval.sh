#!/usr/bin/env bash

set -euo pipefail

root=${Q256_EVAL_ROOT:-/root/q256_eval}
deploy=${root}/deploy
control_socket=${Q256_TRAINING_CONTROL_SOCKET:-${root}/training-control.sock}
eval_arms_csv=${Q256_EVAL_ARMS:-A,B,C,D}
IFS=, read -r -a eval_arms <<<"${eval_arms_csv}"
[[ ${#eval_arms[@]} -gt 0 ]] || { echo "no evaluation arms selected" >&2; exit 2; }
arms_argument=$(printf " '%s'" "${eval_arms[@]}")

[[ -x ${root}/runtime/sandbox/usr/bin/python ]] || { echo "missing extracted runtime" >&2; exit 2; }
[[ -f ${root}/runtime/runtime_integrity.json ]] || { echo "missing runtime receipt" >&2; exit 2; }
[[ -f ${root}/source/q256-evaluator-source-d6aba02.tar ]] || { echo "missing evaluator archive" >&2; exit 2; }
[[ -f ${root}/assets/cifar10-32x32-canonical-08c9ed1b2b1c.zip ]] || { echo "missing canonical dataset" >&2; exit 2; }
[[ -f ${root}/assets/cache/shared/downloads/a866f8d678872dcf6fcf60ddd09807ab_https___nvlabs-fi-cdn.nvidia.com_stylegan2-ada-pytorch_pretrained_metrics_inception-2015-12-05.pt ]] || { echo "missing detector cache" >&2; exit 2; }
ssh -S "${control_socket}" -p 27200 root@px-cloud1.matpool.com true

mkdir -p "${root}"/{inbox,logs,receipts,runs}
mapfile -t gpu_uuids < <(nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader,nounits | sort -n | awk -F', ' '$3 ~ /A100/ && $4 == 40960 {print $2}')
[[ ${#gpu_uuids[@]} -eq 5 ]] || { echo "expected five A100 40GB GPUs" >&2; exit 2; }

for session in q256pull q256durable q256eval14 q256eval15 q256eval16 q256eval17 q256eval18; do
  ! tmux has-session -t "${session}" 2>/dev/null || { echo "existing tmux session: ${session}" >&2; exit 3; }
done

tmux new-session -d -s q256pull \
  "${deploy}/pull_checkpoints.py --root '${root}' --control-socket '${control_socket}' --arms${arms_argument} >>'${root}/logs/checkpoint-pull.log' 2>&1"
tmux new-session -d -s q256durable \
  "Q256_EVAL_ARMS='${eval_arms_csv}' ${deploy}/durable_copy_worker.py >>'${root}/logs/durable-copy.log' 2>&1"

for index in 0 1 2 3 4; do
  seed=$((14 + index))
  base_port=$((52000 + index * 500))
  tmux new-session -d -s "q256eval${seed}" \
    "${deploy}/stream_eval_worker.py --seed '${seed}' --gpu '${gpu_uuids[index]}' --base-port '${base_port}' --root '${root}' --arms${arms_argument} >>'${root}/logs/seed${seed}-worker.log' 2>&1"
done

python3 - "${root}" "${eval_arms_csv}" "${gpu_uuids[@]}" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
arms = sys.argv[2].split(",")
uuids = sys.argv[3:]
payload = {
    "schema": "ect.q256.seed14-18.streaming-evaluation-launch/v1",
    "status": "RUNNING",
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "seed_gpu_mapping": {f"seed{seed}": uuids[seed - 14] for seed in range(14, 19)},
    "arms": arms,
    "budgets_kimg": [384, 512, 640, 768, 896, 1024],
    "nfe": [1, 2],
    "metrics": ["kid50k_full", "fid50k_full"],
    "expected_checkpoint_count": 5 * len(arms) * 6,
    "expected_evaluation_job_count": 5 * len(arms) * 6 * 2,
}
path = root / "launch_manifest.json"
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps(payload, sort_keys=True))
PY
