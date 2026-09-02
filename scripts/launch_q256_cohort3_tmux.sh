#!/usr/bin/env bash
set -euo pipefail
umask 077

container=${ECT_COHORT3_CONTAINER:-ect-q256-cohort3}
repo=/workspace/consistency-gap-optimization
run_root=/data/raw/ECT/ect_runs/q256-target-weight-factorial-cohort3-seed8-12

fail() {
    printf '[launch_q256_cohort3_tmux] ERROR: %s\n' "$*" >&2
    exit 1
}

[[ $# -eq 1 && "$1" == "formal" ]] || fail 'usage: launch_q256_cohort3_tmux.sh formal'
command -v docker >/dev/null 2>&1 || fail 'docker is unavailable'
command -v tmux >/dev/null 2>&1 || fail 'tmux is unavailable'
[[ "$(docker inspect -f '{{.State.Running}}' "${container}" 2>/dev/null)" == true ]] || fail 'exact-runtime container is not running'

for session in q256_c3_seed8 q256_c3_seed9 q256_c3_seed10 q256_c3_seed11 q256_c3_seed12 q256_c3_monitor; do
    if tmux has-session -t "${session}" 2>/dev/null; then
        fail "refusing existing tmux session: ${session}"
    fi
done

docker exec -w "${repo}" "${container}" python scripts/run_q256_cohort3.py status --run-root "${run_root}" >/dev/null

for mapping in 8:0 9:1 10:2 11:3 12:4; do
    seed=${mapping%%:*}
    gpu=${mapping##*:}
    session=q256_c3_seed${seed}
    tmux new-session -d -s "${session}" \
        "docker exec -w ${repo} ${container} python scripts/run_q256_cohort3.py queue --run-root ${run_root} --seed ${seed} --gpu-index ${gpu}"
done

tmux new-session -d -s q256_c3_monitor \
    "docker exec -w ${repo} ${container} python scripts/run_q256_cohort3.py monitor --run-root ${run_root} --interval 180"

printf 'launched sessions:'
tmux list-sessions -F '#{session_name} #{pane_pid}' | grep '^q256_c3_'
