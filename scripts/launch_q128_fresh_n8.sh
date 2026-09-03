#!/usr/bin/env bash
set -Eeuo pipefail

repo="${1:-/root/consistency-gap-optimization}"
run_root="${2:-/root/q128_fresh_regime_history_n8_v1}"
python_bin="${3:-$(command -v python)}"
analysis="${repo}/analysis/q128_fresh_regime_history_n8_v1"
branch="$(git -C "${repo}" symbolic-ref --quiet --short HEAD)"
[[ "${branch}" == "experiment/q128-fresh-regime-history-n8-v1" ]] || { echo "wrong branch" >&2; exit 2; }
"${python_bin}" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="PASS"' "${analysis}/preflight_report.json"
waiver="${analysis}/owner_directed_operational_waiver_003.json"
"${python_bin}" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="ACTIVE"; assert d["waived_gates"]==["seed999_smoke", "exact_resume_switch_parity"]' "${waiver}"
[[ "$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_count"])' "${analysis}/evaluation_manifest.json")" == 272 ]] || { echo "evaluation manifest missing" >&2; exit 2; }
mkdir -p "${run_root}/formal_logs" "${run_root}/runtime_receipts"
freeze_sha="$(git -C "${repo}" rev-parse HEAD)"
freeze_time="$(git -C "${repo}" show -s --format=%cI HEAD)"
declare -a pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  seed=$((201 + gpu))
  log="${run_root}/formal_logs/seed${seed}.log"
  [[ ! -e "${log}" ]] || { echo "refuse existing formal log ${log}" >&2; exit 3; }
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" "${repo}/scripts/run_q128_fresh_regime_history_n8_v1.py" run-seed \
    --repo "${repo}" --run-root "${run_root}" --runtime-python "${python_bin}" --seed "${seed}" --gpu-id "${gpu}" \
    >"${log}" 2>&1 &
  pids+=("$!")
done
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${python_bin}" -c 'import json,subprocess,sys; out=sys.argv[1]; seeds=range(201,209); pids=list(map(int,sys.argv[2].split(","))); rows=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,name,driver_version","--format=csv,noheader"],text=True).splitlines(); waiver=json.load(open(sys.argv[8])); payload={"schema":"ect.q128-fresh-launch-receipt/v1","status":"LAUNCHED_WITH_OWNER_OPERATIONAL_WAIVER","branch":sys.argv[3],"freeze_commit":sys.argv[4],"freeze_commit_timestamp":sys.argv[5],"formal_launch_utc":sys.argv[6],"freeze_precedes_formal_logs":sys.argv[5]<sys.argv[6],"operational_gate_status":{"preflight":"PASS","seed999_smoke":"WAIVED_NOT_PASSED","exact_resume_switch_parity":"WAIVED_NOT_PASSED"},"owner_directed_waiver":waiver,"jobs":[{"seed":s,"gpu_index":s-201,"pid":pids[s-201],"gpu_identity":rows[s-201],"run_dir":f"{sys.argv[7]}/formal/seed{s}","log":f"{sys.argv[7]}/formal_logs/seed{s}.log"} for s in seeds]}; open(out,"w").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")' \
  "${analysis}/launch_receipt.json" "$(IFS=,; echo "${pids[*]}")" "${branch}" "${freeze_sha}" "${freeze_time}" "${started}" "${run_root}" "${waiver}"
cp "${analysis}/launch_receipt.json" "${run_root}/runtime_receipts/launch_receipt.json"
echo "LAUNCHED pids=${pids[*]}"
