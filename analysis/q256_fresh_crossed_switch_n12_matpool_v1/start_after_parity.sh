#!/usr/bin/env bash
set -Eeuo pipefail

base=/root/q256_fresh_crossed_switch_n12_matpool_v1
repo="${base}/formal_repo_v8"
runtime="${base}/runtime"
parity_root="${base}/engineering-parity-v3"
control="${base}/control-v1"
output="${base}/formal-output-v1"
protocol_dir="${repo}/analysis/q256_fresh_crossed_switch_n12_matpool_v1"
python="${runtime}/env/bin/python"
dataset=/mnt/ect_project/q256_seed14_18_eval_assets_20260822/cifar10-32x32-canonical-08c9ed1b2b1c.zip
transfer=/mnt/ect_project/pretrained/edm-cifar10-32x32-uncond-vp.pkl
asset_root=/mnt/ect_project/q256_seed14_18_eval_assets_20260822
cache="${asset_root}/cache"
evaluator_source="${asset_root}/q256-evaluator-source-d6aba02.tar"
detector="${cache}/downloads/a866f8d678872dcf6fcf60ddd09807ab_https___nvlabs-fi-cdn.nvidia.com_stylegan2-ada-pytorch_pretrained_metrics_inception-2015-12-05.pt"
real_features_1="${cache}/gan-metrics/cifar10-32x32-canonical-08c9ed1b2b1c-inception-2015-12-05-0fa1f5d271a9ae22be4cfe13dcd2bb5d.pkl"
real_features_2="${cache}/gan-metrics/cifar10-32x32-canonical-08c9ed1b2b1c-inception-2015-12-05-aad45a63395a8faeaad423fdcda71cb1.pkl"
sample_state=/mnt/ect_project/runs/day1-smoke/training-state-latest.pt
sample_snapshot=/mnt/ect_project/runs/day1-smoke/network-snapshot-000013.pkl
evaluator_repo="${base}/evaluator"
implementation_commit=c12c278b60808e1120c035bb68e7c866c3208df7
gate_log="${base}/gate-tests-621deb9.log"
expected_gatekeeper_commit="${Q256_GATEKEEPER_COMMIT:?set Q256_GATEKEEPER_COMMIT to the frozen launcher commit}"

while tmux has-session -t q256_engineering_parity_v3 2>/dev/null; do
  printf '%s parity still running\n' "$(date -Is)"
  sleep 60
done

test -s "${parity_root}/parity_report.json"
grep -q '^Ran 147 tests' "${gate_log}"
grep -q '^OK (skipped=2)' "${gate_log}"
test "$(git -C "${repo}" rev-parse HEAD)" = "${expected_gatekeeper_commit}"
test -z "$(git -C "${repo}" status --porcelain)"
test ! -e "${output}"
test ! -e "${protocol_dir}/protocol.json"
mkdir -p "${control}"
sha256sum "${gate_log}" > "${control}/gate-tests-621deb9.sha256"

while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; do
  printf '%s waiting for six-GPU exclusivity\n' "$(date -Is)"
  sleep 60
done

cd "${repo}"
env CUDA_VISIBLE_DEVICES='' PYTHONPATH=. "${python}" \
  analysis/q256_fresh_crossed_switch_n12_matpool_v1/experiment.py freeze-protocol \
  --implementation-commit "${implementation_commit}" \
  --runtime-manifest "${runtime}/runtime-manifest.json" \
  --runtime-parity "${parity_root}/parity_report.json" \
  --dataset "${dataset}" --transfer "${transfer}" \
  --evaluator-source "${evaluator_source}" --detector "${detector}" \
  --real-features "${real_features_1}" "${real_features_2}" \
  --storage-sample-state "${sample_state}" \
  --storage-sample-snapshot "${sample_snapshot}" \
  --repo "${repo}" --asset-root "${asset_root}" --evaluator-cache "${cache}" \
  --output-root "${output}" --control-root "${control}" \
  --destination "${protocol_dir}"

git add "${protocol_dir}/protocol.json" "${protocol_dir}/protocol.sha256"
env GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
  GIT_COMMITTER_NAME=Codex GIT_COMMITTER_EMAIL=codex@openai.com \
  git commit -m 'Freeze q256 fresh crossed-switch protocol'
git rev-parse HEAD > "${control}/protocol-commit.txt"
test -z "$(git status --porcelain)"

env CUDA_VISIBLE_DEVICES='' PYTHONPATH=. "${python}" \
  analysis/q256_fresh_crossed_switch_n12_matpool_v1/experiment.py preflight \
  --protocol "${protocol_dir}/protocol.json" \
  --receipt "${control}/preflight.json" --minimum-free-gib 500

exec bash analysis/q256_fresh_crossed_switch_n12_matpool_v1/formal_pipeline.sh \
  "${protocol_dir}/protocol.json" "${control}/preflight.json" \
  "${evaluator_repo}" "${cache}"
