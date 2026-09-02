#!/usr/bin/env bash
set -Eeuo pipefail

repo="${1:?usage: $0 REPO RUNTIME_ROOT}"
runtime_root="${2:?usage: $0 REPO RUNTIME_ROOT}"
prefix="${runtime_root}/env"
requirements="${repo}/analysis/q256_fresh_crossed_switch_n12_matpool_v1/requirements-rebuilt-runtime.txt"
log="${runtime_root}/build.log"

mkdir -p "${runtime_root}"
[[ -f "${requirements}" ]] || {
  echo "missing rebuilt-runtime requirements" >&2
  exit 2
}
if [[ ! -x "${prefix}/bin/python" ]]; then
  conda create --clone /root/miniconda3/envs/myconda --prefix "${prefix}" >>"${log}" 2>&1
fi
"${prefix}/bin/python" -m pip install --disable-pip-version-check \
  --index-url https://mirrors.aliyun.com/pypi/simple/ --timeout 60 --retries 2 \
  --requirement "${requirements}" >>"${log}" 2>&1
conda list --prefix "${prefix}" --explicit >"${runtime_root}/conda-explicit.lock"
"${prefix}/bin/python" -m pip freeze --all >"${runtime_root}/pip-freeze.txt"
env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  "${prefix}/bin/python" \
  "${repo}/analysis/q256_fresh_crossed_switch_n12_matpool_v1/runtime_probe.py" \
  --output "${runtime_root}/runtime-probe.json"
"${prefix}/bin/conda-pack" --prefix "${prefix}" \
  --output "${runtime_root}/q256-rebuilt-runtime-conda-pack.tar.gz" --force >>"${log}" 2>&1
"${prefix}/bin/python" \
  "${repo}/analysis/q256_fresh_crossed_switch_n12_matpool_v1/freeze_runtime.py" \
  --prefix "${prefix}" \
  --archive "${runtime_root}/q256-rebuilt-runtime-conda-pack.tar.gz" \
  --explicit-lock "${runtime_root}/conda-explicit.lock" \
  --pip-freeze "${runtime_root}/pip-freeze.txt" \
  --requirements "${requirements}" \
  --pip-index-url https://mirrors.aliyun.com/pypi/simple/ \
  --probe "${runtime_root}/runtime-probe.json" \
  --output "${runtime_root}/runtime-manifest.json"
