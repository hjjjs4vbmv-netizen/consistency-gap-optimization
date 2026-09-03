#!/usr/bin/env bash
set -Eeuo pipefail

node_id="${1:?usage: $0 node6|node8}"
case "${node_id}" in
  node6|node8) ;;
  *) echo "invalid node id: ${node_id}" >&2; exit 2 ;;
esac

source_host=px-cloud2.matpool.com
source_port=28062
key=/root/q256_eval_transfer_ed25519
known_hosts=/root/q256_eval_known_hosts
archive=/root/q256-training-runtime-py311-torch260.tar.gz
archive_sha_file=${archive}.sha256
base=/root/q256-training-runtime-base
env_dir=/root/q256-training-runtime-env
receipt=/root/q256-training-runtime-transfer-receipt.json
freeze=/root/q256-training-runtime-pip-freeze.txt
runner=/root/firstwave_eval_pool.py

for required in "${key}" "${runner}"; do
  [[ -f "${required}" ]] || { echo "missing required file: ${required}" >&2; exit 2; }
done
[[ ! -e "${archive}" && ! -e "${archive_sha_file}" ]] || {
  echo "runtime transfer target already exists" >&2
  exit 3
}
[[ ! -e "${base}" && ! -e "${env_dir}" && ! -e "${receipt}" ]] || {
  echo "runtime installation target already exists" >&2
  exit 3
}

ssh_args=(
  -i "${key}" -p "${source_port}" -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="${known_hosts}"
)
remote=root@${source_host}
remote_archive=/root/q256-training-runtime-py311-torch260.tar.gz
remote_sha=${remote_archive}.sha256

while ! ssh "${ssh_args[@]}" "${remote}" \
  "test -s '${remote_archive}' -a -s '${remote_sha}'"
do
  printf '[%s] waiting for packed training runtime\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep 60
done

rsync -a --partial --append-verify \
  -e "ssh -i ${key} -p ${source_port} -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=${known_hosts}" \
  "${remote}:${remote_archive}" "${archive}"
rsync -a \
  -e "ssh -i ${key} -p ${source_port} -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=${known_hosts}" \
  "${remote}:${remote_sha}" "${archive_sha_file}"

expected="$(awk '{print $1}' "${archive_sha_file}")"
actual="$(sha256sum "${archive}" | awk '{print $1}')"
[[ "${actual}" == "${expected}" ]] || { echo "runtime archive SHA mismatch" >&2; exit 4; }

mkdir "${base}"
tar -xzf "${archive}" -C "${base}"
"${base}/bin/conda-unpack"
"${base}/bin/python" -m venv --system-site-packages "${env_dir}"
"${env_dir}/bin/python" -m pip install --disable-pip-version-check \
  --index-url https://mirrors.aliyun.com/pypi/simple/ --timeout 60 --retries 2 \
  click==8.2.1 imageio==2.37.0 imageio-ffmpeg==0.6.0 pillow==11.3.0 \
  psutil==7.0.0 requests==2.32.5 scipy==1.16.1 tqdm==4.67.1
"${env_dir}/bin/python" -m pip freeze --all >"${freeze}"

"${env_dir}/bin/python" - "${receipt}" "${archive}" "${actual}" \
  "${freeze}" "${node_id}" <<'PY'
import hashlib
import json
import os
import platform
import sys

import numpy
import scipy
import torch

receipt_path, archive_path, archive_sha, freeze_path, node_id = sys.argv[1:]
probe = {
    "python": platform.python_version(),
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
}
expected = {
    "python": "3.11.13",
    "numpy": "2.1.2",
    "scipy": "1.16.1",
    "torch": "2.6.0+cu124",
    "torch_cuda": "12.4",
}
if probe != expected or not torch.cuda.is_available():
    raise SystemExit(f"runtime probe mismatch: {probe}")
with open(freeze_path, "rb") as handle:
    freeze_sha = hashlib.sha256(handle.read()).hexdigest()
payload = {
    "schema": "ect.q256.training-compatible-evaluation-runtime/v1",
    "status": "PASS",
    "node_id": node_id,
    "archive_path": archive_path,
    "archive_sha256": archive_sha,
    "source_environment": "/root/miniconda3/envs/myconda on training node port 28062",
    "runtime_probe": probe,
    "pip_freeze_path": freeze_path,
    "pip_freeze_sha256": freeze_sha,
    "gpu_count": torch.cuda.device_count(),
    "scientific_protocol_modified": False,
}
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
fd = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
with os.fdopen(fd, "wb") as handle:
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps(payload, sort_keys=True))
PY

exec python "${runner}" --node-id "${node_id}"
