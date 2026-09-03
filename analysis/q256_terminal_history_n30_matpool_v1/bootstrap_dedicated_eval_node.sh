#!/usr/bin/env bash
set -Eeuo pipefail

assets="${1:?usage: $0 ASSET_ROOT}"
archive="${assets}/q256-training-runtime-py311-torch260.tar.gz"
archive_sha_file="${assets}/q256-training-runtime-py311-torch260.tar.gz.sha256"
evaluator_archive="${assets}/q256-evaluator-d6aba02.tar.gz"
dataset="${assets}/cifar10-32x32-eval.zip"
base=/root/q256-training-runtime-base
env_dir=/root/q256-training-runtime-env
evaluator=/root/q256-evaluator-d6aba02
receipt=/root/q256-dedicated-eval-runtime-receipt.json

expected_runtime=c2f2758219964e2b7b79c2f27d01c3edbb2f129bf111d02fab9c1a8b8bc1360a
expected_evaluator=7ef8a1b22af9beab106ad3adbac6474608f27e74c43629a95fcc71738dab0a6f
expected_dataset=08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372
expected_ct_eval=8e17e4cd4e12097e12659a9c8849d42554f24efb25e5255261383d952d878c95

for required in "${archive}" "${archive_sha_file}" "${evaluator_archive}" "${dataset}"; do
  [[ -s "${required}" ]] || { echo "missing asset: ${required}" >&2; exit 2; }
done
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${expected_runtime}" ]]
[[ "$(sha256sum "${evaluator_archive}" | awk '{print $1}')" == "${expected_evaluator}" ]]
[[ "$(sha256sum "${dataset}" | awk '{print $1}')" == "${expected_dataset}" ]]

if [[ ! -d "${evaluator}" ]]; then
  mkdir "${evaluator}"
  tar -xzf "${evaluator_archive}" -C "${evaluator}"
fi
[[ "$(sha256sum "${evaluator}/ct_eval.py" | awk '{print $1}')" == "${expected_ct_eval}" ]]

if [[ ! -d "${base}" ]]; then
  mkdir "${base}"
  tar -xzf "${archive}" -C "${base}"
fi
PATH="${base}/bin:${PATH}" "${base}/bin/conda-unpack"
if [[ ! -d "${env_dir}" ]]; then
  "${base}/bin/python" -m venv --system-site-packages "${env_dir}"
  "${env_dir}/bin/python" -m pip install --disable-pip-version-check \
    --index-url https://mirrors.aliyun.com/pypi/simple/ --timeout 60 --retries 2 \
    click==8.2.1 imageio==2.37.0 imageio-ffmpeg==0.6.0 pillow==11.3.0 \
    psutil==7.0.0 requests==2.32.5 scipy==1.16.1 tqdm==4.67.1
fi

"${env_dir}/bin/python" - "${receipt}" "${assets}" <<'PY'
import json, os, platform, sys
import numpy, scipy, torch

receipt, assets = sys.argv[1:]
probe = {
    "python": platform.python_version(),
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
}
expected = {
    "python": "3.11.13",
    "numpy": "2.1.2",
    "scipy": "1.16.1",
    "torch": "2.6.0+cu124",
    "torch_cuda": "12.4",
    "cuda_available": True,
}
for key, value in expected.items():
    if probe[key] != value:
        raise SystemExit(f"runtime mismatch {key}: {probe[key]!r} != {value!r}")
payload = {
    "schema": "ect.q256.dedicated-evaluation-runtime/v1",
    "status": "PASS",
    "assets": assets,
    "probe": probe,
}
temporary = receipt + ".tmp"
with open(temporary, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush(); os.fsync(handle.fileno())
os.replace(temporary, receipt)
print(json.dumps(payload, sort_keys=True))
PY
