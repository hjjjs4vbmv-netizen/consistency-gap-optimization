#!/usr/bin/env python3
"""Create the one-time integrity receipt for the extracted frozen runtime."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/root/q256_eval")
SIF = ROOT / "runtime" / "ect-pytorch2401-deterministic.sif"
PYTHON = ROOT / "runtime" / "sandbox" / "usr" / "bin" / "python"
OUTPUT = ROOT / "runtime" / "runtime_integrity.json"
EXPECTED = "9d5f2c9e68f1f7dcaa20457bf6e0b6fa46f74a8605edaf5d49fdccf9f6bb62ea"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refuse existing receipt: {OUTPUT}")
    sif_sha = sha256_file(SIF)
    if sif_sha != EXPECTED:
        raise SystemExit("frozen SIF hash mismatch")
    shell = ROOT / "deploy" / "runtime_env.sh"
    command = (
        f"source {shell}; env CUDA_VISIBLE_DEVICES={os.environ['Q256_RUNTIME_GPU']} "
        'PYTHONNOUSERSITE=1 LD_LIBRARY_PATH="$Q256_RUNTIME_LD_LIBRARY_PATH" '
        'PATH="$Q256_RUNTIME_PATH" "$Q256_RUNTIME_PYTHON" -c '
        "'import json,platform,numpy,scipy,torch; print(json.dumps({"
        '"python":platform.python_version(),"numpy":numpy.__version__,'
        '"scipy":scipy.__version__,"torch":torch.__version__,'
        '"torch_cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),'
        '"cuda_available":torch.cuda.is_available(),'
        '"cuda_device_count":torch.cuda.device_count(),'
        '"cuda_device_name":torch.cuda.get_device_name(0)}))'"'"
    )
    runtime = json.loads(
        subprocess.check_output(["bash", "-lc", command], text=True).strip()
    )
    expected_runtime = {
        "python": "3.10.12",
        "numpy": "1.24.4",
        "scipy": "1.12.0",
        "torch": "2.2.0a0+81ea7a4",
        "torch_cuda": "12.3",
        "cudnn": 8907,
        "cuda_available": True,
        "cuda_device_count": 1,
    }
    if any(runtime.get(key) != value for key, value in expected_runtime.items()):
        raise SystemExit(f"runtime mismatch: {runtime}")
    if "A100" not in runtime["cuda_device_name"]:
        raise SystemExit(f"runtime did not expose A100: {runtime}")
    payload = {
        "schema": "ect.q256.portable-frozen-runtime/v1",
        "status": "PASS",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sif": str(SIF),
        "runtime_sif_bytes": SIF.stat().st_size,
        "runtime_sif_sha256": sif_sha,
        "sandbox": str(ROOT / "runtime" / "sandbox"),
        "extraction": "unsquashfs_offset_45056_from_verified_sif",
        "runtime": runtime,
    }
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.link(temporary, OUTPUT)
    temporary.unlink()
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

