#!/usr/bin/env python3
"""Emit an immutable probe for the rebuilt training runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import click
import numpy
import PIL
import psutil
import scipy
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device_count = torch.cuda.device_count()
    payload = {
        "schema": "ect.q256.rebuilt-runtime-probe/v1",
        "status": "PASS" if device_count == 6 else "FAIL",
        "python": platform.python_version(),
        "python_major_minor": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "numpy": numpy.__version__,
        "scipy": scipy.__version__, "pillow": PIL.__version__,
        "click": click.__version__, "psutil": psutil.__version__,
        "cuda_available": torch.cuda.is_available(), "cuda_device_count": device_count,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(device_count)],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    print(hashlib.sha256(encoded).hexdigest())
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
