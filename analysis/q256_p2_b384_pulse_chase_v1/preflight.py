#!/usr/bin/env python3
"""Fail-closed two-A100, assets, runtime, and repository P2 preflight."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training import pulse_chase, reproducibility


def run(*args: str, env: dict | None = None) -> str:
    return subprocess.check_output(args, text=True, env=env).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--runtime-sif", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    protocol = args.protocol.resolve(strict=True)
    failures = []
    protocol_data = json.loads(protocol.read_text(encoding="utf-8"))
    head = run("git", "-C", str(repo), "rev-parse", "HEAD")
    parent = run("git", "-C", str(repo), "rev-parse", "HEAD^")
    dirty = run("git", "-C", str(repo), "status", "--porcelain")
    if dirty:
        failures.append("repository is dirty")
    if parent != protocol_data.get("implementation_commit"):
        failures.append("protocol implementation is not execution HEAD parent")
    assets = {
        "dataset": pulse_chase.sha256_file(args.dataset.resolve(strict=True)),
        "transfer": pulse_chase.sha256_file(args.transfer.resolve(strict=True)),
        "runtime_sif": pulse_chase.sha256_file(args.runtime_sif.resolve(strict=True)),
    }
    if assets != pulse_chase.ASSET_SHA256:
        failures.append("asset SHA256 mismatch")
    gpu_lines = run(
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.used,driver_version,compute_mode",
        "--format=csv,noheader,nounits",
    ).splitlines()
    if len(gpu_lines) != 2:
        failures.append("host does not expose exactly two GPUs")
    gpus = []
    for line in gpu_lines:
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 7:
            failures.append("unexpected nvidia-smi GPU row")
            continue
        gpus.append({
            "index": int(parts[0]), "name": parts[1], "uuid": parts[2],
            "memory_total_mib": int(parts[3]), "memory_used_mib": int(parts[4]),
            "driver_version": parts[5], "compute_mode": parts[6],
        })
    if len(gpus) == 2:
        identities = {(g["name"], g["memory_total_mib"], g["driver_version"],
                       g["compute_mode"]) for g in gpus}
        if len(identities) != 1 or "A100 80GB" not in gpus[0]["name"]:
            failures.append("GPUs cannot run an identical frozen configuration")
    process_rows = run(
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader",
    ).splitlines()
    process_rows = [row for row in process_rows if row.strip()]
    if process_rows:
        failures.append("one or both GPUs have an active compute process")
    runtime = []
    snippet = (
        "import json,torch; p=torch.cuda.get_device_properties(0); "
        "print(json.dumps({'python':__import__('platform').python_version(),"
        "'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
        "'cudnn':torch.backends.cudnn.version(),'name':p.name,"
        "'memory':p.total_memory,'capability':list(p.major_minor) "
        "if hasattr(p,'major_minor') else [p.major,p.minor]}))"
    )
    for gpu in (0, 1):
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONNOUSERSITE="1")
        try:
            payload = run(
                "apptainer", "exec", "--nv", "--bind", "/data:/data",
                str(args.runtime_sif.resolve(strict=True)), "python", "-c", snippet,
                env=env,
            )
            runtime.append(json.loads(payload.splitlines()[-1]))
        except Exception as exc:
            failures.append(f"runtime probe GPU{gpu} failed: {exc}")
    if len(runtime) == 2 and runtime[0] != runtime[1]:
        failures.append("runtime-visible GPU configurations differ")
    payload = {
        "schema": "ect.q256.p2-preflight/v1",
        "status": "PASS" if not failures else "FAIL_CLOSED",
        "repository": str(repo),
        "execution_commit": head,
        "implementation_commit": protocol_data.get("implementation_commit"),
        "protocol_path": str(protocol),
        "protocol_sha256": pulse_chase.sha256_file(protocol),
        "asset_sha256": assets,
        "gpus": gpus,
        "active_compute_processes": process_rows,
        "runtime_probes": runtime,
        "frozen_runtime_config": {
            "global_batch": 128, "batch_gpu": 16, "fp16": True,
            "amp": True, "tf32": False, "optimizer": "RAdam", "lr": 1e-4,
            "deterministic_algorithms": True, "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8",
        },
        "seed_assignment": {"gpu0": list(range(19, 24)),
                            "gpu1": list(range(24, 29))},
        "failures": failures,
    }
    reproducibility.atomic_json_dump(payload, args.output, overwrite=False)
    print(json.dumps({"status": payload["status"], "gpus": gpus,
                      "asset_sha256": assets}))
    return 0 if not failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
