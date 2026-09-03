#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


EVALUATOR_COMMIT = "d6aba02fb88e9db0993623895eb2228ed717d810"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def validate(protocol_path: Path) -> dict:
    protocol_path = protocol_path.resolve(strict=True)
    protocol = json.loads(protocol_path.read_text())
    repo = Path(__file__).resolve().parents[2]
    head = git_value(repo, "rev-parse", "HEAD")
    if git_value(repo, "status", "--porcelain"):
        raise RuntimeError("formal implementation worktree is not clean")
    for record in (protocol["assets"]["dataset"], protocol["assets"]["transfer"],
                   protocol["assets"]["detector"], *protocol["assets"]["real_features"]):
        path = Path(record["path"]).resolve(strict=True)
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"asset SHA256 mismatch: {path}")
    evaluator = Path(protocol["assets"]["evaluator_source"]["path"]).resolve(strict=True)
    if git_value(evaluator, "rev-parse", "HEAD") != EVALUATOR_COMMIT or git_value(evaluator, "status", "--porcelain"):
        raise RuntimeError("evaluator is not the frozen clean commit")
    runtime_path = Path(protocol["paths"]["runtime"]) / "runtime-manifest.json"
    runtime = json.loads(runtime_path.read_text())
    if runtime.get("status") != "PASS" or not (Path(runtime["environment_prefix"]) / "bin" / "python").is_file():
        raise RuntimeError("runtime manifest is not usable")
    query = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader,nounits"
    ], text=True).splitlines()
    if len(query) != 8 or any(not row.startswith(f"{index}, ") or "A100" not in row for index, row in enumerate(query)):
        raise RuntimeError("TASK 2 requires idle A100 GPU indices 0..7")
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"
    ], text=True).strip()
    if apps:
        raise RuntimeError("formal GPUs are not idle")
    training = Path(protocol["paths"]["training"])
    if training.exists():
        raise RuntimeError("formal training destination already exists")
    return {
        "status": "PASS", "protocol_sha256": sha256_file(protocol_path),
        "implementation_commit": head, "evaluator_commit": EVALUATOR_COMMIT,
        "runtime_manifest_sha256": sha256_file(runtime_path), "gpu_count": 8,
    }


def write(protocol_path: Path) -> Path:
    protocol = json.loads(protocol_path.read_text())
    output = Path(protocol["paths"]["evidence"]) / "preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError("preflight receipt already exists")
    temporary = output.with_suffix(".tmp")
    result = validate(protocol_path)
    with temporary.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    path = write(parser.parse_args().protocol.resolve(strict=True))
    print(path)


if __name__ == "__main__":
    main()
