#!/usr/bin/env python3
"""Continuously mirror full q256 n=30 training evidence and checkpoints to storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


PROTOCOL_SHA256 = "317d3ef93102050276c1366d9633e322d60fbc9000cd56c8fc8a24c1d4eef544"
SOURCE_ROOT = Path("/root/q256-terminal-history-n30-v1")
KEY = Path("/root/.ssh/q256_eval_storage_ed25519")
KNOWN_HOSTS = Path("/root/q256_storage_known_hosts")
NODE_CONFIG = {
    "node8": {
        "seeds": tuple(range(50, 66)),
        "firstwave": "eval-node8-firstwave",
        "copy_assets": True,
    },
    "node7": {
        "seeds": tuple(range(66, 80)),
        "firstwave": "eval-node6-firstwave",
        "copy_assets": False,
    },
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def ssh_options(storage_port: int, accept_new: bool = False) -> str:
    strict = "accept-new" if accept_new else "yes"
    return (
        f"ssh -i {KEY} -p {storage_port} -o BatchMode=yes "
        f"-o StrictHostKeyChecking={strict} -o UserKnownHostsFile={KNOWN_HOSTS}"
    )


def ssh(storage_host: str, storage_port: int, command: str, accept_new: bool = False) -> None:
    subprocess.run([
        "ssh", "-i", str(KEY), "-p", str(storage_port), "-o", "BatchMode=yes",
        "-o", f"StrictHostKeyChecking={'accept-new' if accept_new else 'yes'}",
        "-o", f"UserKnownHostsFile={KNOWN_HOSTS}", f"root@{storage_host}", command,
    ], check=True)


def rsync(
    storage_host: str,
    storage_port: int,
    source: str,
    destination: str,
    *,
    includes: list[str] | None = None,
) -> None:
    command = ["rsync", "-aH", "--partial", "--append-verify", "--bwlimit=200000"]
    if includes:
        command += ["--prune-empty-dirs"]
        for pattern in includes:
            command += ["--include", pattern]
        command += ["--exclude", "*"]
    command += ["-e", ssh_options(storage_port), source, f"root@{storage_host}:{destination}"]
    subprocess.run(command, check=True)


def endpoint_terminal(seed: int, cell: str) -> tuple[bool, str, Path | None]:
    seed_dir = SOURCE_ROOT / "training" / f"seed{seed}"
    directory = seed_dir / cell
    compute = directory / "compute_completion_receipt.json"
    trajectory = directory / "trajectory_completion_receipt.json"
    snapshot = directory / "kimg1024" / "network-snapshot.pkl"
    if not compute.is_file():
        prefix_name = "prefix_A" if cell == "AA" else "prefix_B"
        prefix_compute = seed_dir / prefix_name / "compute_completion_receipt.json"
        if prefix_compute.is_file():
            prefix_record = load(prefix_compute)
            if prefix_record.get("status") == "FAIL" or prefix_record.get("exit_code") not in (None, 0):
                return True, "SCIENTIFIC_FAILURE", None
        return False, "WAITING", None
    record = load(compute)
    if record.get("status") == "FAIL" or record.get("exit_code") not in (None, 0):
        return True, "SCIENTIFIC_FAILURE", None
    if record.get("status") == "PASS" and trajectory.is_file() and snapshot.is_file():
        if load(trajectory).get("status") != "PASS":
            raise RuntimeError(f"trajectory receipt not PASS: {trajectory}")
        return True, "PASS", snapshot
    return False, "WAITING_POSTCHECK", None


def sync_metadata(storage_host: str, storage_port: int, storage_root: str, node_id: str, seed: int) -> None:
    source = f"{SOURCE_ROOT}/training/seed{seed}/"
    destination = f"{storage_root}/training_metadata/{node_id}/seed{seed}/"
    rsync(storage_host, storage_port, source, destination, includes=["*/", "*.json", "*.csv", "*.jsonl", "log.txt"])


def write_status(node_id: str, states: list[dict]) -> Path:
    path = SOURCE_ROOT / "control" / f"storage_sync_{node_id}.json"
    payload = {
        "schema": "ect.q256.terminal-history-storage-sync/v1",
        "status": "COMPLETE" if all(item["terminal"] for item in states) else "RUNNING",
        "node_id": node_id,
        "protocol_sha256": PROTOCOL_SHA256,
        "terminal_endpoints": sum(item["terminal"] for item in states),
        "successful_checkpoints": sum(item["state"] == "PASS" for item in states),
        "scientific_failures": [
            {"seed": item["seed"], "cell": item["cell"]}
            for item in states if item["state"] == "SCIENTIFIC_FAILURE"
        ],
        "states": states,
        "updated_at": utc_now(),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return path


def initial_sync(storage_host: str, storage_port: int, storage_root: str, node_id: str) -> None:
    config = NODE_CONFIG[node_id]
    ssh(
        storage_host,
        storage_port,
        f"mkdir -p {storage_root}/training {storage_root}/training_metadata/{node_id} "
        f"{storage_root}/evaluation_firstwave/{node_id} {storage_root}/assets "
        f"{storage_root}/control",
        accept_new=True,
    )
    firstwave = SOURCE_ROOT / "evaluation_firstwave" / config["firstwave"]
    if firstwave.is_dir():
        rsync(storage_host, storage_port, f"{firstwave}/", f"{storage_root}/evaluation_firstwave/{node_id}/")
    rsync(storage_host, storage_port, f"{SOURCE_ROOT}/control/", f"{storage_root}/training_metadata/{node_id}/control/", includes=["*.json", "*.sha256", "*.log"])
    if config["copy_assets"]:
        assets = [
            (Path("/mnt/ect_project/datasets/cifar10-32x32.zip"), "cifar10-32x32-training-original.zip"),
            (Path("/mnt/ect_project/pretrained/edm-cifar10-32x32-uncond-vp.pkl"), "edm-cifar10-32x32-uncond-vp.pkl"),
            (Path("/root/q256-eval-assets/cifar10-32x32-eval.zip"), "cifar10-32x32-eval.zip"),
            (Path("/root/q256-eval-assets/q256-evaluator-d6aba02.tar.gz"), "q256-evaluator-d6aba02.tar.gz"),
            (Path("/root/q256-training-runtime-py311-torch260.tar.gz"), "q256-training-runtime-py311-torch260.tar.gz"),
            (Path("/root/q256-training-runtime-py311-torch260.tar.gz.sha256"), "q256-training-runtime-py311-torch260.tar.gz.sha256"),
        ]
        for source, name in assets:
            if not source.is_file():
                raise RuntimeError(f"missing asset: {source}")
            rsync(storage_host, storage_port, str(source), f"{storage_root}/assets/{name}")
        rsync(storage_host, storage_port, "/root/q256-eval-assets/cache-template/", f"{storage_root}/assets/cache-template/")


def run(node_id: str, storage_host: str, storage_port: int, storage_root: str) -> None:
    if sha256_file(SOURCE_ROOT / "control" / "protocol.json") != PROTOCOL_SHA256:
        raise RuntimeError("protocol SHA mismatch")
    if not KEY.is_file():
        raise RuntimeError("storage transfer key missing")
    initial_sync(storage_host, storage_port, storage_root, node_id)
    while True:
        # Full tree copy preserves every prefix and 640/768/896/1024 checkpoint,
        # including optimizer/RNG training states. Seeds are disjoint by node.
        rsync(storage_host, storage_port, f"{SOURCE_ROOT}/training/", f"{storage_root}/training/")
        states = []
        for seed in NODE_CONFIG[node_id]["seeds"]:
            sync_metadata(storage_host, storage_port, storage_root, node_id, seed)
            for cell in ("AA", "BA"):
                terminal, state, snapshot = endpoint_terminal(seed, cell)
                item = {"seed": seed, "cell": cell, "terminal": terminal, "state": state}
                if state == "PASS":
                    assert snapshot is not None
                    digest = sha256_file(snapshot)
                    item.update(checkpoint_sha256=digest, checkpoint_bytes=snapshot.stat().st_size)
                states.append(item)
        status_path = write_status(node_id, states)
        rsync(storage_host, storage_port, str(status_path), f"{storage_root}/control/{status_path.name}")
        if all(item["terminal"] for item in states):
            return
        time.sleep(60)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-id", choices=tuple(NODE_CONFIG), required=True)
    parser.add_argument("--storage-host", required=True)
    parser.add_argument("--storage-port", required=True, type=int)
    parser.add_argument("--storage-root", default="/root/q256-n30-central-store-v1")
    args = parser.parse_args()
    run(args.node_id, args.storage_host, args.storage_port, args.storage_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
