#!/usr/bin/env python3
"""Evaluate immutable seed14--18 checkpoints as soon as they appear locally."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ARMS = ("A", "B", "C", "D")
BUDGETS = (384, 512, 640, 768, 896, 1024)
NFES = (1, 2)
REPLAY_COMMIT = "f4115a89c764081e01be4290f0868cb8f625825e"
CHECKPOINT_SCHEMA = "ect.q256.seed14-18.checkpoint-transfer/v1"


def load_ready_checkpoint(root: Path, seed: int, arm: str, budget: int):
    directory = root / f"seed{seed}" / f"arm{arm}" / f"{budget}k"
    metadata_path = directory / "metadata.json"
    checkpoint = directory / "network-snapshot.pkl"
    if not metadata_path.is_file() or not checkpoint.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    exact = {
        "schema": CHECKPOINT_SCHEMA,
        "status": "immutable_checkpoint_written",
        "seed": seed,
        "arm": arm,
        "budget_kimg": budget,
        "replay_commit": REPLAY_COMMIT,
        "atomic_directory_publish": True,
    }
    for field, expected in exact.items():
        if metadata.get(field) != expected:
            raise RuntimeError(f"checkpoint metadata mismatch {field}: {directory}")
    digest = metadata.get("snapshot_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"invalid snapshot digest: {directory}")
    transfer = json.loads((directory / "transfer_receipt.json").read_text(encoding="utf-8"))
    if transfer.get("status") != "PASS" or transfer.get("snapshot_sha256") != digest:
        raise RuntimeError(f"transfer receipt mismatch: {directory}")
    return checkpoint, digest


def write_state(path: Path, payload: dict) -> None:
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(14, 15, 16, 17, 18), required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--base-port", type=int, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--root", type=Path, default=Path("/root/q256_eval"))
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    script = args.root / "deploy" / "run_eval_job.sh"
    inbox = args.root / "inbox"
    receipts = args.root / "receipts"
    state_path = args.root / "logs" / f"seed{args.seed}-worker-state.json"
    started = time.time()
    selected_arms = tuple(args.arms)
    expected_jobs = len(selected_arms) * len(BUDGETS) * len(NFES)

    while True:
        completed = []
        launched = False
        for arm_index, arm in enumerate(selected_arms):
            for budget_index, budget in enumerate(BUDGETS):
                ready = load_ready_checkpoint(inbox, args.seed, arm, budget)
                for nfe in NFES:
                    job_id = f"seed{args.seed}-arm{arm}-k{budget}-nfe{nfe}"
                    receipt = receipts / f"{job_id}.json"
                    if receipt.is_file():
                        completed.append(job_id)
                        continue
                    if ready is None:
                        continue
                    checkpoint, digest = ready
                    port = args.base_port + arm_index * 100 + budget_index * 2 + nfe
                    subprocess.run(
                        [
                            str(script), str(args.seed), arm, str(budget), str(nfe),
                            args.gpu, str(checkpoint), digest, str(port),
                        ],
                        check=True,
                    )
                    launched = True
                    break
                if launched:
                    break
            if launched:
                break

        write_state(
            state_path,
            {
                "schema": "ect.q256.seed14-18.streaming-worker-state/v1",
                "status": "PASS" if len(completed) == expected_jobs else "WAITING_OR_RUNNING",
                "seed": args.seed,
                "arms": list(selected_arms),
                "gpu_uuid": args.gpu,
                "completed_job_count": len(completed),
                "completed_job_ids": sorted(completed),
                "elapsed_seconds": round(time.time() - started, 3),
                "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        if len(completed) == expected_jobs:
            print(
                f"[q256-stream-worker] PASS seed={args.seed} jobs={expected_jobs}",
                flush=True,
            )
            return
        if not launched:
            print(
                f"[q256-stream-worker] WAIT seed={args.seed} "
                f"completed={len(completed)}/{expected_jobs}",
                flush=True,
            )
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
