#!/usr/bin/env python3
"""Pull immutable seed14--18 checkpoints through an authenticated control socket."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


SEEDS = range(14, 19)
ARMS = ("A", "B", "C", "D")
BUDGETS = (384, 512, 640, 768, 896, 1024)
REPLAY_COMMIT = "f4115a89c764081e01be4290f0868cb8f625825e"
REMOTE_ROOT = (
    "/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260821/"
    "secondary-precision-extension/"
    "seed14-18-256to1024-learning-curve-replay-recovery-v2"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, capture: bool = False, check: bool = True):
    return subprocess.run(command, check=check, text=True, capture_output=capture)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/q256_eval"))
    parser.add_argument("--remote-host", default="px-cloud1.matpool.com")
    parser.add_argument("--remote-port", type=int, default=27200)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument(
        "--control-socket",
        type=Path,
        default=Path("/root/q256_eval/training-control.sock"),
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    inbox = args.root / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    target = f"root@{args.remote_host}"
    ssh = [
        "ssh", "-S", str(args.control_socket), "-p", str(args.remote_port), target
    ]
    rsh = " ".join(
        ["ssh", "-S", str(args.control_socket), "-p", str(args.remote_port)]
    )
    if run([*ssh, "true"], check=False).returncode != 0:
        raise RuntimeError("training-node SSH control socket is unavailable")
    selected_arms = tuple(args.arms)
    expected_checkpoints = len(SEEDS) * len(selected_arms) * len(BUDGETS)

    while True:
        completed = 0
        for seed in SEEDS:
            for arm in selected_arms:
                for budget in BUDGETS:
                    final = inbox / f"seed{seed}" / f"arm{arm}" / f"{budget}k"
                    if (final / "transfer_receipt.json").is_file():
                        completed += 1
                        continue
                    remote = f"{REMOTE_ROOT}/seed{seed}/arm{arm}/kimg{budget:04d}"
                    milestone = f"{remote}/milestone_receipt.json"
                    snapshot = f"{remote}/network-snapshot.pkl"
                    if run(
                        [*ssh, f"test -f {milestone} -a -f {snapshot}"], check=False
                    ).returncode != 0:
                        continue
                    source_metadata = json.loads(
                        run([*ssh, f"cat {milestone}"], capture=True).stdout
                    )
                    expected = {
                        "schema": "ect.q256.learning-curve-milestone/v1",
                        "seed": seed,
                        "arm": arm,
                        "milestone_kimg": budget,
                        "processed_nimg": budget * 1000,
                        "network_snapshot": "network-snapshot.pkl",
                    }
                    if any(source_metadata.get(key) != value for key, value in expected.items()):
                        raise RuntimeError(f"source milestone mismatch: {remote}")
                    remote_sha = run(
                        [*ssh, f"sha256sum {snapshot}"], capture=True
                    ).stdout.split()[0]

                    temporary = final.parent / f".{budget}k.tmp-{os.getpid()}"
                    if temporary.exists():
                        shutil.rmtree(temporary)
                    temporary.mkdir(parents=True)
                    run(
                        [
                            "rsync", "-a", "--partial", "-e", rsh,
                            f"{target}:{milestone}",
                            f"{target}:{snapshot}",
                            str(temporary) + "/",
                        ]
                    )
                    snapshot_sha = sha256_file(temporary / "network-snapshot.pkl")
                    if snapshot_sha != remote_sha:
                        raise RuntimeError(f"checkpoint transfer hash mismatch: {remote}")
                    metadata = {
                        "schema": "ect.q256.seed14-18.checkpoint-transfer/v1",
                        "status": "immutable_checkpoint_written",
                        "seed": seed,
                        "arm": arm,
                        "budget_kimg": budget,
                        "replay_commit": REPLAY_COMMIT,
                        "atomic_directory_publish": True,
                        "source_path": remote,
                        "source_milestone_receipt_sha256": sha256_file(
                            temporary / "milestone_receipt.json"
                        ),
                        "snapshot_sha256": snapshot_sha,
                    }
                    write_json(temporary / "metadata.json", metadata)
                    transfer = {
                        "schema": "ect.q256.seed14-18.control-socket-pull/v1",
                        "status": "PASS",
                        "source_host": args.remote_host,
                        "source_path": remote,
                        "seed": seed,
                        "arm": arm,
                        "budget_kimg": budget,
                        "snapshot_sha256": snapshot_sha,
                        "copied_utc": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                    }
                    write_json(temporary / "transfer_receipt.json", transfer)
                    final.parent.mkdir(parents=True, exist_ok=True)
                    os.rename(temporary, final)
                    completed += 1
                    print(
                        f"[q256-pull] PASS seed={seed} arm={arm} "
                        f"budget={budget} sha256={snapshot_sha}",
                        flush=True,
                    )

        state = {
            "schema": "ect.q256.seed14-18.checkpoint-pull-state/v1",
            "status": (
                "PASS" if completed == expected_checkpoints else "WAITING_OR_RUNNING"
            ),
            "arms": list(selected_arms),
            "completed_checkpoint_count": completed,
            "expected_checkpoint_count": expected_checkpoints,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        temporary_state = args.root / f".checkpoint_pull_state.json.tmp-{os.getpid()}"
        write_json(temporary_state, state)
        os.replace(temporary_state, args.root / "checkpoint_pull_state.json")
        if completed == expected_checkpoints:
            print(
                f"[q256-pull] PASS checkpoints={expected_checkpoints}", flush=True
            )
            return
        print(
            f"[q256-pull] WAIT completed={completed}/{expected_checkpoints}",
            flush=True,
        )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
