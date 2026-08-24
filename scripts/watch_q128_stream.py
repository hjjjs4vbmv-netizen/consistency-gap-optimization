#!/usr/bin/env python3
"""Export each q128 checkpoint as soon as it lands and emit a relay receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


BUDGETS = (256, 384, 512, 640, 768, 896, 1024)
SEEDS = (3, 4, 5)
ARMS = ("A", "Bsame", "Bmatch", "Cmatch", "Dmatch")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_cells(value: str) -> tuple[tuple[int, str], ...]:
    result = []
    for token in value.split(","):
        seed_text, separator, arm = token.strip().partition(":")
        if separator != ":" or not seed_text.isdigit():
            raise argparse.ArgumentTypeError(f"invalid cell: {token!r}")
        cell = (int(seed_text), arm)
        if cell[0] not in SEEDS or cell[1] not in ARMS:
            raise argparse.ArgumentTypeError(f"unsupported cell: {token!r}")
        result.append(cell)
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("cells must be non-empty and unique")
    return tuple(result)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--cells", type=parse_cells, required=True)
    parser.add_argument("--ready-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise SystemExit("poll-seconds must be positive")

    pending = {
        (seed, arm, budget)
        for seed, arm in args.cells
        for budget in BUDGETS
    }
    while pending:
        progressed = False
        for seed, arm, budget in sorted(pending):
            run_dir = args.run_root / f"seed{seed}" / f"arm{arm}"
            state = run_dir / f"training-state-kimg{budget:06d}.pt"
            receipt = run_dir / f"network-snapshot-kimg{budget:06d}.receipt.json"
            snapshot = run_dir / f"network-snapshot-kimg{budget:06d}.pkl"
            if not state.is_file():
                continue
            if not receipt.is_file() or not snapshot.is_file():
                command = [
                    str(args.runtime_python),
                    str(args.repo / "scripts/export_second_q_ab_snapshots.py"),
                    "--run-root", str(args.run_root),
                    "--cells", f"{seed}:{arm}",
                    "--budgets", str(budget),
                ]
                result = subprocess.run(command, cwd=args.repo, check=False)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"snapshot export failed for seed={seed} arm={arm} budget={budget}"
                    )
            export_receipt = json.loads(receipt.read_text(encoding="utf-8"))
            if export_receipt.get("status") != "PASS":
                raise RuntimeError(f"snapshot receipt is not PASS: {receipt}")
            required_small = [
                run_dir / "training_options.json",
                run_dir / "launch_record.json",
                run_dir / "factorial_training_telemetry_v1.csv",
                run_dir / "train_summary.csv",
                run_dir / "log.txt",
            ]
            if not all(path.is_file() for path in required_small):
                continue
            relay = {
                "schema": "ect.q128-stream-ready/v1",
                "status": "READY",
                "seed": seed,
                "arm": arm,
                "budget_kimg": budget,
                "state_path": str(state),
                "state_sha256": export_receipt["source_state_sha256"],
                "snapshot_path": str(snapshot),
                "snapshot_sha256": export_receipt["snapshot_sha256"],
                "snapshot_receipt_path": str(receipt),
                "small_files": [str(path) for path in required_small],
                "priority": ["snapshot", "snapshot_receipt", "small_files", "full_state"],
            }
            ready = args.ready_root / f"seed{seed}-arm{arm}-kimg{budget:06d}.json"
            atomic_json(ready, relay)
            print(f"STREAM_READY seed={seed} arm={arm} budget={budget}", flush=True)
            pending.remove((seed, arm, budget))
            progressed = True
        if pending and not progressed:
            time.sleep(args.poll_seconds)
    print("STREAM_WATCH_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
