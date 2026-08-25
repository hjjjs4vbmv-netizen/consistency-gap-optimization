#!/usr/bin/env python3
"""Minimal final-state audit for the 1024 kimg q256 continuation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch


ARMS = {
    "A": (1.0, 1.0),
    "B": (1.1, 1.1),
    "C": (1.1, 1.0),
    "D": (1.0, 1.1),
}
REQUIRED = (
    "training-state-latest.pt",
    "network-snapshot-latest.pkl",
    "training_options.json",
    "train_summary.csv",
    "factorial_training_telemetry_v1.csv",
    "log.txt",
    "final.png",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for arm, (target_scale, denominator_scale) in ARMS.items():
        run_dir = args.run_root / f"seed{args.seed}" / f"arm{arm}"
        for name in REQUIRED:
            path = run_dir / name
            assert path.is_file() and path.stat().st_size > 0, f"missing {path}"

        state_path = run_dir / "training-state-latest.pt"
        state = torch.load(state_path, map_location="cpu")
        attempted = int(state["attempted_iteration"])
        accepted = int(state["successful_optimizer_steps"])
        processed_nimg = int(state["cur_nimg"])
        assert attempted == 8000
        assert processed_nimg == 1_024_000
        assert 0 < accepted <= attempted
        assert isinstance(state["optimizer_state"], dict) and state["optimizer_state"]
        assert state.get("ema") is not None
        assert isinstance(state["gradscaler_state"], dict) and state["gradscaler_state"]
        assert isinstance(state["rank_states"], list) and state["rank_states"]
        factorial = state["factorial"]
        assert factorial["arm"] == arm
        assert float(factorial["target_gap_scale"]) == target_scale
        assert float(factorial["denominator_gap_scale"]) == denominator_scale

        with (run_dir / "factorial_training_telemetry_v1.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            last = None
            for last in csv.DictReader(handle):
                pass
        assert last is not None
        assert int(last["attempted_iteration"]) == attempted
        assert int(last["successful_optimizer_steps"]) == accepted
        assert int(last["processed_nimg"]) == processed_nimg
        assert float(last["processed_kimg"]) == 1024.0
        assert last["arm"] == arm

        log_text = (run_dir / "log.txt").read_text(encoding="utf-8", errors="replace")
        assert "tick 102   kimg 1024.0" in log_text
        assert log_text.rstrip().endswith("Exiting...")

        rows.append(
            {
                "seed": args.seed,
                "arm": arm,
                "source_kimg": 256,
                "final_kimg": 1024.0,
                "final_attempts": attempted,
                "accepted_updates": accepted,
                "amp_skips": attempted - accepted,
                "elapsed_seconds_cumulative": float(last["elapsed_sec"]),
                "final_state_path": str(state_path),
                "final_state_bytes": state_path.stat().st_size,
                "final_state_sha256": sha256_file(state_path),
                "network_snapshot_sha256": sha256_file(
                    run_dir / "network-snapshot-latest.pkl"
                ),
                "artifacts": {name: (run_dir / name).stat().st_size for name in REQUIRED},
                "status": "PASS",
            }
        )
        del state

    payload = {
        "schema": "ect.q256.target-weight-1024k-final-audit/v1",
        "seed": args.seed,
        "arm_count": len(rows),
        "all_pass": all(row["status"] == "PASS" for row in rows),
        "arms": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
