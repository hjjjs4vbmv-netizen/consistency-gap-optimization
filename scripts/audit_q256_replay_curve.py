#!/usr/bin/env python3
"""Strict per-seed audit, inventory, and 1024 canonical parity report."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import reproducibility


ARMS = {
    "A": (1.0, 1.0),
    "B": (1.1, 1.1),
    "C": (1.1, 1.0),
    "D": (1.0, 1.1),
}
BUDGETS = (256, 384, 512, 640, 768, 896, 1024)
REPLAY_BUDGETS = BUDGETS[1:]
EXPECTED_STATE_FIELDS = (
    "net",
    "ema",
    "optimizer_state",
    "loss_fn_state",
    "gradscaler_state",
    "rank_states",
    "cur_nimg",
    "attempted_iteration",
    "successful_optimizer_steps",
    "factorial",
    "trajectory_config",
    "trajectory_config_sha256",
)
ZERO_TELEMETRY_FIELDS = (
    "loss_nonfinite_count",
    "sanitized_grad_nonfinite_count",
    "update_nonfinite_count",
    "model_nonfinite_count",
    "ema_nonfinite_count",
    "factor_nonfinite_count",
    "nonpositive_denominator_count",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_fingerprint(state: dict) -> dict:
    attempted = int(state["attempted_iteration"])
    accepted = int(state["successful_optimizer_steps"])
    return {
        "online_model": reproducibility.module_state_sha256(state["net"]),
        "ema_model": reproducibility.module_state_sha256(state["ema"]),
        "optimizer": reproducibility.state_sha256(
            state["optimizer_state"]
        ),
        "gradscaler": reproducibility.state_sha256(
            state["gradscaler_state"]
        ),
        "loss_control": reproducibility.state_sha256(
            state["loss_fn_state"]
        ),
        "rank_rng_sampler": reproducibility.state_sha256(
            state["rank_states"]
        ),
        "trajectory_config": reproducibility.state_sha256(
            state["trajectory_config"]
        ),
        "factorial": reproducibility.state_sha256(state["factorial"]),
        "cur_nimg": int(state["cur_nimg"]),
        "attempted_iteration": attempted,
        "successful_optimizer_steps": accepted,
        "amp_skips": attempted - accepted,
    }


def validate_state(
    state: dict,
    *,
    seed: int,
    arm: str,
    target_scale: float,
    denominator_scale: float,
    budget: int,
) -> None:
    missing = [key for key in EXPECTED_STATE_FIELDS if key not in state]
    assert not missing, missing
    expected_attempts = budget * 1000 // 128
    assert int(state["cur_nimg"]) == budget * 1000
    assert int(state["attempted_iteration"]) == expected_attempts
    accepted = int(state["successful_optimizer_steps"])
    assert 0 < accepted <= expected_attempts
    assert state["net"] is not None and state["ema"] is not None
    optimizer = state["optimizer_state"]
    assert isinstance(optimizer, dict) and optimizer.get("state")
    assert all("step" in value for value in optimizer["state"].values())
    assert state["gradscaler_state"]
    assert state["loss_fn_state"]
    assert isinstance(state["rank_states"], list)
    assert len(state["rank_states"]) == 1
    rank_state = state["rank_states"][0]
    assert rank_state["rng_state"] and rank_state["sampler_state"]
    assert (
        int(rank_state["sampler_state"]["consumed_samples"])
        == budget * 1000
    )
    factorial = state["factorial"]
    assert factorial["arm"] == arm
    assert float(factorial["target_gap_scale"]) == target_scale
    assert (
        float(factorial["denominator_gap_scale"])
        == denominator_scale
    )
    trajectory = state["trajectory_config"]
    assert int(trajectory["seed"]) == seed
    assert int(trajectory["batch_size"]) == 128
    assert int(trajectory["batch_gpu"]) == 16
    assert trajectory["enable_amp"] is True
    assert trajectory["enable_tf32"] is False
    assert trajectory["cudnn_benchmark"] is False
    assert trajectory["metrics"] == []
    assert (
        reproducibility.state_sha256(trajectory)
        == state["trajectory_config_sha256"]
    )


def audit_telemetry(path: Path, *, arm: str) -> dict:
    attempted = 0
    accepted = 0
    last_nimg = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            attempted = int(row["attempted_iteration"])
            accepted = int(row["successful_optimizer_steps"])
            last_nimg = int(row["processed_nimg"])
            assert row["arm"] == arm
            for field in ZERO_TELEMETRY_FIELDS:
                assert int(row[field]) == 0, (path, field, attempted)
    assert attempted == 8000 and last_nimg == 1_024_000
    return {
        "attempted_iteration": attempted,
        "successful_optimizer_steps": accepted,
        "amp_skips": attempted - accepted,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--old-run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(3, 4, 5), required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    inventory = []
    parity = []
    for arm, (target_scale, denominator_scale) in ARMS.items():
        source_path = (
            args.source_root
            / f"seed{args.seed}"
            / f"arm{arm}"
            / "training-state-latest.pt"
        )
        source_sha256 = sha256_file(source_path)
        source_state = torch.load(
            source_path, map_location="cpu", weights_only=False
        )
        validate_state(
            source_state,
            seed=args.seed,
            arm=arm,
            target_scale=target_scale,
            denominator_scale=denominator_scale,
            budget=256,
        )
        source_elapsed = float(source_state.get("elapsed_sec", 0.0))
        del source_state
        gc.collect()

        old_path = (
            args.old_run_root
            / f"seed{args.seed}"
            / f"arm{arm}"
            / "training-state-latest.pt"
        )
        old_state = torch.load(
            old_path, map_location="cpu", weights_only=False
        )
        validate_state(
            old_state,
            seed=args.seed,
            arm=arm,
            target_scale=target_scale,
            denominator_scale=denominator_scale,
            budget=1024,
        )
        old_fingerprint = state_fingerprint(old_state)
        del old_state
        gc.collect()

        run_dir = args.run_root / f"seed{args.seed}" / f"arm{arm}"
        telemetry = audit_telemetry(
            run_dir / "factorial_training_telemetry_v1.csv", arm=arm
        )
        log_text = (run_dir / "log.txt").read_text(
            encoding="utf-8", errors="replace"
        )
        resume_lines = [
            line.strip()
            for line in log_text.splitlines()
            if "Loading training state from" in line
        ]

        for budget in BUDGETS:
            if budget == 256:
                state_path = source_path
            else:
                state_path = (
                    run_dir / f"training-state-kimg{budget:06d}.pt"
                )
            snapshot_path = (
                run_dir / f"network-snapshot-kimg{budget:06d}.pkl"
            )
            receipt_path = snapshot_path.with_suffix(".receipt.json")
            assert state_path.is_file() and snapshot_path.is_file()
            assert receipt_path.is_file()
            state = torch.load(
                state_path, map_location="cpu", weights_only=False
            )
            validate_state(
                state,
                seed=args.seed,
                arm=arm,
                target_scale=target_scale,
                denominator_scale=denominator_scale,
                budget=budget,
            )
            fingerprint = state_fingerprint(state)
            receipt = json.loads(receipt_path.read_text())
            state_sha256 = sha256_file(state_path)
            snapshot_sha256 = sha256_file(snapshot_path)
            assert receipt["snapshot_sha256"] == snapshot_sha256
            assert (
                receipt["ema_canonical_sha256"]
                == fingerprint["ema_model"]
            )
            inventory.append(
                {
                    "seed": args.seed,
                    "arm": arm,
                    "target_gap_scale": target_scale,
                    "denominator_gap_scale": denominator_scale,
                    "budget_kimg": budget,
                    "cur_nimg": int(state["cur_nimg"]),
                    "attempted_iteration": int(
                        state["attempted_iteration"]
                    ),
                    "successful_optimizer_steps": int(
                        state["successful_optimizer_steps"]
                    ),
                    "amp_skips": (
                        int(state["attempted_iteration"])
                        - int(state["successful_optimizer_steps"])
                    ),
                    "source_256_state_path": str(source_path),
                    "source_256_state_sha256": source_sha256,
                    "replay_state_path": str(state_path),
                    "replay_state_sha256": state_sha256,
                    "ema_snapshot_path": str(snapshot_path),
                    "ema_snapshot_sha256": snapshot_sha256,
                    "online_model_canonical_sha256": fingerprint[
                        "online_model"
                    ],
                    "ema_model_canonical_sha256": fingerprint[
                        "ema_model"
                    ],
                    "optimizer_canonical_sha256": fingerprint[
                        "optimizer"
                    ],
                    "git_commit": args.git_commit,
                    "runtime_identity": args.runtime_identity,
                    "dataset_sha256": args.dataset_sha256,
                    "status": "PASS",
                }
            )
            if budget == 1024:
                differences = {
                    key: {
                        "old": old_fingerprint[key],
                        "replay": fingerprint[key],
                    }
                    for key in old_fingerprint
                    if old_fingerprint[key] != fingerprint[key]
                }
                parity.append(
                    {
                        "seed": args.seed,
                        "arm": arm,
                        "status": (
                            "BITWISE_EQUIVALENT"
                            if not differences
                            else "STATE_DIFFERENT"
                        ),
                        "differences": differences,
                        "old_state_path": str(old_path),
                        "old_state_sha256": sha256_file(old_path),
                        "replay_state_path": str(state_path),
                        "replay_state_sha256": state_sha256,
                    }
                )
            final_elapsed = float(state.get("elapsed_sec", 0.0))
            del state
            gc.collect()

        assert telemetry["attempted_iteration"] == 8000
        assert telemetry["successful_optimizer_steps"] == inventory[-1][
            "successful_optimizer_steps"
        ]
        inventory[-1]["replay_elapsed_seconds"] = (
            final_elapsed - source_elapsed
        )
        inventory[-1]["resume_history"] = json.dumps(resume_lines)

    assert len(inventory) == 28
    assert len(parity) == 4
    fields = sorted({key for row in inventory for key in row})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(inventory)
    payload = {
        "schema": "ect.q256.target-weight-replay-curve-audit/v1",
        "seed": args.seed,
        "trajectory_count": 4,
        "replay_milestone_count": 24,
        "ema_snapshot_count": 28,
        "all_pass": True,
        "inventory": inventory,
        "parity": parity,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    equivalent = sum(
        row["status"] == "BITWISE_EQUIVALENT" for row in parity
    )
    print(
        f"REPLAY_AUDIT_PASS seed={args.seed} milestones=24 "
        f"snapshots=28 parity={equivalent}/4"
    )


if __name__ == "__main__":
    main()
