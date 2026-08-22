#!/usr/bin/env python3
"""Verify one completed q256 learning-curve replay trajectory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any

import torch

from training import reproducibility


MILESTONES = (384, 512, 640, 768, 896, 1024)
ATTEMPTS = {kimg: kimg * 1000 // 128 for kimg in MILESTONES}
FACTORS = {
    "A": (1.0, 1.0),
    "B": (1.1, 1.1),
    "C": (1.1, 1.0),
    "D": (1.0, 1.1),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def module_is_finite(module: torch.nn.Module) -> bool:
    for value in module.state_dict().values():
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
            if not bool(torch.isfinite(value).all()):
                return False
    return True


def internal_hashes(state: dict[str, Any]) -> dict[str, Any]:
    rank_states = state["rank_states"]
    return {
        "net": reproducibility.module_state_sha256(state["net"]),
        "ema": reproducibility.module_state_sha256(state["ema"]),
        "optimizer": reproducibility.state_sha256(state["optimizer_state"]),
        "gradscaler": reproducibility.state_sha256(state["gradscaler_state"]),
        "rank_rng": [
            reproducibility.state_sha256(item["rng_state"])
            for item in rank_states
        ],
        "rank_sampler": [
            reproducibility.state_sha256(item["sampler_state"])
            for item in rank_states
        ],
    }


def validate_state(
    state: dict[str, Any], *, seed: int, arm: str, kimg: int
) -> dict[str, Any]:
    expected_attempt = ATTEMPTS[kimg]
    required = (
        "net",
        "ema",
        "optimizer_state",
        "gradscaler_state",
        "attempted_iteration",
        "successful_optimizer_steps",
        "cur_nimg",
        "rank_states",
        "factorial",
        "trajectory_config",
        "trajectory_config_sha256",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise RuntimeError(f"milestone state missing {missing}")
    if int(state["attempted_iteration"]) != expected_attempt:
        raise RuntimeError(f"attempt mismatch at {kimg} kimg")
    if int(state["cur_nimg"]) != kimg * 1000:
        raise RuntimeError(f"processed image mismatch at {kimg} kimg")
    factorial = state["factorial"]
    target, denominator = FACTORS[arm]
    if (
        factorial.get("arm") != arm
        or float(factorial.get("target_gap_scale")) != target
        or float(factorial.get("denominator_gap_scale")) != denominator
    ):
        raise RuntimeError(f"factorial identity mismatch at {kimg} kimg")
    trajectory = state["trajectory_config"]
    if (
        int(trajectory.get("seed", -1)) != seed
        or int(trajectory.get("total_kimg", -1)) != 1024
        or int(trajectory.get("batch_size", -1)) != 128
    ):
        raise RuntimeError(f"trajectory config mismatch at {kimg} kimg")
    if reproducibility.state_sha256(trajectory) != state["trajectory_config_sha256"]:
        raise RuntimeError(f"trajectory config hash mismatch at {kimg} kimg")
    rank_states = state["rank_states"]
    if len(rank_states) != 1:
        raise RuntimeError(f"rank-state count mismatch at {kimg} kimg")
    if int(rank_states[0]["sampler_state"].get("consumed_samples", -1)) != kimg * 1000:
        raise RuntimeError(f"sampler cursor mismatch at {kimg} kimg")
    if not module_is_finite(state["net"]) or not module_is_finite(state["ema"]):
        raise RuntimeError(f"non-finite model/EMA at {kimg} kimg")
    return internal_hashes(state)


def validate_telemetry(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "train_summary.csv"
    telemetry_path = run_dir / "factorial_training_telemetry_v1.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    with telemetry_path.open(newline="", encoding="utf-8") as handle:
        telemetry = list(csv.DictReader(handle))
    if len(summary) != 8000 or len(telemetry) != 8000:
        raise RuntimeError(
            f"expected 8000 summary/telemetry rows, got {len(summary)}/{len(telemetry)}"
        )
    if [int(row["attempted_iteration"]) for row in summary] != list(range(1, 8001)):
        raise RuntimeError("summary attempt sequence is not exact")
    final = summary[-1]
    if (
        int(final["attempted_iteration"]) != 8000
        or int(final["processed_nimg"]) != 1024000
        or not math.isclose(float(final["processed_kimg"]), 1024.0)
        or not math.isfinite(float(final["loss"]))
        or int(final["step_skipped"]) != 0
    ):
        raise RuntimeError("invalid final summary row")
    count_fields = [
        name
        for name in telemetry[0]
        if (
            "nonfinite" in name
            or "nonpositive" in name
            or "mismatch" in name
        )
        and name != "raw_grad_nonfinite_count"
    ]
    nonzero_counts = {
        name: sum(int(float(row[name] or 0)) for row in telemetry)
        for name in count_fields
    }
    if any(nonzero_counts.values()):
        raise RuntimeError(f"nonzero strict telemetry anomaly counts: {nonzero_counts}")
    skips = [int(row["attempted_iteration"]) for row in summary if int(row["step_skipped"])]
    skip_by_attempt = {
        int(row["attempted_iteration"]): bool(int(row["step_skipped"]))
        for row in summary
    }
    raw_grad_mismatches = [
        int(row["attempted_iteration"])
        for row in telemetry
        if bool(int(float(row.get("raw_grad_nonfinite_count") or 0)))
        != skip_by_attempt[int(row["attempted_iteration"])]
    ]
    if raw_grad_mismatches:
        raise RuntimeError(
            f"raw-gradient/AMP-skip mismatch at attempts {raw_grad_mismatches}"
        )
    return {
        "summary_rows": len(summary),
        "telemetry_rows": len(telemetry),
        "final": {
            "attempted_iteration": 8000,
            "successful_optimizer_steps": int(final["successful_optimizer_steps"]),
            "processed_kimg": 1024.0,
            "loss": float(final["loss"]),
            "step_skipped": 0,
        },
        "step_skipped_attempts": skips,
        "raw_gradient_skip_mismatch_attempts": raw_grad_mismatches,
        "strict_anomaly_totals": nonzero_counts,
        "summary_sha256": sha256_file(summary_path),
        "telemetry_sha256": sha256_file(telemetry_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--original-1024-state", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=range(14, 19))
    parser.add_argument("--arm", required=True, choices=tuple(FACTORS))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--replay-commit", required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(strict=True)
    source_state = args.source_state.resolve(strict=True)
    original_state_path = args.original_1024_state.resolve(strict=True)
    manifest_path = run_dir / "checkpoint_manifest.json"
    completion_path = run_dir / "trajectory_completion_receipt.json"
    if manifest_path.exists() or completion_path.exists():
        raise RuntimeError("refusing existing replay manifest/completion")

    telemetry = validate_telemetry(run_dir)
    checkpoints = []
    replay_final_hashes = None
    for kimg in MILESTONES:
        milestone_dir = run_dir / f"kimg{kimg:04d}"
        state_path = milestone_dir / "training-state.pt"
        snapshot_path = milestone_dir / "network-snapshot.pkl"
        receipt_path = milestone_dir / "milestone_receipt.json"
        for path in (state_path, snapshot_path, receipt_path):
            if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"missing milestone artifact: {path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("schema") != "ect.q256.learning-curve-milestone/v1"
            or receipt.get("seed") != args.seed
            or receipt.get("arm") != args.arm
            or receipt.get("milestone_kimg") != kimg
        ):
            raise RuntimeError(f"milestone receipt mismatch: {receipt_path}")
        state = torch.load(state_path, map_location="cpu")
        hashes = validate_state(state, seed=args.seed, arm=args.arm, kimg=kimg)
        with snapshot_path.open("rb") as handle:
            snapshot = pickle.load(handle)
        if "ema" not in snapshot or not module_is_finite(snapshot["ema"]):
            raise RuntimeError(f"invalid snapshot EMA at {kimg} kimg")
        snapshot_ema_sha = reproducibility.module_state_sha256(snapshot["ema"])
        if snapshot_ema_sha != hashes["ema"]:
            raise RuntimeError(f"snapshot/state EMA mismatch at {kimg} kimg")
        checkpoints.append(
            {
                "kimg": kimg,
                "attempted_iteration": ATTEMPTS[kimg],
                "directory": str(milestone_dir),
                "training_state": {
                    "path": str(state_path),
                    "bytes": state_path.stat().st_size,
                    "sha256": sha256_file(state_path),
                },
                "evaluation_snapshot": {
                    "path": str(snapshot_path),
                    "bytes": snapshot_path.stat().st_size,
                    "sha256": sha256_file(snapshot_path),
                    "ema_state_sha256": snapshot_ema_sha,
                },
                "milestone_receipt_sha256": sha256_file(receipt_path),
                "internal_state_sha256": hashes,
            }
        )
        if kimg == 1024:
            replay_final_hashes = hashes
        del state, snapshot

    original_state = torch.load(original_state_path, map_location="cpu")
    original_hashes = validate_state(
        original_state, seed=args.seed, arm=args.arm, kimg=1024
    )
    comparison_fields = ("net", "ema", "optimizer", "gradscaler", "rank_rng", "rank_sampler")
    field_match = {
        name: replay_final_hashes[name] == original_hashes[name]
        for name in comparison_fields
    }
    comparison = {
        "original_1024_state": str(original_state_path),
        "original_1024_state_sha256": sha256_file(original_state_path),
        "field_match": field_match,
        "computational_state_match": all(field_match.values()),
        "verdict": (
            "BITWISE_COMPUTATIONAL_MATCH"
            if all(field_match.values())
            else "DIVERGENCE_RECORDED_REQUIRES_METRIC_CHECK"
        ),
    }
    source_record = {
        "path": str(source_state),
        "bytes": source_state.stat().st_size,
        "sha256": sha256_file(source_state),
        "kimg": 256,
    }
    manifest = {
        "schema": "ect.q256.learning-curve-checkpoint-manifest/v1",
        "status": "PASS",
        "seed": args.seed,
        "arm": args.arm,
        "source_commit": args.source_commit,
        "replay_commit": args.replay_commit,
        "source_256": source_record,
        "milestones": checkpoints,
        "telemetry": telemetry,
        "original_vs_replay_1024": comparison,
    }
    write_json_exclusive(manifest_path, manifest)
    completion = {
        "schema": "ect.q256.learning-curve-trajectory-completion/v1",
        "status": "PASS",
        "seed": args.seed,
        "arm": args.arm,
        "checkpoint_count": len(checkpoints),
        "checkpoint_manifest": str(manifest_path),
        "checkpoint_manifest_sha256": sha256_file(manifest_path),
        "original_vs_replay_1024": comparison,
    }
    write_json_exclusive(completion_path, completion)
    print(
        json.dumps(
            {
                "status": "PASS",
                "seed": args.seed,
                "arm": args.arm,
                "checkpoints": len(checkpoints),
                "comparison": comparison["verdict"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
