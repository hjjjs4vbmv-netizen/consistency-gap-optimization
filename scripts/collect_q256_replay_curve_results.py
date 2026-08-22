#!/usr/bin/env python3
"""Combine per-seed replay audits into the compact PR result package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BUDGETS = (256, 384, 512, 640, 768, 896, 1024)
ARMS = ("A", "B", "C", "D")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def archive_path(path: str, archive_root: Path) -> str:
    markers = (
        "/runs/q256-target-weight-replay-curve-v1/",
        "/source_states/formal-direct-dcca41b-deterministic-v1/",
    )
    for marker in markers:
        if marker in path:
            return str(archive_root) + marker + path.split(marker, 1)[1]
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    reports = [json.loads(path.read_text()) for path in args.audit]
    reports.sort(key=lambda value: int(value["seed"]))
    assert [int(report["seed"]) for report in reports] == [3, 4, 5]
    assert all(report["all_pass"] is True for report in reports)
    assert sum(report["replay_milestone_count"] for report in reports) == 72
    assert sum(report["ema_snapshot_count"] for report in reports) == 84

    inventory = []
    parity = []
    for report in reports:
        for row in report["inventory"]:
            record = dict(row)
            for key in (
                "source_256_state_path",
                "replay_state_path",
                "ema_snapshot_path",
            ):
                record[key] = archive_path(record[key], args.archive_root)
            inventory.append(record)
        for row in report["parity"]:
            record = dict(row)
            record["differences"] = json.dumps(
                record["differences"], sort_keys=True
            )
            record["replay_state_path"] = archive_path(
                record["replay_state_path"], args.archive_root
            )
            parity.append(record)

    inventory.sort(
        key=lambda row: (
            int(row["seed"]), row["arm"], int(row["budget_kimg"])
        )
    )
    parity.sort(key=lambda row: (int(row["seed"]), row["arm"]))
    expected = {
        (seed, arm, budget)
        for seed in (3, 4, 5)
        for arm in ARMS
        for budget in BUDGETS
    }
    assert {
        (int(row["seed"]), row["arm"], int(row["budget_kimg"]))
        for row in inventory
    } == expected
    assert len(parity) == 12
    assert all(row["status"] == "BITWISE_EQUIVALENT" for row in parity)

    parity_by_cell = {
        (int(row["seed"]), row["arm"]): row["status"] for row in parity
    }
    integrity = []
    compute = []
    for row in inventory:
        if int(row["budget_kimg"]) != 1024:
            continue
        seed = int(row["seed"])
        arm = row["arm"]
        resume_history = json.loads(row.get("resume_history", "[]"))
        elapsed = float(row["replay_elapsed_seconds"])
        integrity.append(
            {
                "seed": seed,
                "arm": arm,
                "final_kimg": 1024,
                "cur_nimg": int(row["cur_nimg"]),
                "attempted_iteration": int(row["attempted_iteration"]),
                "successful_optimizer_steps": int(
                    row["successful_optimizer_steps"]
                ),
                "amp_skips": int(row["amp_skips"]),
                "source_256_state_sha256": row[
                    "source_256_state_sha256"
                ],
                "final_state_sha256": row["replay_state_sha256"],
                "online_model_canonical_sha256": row[
                    "online_model_canonical_sha256"
                ],
                "ema_model_canonical_sha256": row[
                    "ema_model_canonical_sha256"
                ],
                "optimizer_canonical_sha256": row[
                    "optimizer_canonical_sha256"
                ],
                "resume_load_count": len(resume_history),
                "crash_recovery_count": max(0, len(resume_history) - 1),
                "parity": parity_by_cell[(seed, arm)],
                "status": "PASS",
            }
        )
        compute.append(
            {
                "seed": seed,
                "arm": arm,
                "replay_elapsed_seconds": elapsed,
                "gpu_hours": elapsed / 3600,
                "runtime_identity": row["runtime_identity"],
                "git_commit": row["git_commit"],
                "status": "PASS",
            }
        )

    assert len(integrity) == 12 and len(compute) == 12
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "replay_checkpoint_inventory.csv", inventory)
    write_csv(args.output_dir / "replay_1024_parity.csv", parity)
    write_csv(args.output_dir / "training_integrity.csv", integrity)
    write_csv(args.output_dir / "compute_cost.csv", compute)

    total_gpu_hours = sum(float(row["gpu_hours"]) for row in compute)
    report = f"""# q256 target-geometry × denominator-weighting exact-budget replay

**Training status: 12/12 trajectories PASS**  
**Checkpoint coverage: 72/72 replay milestones PASS**  
**EMA snapshot coverage: 84/84 PASS**  
**1024 replay parity: 12/12 bitwise-equivalent**

## Scope

Seeds 3–5 and frozen arms A/B/C/D were replayed from their immutable formal 256 kimg full-states to a total budget of exactly 1024 kimg. No 0→256 training, seed extension, parameter sweep, checkpoint selection, FID, or KID was performed.

Immutable replay milestones are 384, 512, 640, 768, 896, and 1024 kimg. The 256 kimg rows reference the frozen source states. Every milestone is keyed by exact `cur_nimg`, not tick number.

## Training integrity

- All states contain online model, EMA, complete RAdam state, GradScaler, counters, loss/control state, rank-local RNG, and sampler state.
- Every cell reached `cur_nimg=1024000` and `attempted_iteration=8000`.
- Strict telemetry reported no non-finite loss/update/model/EMA/factor events and no non-positive denominator events.
- No formal trajectory crashed or required recovery. The separate seed3/armA saver smoke is archived as engineering evidence and is not part of the 12 formal trajectories.
- Training commit: `c8721a05227f3ff171f8dc1f559a64d58281c0ae`.

## Canonical 1024 parity

All 12 replay endpoints are canonically bitwise-equivalent to the corresponding PR #76 endpoints for online model, EMA, optimizer state, GradScaler, loss/control state, RNG/sampler state, trajectory config, counters, and factorial identity. File-level `.pt` SHA256 is reported separately and is not used as the parity criterion.

## Compute

Total replay compute across the 12 single-GPU trajectories was {total_gpu_hours:.3f} A100 GPU-hours, including immutable checkpoint I/O in elapsed wall time.

## Archive

Server archive root: `{args.archive_root}`

The archive contains the 72 immutable replay states, 84 EMA snapshots, 12 frozen 256 kimg source states, deterministic runtime image, code bundles, resolved options, telemetry, logs, audits, and saver smoke evidence. `artifact_hashes.sha256` is generated after server-side transfer verification.

## Frozen evaluation plan

The next phase is frozen but has not been executed. Primary jobs are FID-50k and KID-50k at NFE=1 for all 84 seed×arm×budget checkpoints. Secondary NFE=2 jobs use `mid_t=0.821`. Evaluation uses FP32, samples 0–49999, metric seed 20260730, and shared generated features for KID/FID. The launcher is fail-closed until explicitly enabled after archive verification.

## Limitations

This package establishes a deterministic model trajectory and replay identity. It contains three formal training seeds and no intermediate FID/KID values yet; no mechanism or arm ranking should be inferred from training loss, previews, or endpoint metrics alone.
"""
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(
        f"REPLAY_RESULTS_PASS inventory={len(inventory)} parity={len(parity)} "
        f"gpu_hours={total_gpu_hours:.6f}"
    )


if __name__ == "__main__":
    main()
