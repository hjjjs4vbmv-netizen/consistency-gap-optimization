#!/usr/bin/env python3
"""Verify the mandatory 10-cell 512->640 no-op resume parity gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training import reproducibility, schedule_switch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_summary(state: dict) -> dict:
    ranks = state["rank_states"]
    return {
        "internal_state_sha256": schedule_switch.internal_state_hashes(state),
        "attempted_iteration": int(state["attempted_iteration"]),
        "successful_optimizer_steps": int(state["successful_optimizer_steps"]),
        "cur_nimg": int(state["cur_nimg"]),
        "sampler_consumed_samples": [
            int(item["sampler_state"]["consumed_samples"])
            for item in ranks
        ],
    }


def telemetry_rows(path: Path, start: int, end: int) -> list[dict]:
    with path.open("rt", newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if start <= int(row["attempted_iteration"]) <= end
        ]
    attempts = [int(row["attempted_iteration"]) for row in rows]
    if attempts != list(range(start, end + 1)):
        raise RuntimeError(f"telemetry attempt sequence mismatch: {path}")
    return rows


def verify_cell(parity_root: Path, inventory_cell: dict) -> dict:
    seed = inventory_cell["seed"]
    arm = inventory_cell["origin_arm"]
    branch = f"{arm}_to_{arm}"
    run_dir = parity_root / f"seed{seed}" / branch
    manifest = schedule_switch.load_run_manifest(
        run_dir / "formal_run_manifest.json"
    )
    new_path = run_dir / "kimg0640" / "training-state.pt"
    archived_record = next(
        item for item in inventory_cell["archived_controls"]
        if item["kimg"] == 640
    )
    archived_path = Path(archived_record["training_state"]["path"])
    new_state = torch.load(new_path, map_location="cpu", weights_only=False)
    archived_state = torch.load(
        archived_path, map_location="cpu", weights_only=False
    )
    schedule_switch.verify_switched_state(new_state, manifest)
    new_summary = state_summary(new_state)
    archived_summary = state_summary(archived_state)
    state_match = new_summary == archived_summary
    new_rows = telemetry_rows(
        run_dir / "schedule_switch_training_telemetry_v1.csv", 4001, 5000
    )
    archived_rows = telemetry_rows(
        Path(inventory_cell["canonical_cell_dir"])
        / "factorial_training_telemetry_v1.csv",
        4001,
        5000,
    )
    paired_fields = (
        "attempted_iteration", "batch_sha256", "t_sha256", "base_r_sha256"
    )
    mismatches = []
    for new, archived in zip(new_rows, archived_rows):
        for field in paired_fields:
            if new[field] != archived[field]:
                mismatches.append({
                    "attempted_iteration": int(new["attempted_iteration"]),
                    "field": field,
                    "new": new[field],
                    "archived": archived[field],
                })
                break
    return {
        "seed": seed,
        "branch": branch,
        "new_state_path": str(new_path),
        "new_state_sha256": sha256_file(new_path),
        "archived_state_path": str(archived_path),
        "archived_state_sha256": sha256_file(archived_path),
        "state_field_match": {
            key: new_summary[key] == archived_summary[key]
            for key in new_summary
        },
        "computational_state_match": state_match,
        "paired_telemetry_rows": len(new_rows),
        "paired_telemetry_mismatch_count": len(mismatches),
        "paired_telemetry_mismatches": mismatches[:20],
        "status": "PASS" if state_match and not mismatches else "FAIL",
    }


def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# q256 schedule-switch no-op parity report",
        "",
        f"Gate status: **{payload['status']}**",
        "",
        f"Computational-state matches: **{payload['pass_count']}/10**",
        "",
        "| Seed | Branch | State | Paired batch/t/base-r | Status |",
        "|---:|---|---|---|---|",
    ]
    for cell in payload["cells"]:
        lines.append(
            f"| {cell['seed']} | {cell['branch']} | "
            f"{'MATCH' if cell['computational_state_match'] else 'MISMATCH'} | "
            f"{cell['paired_telemetry_mismatch_count']} mismatches | "
            f"{cell['status']} |"
        )
    lines.extend([
        "",
        "Formal crossed training is authorized only when this report is 10/10 PASS.",
        "",
    ])
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    cells = []
    for inventory_cell in inventory["cells"]:
        try:
            cells.append(verify_cell(args.parity_root, inventory_cell))
        except Exception as exc:  # Preserve every cell verdict in fail-closed report.
            cells.append({
                "seed": inventory_cell["seed"],
                "branch": f"{inventory_cell['origin_arm']}_to_{inventory_cell['origin_arm']}",
                "computational_state_match": False,
                "paired_telemetry_mismatch_count": None,
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            })
    pass_count = sum(cell["status"] == "PASS" for cell in cells)
    payload = {
        "schema": "ect.q256.schedule-switch-parity-gate/v1",
        "status": "PASS" if pass_count == 10 else "FAIL",
        "verdict": (
            "10/10 COMPUTATIONAL_STATE_MATCH"
            if pass_count == 10 else "FAIL_CLOSED"
        ),
        "pass_count": pass_count,
        "expected_count": 10,
        "cells": cells,
    }
    reproducibility.atomic_json_dump(payload, args.output_json, overwrite=False)
    write_report(args.output_report, payload)
    print(json.dumps({"status": payload["status"], "pass_count": pass_count}))
    return 0 if pass_count == 10 else 2


if __name__ == "__main__":
    raise SystemExit(main())
