#!/usr/bin/env python3
"""Inventory the heterogeneous but frozen seed3-7 A/B 512-kimg sources."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training import reproducibility, schedule_switch


SEEDS = (3, 4, 5, 6, 7)
ARMS = ("A", "B")
CONTROL_BUDGETS = (640, 768, 896, 1024)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def staged_path(prefix: Path, original: str) -> Path:
    path = prefix / original.lstrip("/")
    return path.resolve(strict=True)


def file_record(path: Path, *, original_path: str | None = None) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing regular source artifact: {path}")
    result = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if original_path is not None:
        result["original_path"] = original_path
    return result


def state_record(prefix: Path, original_path: str) -> dict:
    return file_record(
        staged_path(prefix, original_path), original_path=original_path
    )


def source_manifest_path(prefix: Path, cohort: dict, seed: int) -> tuple[Path, str]:
    root = Path(cohort["canonical_root"])
    if seed <= 5:
        original = str(root.parent.parent / "audits" / f"seed{seed}-replay-inventory.csv")
    else:
        original = str(
            root / "consolidated_results_v1" / "reports"
            / "checkpoint_inventory_seed6_7_ab_128k.csv"
        )
    return staged_path(prefix, original), original


def load_manifest_rows(path: Path) -> list[dict]:
    with path.open("rt", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty source manifest: {path}")
    return rows


def manifest_row(rows: list[dict], *, seed: int, arm: str, budget: int) -> dict:
    matches = [
        row for row in rows
        if int(row["seed"]) == seed and row["arm"] == arm
        and int(row["budget_kimg"]) == budget
    ]
    if len(matches) != 1 or matches[0].get("status") != "PASS":
        raise RuntimeError(
            f"missing unique PASS manifest row: seed={seed} arm={arm} budget={budget}"
        )
    return matches[0]


def row_state_identity(row: dict, *, seed: int) -> tuple[str, int | None, str]:
    if seed <= 5:
        return row["replay_state_path"], None, row["replay_state_sha256"]
    return (
        row["training_state"], int(row["training_state_bytes"]),
        row["training_state_sha256"],
    )


def row_snapshot_identity(row: dict, *, seed: int) -> tuple[str, int | None, str]:
    if seed <= 5:
        return row["ema_snapshot_path"], None, row["ema_snapshot_sha256"]
    return row["snapshot"], int(row["snapshot_bytes"]), row["snapshot_sha256"]


def canonical_cell_dir(prefix: Path, cohort: dict, seed: int, arm: str) -> tuple[Path, str]:
    original = str(Path(cohort["canonical_root"]) / f"seed{seed}" / f"arm{arm}")
    return staged_path(prefix, original), original


def format_pattern(cohort: dict, key: str, *, seed: int, arm: str, budget: int | None = None) -> str:
    values = {"seed": seed, "arm": arm}
    if budget is not None:
        values["budget"] = budget
    relative = cohort[key].format(**values)
    return str(Path(cohort["canonical_root"]) / relative)


def normalize_trajectory_config(config: dict) -> dict:
    value = reproducibility.canonical_json_data(copy.deepcopy(config))
    value.pop("seed", None)
    value.pop("rank_seed", None)
    dataset = value.get("dataset_kwargs")
    if isinstance(dataset, dict):
        dataset.pop("path", None)
    loss = value.get("loss_kwargs")
    if isinstance(loss, dict):
        for key in ("arm", "target_gap_scale", "denominator_gap_scale"):
            loss.pop(key, None)
    return value


def validate_state(state: dict, *, seed: int, arm: str) -> dict:
    required = (
        "net", "ema", "optimizer_state", "gradscaler_state",
        "attempted_iteration", "successful_optimizer_steps", "cur_nimg",
        "rank_states", "factorial", "trajectory_config",
        "trajectory_config_sha256",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise RuntimeError(f"source state missing fields: {missing}")
    if int(state["cur_nimg"]) != 512000 or int(state["attempted_iteration"]) != 4000:
        raise RuntimeError("source state is not exact 512 kimg / attempt 4000")
    if reproducibility.state_sha256(state["trajectory_config"]) != state["trajectory_config_sha256"]:
        raise RuntimeError("source trajectory-config SHA mismatch")
    if int(state["trajectory_config"].get("seed", -1)) != seed:
        raise RuntimeError("source state seed mismatch")
    target = 1.0 if arm == "A" else 1.1
    factorial = state["factorial"]
    if (
        factorial.get("protocol") != "q256_target_weight_v1"
        or factorial.get("arm") != arm
        or float(factorial.get("target_gap_scale")) != target
        or float(factorial.get("denominator_gap_scale")) != target
    ):
        raise RuntimeError("source factorial identity mismatch")
    ranks = state["rank_states"]
    if len(ranks) != 1 or int(ranks[0]["sampler_state"].get("consumed_samples", -1)) != 512000:
        raise RuntimeError("source sampler cursor mismatch")
    return schedule_switch.internal_state_hashes(state)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-prefix", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    prefix = args.source_prefix.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol["protocol"] not in {
            schedule_switch.SEED3_7_PROTOCOL,
            schedule_switch.SEED3_7_PROTOCOL_V2,
        }
        or tuple(protocol["seeds"]) != SEEDS
    ):
        raise RuntimeError("wrong frozen protocol for seed3-7 inventory")
    protocol_sha = sha256_file(protocol_path)
    cohorts = protocol["source_inventory"]["cohorts"]
    cells = []
    compatibility = {}
    for seed in SEEDS:
        cohort = next(item for item in cohorts if seed in item["seeds"])
        manifest_path, manifest_original = source_manifest_path(prefix, cohort, seed)
        manifest_record = file_record(manifest_path, original_path=manifest_original)
        manifest_rows = load_manifest_rows(manifest_path)
        for arm in ARMS:
            cell_dir, cell_original = canonical_cell_dir(prefix, cohort, seed, arm)
            source_state_original = format_pattern(cohort, "source_state_pattern", seed=seed, arm=arm)
            source_snapshot_original = format_pattern(cohort, "source_snapshot_pattern", seed=seed, arm=arm)
            source_state = state_record(prefix, source_state_original)
            source_snapshot = state_record(prefix, source_snapshot_original)
            source_manifest_row = manifest_row(
                manifest_rows, seed=seed, arm=arm, budget=512
            )
            _, expected_source_bytes, expected_source_sha = row_state_identity(
                source_manifest_row, seed=seed
            )
            _, expected_snapshot_bytes, expected_snapshot_sha = row_snapshot_identity(
                source_manifest_row, seed=seed
            )
            if source_state["sha256"] != expected_source_sha or (
                expected_source_bytes is not None
                and source_state["bytes"] != expected_source_bytes
            ):
                raise RuntimeError("source state differs from archived PASS manifest")
            if source_snapshot["sha256"] != expected_snapshot_sha or (
                expected_snapshot_bytes is not None
                and source_snapshot["bytes"] != expected_snapshot_bytes
            ):
                raise RuntimeError("source snapshot differs from archived PASS manifest")
            state = torch.load(source_state["path"], map_location="cpu", weights_only=False)
            internal = validate_state(state, seed=seed, arm=arm)
            with open(source_snapshot["path"], "rb") as handle:
                snapshot = pickle.load(handle)
            if reproducibility.module_state_sha256(snapshot["ema"]) != internal["ema"]:
                raise RuntimeError("source snapshot EMA does not match full state")
            normalized = normalize_trajectory_config(state["trajectory_config"])
            normalized_sha = reproducibility.state_sha256(normalized)
            compatibility[(seed, arm)] = normalized_sha
            controls = []
            for budget in CONTROL_BUDGETS:
                control_state_original = format_pattern(
                    cohort, "archived_control_state_pattern",
                    seed=seed, arm=arm, budget=budget,
                )
                control_snapshot_original = format_pattern(
                    cohort, "archived_control_snapshot_pattern",
                    seed=seed, arm=arm, budget=budget,
                )
                control_manifest_row = manifest_row(
                    manifest_rows, seed=seed, arm=arm, budget=budget
                )
                original_manifest_state, manifest_state_bytes, manifest_state_sha = (
                    row_state_identity(control_manifest_row, seed=seed)
                )
                _, manifest_snapshot_bytes, manifest_snapshot_sha = (
                    row_snapshot_identity(control_manifest_row, seed=seed)
                )
                control_snapshot = state_record(prefix, control_snapshot_original)
                if control_snapshot["sha256"] != manifest_snapshot_sha or (
                    manifest_snapshot_bytes is not None
                    and control_snapshot["bytes"] != manifest_snapshot_bytes
                ):
                    raise RuntimeError("control snapshot differs from PASS manifest")
                if budget == 640:
                    control_state = state_record(prefix, control_state_original)
                    if control_state["sha256"] != manifest_state_sha or (
                        manifest_state_bytes is not None
                        and control_state["bytes"] != manifest_state_bytes
                    ):
                        raise RuntimeError("640 control state differs from PASS manifest")
                else:
                    control_state = {
                        "path": original_manifest_state,
                        "original_path": control_state_original,
                        "bytes": manifest_state_bytes,
                        "sha256": manifest_state_sha,
                        "staged": False,
                        "verified_via_source_manifest": manifest_record["sha256"],
                    }
                controls.append({
                    "kimg": budget,
                    "training_state": control_state,
                    "network_snapshot": control_snapshot,
                })
            required_history = {}
            for name in (
                "training_options.json", "train_summary.csv",
                "factorial_training_telemetry_v1.csv",
                "initial_state_receipt_v1.json",
            ):
                original = str(Path(cell_original) / name)
                required_history[name] = state_record(prefix, original)
            cells.append({
                "status": "PASS",
                "seed": seed,
                "origin_arm": arm,
                "canonical_cell_dir": str(cell_dir),
                "original_canonical_cell_dir": cell_original,
                "source_kimg": 512,
                "cur_nimg": int(state["cur_nimg"]),
                "attempted_iteration": int(state["attempted_iteration"]),
                "successful_optimizer_steps": int(state["successful_optimizer_steps"]),
                "sampler_consumed_samples": int(
                    state["rank_states"][0]["sampler_state"]["consumed_samples"]
                ),
                "factorial": reproducibility.canonical_json_data(state["factorial"]),
                "trajectory_config_sha256": state["trajectory_config_sha256"],
                "normalized_trajectory_config_sha256": normalized_sha,
                "source_checkpoint_manifest": manifest_record,
                "source_artifacts": {
                    "training_state": source_state,
                    "network_snapshot": source_snapshot,
                },
                "internal_state_sha256": internal,
                "required_history": required_history,
                "archived_controls": controls,
            })
            del state, snapshot
    if len(cells) != 10:
        raise RuntimeError("source inventory is not exactly 10 cells")
    unique_compatibility = sorted(set(compatibility.values()))
    compatibility_status = "PASS" if len(unique_compatibility) == 1 else "FAIL"
    payload = {
        "schema": "ect.q256.schedule-switch-source-inventory/v1",
        "status": "PASS" if compatibility_status == "PASS" else "FAIL_CLOSED",
        "protocol_sha256": protocol_sha,
        "source_prefix": str(prefix),
        "seeds": list(SEEDS),
        "origin_arms": list(ARMS),
        "source_kimg": 512,
        "archived_control_kimg": list(CONTROL_BUDGETS),
        "cross_cohort_compatibility": {
            "status": compatibility_status,
            "normalized_trajectory_config_sha256": unique_compatibility,
            "cells": [
                {"seed": seed, "arm": arm, "sha256": compatibility[(seed, arm)]}
                for seed in SEEDS for arm in ARMS
            ],
        },
        "cells": cells,
    }
    reproducibility.atomic_json_dump(payload, args.output_json, overwrite=False)
    lines = [
        "# q256 seed3-7 schedule-switch source inventory", "",
        f"Status: **{payload['status']}**", "",
        f"Protocol SHA256: `{protocol_sha}`", "",
        f"Cross-cohort compatibility: **{compatibility_status}**", "",
        "| Seed | Arm | Attempts | Successful steps | Source state SHA256 |",
        "|---:|:---:|---:|---:|---|",
    ]
    for cell in cells:
        lines.append(
            f"| {cell['seed']} | {cell['origin_arm']} | "
            f"{cell['attempted_iteration']} | {cell['successful_optimizer_steps']} | "
            f"`{cell['source_artifacts']['training_state']['sha256']}` |"
        )
    lines.extend(["", "Parity is authorized only when this report is PASS.", ""])
    with args.output_report.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines)); handle.flush(); os.fsync(handle.fileno())
    print(json.dumps({"status": payload["status"], "cells": len(cells),
                      "compatibility": compatibility_status}))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
