#!/usr/bin/env python3
"""Build the immutable 10-cell q256 512-kimg source inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training import reproducibility, schedule_switch


SEEDS = tuple(range(14, 19))
ARMS = ("A", "B")
SOURCE_KIMG = 512
CONTROL_KIMG = (640, 768, 896, 1024)
SOURCE_TRAINING_COMMIT = "dcca41b19e7c45512b5fbe98776520396a1bf9ac"
SOURCE_REPLAY_COMMIT = "f4115a89c764081e01be4290f0868cb8f625825e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing immutable source artifact: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_checkpoint_manifest(cell_dir: Path) -> tuple[dict, dict]:
    path = cell_dir / "checkpoint_manifest.json"
    record = file_record(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("status") == "PASS", f"checkpoint manifest not PASS: {path}")
    replay_commit = payload.get("replay_commit") or payload.get("implementation_commit")
    source_commit = payload.get("source_commit")
    require(replay_commit == SOURCE_REPLAY_COMMIT,
            f"unexpected replay commit in {path}: {replay_commit}")
    require(source_commit == SOURCE_TRAINING_COMMIT,
            f"unexpected training source commit in {path}: {source_commit}")
    return payload, record


def verify_milestone(cell_dir: Path, seed: int, arm: str, kimg: int) -> dict:
    milestone = cell_dir / f"kimg{kimg:04d}"
    state_path = milestone / "training-state.pt"
    snapshot_path = milestone / "network-snapshot.pkl"
    receipt_path = milestone / "milestone_receipt.json"
    state_record = file_record(state_path)
    snapshot_record = file_record(snapshot_path)
    receipt_record = file_record(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("seed") == seed and receipt.get("arm") == arm,
            f"milestone receipt cell mismatch: {receipt_path}")
    require(receipt.get("milestone_kimg") == kimg,
            f"milestone receipt budget mismatch: {receipt_path}")
    return {
        "kimg": kimg,
        "training_state": state_record,
        "network_snapshot": snapshot_record,
        "milestone_receipt": receipt_record,
    }


def inventory_cell(root: Path, seed: int, arm: str) -> dict:
    cell_dir = root / f"seed{seed}" / f"arm{arm}"
    require(cell_dir.is_dir() and not cell_dir.is_symlink(),
            f"missing canonical cell directory: {cell_dir}")
    checkpoint_manifest, checkpoint_manifest_record = load_checkpoint_manifest(cell_dir)
    source_artifacts = verify_milestone(cell_dir, seed, arm, SOURCE_KIMG)
    controls = [verify_milestone(cell_dir, seed, arm, kimg) for kimg in CONTROL_KIMG]
    state_path = Path(source_artifacts["training_state"]["path"])
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    manifest_stub = {
        "seed": seed,
        "origin_arm": arm,
        "source_state": {
            "internal_state_sha256": schedule_switch.internal_state_hashes(state)
        },
    }
    internal = schedule_switch.verify_source_state(state, manifest_stub)
    trajectory = state["trajectory_config"]
    with Path(source_artifacts["network_snapshot"]["path"]).open("rb") as handle:
        snapshot = pickle.load(handle)
    require("ema" in snapshot, "source evaluation snapshot is missing EMA")
    require(reproducibility.module_state_sha256(snapshot["ema"]) == internal["ema"],
            "source snapshot EMA does not match full-state EMA")
    rank_states = state["rank_states"]
    result = {
        "status": "PASS",
        "seed": seed,
        "origin_arm": arm,
        "canonical_cell_dir": str(cell_dir.resolve()),
        "source_kimg": SOURCE_KIMG,
        "cur_nimg": int(state["cur_nimg"]),
        "attempted_iteration": int(state["attempted_iteration"]),
        "successful_optimizer_steps": int(state["successful_optimizer_steps"]),
        "sampler_consumed_samples": int(
            rank_states[0]["sampler_state"]["consumed_samples"]
        ),
        "factorial": reproducibility.canonical_json_data(state["factorial"]),
        "trajectory_config_sha256": state["trajectory_config_sha256"],
        "source_training_commit": SOURCE_TRAINING_COMMIT,
        "source_replay_commit": SOURCE_REPLAY_COMMIT,
        "source_checkpoint_manifest": checkpoint_manifest_record,
        "source_checkpoint_manifest_schema": checkpoint_manifest.get("schema"),
        "source_artifacts": source_artifacts,
        "internal_state_sha256": internal,
        "archived_controls": controls,
    }
    del state, snapshot
    return result


def write_report(path: Path, inventory: dict) -> None:
    lines = [
        "# q256 512-kimg schedule-switch source inventory",
        "",
        f"Status: **{inventory['status']}**",
        "",
        f"Canonical root: `{inventory['canonical_root']}`",
        "",
        f"Protocol SHA256: `{inventory['protocol_sha256']}`",
        "",
        "All ten seed × origin-arm source states are exact 512-kimg full states. "
        "Each has a matching EMA snapshot, milestone receipt, complete optimizer/"
        "GradScaler/RNG/sampler state, and archived 640/768/896/1024 controls.",
        "",
        "| Seed | Arm | Attempts | Successful steps | State SHA256 | Checkpoint manifest SHA256 |",
        "|---:|:---:|---:|---:|---|---|",
    ]
    for cell in inventory["cells"]:
        lines.append(
            f"| {cell['seed']} | {cell['origin_arm']} | "
            f"{cell['attempted_iteration']} | {cell['successful_optimizer_steps']} | "
            f"`{cell['source_artifacts']['training_state']['sha256']}` | "
            f"`{cell['source_checkpoint_manifest']['sha256']}` |"
        )
    lines.extend([
        "",
        "No source file was modified, moved, replaced, or synthesized during this audit.",
        "",
    ])
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    root = args.canonical_root.resolve(strict=True)
    protocol = args.protocol.resolve(strict=True)
    protocol_sha = sha256_file(protocol)
    cells = [inventory_cell(root, seed, arm) for seed in SEEDS for arm in ARMS]
    require(len(cells) == 10 and all(cell["status"] == "PASS" for cell in cells),
            "source inventory is incomplete")
    payload = {
        "schema": "ect.q256.schedule-switch-source-inventory/v1",
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "canonical_root": str(root),
        "seeds": list(SEEDS),
        "origin_arms": list(ARMS),
        "source_kimg": SOURCE_KIMG,
        "archived_control_kimg": list(CONTROL_KIMG),
        "cells": cells,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    reproducibility.atomic_json_dump(payload, args.output_json, overwrite=False)
    write_report(args.output_report, payload)
    print(json.dumps({"status": "PASS", "cells": len(cells),
                      "protocol_sha256": protocol_sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
