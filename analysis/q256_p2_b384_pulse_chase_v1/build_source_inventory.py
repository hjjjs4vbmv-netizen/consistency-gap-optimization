#!/usr/bin/env python3
"""Audit one fresh B@384 source state and write an exclusive inventory."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training import pulse_chase, reproducibility


def floating_nonfinite(value) -> int:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            return int((~torch.isfinite(value)).sum())
        return 0
    if isinstance(value, dict):
        return sum(floating_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(floating_nonfinite(item) for item in value)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(18, *pulse_chase.SEEDS), required=True)
    parser.add_argument("--run-kind", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--runtime-sif", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_path = args.source_state.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    failures: list[str] = []
    if source_path.is_symlink():
        failures.append("source state is a symlink")
    assets = {
        "dataset": pulse_chase.sha256_file(args.dataset.resolve(strict=True)),
        "transfer": pulse_chase.sha256_file(args.transfer.resolve(strict=True)),
        "runtime_sif": pulse_chase.sha256_file(args.runtime_sif.resolve(strict=True)),
    }
    if assets != pulse_chase.ASSET_SHA256:
        failures.append("asset SHA256 mismatch")
    state = torch.load(source_path, map_location="cpu", weights_only=False)
    required = {
        "net", "ema", "optimizer_state", "gradscaler_state", "loss_fn_state",
        "rank_states", "factorial", "trajectory_config",
        "trajectory_config_sha256", "cur_nimg", "attempted_iteration",
        "successful_optimizer_steps", "cur_tick", "tick_start_nimg",
        "snapshot_grid_z", "snapshot_grid_c", "snapshot_grid_size",
    }
    missing = sorted(required - set(state))
    if missing:
        failures.append("missing fields: " + ", ".join(missing))
    if state.get("factorial") != pulse_chase.factorial_for_arm("B"):
        failures.append("source factorial is not B")
    if int(state.get("cur_nimg", -1)) != pulse_chase.SOURCE_KIMG * 1000:
        failures.append("source cur_nimg is not 384000")
    if int(state.get("attempted_iteration", -1)) != pulse_chase.SOURCE_ATTEMPT:
        failures.append("source attempted iteration is not 3000")
    trajectory = state.get("trajectory_config", {})
    if reproducibility.state_sha256(trajectory) != state.get(
        "trajectory_config_sha256"
    ):
        failures.append("trajectory hash is invalid")
    if int(trajectory.get("seed", -1)) != args.seed:
        failures.append("trajectory seed mismatch")
    ranks = state.get("rank_states", [])
    if len(ranks) != 1:
        failures.append("source rank-state count is not one")
    elif int(ranks[0]["sampler_state"].get("consumed_samples", -1)) != 384000:
        failures.append("source sampler cursor is not 384000")
    optimizer_steps = []
    for item in state.get("optimizer_state", {}).get("state", {}).values():
        value = item.get("step")
        if value is None:
            failures.append("RAdam state lacks step")
            continue
        optimizer_steps.append(
            int(value.item()) if isinstance(value, torch.Tensor) else int(value)
        )
    if not optimizer_steps:
        failures.append("RAdam step state is empty")
    nonfinite = floating_nonfinite({
        "net": state.get("net").state_dict() if state.get("net") else {},
        "ema": state.get("ema").state_dict() if state.get("ema") else {},
        "optimizer": state.get("optimizer_state", {}),
        "gradscaler": state.get("gradscaler_state", {}),
    })
    if nonfinite:
        failures.append(f"{nonfinite} non-finite state values")
    internal = pulse_chase.internal_state_hashes(state)
    record = pulse_chase.state_record(source_path, state)
    payload = {
        "schema": "ect.q256.p2-b384-source-inventory/v1",
        "status": "PASS" if not failures else "FAIL_CLOSED",
        "seed": args.seed,
        "run_kind": args.run_kind,
        "origin_arm": "B",
        "source_kimg": 384,
        "protocol_path": str(protocol_path),
        "protocol_sha256": pulse_chase.sha256_file(protocol_path),
        "implementation_commit": args.implementation_commit,
        "asset_sha256": assets,
        "source_state": record,
        "internal_state_sha256": internal,
        "radam_step_summary": {
            "parameter_states": len(optimizer_steps),
            "minimum": min(optimizer_steps) if optimizer_steps else None,
            "maximum": max(optimizer_steps) if optimizer_steps else None,
            "unique": sorted(set(optimizer_steps)),
        },
        "floating_nonfinite_count": nonfinite,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reproducibility.atomic_json_dump(payload, args.output, overwrite=False)
    print(json.dumps({"status": payload["status"], "seed": args.seed,
                      "source_sha256": record["sha256"]}))
    return 0 if not failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
