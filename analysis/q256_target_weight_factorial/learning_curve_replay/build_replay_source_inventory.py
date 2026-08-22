#!/usr/bin/env python3
"""Build the immutable 20-cell q256 replay source-state inventory."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any

import torch

from training import reproducibility


SEEDS = tuple(range(14, 19))
ARMS = ("A", "B", "C", "D")
FACTORS = {
    "A": (1.0, 1.0),
    "B": (1.1, 1.1),
    "C": (1.1, 1.0),
    "D": (1.0, 1.1),
}
SOURCE_COMMIT = "dcca41b19e7c45512b5fbe98776520396a1bf9ac"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_is_finite(module: torch.nn.Module) -> bool:
    for value in module.state_dict().values():
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
            if not bool(torch.isfinite(value).all()):
                return False
    return True


def load_last_summary(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty summary: {path}")
    row = rows[-1]
    result = {
        "attempted_iteration": int(row["attempted_iteration"]),
        "successful_optimizer_steps": int(row["successful_optimizer_steps"]),
        "processed_nimg": int(row["processed_nimg"]),
        "processed_kimg": float(row["processed_kimg"]),
        "loss": float(row["loss"]),
        "step_skipped": int(row["step_skipped"]),
    }
    if (
        result["attempted_iteration"] != 2000
        or not (0 < result["successful_optimizer_steps"] <= 2000)
        or result["processed_nimg"] != 256000
        or not math.isclose(result["processed_kimg"], 256.0)
        or not math.isfinite(result["loss"])
        or result["step_skipped"] != 0
    ):
        raise RuntimeError(f"invalid 256 kimg summary endpoint: {path}: {result}")
    return result


def inspect_cell(source_root: Path, seed: int, arm: str) -> dict[str, Any]:
    run_dir = source_root / f"seed{seed}" / f"arm{arm}"
    state_path = run_dir / "training-state-latest.pt"
    snapshot_path = run_dir / "network-snapshot-latest.pkl"
    options_path = run_dir / "training_options.json"
    summary_path = run_dir / "train_summary.csv"
    telemetry_path = run_dir / "factorial_training_telemetry_v1.csv"
    initial_receipt_path = run_dir / "initial_state_receipt_v1.json"
    log_path = run_dir / "log.txt"
    required = (
        state_path,
        snapshot_path,
        options_path,
        summary_path,
        telemetry_path,
        initial_receipt_path,
        log_path,
    )
    for path in required:
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"missing or invalid source artifact: {path}")

    summary = load_last_summary(summary_path)
    options = json.loads(options_path.read_text(encoding="utf-8"))
    initial_receipt = json.loads(initial_receipt_path.read_text(encoding="utf-8"))
    state = torch.load(state_path, map_location="cpu")
    expected_target, expected_denominator = FACTORS[arm]
    required_state = (
        "net",
        "ema",
        "optimizer_state",
        "gradscaler_state",
        "attempted_iteration",
        "successful_optimizer_steps",
        "cur_nimg",
        "cur_tick",
        "tick_start_nimg",
        "rank_states",
        "factorial",
        "trajectory_config",
        "trajectory_config_sha256",
        "loss_fn_state",
        "snapshot_grid_z",
        "snapshot_grid_c",
        "snapshot_grid_size",
    )
    missing = [key for key in required_state if key not in state]
    if missing:
        raise RuntimeError(f"source state missing keys {missing}: {state_path}")
    factorial = state["factorial"]
    trajectory = state["trajectory_config"]
    if (
        factorial.get("arm") != arm
        or float(factorial.get("target_gap_scale")) != expected_target
        or float(factorial.get("denominator_gap_scale")) != expected_denominator
    ):
        raise RuntimeError(f"factorial identity mismatch: seed{seed}/arm{arm}")
    if (
        int(trajectory.get("seed", -1)) != seed
        or int(trajectory.get("total_kimg", -1)) != 256
        or int(trajectory.get("batch_size", -1)) != 128
    ):
        raise RuntimeError(f"trajectory identity mismatch: seed{seed}/arm{arm}")
    if reproducibility.state_sha256(trajectory) != state["trajectory_config_sha256"]:
        raise RuntimeError(f"trajectory hash mismatch: seed{seed}/arm{arm}")
    if (
        int(state["attempted_iteration"]) != 2000
        or int(state["cur_nimg"]) != 256000
        or int(state["successful_optimizer_steps"])
        != summary["successful_optimizer_steps"]
    ):
        raise RuntimeError(f"state/summary endpoint mismatch: seed{seed}/arm{arm}")
    rank_states = state["rank_states"]
    if len(rank_states) != 1:
        raise RuntimeError(f"unexpected rank-state count: seed{seed}/arm{arm}")
    rank_state = rank_states[0]
    if "rng_state" not in rank_state or "sampler_state" not in rank_state:
        raise RuntimeError(f"missing RNG/sampler state: seed{seed}/arm{arm}")
    if int(rank_state["sampler_state"].get("consumed_samples", -1)) != 256000:
        raise RuntimeError(f"sampler consumption mismatch: seed{seed}/arm{arm}")
    if not module_is_finite(state["net"]) or not module_is_finite(state["ema"]):
        raise RuntimeError(f"non-finite model/EMA state: seed{seed}/arm{arm}")

    with snapshot_path.open("rb") as handle:
        snapshot = pickle.load(handle)
    if "ema" not in snapshot or not module_is_finite(snapshot["ema"]):
        raise RuntimeError(f"invalid evaluation snapshot: seed{seed}/arm{arm}")
    state_ema_sha = reproducibility.module_state_sha256(state["ema"])
    if reproducibility.module_state_sha256(snapshot["ema"]) != state_ema_sha:
        raise RuntimeError(f"snapshot/state EMA mismatch: seed{seed}/arm{arm}")
    if int(initial_receipt.get("seed", -1)) != seed:
        raise RuntimeError(f"initial receipt seed mismatch: seed{seed}/arm{arm}")
    if initial_receipt.get("factorial", {}).get("arm") != arm:
        raise RuntimeError(f"initial receipt arm mismatch: seed{seed}/arm{arm}")
    if int(options.get("seed", -1)) != seed:
        raise RuntimeError(f"training options seed mismatch: seed{seed}/arm{arm}")

    internal = {
        "model_state_sha256": reproducibility.module_state_sha256(state["net"]),
        "ema_state_sha256": state_ema_sha,
        "optimizer_state_sha256": reproducibility.state_sha256(state["optimizer_state"]),
        "gradscaler_state_sha256": reproducibility.state_sha256(state["gradscaler_state"]),
        "rank_rng_sha256": reproducibility.state_sha256(rank_state["rng_state"]),
        "rank_sampler_sha256": reproducibility.state_sha256(rank_state["sampler_state"]),
    }
    files = {
        path.name: {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in required
    }
    del state, snapshot
    gc.collect()
    return {
        "seed": seed,
        "arm": arm,
        "status": "PASS",
        "source_commit": SOURCE_COMMIT,
        "summary_endpoint": summary,
        "factorial": {
            "target_gap_scale": expected_target,
            "denominator_gap_scale": expected_denominator,
        },
        "full_state_fields": list(required_state),
        "rng_and_sampler_present": True,
        "trajectory_config_sha256": initial_receipt["trajectory_config_sha256"],
        "internal_state_sha256": internal,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve(strict=True)
    out = args.out.resolve()
    if out.exists() or out.is_symlink():
        raise RuntimeError(f"refusing existing inventory: {out}")
    cells = []
    for seed in SEEDS:
        for arm in ARMS:
            try:
                cells.append(inspect_cell(source_root, seed, arm))
            except BaseException as exc:
                cells.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "status": "BLOCKED",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    pass_count = sum(cell["status"] == "PASS" for cell in cells)
    payload = {
        "schema": "ect.q256.learning-curve-replay-source-inventory/v1",
        "status": "PASS" if pass_count == 20 else "PARTIAL_BLOCKED",
        "source_root": str(source_root),
        "source_commit": SOURCE_COMMIT,
        "required_endpoint_kimg": 256,
        "cell_count": 20,
        "pass_count": pass_count,
        "blocked_count": 20 - pass_count,
        "cells": cells,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"status": payload["status"], "pass": pass_count, "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
