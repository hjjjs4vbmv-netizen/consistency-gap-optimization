#!/usr/bin/env python3
"""Adopt a fail-closed 384 kimg checkpoint into a fresh recovery root."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path

import torch

from training import reproducibility


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_module(module: torch.nn.Module) -> bool:
    return all(
        bool(torch.isfinite(value).all())
        for value in module.state_dict().values()
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex())
    )


def write_json_exclusive(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-run-dir", type=Path, required=True)
    parser.add_argument("--recovery-run-dir", type=Path, required=True)
    parser.add_argument("--clean-source-log", type=Path, required=True)
    parser.add_argument("--failure-log", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=range(14, 19))
    args = parser.parse_args()
    failed = args.failed_run_dir.resolve(strict=True)
    recovery = args.recovery_run_dir.resolve()
    clean_log = args.clean_source_log.resolve(strict=True)
    failure_log = args.failure_log.resolve(strict=True)
    if recovery.exists() or recovery.is_symlink():
        raise RuntimeError(f"refusing existing recovery run: {recovery}")
    failure_text = failure_log.read_text(encoding="utf-8", errors="replace")
    if "AttributeError: module 'torch_utils.distributed' has no attribute 'barrier'" not in failure_text:
        raise RuntimeError("failed run does not have the authorized barrier failure")

    state_source = failed / "training-state-latest.pt"
    snapshot_source = failed / "network-snapshot-latest.pkl"
    with (failed / "train_summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    final = summary[-1]
    if (
        int(final["attempted_iteration"]) != 3000
        or int(final["processed_nimg"]) != 384000
        or float(final["processed_kimg"]) != 384.0
        or int(final["step_skipped"]) != 0
    ):
        raise RuntimeError(f"failed run did not stop at exact 384 kimg: {final}")
    state = torch.load(state_source, map_location="cpu")
    if (
        int(state["attempted_iteration"]) != 3000
        or int(state["cur_nimg"]) != 384000
        or state["factorial"].get("arm") != "A"
        or int(state["trajectory_config"].get("seed", -1)) != args.seed
        or not finite_module(state["net"])
        or not finite_module(state["ema"])
    ):
        raise RuntimeError("failed run latest state is not a valid seed/armA 384 checkpoint")
    with snapshot_source.open("rb") as handle:
        snapshot = pickle.load(handle)
    if not finite_module(snapshot["ema"]):
        raise RuntimeError("failed run latest snapshot contains non-finite EMA")
    state_ema_sha = reproducibility.module_state_sha256(state["ema"])
    if reproducibility.module_state_sha256(snapshot["ema"]) != state_ema_sha:
        raise RuntimeError("failed run state/snapshot EMA mismatch")

    recovery.parent.mkdir(parents=True, exist_ok=True)
    recovery.mkdir(mode=0o750)
    for name in (
        "training_options.json",
        "train_summary.csv",
        "factorial_training_telemetry_v1.csv",
        "initial_state_receipt_v1.json",
        "stats.jsonl",
    ):
        source = failed / name
        if source.is_file():
            shutil.copy2(source, recovery / name)
    shutil.copy2(clean_log, recovery / "log.txt")
    shutil.copy2(state_source, recovery / "training-state-latest.pt")
    shutil.copy2(snapshot_source, recovery / "network-snapshot-latest.pkl")
    milestone = recovery / "kimg0384"
    milestone.mkdir(mode=0o750)
    shutil.copy2(state_source, milestone / "training-state.pt")
    shutil.copy2(snapshot_source, milestone / "network-snapshot.pkl")
    receipt = {
        "schema": "ect.q256.learning-curve-milestone/v1",
        "seed": args.seed,
        "arm": "A",
        "milestone_kimg": 384,
        "attempted_iteration": 3000,
        "successful_optimizer_steps": int(state["successful_optimizer_steps"]),
        "processed_nimg": 384000,
        "processed_kimg": 384.0,
        "network_snapshot": "network-snapshot.pkl",
        "training_state": "training-state.pt",
        "trajectory_config_sha256": state["trajectory_config_sha256"],
        "natural_maintenance_due": False,
        "recovery_kind": "hash-identical adoption after output-only barrier failure",
    }
    write_json_exclusive(milestone / "milestone_receipt.json", receipt)
    provenance = {
        "schema": "ect.q256.learning-curve-384-recovery/v1",
        "status": "PASS",
        "seed": args.seed,
        "arm": "A",
        "failed_run_dir": str(failed),
        "failure_log": str(failure_log),
        "failure_log_sha256": sha256_file(failure_log),
        "state_source_sha256": sha256_file(state_source),
        "snapshot_source_sha256": sha256_file(snapshot_source),
        "adopted_state_sha256": sha256_file(milestone / "training-state.pt"),
        "adopted_snapshot_sha256": sha256_file(milestone / "network-snapshot.pkl"),
        "computational_state": {
            "net": reproducibility.module_state_sha256(state["net"]),
            "ema": state_ema_sha,
            "optimizer": reproducibility.state_sha256(state["optimizer_state"]),
            "gradscaler": reproducibility.state_sha256(state["gradscaler_state"]),
        },
    }
    if provenance["state_source_sha256"] != provenance["adopted_state_sha256"]:
        raise RuntimeError("adopted 384 training state is not byte-identical")
    if provenance["snapshot_source_sha256"] != provenance["adopted_snapshot_sha256"]:
        raise RuntimeError("adopted 384 snapshot is not byte-identical")
    write_json_exclusive(recovery / "recovery_384_provenance.json", provenance)
    print(json.dumps({"status": "PASS", "seed": args.seed, "kimg": 384}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
