#!/usr/bin/env python3
"""Validate a completed training run and emit a formal-evaluation receipt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any

import torch


CHECKER_VERSION = "1"


def fail(message: str) -> None:
    raise SystemExit(f"[check_training_integrity] ERROR: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {label} {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must contain a JSON object: {path}")
    return value


def metric_mean(record: dict[str, Any], name: str) -> float:
    try:
        value = float(record[name]["mean"])
    except (KeyError, TypeError, ValueError) as exc:
        fail(f"stats record lacks finite {name}.mean: {exc}")
    if not math.isfinite(value):
        fail(f"stats record has non-finite {name}.mean")
    return value


def inspect_stats(path: Path, budget_kimg: int) -> dict[str, Any]:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        fail(f"cannot read stats {path}: {exc}")
    if not lines:
        fail(f"stats is empty: {path}")
    try:
        records = [json.loads(line) for line in lines]
    except json.JSONDecodeError as exc:
        fail(f"stats contains invalid JSON: {path}: {exc}")
    if not all(isinstance(record, dict) for record in records):
        fail(f"stats contains a non-object record: {path}")
    for record in records:
        metric_mean(record, "Loss/loss")
    final_kimg = metric_mean(records[-1], "Progress/kimg")
    if final_kimg < budget_kimg:
        fail(f"stats stops at {final_kimg} kimg, below declared {budget_kimg} kimg")
    return {"records": len(records), "final_kimg": final_kimg}


def inspect_summary(path: Path, budget_kimg: int) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        fail(f"cannot read training summary {path}: {exc}")
    if not rows:
        fail(f"training summary is empty: {path}")
    required = {"processed_kimg", "loss"}
    if not required.issubset(rows[0]):
        fail(f"training summary lacks required columns {sorted(required)}: {path}")
    for row in rows:
        try:
            loss = float(row["loss"])
        except (TypeError, ValueError) as exc:
            fail(f"training summary has invalid loss: {exc}")
        if not math.isfinite(loss):
            fail("training summary has non-finite loss")
    try:
        final_kimg = float(rows[-1]["processed_kimg"])
    except (TypeError, ValueError) as exc:
        fail(f"training summary has invalid final processed_kimg: {exc}")
    if not math.isfinite(final_kimg) or final_kimg < budget_kimg:
        fail(f"training summary stops at {final_kimg} kimg, below declared {budget_kimg} kimg")
    return {"rows": len(rows), "final_kimg": final_kimg}


def inspect_log(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        fail(f"cannot read training log {path}: {exc}")
    if "Exiting..." not in text:
        fail(f"training log does not record a clean exit: {path}")
    if "Traceback (most recent call last)" in text:
        fail(f"training log contains a traceback: {path}")
    return {"clean_exit_marker": "Exiting..."}


def inspect_state(path: Path, budget_kimg: int) -> dict[str, Any]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        fail(f"cannot load training state {path}: {exc}")
    if not isinstance(state, dict):
        fail(f"training state must be a dictionary: {path}")
    cur_nimg = state.get("cur_nimg")
    if not isinstance(cur_nimg, (int, float)) or cur_nimg < budget_kimg * 1000:
        fail(f"training state cur_nimg={cur_nimg!r} is below {budget_kimg * 1000}")
    stack: list[Any] = [state]
    tensors_checked = 0
    while stack:
        value = stack.pop()
        if isinstance(value, torch.Tensor):
            tensors_checked += 1
            if value.is_floating_point() or value.is_complex():
                if not torch.isfinite(value).all().item():
                    fail(f"training state contains a non-finite tensor: {path}")
        elif isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple, set)):
            stack.extend(value)
        elif isinstance(value, float) and not math.isfinite(value):
            fail(f"training state contains a non-finite scalar: {path}")
    return {"cur_nimg": int(cur_nimg), "tensors_checked": tensors_checked}


def build_receipt(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else run_dir / "network-snapshot-latest.pkl"
    paths = {
        "checkpoint": checkpoint,
        "training_options": run_dir / "training_options.json",
        "stats": run_dir / "stats.jsonl",
        "summary": run_dir / "train_summary.csv",
        "log": run_dir / "log.txt",
        "state": run_dir / "training-state-latest.pt",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        fail(f"run is missing required artifacts {missing}: {run_dir}")
    options = load_json(paths["training_options"], "training options")
    if options.get("total_kimg") != args.budget_kimg:
        fail(f"training_options total_kimg={options.get('total_kimg')!r}, expected {args.budget_kimg}")
    if options.get("seed") != args.training_seed:
        fail(f"training_options seed={options.get('seed')!r}, expected {args.training_seed}")
    if args.expected_training_commit:
        commit_file = run_dir / "commit_sha.txt"
        if not commit_file.is_file() or args.expected_training_commit not in commit_file.read_text(encoding="utf-8"):
            fail(f"run does not attest expected training commit {args.expected_training_commit}")
    stats = inspect_stats(paths["stats"], args.budget_kimg)
    summary = inspect_summary(paths["summary"], args.budget_kimg)
    log = inspect_log(paths["log"])
    state = inspect_state(paths["state"], args.budget_kimg)
    if abs(stats["final_kimg"] - summary["final_kimg"]) > 1e-6:
        fail("stats and training summary final kimg disagree")
    if abs(stats["final_kimg"] * 1000 - state["cur_nimg"]) > 1e-6:
        fail("stats and training state progress disagree")
    checkpoint_sha256 = sha256_file(checkpoint)
    return {
        "schema_version": 1,
        "status": "passed",
        "checkpoint_id": args.checkpoint_id,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "training_run_id": args.training_run_id,
        "method": args.method,
        "training_seed": args.training_seed,
        "budget_kimg": args.budget_kimg,
        "completion_passed": True,
        "logs_state_consistent": True,
        "finite_loss_state_passed": True,
        "checker_version": args.checker_version,
        "checker_git_commit": git_head(),
        "checked_at_unix": time.time(),
        "evidence": {
            "run_directory": str(run_dir),
            "training_options": str(paths["training_options"]),
            "stats": stats,
            "training_summary": summary,
            "log": log,
            "training_state": state,
            "expected_training_commit": args.expected_training_commit,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--budget-kimg", type=int, required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--expected-training-commit")
    parser.add_argument("--checker-version", default=CHECKER_VERSION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.training_run_id:
        fail("training_run_id must be non-empty")
    receipt = build_receipt(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
