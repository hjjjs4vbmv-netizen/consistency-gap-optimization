#!/usr/bin/env python3
"""Verify one completed A/B/C seed-replication run and hash its artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ARM_CONTRACT = {
    "A": (1.0, 0.0001),
    "B": (1.3, 0.0001),
    "C": (1.3, 0.00012963523762588692),
}
NUMBERED_IDS = [f"{index:06d}" for index in range(1, 9)]


def fail(message: str) -> None:
    raise SystemExit("SEED REPLICATION RUN REJECTED: " + message)


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        fail(f"{label} must be numeric: {exc}")
    if not math.isfinite(result):
        fail(f"{label} is non-finite")
    return result


def close(value: Any, expected: float, label: str, tolerance: float = 1e-18) -> None:
    observed = finite(value, label)
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance):
        fail(f"{label}={observed!r}, expected {expected!r}")


def require_files(run_dir: Path) -> dict[str, Path]:
    paths = {
        "training_options": run_dir / "training_options.json",
        "stats": run_dir / "stats.jsonl",
        "train_summary": run_dir / "train_summary.csv",
        "log": run_dir / "log.txt",
        "model_init_image": run_dir / "model_init.png",
        "data_image": run_dir / "data.png",
        "final_ema_snapshot": run_dir / "network-snapshot-latest.pkl",
        "final_training_state": run_dir / "training-state-latest.pt",
        "protocol_commit": run_dir / "protocol_commit.txt",
        "training_code_commit": run_dir / "training_code_commit.txt",
        "source_audit_receipt_sha256": run_dir / "source_audit_receipt_sha256.txt",
    }
    for artifact_id in NUMBERED_IDS:
        paths[f"network_snapshot_{artifact_id}"] = run_dir / f"network-snapshot-{artifact_id}.pkl"
        paths[f"training_state_{artifact_id}"] = run_dir / f"training-state-{artifact_id}.pt"
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        fail(f"missing required artifacts: {missing}")
    return paths


def validate_options(options: dict[str, Any], run_dir: Path, arm: str, seed: int) -> None:
    expected_top = {
        "batch_gpu": 16,
        "batch_size": 128,
        "double_ticks": 10000,
        "ema_beta": 0.9993,
        "enable_amp": True,
        "enable_tf32": False,
        "metrics": [],
        "sample_ticks": 9999,
        "seed": seed,
        "snapshot_ticks": 1,
        "state_dump_ticks": 1,
        "ckpt_ticks": 1,
        "total_kimg": 256,
        "run_dir": str(run_dir),
    }
    for key, expected in expected_top.items():
        if options.get(key) != expected:
            fail(f"training_options.{key}={options.get(key)!r}, expected {expected!r}")
    network = options.get("network_kwargs", {})
    if network.get("use_fp16") is not True or network.get("dropout") != 0.2:
        fail("network FP16/dropout contract changed")
    loss = options.get("loss_kwargs", {})
    expected_loss = {"adj": "global_sigmoid", "q": 128.0, "k": 8.0, "b": 1.0, "c": 0.0}
    for key, expected in expected_loss.items():
        if loss.get(key) != expected:
            fail(f"loss_kwargs.{key} changed")
    gap, lr = ARM_CONTRACT[arm]
    close(loss.get("global_gap_scale"), gap, "global gap scale")
    optimizer = options.get("optimizer_kwargs", {})
    if optimizer.get("class_name") != "torch.optim.RAdam":
        fail("optimizer is not RAdam")
    close(optimizer.get("lr"), lr, "learning rate")
    if optimizer.get("betas") != [0.9, 0.999] or optimizer.get("eps") != 1e-08:
        fail("RAdam hyperparameters changed")
    dataset = options.get("dataset_kwargs", {})
    if dataset.get("path") != "/data/raw/ECT/datasets/cifar10-32x32.zip":
        fail("dataset path changed")
    if options.get("resume_pkl") != "/data/raw/ECT/pretrained/edm-cifar10-32x32-uncond-vp.pkl":
        fail("transfer checkpoint path changed")


def validate_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        fail("train_summary.csv is empty")
    for index, row in enumerate(rows):
        finite(row.get("loss"), f"summary loss row {index}")
    final_kimg = finite(rows[-1].get("processed_kimg"), "final processed_kimg")
    if final_kimg != 256.0:
        fail(f"final processed_kimg={final_kimg}, expected 256.0")
    return {"rows": len(rows), "final_processed_kimg": final_kimg}


def validate_stats(path: Path) -> dict[str, Any]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        fail("stats.jsonl is empty")
    for index, row in enumerate(records):
        finite(row["Loss/loss"]["mean"], f"stats loss row {index}")
    final_kimg = finite(records[-1]["Progress/kimg"]["mean"], "stats final kimg")
    if abs(final_kimg - 256.0) * 1000 > 1.0:
        fail(f"stats final kimg={final_kimg}, expected 256")
    return {"records": len(records), "final_kimg": final_kimg}


def validate_state(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        fail("final training state is not a dictionary")
    if state.get("cur_nimg") != 256000:
        fail(f"final state cur_nimg={state.get('cur_nimg')!r}, expected 256000")
    for key in ("net", "optimizer_state", "gradscaler_state", "loss_fn_state"):
        if key not in state:
            fail(f"final state lacks {key}")
    stack = [state]
    tensors_checked = 0
    while stack:
        value = stack.pop()
        if isinstance(value, torch.Tensor):
            tensors_checked += 1
            if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
                fail("final state contains a non-finite tensor")
        elif isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
        elif isinstance(value, float) and not math.isfinite(value):
            fail("final state contains a non-finite scalar")
    scaler = state.get("gradscaler_state", {})

    def scalar(value: Any) -> Any:
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return value.item()
        return value

    return {
        "cur_nimg": state["cur_nimg"],
        "attempted_iteration": scalar(state.get("attempted_iteration")),
        "successful_optimizer_steps": scalar(state.get("successful_optimizer_steps")),
        "gradscaler_scale": scalar(scaler.get("scale")),
        "optimizer_parameter_states": len(state["optimizer_state"].get("state", {})),
        "tensors_checked": tensors_checked,
    }


def validate_snapshot(path: Path, expected_gap: float) -> dict[str, Any]:
    with path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if not isinstance(checkpoint, dict):
        fail("final snapshot is not a dictionary")
    ema = checkpoint.get("ema")
    if not isinstance(ema, torch.nn.Module):
        fail("final snapshot lacks EMA module")
    tensors_checked = 0
    for name, tensor in list(ema.named_parameters()) + list(ema.named_buffers()):
        if tensor.is_floating_point() or tensor.is_complex():
            tensors_checked += 1
            if not torch.isfinite(tensor).all().item():
                fail(f"EMA tensor {name} is non-finite")
    loss_fn = checkpoint.get("loss_fn")
    schedule = getattr(loss_fn, "schedule", None)
    if getattr(schedule, "name", None) != "global_sigmoid":
        fail("snapshot schedule is not global_sigmoid")
    close(getattr(schedule, "global_gap_scale", None), expected_gap, "snapshot gap scale", tolerance=1e-12)
    return {"ema_present": True, "ema_finite": True, "ema_tensors_checked": tensors_checked}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=sorted(ARM_CONTRACT))
    parser.add_argument("--seed", required=True, type=int, choices=[4, 5])
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--training-code-commit", required=True)
    parser.add_argument("--source-audit-receipt-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    paths = require_files(run_dir)
    options = json.loads(paths["training_options"].read_text(encoding="utf-8"))
    validate_options(options, run_dir, args.arm, args.seed)
    if paths["protocol_commit"].read_text(encoding="utf-8").strip() != args.protocol_commit:
        fail("per-run protocol commit mismatch")
    if paths["training_code_commit"].read_text(encoding="utf-8").strip() != args.training_code_commit:
        fail("per-run training code commit mismatch")
    if paths["source_audit_receipt_sha256"].read_text(encoding="utf-8").strip() != args.source_audit_receipt_sha256:
        fail("per-run source audit receipt hash mismatch")
    log_text = paths["log"].read_text(encoding="utf-8", errors="replace")
    if "Exiting..." not in log_text or "Traceback (most recent call last)" in log_text:
        fail("training log does not record a clean, traceback-free exit")

    gap, lr = ARM_CONTRACT[args.arm]
    summary = validate_summary(paths["train_summary"])
    stats = validate_stats(paths["stats"])
    state = validate_state(paths["final_training_state"])
    snapshot = validate_snapshot(paths["final_ema_snapshot"], gap)
    hashes = {name: sha256_file(path) for name, path in sorted(paths.items())}
    sizes = {name: path.stat().st_size for name, path in sorted(paths.items())}
    receipt = {
        "schema_version": 1,
        "receipt_type": "gap_lr_seed_replication_run_integrity",
        "status": "passed",
        "experiment_id": "gap_lr_matched_q128_s45_replication_v1",
        "arm": args.arm,
        "seed": args.seed,
        "run_dir": str(run_dir),
        "gap_scale": gap,
        "learning_rate": lr,
        "protocol_commit": args.protocol_commit,
        "training_code_commit": args.training_code_commit,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "completion": {"budget_kimg": 256, "summary": summary, "stats": stats},
        "final_training_state": state,
        "final_ema_snapshot": snapshot,
        "artifact_sha256": hashes,
        "artifact_size_bytes": sizes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        fail(str(exc))
