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

from scripts import gap_lr_seed_replication_contract as audit_contract


ARM_CONTRACT = {
    "A": (1.0, 0.0001),
    "B": (1.3, 0.0001),
    "C": (1.3, 0.00012963523762588692),
}
NUMBERED_IDS = [f"{index:06d}" for index in range(1, 9)]
EXPECTED_ATTEMPTED_ITERATIONS = 2000
MAX_AMP_SKIPS = 16


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


def expected_options(run_dir: Path, arm: str, seed: int) -> dict[str, Any]:
    gap, lr = ARM_CONTRACT[arm]
    return {
        "adaptive_update_kimg": 0.5,
        "batch_gpu": 16,
        "batch_size": 128,
        "ckpt_ticks": 1,
        "cudnn_benchmark": True,
        "data_loader_kwargs": {
            "num_workers": 1,
            "pin_memory": True,
            "prefetch_factor": 2,
        },
        "dataset_kwargs": {
            "cache": True,
            "class_name": "training.dataset.ImageFolderDataset",
            "max_size": 50000,
            "path": "/data/raw/ECT/datasets/cifar10-32x32.zip",
            "resolution": 32,
            "use_labels": False,
            "xflip": False,
        },
        "double_ticks": 10000,
        "ema_beta": 0.9993,
        "ema_halflife_kimg": None,
        "ema_rampup_ratio": None,
        "enable_amp": True,
        "enable_tf32": False,
        "eval_ticks": 50,
        "kimg_per_tick": 32.0,
        "loss_kwargs": {
            "P_mean": -1.1,
            "P_std": 2.0,
            "adaptive_loss_ema_beta": 0.9,
            "adaptive_max_adjust": 0.05,
            "adaptive_min_gap": 0.001,
            "adaptive_warmup_updates": 2,
            "adj": "global_sigmoid",
            "b": 1.0,
            "c": 0.0,
            "class_name": "training.loss.ECMLoss",
            "global_gap_scale": gap,
            "k": 8.0,
            "local_tbin_deadband": 0.02,
            "local_tbin_gain": 0.5,
            "local_tbin_long_beta": 0.99,
            "local_tbin_max_scale": 1.5,
            "local_tbin_min_gap": 0.001,
            "local_tbin_min_scale": 0.75,
            "local_tbin_num_bins": 4,
            "local_tbin_short_beta": 0.9,
            "local_tbin_warmup_updates": 32,
            "q": 128.0,
        },
        "loss_scaling": 1.0,
        "metrics": [],
        "mid_t": [0.821],
        "network_kwargs": {
            "channel_mult": [2, 2, 2],
            "channel_mult_noise": 1,
            "class_name": "training.networks.ECMPrecond",
            "decoder_type": "standard",
            "dropout": 0.2,
            "embedding_type": "positional",
            "encoder_type": "standard",
            "model_channels": 128,
            "model_type": "SongUNet",
            "resample_filter": [1, 1],
            "use_fp16": True,
        },
        "optimizer_kwargs": {
            "betas": [0.9, 0.999],
            "class_name": "torch.optim.RAdam",
            "eps": 1e-08,
            "lr": lr,
        },
        "resume_pkl": (
            "/data/raw/ECT/pretrained/edm-cifar10-32x32-uncond-vp.pkl"
        ),
        "run_dir": str(run_dir),
        "sample_ticks": 9999,
        "seed": seed,
        "snapshot_ticks": 1,
        "state_dump_ticks": 1,
        "total_kimg": 256,
    }


def validate_options(options: dict[str, Any], run_dir: Path, arm: str, seed: int) -> None:
    expected = expected_options(run_dir, arm, seed)
    if not audit_contract.exact_json_equal(options, expected):
        fail("training_options differ from the exact frozen arm/seed contract")


def validate_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_ATTEMPTED_ITERATIONS:
        fail(
            f"train_summary.csv has {len(rows)} rows, expected "
            f"{EXPECTED_ATTEMPTED_ITERATIONS}"
        )
    skipped_steps = 0
    successful_steps = 0
    final_grad_scale = None
    for index, row in enumerate(rows, start=1):
        finite(row.get("loss"), f"summary loss row {index}")
        attempted_value = finite(
            row.get("attempted_iteration"), f"attempted row {index}"
        )
        successful_value = finite(
            row.get("successful_optimizer_steps"), f"successful row {index}"
        )
        skipped_value = finite(row.get("step_skipped"), f"step_skipped row {index}")
        if not all(
            value.is_integer()
            for value in (attempted_value, successful_value, skipped_value)
        ):
            fail(f"summary counters row {index} must be exact integers")
        attempted = int(attempted_value)
        successful = int(successful_value)
        skipped = int(skipped_value)
        grad_scale = finite(row.get("grad_scale"), f"grad_scale row {index}")
        if attempted != index:
            fail(f"attempted_iteration row {index} is {attempted}, expected {index}")
        if skipped not in (0, 1):
            fail(f"step_skipped row {index} must be 0 or 1")
        if grad_scale <= 0:
            fail(f"grad_scale row {index} must be positive")
        skipped_steps += skipped
        successful_steps += 1 - skipped
        if successful != successful_steps:
            fail(
                f"successful_optimizer_steps row {index} is {successful}, "
                f"expected {successful_steps}"
            )
        if row.get("schedule") != "global_sigmoid":
            fail(f"schedule row {index} is not global_sigmoid")
        final_grad_scale = grad_scale
    if skipped_steps > MAX_AMP_SKIPS:
        fail(f"AMP skipped {skipped_steps} steps, maximum is {MAX_AMP_SKIPS}")
    final_kimg = finite(rows[-1].get("processed_kimg"), "final processed_kimg")
    if final_kimg != 256.0:
        fail(f"final processed_kimg={final_kimg}, expected 256.0")
    return {
        "rows": len(rows),
        "final_processed_kimg": final_kimg,
        "attempted_iterations": EXPECTED_ATTEMPTED_ITERATIONS,
        "successful_optimizer_steps": successful_steps,
        "amp_skipped_steps": skipped_steps,
        "max_allowed_amp_skips": MAX_AMP_SKIPS,
        "final_gradscaler_scale": final_grad_scale,
        "amp_contract_passed": True,
    }


def validate_stats(path: Path) -> dict[str, Any]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = audit_contract.loads_strict(line)
            if not isinstance(value, dict):
                fail("stats row is not a JSON object")
            records.append(value)
    if not records:
        fail("stats.jsonl is empty")
    for index, row in enumerate(records):
        finite(row["Loss/loss"]["mean"], f"stats loss row {index}")
    final_kimg = finite(records[-1]["Progress/kimg"]["mean"], "stats final kimg")
    if abs(final_kimg - 256.0) * 1000 > 1.0:
        fail(f"stats final kimg={final_kimg}, expected 256")
    return {"records": len(records), "final_kimg": final_kimg}


def validate_state(path: Path, expected_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        fail("final training state is not a dictionary")
    cur_nimg = state.get("cur_nimg")
    if (
        type(cur_nimg) not in (int, float)
        or not math.isfinite(cur_nimg)
        or not float(cur_nimg).is_integer()
        or int(cur_nimg) != 256000
    ):
        fail(f"final state cur_nimg={cur_nimg!r}, expected integral 256000")
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

    result = {
        "cur_nimg": int(cur_nimg),
        "attempted_iteration": scalar(state.get("attempted_iteration")),
        "successful_optimizer_steps": scalar(state.get("successful_optimizer_steps")),
        "gradscaler_scale": scalar(scaler.get("scale")),
        "optimizer_parameter_states": len(state["optimizer_state"].get("state", {})),
        "tensors_checked": tensors_checked,
    }
    if expected_summary is not None:
        expected_pairs = {
            "attempted_iteration": expected_summary["attempted_iterations"],
            "successful_optimizer_steps": expected_summary["successful_optimizer_steps"],
            "gradscaler_scale": expected_summary["final_gradscaler_scale"],
        }
        for key, expected in expected_pairs.items():
            if result[key] != expected:
                fail(f"final state {key}={result[key]!r}, expected {expected!r}")
        if result["optimizer_parameter_states"] != 416:
            fail(
                "final state optimizer parameter-state count is "
                f"{result['optimizer_parameter_states']}, expected 416"
            )
    return result


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
    options = audit_contract.load_json_object(paths["training_options"])
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
    state = validate_state(paths["final_training_state"], summary)
    snapshot = validate_snapshot(paths["final_ema_snapshot"], gap)
    hashes = {name: sha256_file(path) for name, path in sorted(paths.items())}
    sizes = {name: path.stat().st_size for name, path in sorted(paths.items())}
    receipt = {
        "schema_version": 2,
        "receipt_type": "gap_lr_seed_replication_run_integrity",
        "status": "passed",
        "experiment_id": "gap_lr_matched_q128_s45_replication_v1",
        "arm": args.arm,
        "seed": args.seed,
        "run_dir": str(run_dir),
        "gap_scale": gap,
        "learning_rate": lr,
        "execution_protocol_commit": args.protocol_commit,
        "protocol_commit": args.protocol_commit,
        "training_code_commit": args.training_code_commit,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "verifier": {
            "path": "scripts/verify_gap_lr_seed_replication_run.py",
            "source_sha256": sha256_file(Path(__file__)),
        },
        "completion": {"budget_kimg": 256, "summary": summary, "stats": stats},
        "final_training_state": state,
        "final_ema_snapshot": snapshot,
        "artifact_sha256": hashes,
        "artifact_size_bytes": sizes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        fail(str(exc))
