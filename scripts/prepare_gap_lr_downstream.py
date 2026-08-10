#!/usr/bin/env python3
"""Validate and index the formal gap/LR artifacts for Role D and Role E.

The committed post-run receipt intentionally contains no absolute server
paths.  This server-side helper binds those logical run IDs to one explicit
experiment root, recomputes file hashes, and optionally deserializes the exact
states/snapshots that downstream jobs will consume.  It never modifies a
training artifact.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT = (
    REPO_ROOT / "results/gap_lr_matched/collaborator_training_state_receipt.json"
)
EXPERIMENT_ID = "gap_lr_matched_q128_s3_v1"
STATE_IDS = ("000001", "000002", "000004", "000008")
FINAL_STATE_ID = "000008"
REQUIRED_STATE_KEYS = {
    "attempted_iteration",
    "cur_nimg",
    "cur_tick",
    "elapsed_sec",
    "gradscaler_state",
    "loss_fn_state",
    "net",
    "optimizer_state",
    "successful_optimizer_steps",
    "tick_start_nimg",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_load(path: Path):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised on the server env.
        raise RuntimeError("deserialization requires the ECT PyTorch environment") from exc
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch 2.3 does not expose weights_only on every build.
        return torch.load(path, map_location="cpu")


def inspect_training_state(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    data = _torch_load(path)
    try:
        if not isinstance(data, dict):
            raise RuntimeError("training state payload is not a dict")
        missing = sorted(REQUIRED_STATE_KEYS - set(data))
        if missing:
            raise RuntimeError(f"training state missing keys: {missing}")
        optimizer = data["optimizer_state"]
        optimizer_states = optimizer.get("state", {})
        if len(optimizer_states) != 416:
            raise RuntimeError(
                f"expected 416 optimizer parameter states, got {len(optimizer_states)}"
            )
        required_optimizer_fields = {"step", "exp_avg", "exp_avg_sq"}
        bad_fields = sum(
            not required_optimizer_fields.issubset(parameter_state)
            for parameter_state in optimizer_states.values()
        )
        if bad_fields:
            raise RuntimeError(
                f"{bad_fields} optimizer parameter states lack step/exp_avg/exp_avg_sq"
            )
        actual_steps = int(data["successful_optimizer_steps"])
        expected_steps = int(expected["successful_optimizer_steps"])
        if actual_steps != expected_steps:
            raise RuntimeError(
                f"successful_optimizer_steps {actual_steps} != receipt {expected_steps}"
            )
        expected_nimg = int(round(float(expected["actual_kimg"]) * 1000))
        actual_nimg = int(data["cur_nimg"])
        if actual_nimg != expected_nimg:
            raise RuntimeError(f"cur_nimg {actual_nimg} != receipt {expected_nimg}")
        scaler = data["gradscaler_state"]
        if not isinstance(scaler, dict) or "scale" not in scaler:
            raise RuntimeError("GradScaler state is absent or malformed")
        return {
            "deserialized": True,
            "top_level_keys": sorted(data),
            "optimizer_param_state_count": len(optimizer_states),
            "successful_optimizer_steps": actual_steps,
            "cur_nimg": actual_nimg,
            "gradscaler_scale": float(scaler["scale"]),
        }
    finally:
        del data
        gc.collect()


def inspect_network_snapshot(path: Path, expected_gap: float) -> dict[str, Any]:
    # Formal snapshots are trusted experiment artifacts produced by this repo.
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    try:
        if not isinstance(payload, dict):
            raise RuntimeError("network snapshot payload is not a dict")
        if payload.get("ema") is None:
            raise RuntimeError("network snapshot has no EMA network")
        if payload.get("loss_fn") is None:
            raise RuntimeError("network snapshot has no loss_fn")
        if payload.get("augment_pipe") is not None:
            raise RuntimeError("formal snapshot unexpectedly enables augmentation")
        loss = payload["loss_fn"]
        schedule = getattr(loss, "schedule", None)
        name = getattr(schedule, "name", None)
        if name != "global_sigmoid":
            raise RuntimeError(f"expected global_sigmoid snapshot, got {name!r}")
        scale = float(getattr(schedule, "global_gap_scale", math.nan))
        if not math.isclose(scale, expected_gap, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"snapshot gap scale {scale} != registered arm gap {expected_gap}"
            )
        return {
            "deserialized": True,
            "ema_present": True,
            "loss_fn_present": True,
            "schedule_name": name,
            "global_gap_scale": scale,
            "stage": int(getattr(loss, "stage", -1)),
        }
    finally:
        del payload
        gc.collect()


def selected_states(scope: str, arms: dict[str, Any]):
    if scope == "role-d":
        return [("A", state_id) for state_id in STATE_IDS]
    if scope == "role-e":
        return [(arm, FINAL_STATE_ID) for arm in arms]
    return [(arm, state_id) for arm in arms for state_id in STATE_IDS]


def build_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    errors: list[str] = []
    if receipt.get("experiment_id") != EXPERIMENT_ID:
        errors.append("receipt experiment_id mismatch")
    if receipt.get("status") != "passed":
        errors.append("receipt status is not passed")

    source = receipt["source"]
    assets = {}
    for label, path, expected_hash in (
        ("dataset", args.data, source["dataset_sha256"]),
        ("transfer", args.transfer, source["transfer_sha256"]),
    ):
        record: dict[str, Any] = {"path": str(path.resolve()), "expected_sha256": expected_hash}
        if not path.is_file():
            record["status"] = "missing"
            errors.append(f"missing {label}: {path}")
        else:
            actual_hash = sha256_file(path)
            record.update(actual_sha256=actual_hash, status="passed" if actual_hash == expected_hash else "failed")
            if actual_hash != expected_hash:
                errors.append(f"{label} SHA256 mismatch")
        assets[label] = record

    artifact_rows = []
    arms = receipt["arms"]
    for arm, state_id in selected_states(args.scope, arms):
        arm_receipt = arms[arm]
        state_receipt = arm_receipt["states"][state_id]
        run_dir = args.experiment_root / arm_receipt["run_id"]
        state_path = run_dir / f"training-state-{state_id}.pt"
        snapshot_path = run_dir / f"network-snapshot-{state_id}.pkl"
        row: dict[str, Any] = {
            "arm": arm,
            "run_id": arm_receipt["run_id"],
            "gap_scale": float(arm_receipt["gap_scale"]),
            "learning_rate": float(arm_receipt["learning_rate"]),
            "state_id": state_id,
            "actual_kimg": float(state_receipt["actual_kimg"]),
            "training_state": str(state_path.resolve()),
            "network_snapshot": str(snapshot_path.resolve()),
            "expected_training_state_sha256": state_receipt["sha256"],
        }
        row_errors = []
        if not state_path.is_file():
            row_errors.append("missing training state")
        if not snapshot_path.is_file():
            row_errors.append("missing network snapshot")
        if not row_errors:
            state_hash = sha256_file(state_path)
            snapshot_hash = sha256_file(snapshot_path)
            row["training_state_sha256"] = state_hash
            row["network_snapshot_sha256"] = snapshot_hash
            if state_hash != state_receipt["sha256"]:
                row_errors.append("training state SHA256 mismatch")
            if args.deserialize:
                try:
                    row["training_state_inspection"] = inspect_training_state(
                        state_path, state_receipt
                    )
                except Exception as exc:  # fail closed while retaining the full index.
                    row_errors.append(f"training state deserialize failed: {exc}")
                try:
                    row["network_snapshot_inspection"] = inspect_network_snapshot(
                        snapshot_path, float(arm_receipt["gap_scale"])
                    )
                except Exception as exc:
                    row_errors.append(f"network snapshot deserialize failed: {exc}")
        row["status"] = "passed" if not row_errors else "failed"
        row["errors"] = row_errors
        errors.extend(f"{arm}/{state_id}: {error}" for error in row_errors)
        artifact_rows.append(row)

    role_d = (
        [row for row in artifact_rows if row["arm"] == "A"]
        if args.scope in {"role-d", "all"} else []
    )
    role_e = (
        [row for row in artifact_rows if row["state_id"] == FINAL_STATE_ID]
        if args.scope in {"role-e", "all"} else []
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "scope": args.scope,
        "status": "passed" if not errors else "failed",
        "read_only_verification": True,
        "experiment_root": str(args.experiment_root.resolve()),
        "assets": assets,
        "artifacts": artifact_rows,
        "role_d_inputs": role_d,
        "role_d_contract": {
            "reference_arm": "A",
            "state_ids": list(STATE_IDS),
            "same_minibatch_t_noise_dropout_seed": 20260810,
            "mix_arms_across_k": False,
        },
        "role_e_inputs": role_e,
        "role_e_contract": {
            "checkpoint_state_id": FINAL_STATE_ID,
            "nfe": 1,
            "metrics": ["fid5k_full", "kid5k_full"],
            "sample_seeds": "0-4999",
            "evaluator_seed": 20260730,
            "precision": "fp32",
        },
        "errors": errors,
    }
    return manifest, errors


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--transfer", required=True, type=Path)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--scope", choices=("role-d", "role-e", "all"), default="all")
    parser.add_argument(
        "--deserialize", action=argparse.BooleanOptionalAction, default=True,
        help="load states/snapshots in the ECT environment (default: true)",
    )
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest, errors = build_manifest(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{manifest['status']}: wrote {args.out}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
