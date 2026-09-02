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
import csv
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
ROLE_D_ARM = "A"
ROLE_D_RUN_LOG = Path("logs/A.log")
ROLE_D_LAUNCH_PROVENANCE = Path("launch_provenance.txt")
ROLE_D_RUN_PROVENANCE = ("train_summary.csv", "training_options.json")
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
    print(f"hashing: {path}", flush=True)
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
    print(f"deserializing training state: {path}", flush=True)
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
        parameter_steps = {
            int(parameter_state["step"].item())
            if hasattr(parameter_state["step"], "item")
            else int(parameter_state["step"])
            for parameter_state in optimizer_states.values()
        }
        if parameter_steps != {actual_steps}:
            raise RuntimeError(
                "optimizer parameter steps do not all match "
                f"successful_optimizer_steps {actual_steps}: {sorted(parameter_steps)}"
            )
        expected_nimg = int(round(float(expected["actual_kimg"]) * 1000))
        actual_nimg = int(data["cur_nimg"])
        if actual_nimg != expected_nimg:
            raise RuntimeError(f"cur_nimg {actual_nimg} != receipt {expected_nimg}")
        scaler = data["gradscaler_state"]
        if not isinstance(scaler, dict) or "scale" not in scaler:
            raise RuntimeError("GradScaler state is absent or malformed")
        scaler_scale = float(scaler["scale"])
        expected_scaler_scale = float(expected["gradscaler_scale"])
        if not math.isclose(
            scaler_scale, expected_scaler_scale, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(
                f"GradScaler scale {scaler_scale} != receipt {expected_scaler_scale}"
            )
        return {
            "deserialized": True,
            "top_level_keys": sorted(data),
            "optimizer_param_state_count": len(optimizer_states),
            "successful_optimizer_steps": actual_steps,
            "cur_nimg": actual_nimg,
            "optimizer_parameter_steps": sorted(parameter_steps),
            "gradscaler_scale": scaler_scale,
        }
    finally:
        del data
        gc.collect()


def inspect_network_snapshot(path: Path, expected_gap: float) -> dict[str, Any]:
    # Formal snapshots are trusted experiment artifacts produced by this repo.
    print(f"deserializing network snapshot: {path}", flush=True)
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


def index_file(path: Path) -> tuple[dict[str, Any], list[str]]:
    record: dict[str, Any] = {"path": str(path.resolve())}
    if not path.is_file():
        record["status"] = "missing"
        return record, [f"missing provenance file: {path}"]
    record.update(
        status="passed",
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )
    return record, []


def inspect_train_summary(
    path: Path, expected_states: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"row_count": 0, "checkpoint_rows": {}}, ["train summary is empty"]
    required_columns = {
        "attempted_iteration",
        "successful_optimizer_steps",
        "processed_kimg",
        "schedule",
    }
    missing_columns = sorted(required_columns - set(rows[0]))
    if missing_columns:
        return {
            "row_count": len(rows),
            "checkpoint_rows": {},
        }, [f"train summary missing columns: {missing_columns}"]

    checkpoint_rows: dict[str, Any] = {}
    for state_id in STATE_IDS:
        expected = expected_states[state_id]
        expected_kimg = float(expected["actual_kimg"])
        matches = [
            row
            for row in rows
            if math.isclose(
                float(row["processed_kimg"]), expected_kimg, rel_tol=0.0, abs_tol=5e-7
            )
        ]
        if len(matches) != 1:
            errors.append(
                f"train summary has {len(matches)} rows for state {state_id} "
                f"at {expected_kimg} kimg"
            )
            continue
        row = matches[0]
        actual_steps = int(row["successful_optimizer_steps"])
        expected_steps = int(expected["successful_optimizer_steps"])
        if actual_steps != expected_steps:
            errors.append(
                f"train summary state {state_id} successful steps "
                f"{actual_steps} != receipt {expected_steps}"
            )
        if row["schedule"] != "global_sigmoid":
            errors.append(
                f"train summary state {state_id} schedule {row['schedule']!r} "
                "!= 'global_sigmoid'"
            )
        checkpoint_rows[state_id] = {
            "attempted_iteration": int(row["attempted_iteration"]),
            "successful_optimizer_steps": actual_steps,
            "processed_kimg": float(row["processed_kimg"]),
            "schedule": row["schedule"],
        }
    return {
        "row_count": len(rows),
        "checkpoint_rows": checkpoint_rows,
        "final_processed_kimg": float(rows[-1]["processed_kimg"]),
        "final_successful_optimizer_steps": int(
            rows[-1]["successful_optimizer_steps"]
        ),
    }, errors


def inspect_training_options(
    path: Path, run_dir: Path, arm_receipt: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    options = json.loads(path.read_text(encoding="utf-8"))
    recorded_run_dir = Path(str(options.get("run_dir", "")))
    if (
        not recorded_run_dir.is_absolute()
        or recorded_run_dir.resolve() != run_dir.resolve()
    ):
        errors.append(
            "training_options.run_dir does not identify the selected formal Arm A run"
        )
    loss_kwargs = options.get("loss_kwargs", {})
    if loss_kwargs.get("adj") != "global_sigmoid":
        errors.append("training_options loss_kwargs.adj is not global_sigmoid")
    gap = float(loss_kwargs.get("global_gap_scale", math.nan))
    expected_gap = float(arm_receipt["gap_scale"])
    if not math.isclose(gap, expected_gap, rel_tol=0.0, abs_tol=1e-12):
        errors.append(
            f"training_options global_gap_scale {gap} != receipt {expected_gap}"
        )
    optimizer_kwargs = options.get("optimizer_kwargs", {})
    lr = float(optimizer_kwargs.get("lr", math.nan))
    expected_lr = float(arm_receipt["learning_rate"])
    if not math.isclose(lr, expected_lr, rel_tol=0.0, abs_tol=1e-15):
        errors.append(f"training_options lr {lr} != receipt {expected_lr}")
    return {
        "run_dir": str(recorded_run_dir),
        "schedule": loss_kwargs.get("adj"),
        "global_gap_scale": gap,
        "learning_rate": lr,
        "seed": options.get("seed"),
        "total_kimg": options.get("total_kimg"),
        "batch_size": options.get("batch_size"),
        "batch_gpu": options.get("batch_gpu"),
        "enable_amp": options.get("enable_amp"),
    }, errors


def build_role_d_provenance(
    args: argparse.Namespace, arm_receipt: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    run_dir = args.experiment_root / arm_receipt["run_id"]
    paths = {
        "train_summary": run_dir / ROLE_D_RUN_PROVENANCE[0],
        "training_options": run_dir / ROLE_D_RUN_PROVENANCE[1],
        "run_log": args.experiment_root / ROLE_D_RUN_LOG,
        "launch_provenance": args.experiment_root / ROLE_D_LAUNCH_PROVENANCE,
    }
    if args.launcher_log is None:
        errors.append("--launcher-log is required for Role D provenance")
    else:
        paths["launcher_log"] = args.launcher_log

    files = {}
    for label, path in paths.items():
        record, file_errors = index_file(path)
        files[label] = record
        errors.extend(file_errors)

    summary_inspection = None
    if files["train_summary"]["status"] == "passed":
        summary_inspection, summary_errors = inspect_train_summary(
            paths["train_summary"], arm_receipt["states"]
        )
        errors.extend(summary_errors)

    options_inspection = None
    if files["training_options"]["status"] == "passed":
        options_inspection, options_errors = inspect_training_options(
            paths["training_options"], run_dir, arm_receipt
        )
        errors.extend(options_errors)

    if files["run_log"]["status"] == "passed":
        log_text = paths["run_log"].read_text(encoding="utf-8", errors="replace")
        fatal_markers = [
            marker
            for marker in ("Traceback (most recent call last)", "OutOfMemoryError")
            if marker in log_text
        ]
        files["run_log"]["fatal_markers"] = fatal_markers
        if fatal_markers:
            errors.append(f"Arm A run log contains fatal markers: {fatal_markers}")

    if "launcher_log" in files and files["launcher_log"]["status"] == "passed":
        launcher_text = paths["launcher_log"].read_text(
            encoding="utf-8", errors="replace"
        )
        completed = "ALL FORMAL ARMS COMPLETE" in launcher_text
        files["launcher_log"]["completion_marker_present"] = completed
        if not completed:
            errors.append("launcher log lacks ALL FORMAL ARMS COMPLETE")

    return {
        "owner": "Collaborator",
        "consumer": "Role D",
        "trajectory_id": arm_receipt["run_id"],
        "run_dir": str(run_dir.resolve()),
        "single_uninterrupted_run": True,
        "reference_arm": ROLE_D_ARM,
        "gap_scale": float(arm_receipt["gap_scale"]),
        "state_ids": list(STATE_IDS),
        "actual_kimg": [
            float(arm_receipt["states"][state_id]["actual_kimg"])
            for state_id in STATE_IDS
        ],
        "files": files,
        "train_summary_inspection": summary_inspection,
        "training_options_inspection": options_inspection,
        "role_d_may_substitute_or_mix_runs": False,
        "status": "passed" if not errors else "failed",
    }, errors


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
    role_d_provenance = None
    if args.scope in {"role-d", "all"}:
        role_d_provenance, provenance_errors = build_role_d_provenance(
            args, arms[ROLE_D_ARM]
        )
        errors.extend(f"Role D provenance: {error}" for error in provenance_errors)
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
        "role_d_provenance": role_d_provenance,
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
    parser.add_argument(
        "--launcher-log",
        type=Path,
        help="detached formal launcher log; required by the Role D launcher",
    )
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
