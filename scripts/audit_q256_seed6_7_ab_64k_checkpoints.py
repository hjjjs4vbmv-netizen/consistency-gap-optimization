#!/usr/bin/env python3
"""Audit seed6/7 A/B 256->1024k continuation checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import stat
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from training import q256_budget_checkpoints as schedule
from training import reproducibility


SEEDS = (6, 7)
ARMS = {"A": (1.0, 1.0), "B": (1.1, 1.1)}
SOURCE_KIMG = 256
SOURCE_ATTEMPTS = 2000
SCHEMA_SOURCE = "ect.q256.seed6-7-ab-source-state-audit/v1"
SCHEMA_INVENTORY = "ect.q256.seed6-7-ab-64k-checkpoint-inventory/v1"
CHECKPOINT_SCHEMA = "ect.q256.seed6-7-ab-budget-checkpoint/v1"
CLASSIFICATION = "secondary_precision_extension_not_original_preregistration"
REQUIRED_STATE_FIELDS = {
    "net",
    "ema",
    "optimizer_state",
    "gradscaler_state",
    "rank_states",
    "cur_nimg",
    "attempted_iteration",
    "successful_optimizer_steps",
    "factorial",
    "trajectory_config",
    "trajectory_config_sha256",
    "reproducibility_schema",
    "loss_fn_state",
    "snapshot_grid_z",
    "snapshot_grid_c",
    "snapshot_grid_size",
}


class AuditError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise AuditError(message)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        fail(f"missing regular JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        fail(f"JSON root is not an object: {path}")
    return value


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o640,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            fail(f"refuse to overwrite receipt: {path}")
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        fail(f"missing regular {label}: {path}")
    return path


def load_state(path: Path) -> dict[str, Any]:
    regular_file(path, "training state")
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        fail(f"training state is not a dictionary: {path}")
    return state


def validate_rank_states(rank_states: Any, path: Path) -> None:
    if not isinstance(rank_states, list) or len(rank_states) != 1:
        fail(f"state must contain one rank state: {path}")
    rank = rank_states[0]
    if not isinstance(rank, dict):
        fail(f"rank state is not a dictionary: {path}")
    required = {"rank", "world_size", "rng_state", "sampler_state"}
    if not required.issubset(rank):
        fail(f"rank state lacks RNG/sampler fields: {path}")
    if rank["rank"] != 0 or rank["world_size"] != 1:
        fail(f"rank/world-size identity mismatch: {path}")


def validate_state(
    state: dict[str, Any],
    *,
    path: Path,
    seed: int,
    arm: str,
    budget_kimg: int,
    expected_total_kimg: int,
) -> dict[str, Any]:
    missing = sorted(REQUIRED_STATE_FIELDS - set(state))
    if missing:
        fail(f"training state lacks {missing}: {path}")
    expected_nimg = budget_kimg * 1000
    expected_attempts = expected_nimg // 128
    if state["cur_nimg"] != expected_nimg:
        fail(f"cur_nimg mismatch at {path}: {state['cur_nimg']} != {expected_nimg}")
    if state["attempted_iteration"] != expected_attempts:
        fail(
            f"attempted_iteration mismatch at {path}: "
            f"{state['attempted_iteration']} != {expected_attempts}"
        )
    accepted = state["successful_optimizer_steps"]
    if isinstance(accepted, bool) or not isinstance(accepted, int):
        fail(f"successful_optimizer_steps is invalid: {path}")
    if accepted <= 0 or accepted > expected_attempts:
        fail(f"successful_optimizer_steps is out of range: {path}")
    if not isinstance(state["optimizer_state"], dict) or not state["optimizer_state"]:
        fail(f"optimizer state is missing: {path}")
    if not isinstance(state["gradscaler_state"], dict) or not state["gradscaler_state"]:
        fail(f"GradScaler state is missing: {path}")
    if state["net"] is None or state["ema"] is None:
        fail(f"online/EMA model is missing: {path}")
    validate_rank_states(state["rank_states"], path)
    if state["reproducibility_schema"] != reproducibility.TRAINING_STATE_SCHEMA:
        fail(f"reproducibility schema mismatch: {path}")
    trajectory = state["trajectory_config"]
    if not isinstance(trajectory, dict):
        fail(f"trajectory config is missing: {path}")
    if reproducibility.state_sha256(trajectory) != state["trajectory_config_sha256"]:
        fail(f"trajectory config digest mismatch: {path}")
    if trajectory.get("seed") != seed or trajectory.get("total_kimg") != expected_total_kimg:
        fail(f"trajectory seed/total budget mismatch: {path}")
    factorial = state["factorial"]
    target_scale, denominator_scale = ARMS[arm]
    if (
        not isinstance(factorial, dict)
        or factorial.get("protocol") != "q256_target_weight_v1"
        or factorial.get("arm") != arm
        or float(factorial.get("target_gap_scale")) != target_scale
        or float(factorial.get("denominator_gap_scale")) != denominator_scale
    ):
        fail(f"factorial identity mismatch: {path}")
    return {
        "attempted_iteration": expected_attempts,
        "successful_optimizer_steps": accepted,
        "amp_skips": expected_attempts - accepted,
        "cur_nimg": expected_nimg,
        "trajectory_config_sha256": state["trajectory_config_sha256"],
        "ema_state_sha256": reproducibility.module_state_sha256(state["ema"]),
        "state_fields": sorted(state),
    }


def validate_snapshot(path: Path, expected_ema_sha256: str) -> dict[str, Any]:
    regular_file(path, "EMA snapshot")
    with path.open("rb") as handle:
        snapshot = pickle.load(handle)
    if not isinstance(snapshot, dict) or snapshot.get("ema") is None:
        fail(f"snapshot lacks EMA: {path}")
    observed = reproducibility.module_state_sha256(snapshot["ema"])
    if observed != expected_ema_sha256:
        fail(f"full-state/snapshot EMA mismatch: {path}")
    return {"ema_state_sha256": observed, "snapshot_fields": sorted(snapshot)}


def source_command(args: argparse.Namespace) -> None:
    root = args.source_root.resolve(strict=True)
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            run_dir = root / f"seed{seed}" / f"arm{arm}"
            state_path = run_dir / "training-state-latest.pt"
            state = load_state(state_path)
            state_record = validate_state(
                state,
                path=state_path,
                seed=seed,
                arm=arm,
                budget_kimg=SOURCE_KIMG,
                expected_total_kimg=SOURCE_KIMG,
            )
            initial = load_json(run_dir / "initial_state_receipt_v1.json")
            if initial.get("seed") != seed or initial.get("factorial", {}).get("arm") != arm:
                fail(f"initial-state seed/arm mismatch: {run_dir}")
            state_sha256 = sha256_file(state_path)
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "run_dir": str(run_dir),
                    "source_state": str(state_path),
                    "source_state_bytes": state_path.stat().st_size,
                    "source_state_sha256": state_sha256,
                    "initial_state_receipt": str(run_dir / "initial_state_receipt_v1.json"),
                    "initial_common_state_sha256": initial.get(
                        "common_initial_state_sha256"
                    ),
                    **state_record,
                    "status": "PASS",
                }
            )
            del state
    payload = {
        "schema": SCHEMA_SOURCE,
        "status": "PASS",
        "created_utc": utc_now(),
        "extension_classification": CLASSIFICATION,
        "replaces_preregistered_seed": False,
        "source_root": str(root),
        "cell_count": len(rows),
        "cells": rows,
    }
    write_json_exclusive(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def inventory_command(args: argparse.Namespace) -> None:
    artifact_root = args.artifact_root.resolve(strict=True)
    source_audit = load_json(args.source_audit.resolve(strict=True))
    if source_audit.get("schema") != SCHEMA_SOURCE or source_audit.get("status") != "PASS":
        fail("source audit is not PASS")
    source_by_cell = {
        (int(row["seed"]), str(row["arm"])): row
        for row in source_audit["cells"]
    }
    seeds = (args.seed,) if args.seed is not None else SEEDS
    rows = []
    arm_runtimes = []
    for seed in seeds:
        if seed not in SEEDS:
            fail(f"unsupported seed: {seed}")
        for arm, (target_scale, denominator_scale) in ARMS.items():
            run_dir = artifact_root / f"seed{seed}" / f"arm{arm}"
            checkpoint_root = run_dir / "checkpoints"
            if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
                fail(f"missing checkpoint root: {checkpoint_root}")
            temporary = sorted(checkpoint_root.glob(".*.tmp-*"))
            if temporary:
                fail(f"partial checkpoint directories remain: {temporary}")
            for budget_kimg in schedule.BUDGETS_KIMG:
                checkpoint_dir = checkpoint_root / f"{budget_kimg}k"
                if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
                    fail(f"missing immutable checkpoint: {checkpoint_dir}")
                if stat.S_IMODE(checkpoint_dir.stat().st_mode) != 0o550:
                    fail(f"checkpoint directory mode is not immutable: {checkpoint_dir}")
                metadata_path = checkpoint_dir / "metadata.json"
                state_path = checkpoint_dir / "training-state.pt"
                snapshot_path = checkpoint_dir / "network-snapshot.pkl"
                metadata = load_json(metadata_path)
                source = source_by_cell[(seed, arm)]
                exact = {
                    "schema": CHECKPOINT_SCHEMA,
                    "status": "immutable_checkpoint_written",
                    "extension_classification": CLASSIFICATION,
                    "replaces_preregistered_seed": False,
                    "seed": seed,
                    "arm": arm,
                    "budget_kimg": budget_kimg,
                    "attempted_iteration": budget_kimg * 1000 // 128,
                    "cur_nimg": budget_kimg * 1000,
                    "target_gap_scale": target_scale,
                    "denominator_gap_scale": denominator_scale,
                    "source_checkpoint_sha256": source["source_state_sha256"],
                    "training_commit": args.training_commit,
                    "checkpoint_interval_kimg": 64,
                    "atomic_directory_publish": True,
                }
                for field, expected in exact.items():
                    if metadata.get(field) != expected:
                        fail(
                            f"metadata {field} mismatch at {checkpoint_dir}: "
                            f"{metadata.get(field)!r} != {expected!r}"
                        )
                for path in (metadata_path, state_path, snapshot_path):
                    regular_file(path, "checkpoint artifact")
                    if stat.S_IMODE(path.stat().st_mode) != 0o440:
                        fail(f"checkpoint file mode is not immutable: {path}")
                if (
                    metadata.get("training_state_bytes") != state_path.stat().st_size
                    or metadata.get("training_state_sha256") != sha256_file(state_path)
                    or metadata.get("snapshot_bytes") != snapshot_path.stat().st_size
                    or metadata.get("snapshot_sha256") != sha256_file(snapshot_path)
                ):
                    fail(f"checkpoint artifact hash/size mismatch: {checkpoint_dir}")
                state = load_state(state_path)
                state_record = validate_state(
                    state,
                    path=state_path,
                    seed=seed,
                    arm=arm,
                    budget_kimg=budget_kimg,
                    expected_total_kimg=1024,
                )
                if metadata.get("training_state_fields") != state_record["state_fields"]:
                    fail(f"metadata state-field inventory mismatch: {checkpoint_dir}")
                snapshot_record = validate_snapshot(
                    snapshot_path, state_record["ema_state_sha256"]
                )
                rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "budget_kimg": budget_kimg,
                        "checkpoint_dir": str(checkpoint_dir),
                        "training_state": str(state_path),
                        "training_state_bytes": state_path.stat().st_size,
                        "training_state_sha256": metadata["training_state_sha256"],
                        "snapshot": str(snapshot_path),
                        "snapshot_bytes": snapshot_path.stat().st_size,
                        "snapshot_sha256": metadata["snapshot_sha256"],
                        "metadata": str(metadata_path),
                        "metadata_sha256": sha256_file(metadata_path),
                        **state_record,
                        **snapshot_record,
                        "status": "PASS",
                    }
                )
                del state
            runtime_path = run_dir / "arm_runtime.json"
            runtime = load_json(runtime_path)
            if (
                runtime.get("status") != "PASS"
                or runtime.get("seed") != seed
                or runtime.get("arm") != arm
                or not math.isfinite(float(runtime.get("wall_seconds", math.nan)))
            ):
                fail(f"invalid arm runtime receipt: {runtime_path}")
            arm_runtimes.append(runtime)
    expected_count = len(seeds) * len(ARMS) * len(schedule.BUDGETS_KIMG)
    if len(rows) != expected_count:
        fail(f"inventory row count mismatch: {len(rows)} != {expected_count}")
    training_gpu_hours = sum(float(row["wall_seconds"]) for row in arm_runtimes) / 3600
    payload = {
        "schema": SCHEMA_INVENTORY,
        "status": "PASS",
        "created_utc": utc_now(),
        "extension_classification": CLASSIFICATION,
        "replaces_preregistered_seed": False,
        "artifact_root": str(artifact_root),
        "training_commit": args.training_commit,
        "seeds": list(seeds),
        "arms": list(ARMS),
        "budgets_kimg": list(schedule.BUDGETS_KIMG),
        "checkpoint_count": len(rows),
        "training_gpu_hours": training_gpu_hours,
        "arm_runtimes": arm_runtimes,
        "checkpoints": rows,
    }
    write_json_exclusive(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def select_resume_command(args: argparse.Namespace) -> None:
    if args.seed not in SEEDS or args.arm not in ARMS:
        fail("resume selector received an unsupported seed/arm")
    source_path = (
        args.source_root.resolve(strict=True)
        / f"seed{args.seed}"
        / f"arm{args.arm}"
        / "training-state-latest.pt"
    )
    run_dir = args.artifact_root.resolve(strict=True) / f"seed{args.seed}" / f"arm{args.arm}"
    candidates: list[tuple[int, Path]] = []
    source_state = load_state(source_path)
    validate_state(
        source_state,
        path=source_path,
        seed=args.seed,
        arm=args.arm,
        budget_kimg=SOURCE_KIMG,
        expected_total_kimg=SOURCE_KIMG,
    )
    candidates.append((SOURCE_KIMG * 1000, source_path))
    del source_state
    latest_path = run_dir / "training-state-latest.pt"
    if latest_path.is_file() and not latest_path.is_symlink():
        latest_state = load_state(latest_path)
        latest_nimg = int(latest_state.get("cur_nimg", -1))
        if latest_nimg % 1000 != 0 or not (256_000 < latest_nimg <= 1_024_000):
            fail(f"run latest has an invalid image count: {latest_path}")
        validate_state(
            latest_state,
            path=latest_path,
            seed=args.seed,
            arm=args.arm,
            budget_kimg=latest_nimg // 1000,
            expected_total_kimg=1024,
        )
        candidates.append((latest_nimg, latest_path))
        del latest_state
    checkpoint_root = run_dir / "checkpoints"
    if checkpoint_root.is_dir() and not checkpoint_root.is_symlink():
        for budget_kimg in schedule.BUDGETS_KIMG:
            checkpoint_dir = checkpoint_root / f"{budget_kimg}k"
            metadata_path = checkpoint_dir / "metadata.json"
            state_path = checkpoint_dir / "training-state.pt"
            if not metadata_path.exists() and not state_path.exists():
                continue
            metadata = load_json(metadata_path)
            if (
                metadata.get("status") != "immutable_checkpoint_written"
                or metadata.get("seed") != args.seed
                or metadata.get("arm") != args.arm
                or metadata.get("budget_kimg") != budget_kimg
                or metadata.get("training_state_sha256") != sha256_file(state_path)
            ):
                fail(f"invalid resume checkpoint metadata: {checkpoint_dir}")
            checkpoint_state = load_state(state_path)
            validate_state(
                checkpoint_state,
                path=state_path,
                seed=args.seed,
                arm=args.arm,
                budget_kimg=budget_kimg,
                expected_total_kimg=1024,
            )
            candidates.append((budget_kimg * 1000, state_path))
            del checkpoint_state
    selected_nimg, selected_path = max(candidates, key=lambda item: item[0])
    payload = {
        "status": "PASS",
        "seed": args.seed,
        "arm": args.arm,
        "selected_cur_nimg": selected_nimg,
        "selected_resume_state": str(selected_path),
    }
    print(str(selected_path) if args.path_only else json.dumps(payload, sort_keys=True))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    source = subparsers.add_parser("source")
    source.add_argument("--source-root", type=Path, required=True)
    source.add_argument("--output", type=Path, required=True)
    source.set_defaults(handler=source_command)
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--artifact-root", type=Path, required=True)
    inventory.add_argument("--source-audit", type=Path, required=True)
    inventory.add_argument("--training-commit", required=True)
    inventory.add_argument("--seed", type=int)
    inventory.add_argument("--output", type=Path, required=True)
    inventory.set_defaults(handler=inventory_command)
    select_resume = subparsers.add_parser("select-resume")
    select_resume.add_argument("--source-root", type=Path, required=True)
    select_resume.add_argument("--artifact-root", type=Path, required=True)
    select_resume.add_argument("--seed", type=int, required=True)
    select_resume.add_argument("--arm", required=True)
    select_resume.add_argument("--path-only", action="store_true")
    select_resume.set_defaults(handler=select_resume_command)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        args.handler(args)
        return 0
    except (AuditError, AssertionError, KeyError, OSError, ValueError) as exc:
        print(f"[q256-seed6-7-ab-64k-audit] ERROR: {exc}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
