#!/usr/bin/env python3
"""Build the M1 first-16 source inventory from PR101 prefix directories."""

import argparse
import copy
import csv
import json
import math
import re
from pathlib import Path

import torch

from training import ct_training_loop, reproducibility, schedule_switch


ARMS = {"A": (1.0, 1.0), "B": (1.1, 1.1)}
REQUIRED_STATE = {
    "net", "ema", "optimizer_state", "gradscaler_state", "loss_fn_state",
    "rank_states", "factorial", "attempted_iteration",
    "successful_optimizer_steps", "cur_nimg", "cur_tick", "tick_start_nimg",
    "snapshot_grid_z", "snapshot_grid_c", "snapshot_grid_size",
    "trajectory_config", "trajectory_config_sha256", "reproducibility_schema",
}


def sha256_file(path: Path) -> str:
    return schedule_switch.sha256_file(str(path))


def _finite(value) -> bool:
    if isinstance(value, torch.Tensor):
        return not (value.is_floating_point() or value.is_complex()) or bool(
            torch.isfinite(value).all()
        )
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _module_finite(module) -> bool:
    return isinstance(module, torch.nn.Module) and _finite(module.state_dict())


def _load_source(state_path: Path, receipt_path: Path, seed: int, arm: str):
    state_path = state_path.resolve(strict=True)
    receipt_path = receipt_path.resolve(strict=True)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if set(receipt) != {
        "schema", "seed", "attempted_iteration", "processed_nimg",
        "factorial", "dataset_path", "transfer_path", "world_size",
        "batch_size", "batch_gpu", "trajectory_config",
        "trajectory_config_sha256", "hashes",
        "common_initial_state_sha256", "rank_states",
    }:
        raise RuntimeError("initial provenance receipt fields mismatch")
    if (
        receipt.get("schema") != reproducibility.INITIAL_RECEIPT_SCHEMA
        or receipt.get("seed") != seed
        or receipt.get("attempted_iteration") != 0
        or receipt.get("processed_nimg") != 0
        or receipt.get("world_size") != 1
        or receipt.get("batch_size") != 128
        or receipt.get("batch_gpu") != 16
    ):
        raise RuntimeError("initial provenance receipt identity mismatch")
    target, denominator = ARMS[arm]
    factorial = receipt.get("factorial", {})
    if (
        factorial.get("protocol") != "q256_target_weight_v1"
        or factorial.get("arm") != arm
        or float(factorial.get("target_gap_scale", -1)) != target
        or float(factorial.get("denominator_gap_scale", -1)) != denominator
    ):
        raise RuntimeError("initial provenance receipt arm/scale mismatch")
    receipt_trajectory = receipt.get("trajectory_config")
    if (
        not isinstance(receipt_trajectory, dict)
        or receipt_trajectory.get("schema")
        != reproducibility.TRAJECTORY_CONFIG_SCHEMA
        or receipt_trajectory.get("seed") != seed
        or receipt_trajectory.get("world_size") != 1
        or receipt_trajectory.get("batch_size") != 128
        or receipt_trajectory.get("batch_gpu") != 16
        or reproducibility.state_sha256(receipt_trajectory)
        != receipt.get("trajectory_config_sha256")
    ):
        raise RuntimeError("initial provenance trajectory hash is invalid")
    hashes = receipt.get("hashes")
    ranks = receipt.get("rank_states")
    if not isinstance(ranks, list) or len(ranks) != 1:
        raise RuntimeError("initial provenance receipt requires one rank")
    rank = ranks[0]
    if not isinstance(rank, dict) or set(rank) != {
        "rank", "world_size", "rng_sha256", "sampler_sha256", "sampler_state"
    }:
        raise RuntimeError("initial provenance rank fields mismatch")
    sampler = rank["sampler_state"]
    if (
        rank["rank"] != 0 or rank["world_size"] != 1
        or not isinstance(sampler, dict)
        or set(sampler) != {
            "schema", "dataset_size", "rank", "num_replicas", "shuffle", "seed",
            "window_size", "consumed_samples",
        }
        or sampler.get("schema") != "ect.infinite-sampler/v1"
        or sampler.get("dataset_size") != 50000
        or sampler.get("rank") != 0 or sampler.get("num_replicas") != 1
        or sampler.get("shuffle") is not True
        or sampler.get("seed") != seed or sampler.get("consumed_samples") != 0
        or float(sampler.get("window_size", -1)) != 0.5
        or reproducibility.state_sha256(sampler) != rank["sampler_sha256"]
    ):
        raise RuntimeError("initial provenance rank/sampler binding mismatch")
    hex64 = re.compile(r"^[0-9a-f]{64}$")
    if (
        not isinstance(hashes, dict)
        or set(hashes) != {
            "model", "ema", "optimizer", "gradscaler",
            "rank_rng", "rank_sampler",
        }
        or any(hex64.fullmatch(str(hashes[name])) is None
               for name in ("model", "ema", "optimizer", "gradscaler"))
        or hashes["model"] != hashes["ema"]
        or hex64.fullmatch(str(rank["rng_sha256"])) is None
        or hex64.fullmatch(str(rank["sampler_sha256"])) is None
        or hashes["rank_rng"] != [rank["rng_sha256"]]
        or hashes["rank_sampler"] != [rank["sampler_sha256"]]
        or reproducibility.state_sha256(hashes)
        != receipt.get("common_initial_state_sha256")
    ):
        raise RuntimeError("initial provenance common-state hash is invalid")
    size = state_path.stat().st_size
    digest = sha256_file(state_path)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("trajectory_config_sha256") != receipt.get("trajectory_config_sha256"):
        raise RuntimeError("initial receipt and 512 state trajectory differ")
    record = {
        "source_state_path": str(state_path),
        "source_state_bytes": size,
        "source_state_sha256": digest,
        "provenance_receipt_path": str(receipt_path),
        "provenance_receipt_sha256": sha256_file(receipt_path),
        "internal_state_sha256": schedule_switch.internal_state_hashes(state),
    }
    return record, state, receipt["common_initial_state_sha256"]


def _validate_support_csv(prefix: Path, arm: str) -> tuple[dict, int]:
    contracts = (
        (
            prefix / "train_summary.csv",
            ct_training_loop._TRAIN_SUMMARY_FIELDS,
            None,
        ),
        (
            prefix / "factorial_training_telemetry_v1.csv",
            ct_training_loop._FACTORIAL_TELEMETRY_FIELDS,
            arm,
        ),
    )
    identities = {}
    terminal_successful_steps = set()
    for path, expected_fields, expected_arm in contracts:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != expected_fields:
                raise RuntimeError(f"source support CSV schema mismatch: {path}")
            rows = list(reader)
        if len(rows) != 4000:
            raise RuntimeError(f"source support CSV must contain 4000 rows: {path}")
        attempts = [int(row["attempted_iteration"]) for row in rows]
        if attempts != list(range(1, 4001)):
            raise RuntimeError(f"source support CSV attempt coverage mismatch: {path}")
        if any(int(row["processed_nimg"]) != attempt * 128
               for row, attempt in zip(rows, attempts)):
            raise RuntimeError(f"source support CSV image progress mismatch: {path}")
        terminal_steps = int(rows[-1]["successful_optimizer_steps"])
        if terminal_steps < 0 or terminal_steps > 4000:
            raise RuntimeError("source support CSV successful-step counter is invalid")
        terminal_successful_steps.add(terminal_steps)
        if expected_arm is not None and any(
            row.get("schema") != "ect.q256.target-weight-training-telemetry/v1"
            or row.get("protocol") != "q256_target_weight_v1"
            or row.get("arm") != expected_arm
            for row in rows
        ):
            raise RuntimeError(f"source support CSV arm/protocol mismatch: {path}")
        identities[path.name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    options = prefix / "training_options.json"
    json.loads(options.read_text(encoding="utf-8"))
    identities[options.name] = {
        "path": str(options.resolve()),
        "bytes": options.stat().st_size,
        "sha256": sha256_file(options),
    }
    if len(terminal_successful_steps) != 1:
        raise RuntimeError("source support CSV successful-step ledgers disagree")
    return identities, terminal_successful_steps.pop()


def _validate_state(state: dict, seed: int, arm: str) -> None:
    missing = sorted(REQUIRED_STATE - set(state))
    if missing:
        raise RuntimeError("full-state fields missing: " + ", ".join(missing))
    if state.get("reproducibility_schema") != reproducibility.TRAINING_STATE_SCHEMA:
        raise RuntimeError("source is not a versioned full training state")
    if int(state["cur_nimg"]) != 512000 or int(state["attempted_iteration"]) != 4000:
        raise RuntimeError("source progress is not 512 kimg / attempt 4000")
    successful_steps = state["successful_optimizer_steps"]
    if (
        isinstance(successful_steps, bool)
        or not isinstance(successful_steps, int)
        or successful_steps < 0 or successful_steps > 4000
    ):
        raise RuntimeError("source successful-step counter is invalid")
    factorial = state["factorial"]
    target, denominator = ARMS[arm]
    if (
        factorial.get("protocol") != "q256_target_weight_v1"
        or factorial.get("arm") != arm
        or float(factorial.get("target_gap_scale", -1)) != target
        or float(factorial.get("denominator_gap_scale", -1)) != denominator
    ):
        raise RuntimeError("source arm/scale mismatch")
    trajectory = state["trajectory_config"]
    if reproducibility.state_sha256(trajectory) != state["trajectory_config_sha256"]:
        raise RuntimeError("source trajectory hash is invalid")
    expected = {
        "seed": seed, "world_size": 1, "batch_size": 128, "batch_gpu": 16,
        "total_kimg": 1024, "ema_beta": 0.9993, "enable_tf32": False,
        "enable_amp": True, "cudnn_benchmark": False,
        "num_accumulation_rounds": 8, "double_ticks": 10000,
        "cudnn_deterministic": True, "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8", "loss_scaling": 1.0,
        "kimg_per_tick": 10.0, "snapshot_ticks": None,
        "state_dump_ticks": None, "ckpt_ticks": 10,
        "sample_ticks": 26, "eval_ticks": 50,
        "adaptive_update_kimg": 0.5, "metrics": [],
        "ema_halflife_kimg": None, "ema_rampup_ratio": None,
        "lr_rampup_kimg": 0, "mid_t": [0.821], "device": "cuda",
    }
    if any(trajectory.get(key) != value for key, value in expected.items()):
        raise RuntimeError("source trajectory differs from M1 fixed settings")
    loss = trajectory.get("loss_kwargs", {})
    optimizer = trajectory.get("optimizer_kwargs", {})
    network = trajectory.get("network_kwargs", {})
    dataset = trajectory.get("dataset_kwargs", {})
    loader = trajectory.get("data_loader_kwargs", {})
    if (
        trajectory.get("schema") != reproducibility.TRAJECTORY_CONFIG_SCHEMA
        or
        loss.get("factorial_protocol") != "q256_target_weight_v1"
        or int(loss.get("q", -1)) != 256
        or loss.get("adj") != "sigmoid"
        or float(loss.get("P_mean", 0)) != -1.1
        or float(loss.get("P_std", 0)) != 2.0
        or float(loss.get("target_gap_scale", -1)) != target
        or float(loss.get("denominator_gap_scale", -1)) != denominator
        or not str(optimizer.get("class_name", "")).endswith(".RAdam")
        or float(optimizer.get("lr", -1)) != 1e-4
        or list(optimizer.get("betas", [])) != [0.9, 0.999]
        or float(optimizer.get("eps", -1)) != 1e-8
        or float(network.get("dropout", -1)) != 0.2
        or network.get("model_type") != "SongUNet"
        or network.get("use_fp16") is not True
        or dataset.get("use_labels") is not False
        or dataset.get("xflip") is not False
        or int(dataset.get("resolution", -1)) != 32
        or int(loader.get("num_workers", -1)) != 1
    ):
        raise RuntimeError("source loss/optimizer/network config mismatch")
    ranks = state["rank_states"]
    if (
        not isinstance(ranks, list) or len(ranks) != 1
        or int(ranks[0].get("sampler_state", {}).get("consumed_samples", -1))
        != 512000
    ):
        raise RuntimeError("source rank/sampler state mismatch")
    if not (
        _module_finite(state["net"])
        and _module_finite(state["ema"])
        and _finite(state["optimizer_state"])
        and _finite(state["gradscaler_state"])
        and _finite(state["loss_fn_state"])
    ):
        raise RuntimeError("source contains non-finite initialized state")


def _comparable_config(state: dict) -> dict:
    value = copy.deepcopy(state["trajectory_config"])
    loss = value.get("loss_kwargs", {})
    for key in ("arm", "target_gap_scale", "denominator_gap_scale"):
        loss.pop(key, None)
    return reproducibility.canonical_json_data(value)


def select_source_state(prefix: Path) -> Path:
    primary = prefix / "training-state-kimg000512.pt"
    fallback = prefix / "training-state-latest.pt"
    return next((path for path in (primary, fallback) if path.is_file()), primary)


def inspect_candidate(source_root: Path, seed: int, audit: dict) -> dict:
    sources, states, paths, common_initial = {}, {}, {}, {}
    for arm in ARMS:
        prefix = source_root / f"seed{seed}" / f"prefix_{arm}"
        primary_state = prefix / "training-state-kimg000512.pt"
        state_path = select_source_state(prefix)
        receipt = prefix / "initial_state_receipt_v1.json"
        paths[arm] = (prefix, state_path, receipt)
        audit[arm] = {
            "expected": {
                "source_state_path": str(primary_state.resolve()),
                "source_state_bytes": None, "source_state_sha256": None,
                "provenance_receipt_path": str(receipt.resolve()),
                "provenance_receipt_sha256": None,
            },
            "actual": {
                "source_state_path": (
                    str(state_path.resolve()) if state_path.is_file() else None
                ),
                "source_state_bytes": (
                    state_path.stat().st_size if state_path.is_file() else None
                ),
                "source_state_sha256": None,
                "provenance_receipt_path": (
                    str(receipt.resolve()) if receipt.is_file() else None
                ),
                "provenance_receipt_sha256": (
                    sha256_file(receipt) if receipt.is_file() else None
                ),
            },
        }
    for arm in ARMS:
        prefix, state_path, receipt = paths[arm]
        for name in (
            "train_summary.csv", "factorial_training_telemetry_v1.csv",
            "training_options.json",
        ):
            support = prefix / name
            if not support.is_file() or support.is_symlink() or support.stat().st_size == 0:
                raise RuntimeError(f"source resume history is unavailable: {support}")
        support_files, support_successful_steps = _validate_support_csv(prefix, arm)
        record, state, common_initial[arm] = _load_source(
            state_path, receipt, seed, arm
        )
        record["support_files"] = support_files
        record["common_initial_state_sha256"] = common_initial[arm]
        audit[arm]["actual"].update({
            key: record[key] for key in (
                "source_state_path", "source_state_bytes", "source_state_sha256",
                "provenance_receipt_path", "provenance_receipt_sha256",
            )
        })
        sources[arm], states[arm] = record, state
        _validate_state(state, seed, arm)
        if int(state["successful_optimizer_steps"]) != support_successful_steps:
            raise RuntimeError("source support CSV and full-state step ledgers disagree")
    if _comparable_config(states["A"]) != _comparable_config(states["B"]):
        raise RuntimeError("paired A/B trajectory configs differ beyond arm scales")
    if common_initial["A"] != common_initial["B"]:
        raise RuntimeError("paired A/B provenance common initial state mismatch")
    for field in ("rng_state", "sampler_state"):
        left = states["A"]["rank_states"][0][field]
        right = states["B"]["rank_states"][0][field]
        if reproducibility.state_sha256(left) != reproducibility.state_sha256(right):
            raise RuntimeError(f"paired A/B {field} mismatch")
    return sources


def build_inventory(source_root: Path) -> dict:
    candidates, qualified = [], 0
    for seed in range(50, 80):
        if qualified == 16:
            candidates.append({
                "seed": seed, "checked": False, "qualified": False,
                "reason": "NOT_CHECKED_AFTER_ROSTER_FILLED",
            })
            continue
        audit = {}
        try:
            sources = inspect_candidate(source_root, seed, audit)
            qualified += 1
            candidates.append({
                "seed": seed, "checked": True, "qualified": True,
                "reason": "QUALIFIED", "sources": sources,
            })
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            candidates.append({
                "seed": seed, "checked": True, "qualified": False,
                "reason": "SOURCE_IDENTITY_UNAVAILABLE",
                "detail": str(exc), "sources": audit,
            })
        except Exception as exc:
            candidates.append({
                "seed": seed, "checked": True, "qualified": False,
                "reason": "SOURCE_VALIDATION_FAILED",
                "detail": str(exc), "sources": audit,
            })
    status = "PASS" if qualified == 16 else "BLOCKED_SOURCE_INVENTORY"
    return {"schema": "ect.m1.source-inventory/v1", "status": status, "candidates": candidates}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = build_inventory(args.source_root.resolve(strict=True))
    reproducibility.atomic_json_dump(inventory, args.output.resolve(), overwrite=False)
    print(f"M1_SOURCE_INVENTORY_{inventory['status']}")
    return 0 if inventory["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
