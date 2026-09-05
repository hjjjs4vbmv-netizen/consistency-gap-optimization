#!/usr/bin/env python3
"""Create the single parameterized 64-trajectory M1 training manifest."""

import argparse
import json
import re
from pathlib import Path

from scripts.build_m1_evaluation_slots import SlotError, normalize_roster
from training import reproducibility, schedule_switch


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PROTOCOL = (
    REPO_ROOT / "analysis" / "q256_terminal_history_n30_matpool_v1" / "protocol.json"
)
TRAINING_DATASET_SHA256 = (
    "9818e4b801a52eac437485bc8a69e40b54e9ae9c5d1427467343c91de868f1b3"
)
ORDERS = (
    ("K_A", "K_B", "R_A", "R_B"),
    ("K_B", "R_A", "R_B", "K_A"),
    ("R_A", "R_B", "K_A", "K_B"),
    ("R_B", "K_A", "K_B", "R_A"),
)
BRANCHES = {
    "K_A": {"origin_arm": "A", "continuation_arm": "A", "optimizer": "keep"},
    "K_B": {"origin_arm": "B", "continuation_arm": "A", "optimizer": "keep"},
    "R_A": {"origin_arm": "A", "continuation_arm": "A", "optimizer": "reset"},
    "R_B": {"origin_arm": "B", "continuation_arm": "A", "optimizer": "reset"},
}
SOURCE_FIELDS = {
    "source_state_path", "source_state_bytes", "source_state_sha256",
    "provenance_receipt_path", "provenance_receipt_sha256",
    "internal_state_sha256", "support_files", "common_initial_state_sha256",
}
TRAINING = {
    "arch": "ddpmpp", "precond": "ect", "conditional": False,
    "batch": 128, "batch_gpu": 16, "world_size": 1,
    "optimizer": "RAdam", "lr": 1e-4, "betas": [0.9, 0.999],
    "eps": 1e-8, "weight_decay": 0.0,
    "q": 256, "k": 8, "b": 1, "c": 0,
    "mapping": "sigmoid", "mean": -1.1, "std": 2.0,
    "global_gap_scale": 1.0,
    "target_gap_scale": 1.0, "denominator_gap_scale": 1.0,
    "dropout": 0.2, "augment": 0, "xflip": False,
    "ema_beta": 0.9993, "enable_amp": True, "fp16": True,
    "ema_halflife_kimg": None, "ema_rampup_ratio": None,
    "lr_rampup_kimg": 0,
    "loss_scaling": 1.0, "tf32": False,
    "cudnn_benchmark": False, "deterministic_algorithms": True,
    "cublas_workspace_config": ":4096:8", "workers": 1, "cache": True,
    "managed_loss_overflow": False,
    "gradient_nonfinite_replacement": {"nan": 0.0, "posinf": 1e5, "neginf": -1e5},
    "double_ticks": 10000, "tick": 10, "checkpoint_tick": 10,
    "snapshot_tick": 0, "state_dump_tick": 0,
    "sample_every": 26, "eval_every": 50,
    "mid_t": 0.821, "adaptive_update_kimg": 0.5,
    "start_attempt": 4000, "final_attempt": 8000,
    "final_kimg": 1024, "milestone_kimg": [640, 768, 896, 1024],
    "metrics": "none",
}
RUNTIME_CONTRACT = {
    "python": "3.11.13", "torch": "2.6.0+cu124", "cuda": "12.4",
    "numpy": "2.1.2", "scipy": "1.16.1",
}
GPU_CONTRACT = {
    "name_contains": "A100", "minimum_free_mib": 35000,
    "maximum_utilization_percent": 5,
}


def sha256_file(path: Path) -> str:
    return schedule_switch.sha256_file(str(path))


def validate_runtime_receipt(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    probe = value.get("runtime_probe", {})
    pip_freeze = value.get("pip_freeze", {})
    if (
        value.get("schema") != "ect.m1.rebuilt-training-runtime/v1"
        or value.get("status") != "PASS"
        or value.get("runtime_origin") != "REBUILT_NOT_BYTE_IDENTICAL"
        or any(probe.get(key) != expected for key, expected in RUNTIME_CONTRACT.items())
        or not isinstance(probe.get("cudnn"), int)
        or set(pip_freeze) != {"path", "sha256"}
    ):
        raise RuntimeError("invalid rebuilt M1 runtime receipt")
    freeze = Path(pip_freeze["path"])
    if (
        not freeze.is_absolute() or not freeze.is_file()
        or sha256_file(freeze) != pip_freeze["sha256"]
    ):
        raise RuntimeError("M1 pip-freeze binding mismatch")
    return {key: probe[key] for key in ("python", "torch", "cuda", "cudnn", "numpy", "scipy")}


def selected_roster(inventory: dict) -> list[dict]:
    try:
        roster = normalize_roster(inventory)
    except SlotError as exc:
        raise RuntimeError(f"invalid mechanical source roster: {exc}") from exc
    candidates = {row["seed"]: row for row in inventory["candidates"]}
    result = []
    for index, identity in enumerate(roster):
        row = candidates[identity["seed"]]
        sources = row.get("sources")
        if not isinstance(sources, dict) or set(sources) != {"A", "B"}:
            raise RuntimeError("qualified inventory row lacks A/B sources")
        for arm, source in sources.items():
            if not isinstance(source, dict) or set(source) != SOURCE_FIELDS:
                raise RuntimeError(f"qualified {arm} source identity is incomplete")
            if not Path(source["source_state_path"]).is_absolute():
                raise RuntimeError(f"qualified {arm} source path is not absolute")
            if not Path(source["provenance_receipt_path"]).is_absolute():
                raise RuntimeError(f"qualified {arm} provenance path is not absolute")
        result.append({
            "roster_slot": identity["roster_slot"],
            "seed": identity["seed"],
            "order": list(ORDERS[index % len(ORDERS)]),
            "sources": sources,
        })
    return result


def build_manifest(
    inventory: dict, inventory_path: Path, protocol_path: Path, *,
    implementation_commit: str, dataset_path: Path, dataset_sha256: str,
    runtime_python: Path, runtime_receipt: Path, output_root: Path,
) -> dict:
    if re.fullmatch(r"[0-9a-f]{40}", implementation_commit) is None:
        raise RuntimeError("implementation commit must be 40 lowercase hex digits")
    if re.fullmatch(r"[0-9a-f]{64}", dataset_sha256) is None:
        raise RuntimeError("dataset SHA256 must be 64 lowercase hex digits")
    if dataset_sha256 != TRAINING_DATASET_SHA256:
        raise RuntimeError("dataset SHA256 is not the frozen M1 training archive")
    if sha256_file(dataset_path) != dataset_sha256:
        raise RuntimeError("dataset file does not match supplied SHA256")
    if REPO_ROOT == output_root or REPO_ROOT in output_root.parents:
        raise RuntimeError("M1 output root must be outside the implementation repo")
    runtime_contract = validate_runtime_receipt(runtime_receipt)
    return {
        "schema": "ect.m1.training-run-manifest/v1",
        "experiment_protocol": schedule_switch.M1_HISTORY_PERSISTENCE_PROTOCOL,
        "implementation_commit": implementation_commit,
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "baseline_protocol": {
            "path": str(BASELINE_PROTOCOL), "sha256": sha256_file(BASELINE_PROTOCOL),
        },
        "source_inventory": {"path": str(inventory_path), "sha256": sha256_file(inventory_path)},
        "dataset": {"path": str(dataset_path), "sha256": dataset_sha256},
        "runtime_python": str(runtime_python),
        "runtime_receipt": {
            "path": str(runtime_receipt), "sha256": sha256_file(runtime_receipt),
        },
        "runtime_contract": runtime_contract,
        "gpu_contract": GPU_CONTRACT,
        "output_root": str(output_root),
        "branches": BRANCHES,
        "roster": selected_roster(inventory),
        "training": TRAINING,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory_path = args.inventory.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    manifest = build_manifest(
        inventory, inventory_path, protocol_path,
        implementation_commit=args.implementation_commit,
        dataset_path=args.dataset.resolve(strict=True),
        dataset_sha256=args.dataset_sha256,
        runtime_python=args.runtime_python.resolve(strict=True),
        runtime_receipt=args.runtime_receipt.resolve(strict=True),
        output_root=args.output_root.resolve(),
    )
    reproducibility.atomic_json_dump(manifest, args.output.resolve(), overwrite=False)
    print("M1_TRAINING_MANIFEST_PASS trajectories=64")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
