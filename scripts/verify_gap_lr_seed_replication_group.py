#!/usr/bin/env python3
"""Verify paired A/B/C configuration and seed-only differences."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from PIL import Image, ImageChops


RUN_IDS = {
    "A": "arm_a_g1_0_lr_fixed_s{seed}",
    "B": "arm_b_g1_3_lr_fixed_s{seed}",
    "C": "arm_c_g1_3_lr_matched_s{seed}",
}


def fail(message: str) -> None:
    raise SystemExit("SEED REPLICATION GROUP REJECTED: " + message)


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        fail(f"expected JSON object: {path}")
    return value


def normalized_within_seed(options: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(options)
    value.pop("run_dir", None)
    value["loss_kwargs"].pop("global_gap_scale", None)
    value["optimizer_kwargs"].pop("lr", None)
    return value


def normalized_between_seeds(options: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(options)
    value.pop("run_dir", None)
    value.pop("seed", None)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--completed-seeds", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    seeds = [int(item) for item in args.completed_seeds.split(",") if item]
    if seeds not in ([4], [4, 5]):
        fail("completed seeds must be exactly 4 or 4,5")

    root = args.experiment_root.resolve()
    options_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    seed_receipts: dict[str, Any] = {}
    for seed in seeds:
        options_by_seed[seed] = {}
        model_init_paths = []
        data_image_hashes = []
        for arm, template in RUN_IDS.items():
            run_id = template.format(seed=seed)
            run_dir = root / run_id
            receipt_path = root / "integrity_receipts" / f"seed{seed}_{arm}.integrity.json"
            receipt = load(receipt_path)
            if receipt.get("status") != "passed" or receipt.get("seed") != seed or receipt.get("arm") != arm:
                fail(f"invalid per-run receipt for seed {seed} arm {arm}")
            options = load(run_dir / "training_options.json")
            options_by_seed[seed][arm] = options
            model_init_paths.append(run_dir / "model_init.png")
            data_image_hashes.append(sha256_file(run_dir / "data.png"))
            seed_receipts[f"seed{seed}_{arm}"] = {
                "path": str(receipt_path),
                "sha256": sha256_file(receipt_path),
            }

        normalized = [normalized_within_seed(options_by_seed[seed][arm]) for arm in ("A", "B", "C")]
        if not (normalized[0] == normalized[1] == normalized[2]):
            fail(f"seed {seed} A/B/C differ outside gap/LR/run_dir")
        if any(any(high > 1 for _, high in ImageChops.difference(Image.open(model_init_paths[0]).convert("RGB"), Image.open(path).convert("RGB")).getextrema()) for path in model_init_paths[1:]):
            fail(f"seed {seed} A/B/C model initialization images differ")
        if len(set(data_image_hashes)) != 1:
            fail(f"seed {seed} A/B/C data images differ")

    if seeds == [4, 5]:
        for arm in ("A", "B", "C"):
            if normalized_between_seeds(options_by_seed[4][arm]) != normalized_between_seeds(options_by_seed[5][arm]):
                fail(f"arm {arm} differs across seeds outside seed/run_dir")

    launcher_logs = {}
    for path in sorted((root / "logs").glob("*.log")):
        if "group.verification" in path.name:
            continue
        launcher_logs[path.name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    receipt = {
        "schema_version": 1,
        "receipt_type": "gap_lr_seed_replication_group_integrity",
        "status": "passed",
        "experiment_id": "gap_lr_matched_q128_s45_replication_v1",
        "completed_seeds": seeds,
        "required_arms_per_seed": ["A", "B", "C"],
        "within_seed_allowed_differences": [
            "loss_kwargs.global_gap_scale", "optimizer_kwargs.lr", "run_dir"
        ],
        "within_seed_contract_passed": True,
        "between_seed_allowed_differences": ["seed", "run_dir"],
        "between_seed_contract_passed": seeds == [4, 5],
        "model_initialization_pairing_passed": True,
        "model_initialization_image_tolerance_lsb": 1,
        "data_image_pairing_passed": True,
        "per_run_receipts": seed_receipts,
        "launcher_logs": launcher_logs,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        fail(str(exc))
