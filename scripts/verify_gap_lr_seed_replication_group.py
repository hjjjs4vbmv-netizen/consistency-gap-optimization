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


RUN_IDS = {
    "A": "arm_a_g1_0_lr_fixed_s{seed}",
    "B": "arm_b_g1_3_lr_fixed_s{seed}",
    "C": "arm_c_g1_3_lr_matched_s{seed}",
}
EXPERIMENT_ID = "gap_lr_matched_q128_s45_replication_v1"
EXECUTION_PROTOCOL_COMMIT = "583c2fe0f914fc1191903d747737fd54b4ba1eef"
TRAINING_CODE_COMMIT = "2357bb1d2531a343bdb4397f5a08f4d42a2d135b"
NUMBERED_IDS = [f"{index:06d}" for index in range(1, 9)]


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


def artifact_paths(run_dir: Path) -> dict[str, Path]:
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
        paths[f"network_snapshot_{artifact_id}"] = (
            run_dir / f"network-snapshot-{artifact_id}.pkl"
        )
        paths[f"training_state_{artifact_id}"] = (
            run_dir / f"training-state-{artifact_id}.pt"
        )
    return paths


def validate_and_rehash_receipt(
    receipt: dict[str, Any], receipt_path: Path, run_dir: Path, seed: int, arm: str
) -> None:
    if (
        receipt.get("schema_version") != 2
        or receipt.get("receipt_type") != "gap_lr_seed_replication_run_integrity"
        or receipt.get("status") != "passed"
        or receipt.get("experiment_id") != EXPERIMENT_ID
        or receipt.get("seed") != seed
        or receipt.get("arm") != arm
        or receipt.get("execution_protocol_commit") != EXECUTION_PROTOCOL_COMMIT
        or receipt.get("training_code_commit") != TRAINING_CODE_COMMIT
        or Path(receipt.get("run_dir", "")).resolve() != run_dir.resolve()
    ):
        fail(f"invalid per-run receipt for seed {seed} arm {arm}")
    summary = receipt.get("completion", {}).get("summary", {})
    state = receipt.get("final_training_state", {})
    if (
        receipt.get("completion", {}).get("budget_kimg") != 256
        or summary.get("rows") != 2000
        or summary.get("attempted_iterations") != 2000
        or summary.get("amp_contract_passed") is not True
        or state.get("cur_nimg") != 256000
        or state.get("attempted_iteration") != 2000
        or state.get("successful_optimizer_steps")
        != summary.get("successful_optimizer_steps")
    ):
        fail(f"incomplete per-run receipt for seed {seed} arm {arm}")
    paths = artifact_paths(run_dir)
    hashes = receipt.get("artifact_sha256", {})
    sizes = receipt.get("artifact_size_bytes", {})
    if set(hashes) != set(paths) or set(sizes) != set(paths):
        fail(f"artifact manifest mismatch for seed {seed} arm {arm}")
    for name, path in sorted(paths.items()):
        if not path.is_file():
            fail(f"missing {name} for seed {seed} arm {arm}")
        if sizes[name] != path.stat().st_size or hashes[name] != sha256_file(path):
            fail(f"artifact binding mismatch for seed {seed} arm {arm}: {name}")


def preview_difference(first: Path, second: Path) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    left = np.asarray(Image.open(first).convert("RGB"), dtype=np.int16)
    right = np.asarray(Image.open(second).convert("RGB"), dtype=np.int16)
    if left.shape != right.shape:
        fail("model-init preview dimensions differ")
    delta = left - right
    absolute = np.abs(delta)
    return {
        "first_run_id": first.parent.name,
        "second_run_id": second.parent.name,
        "exact_pixel_values_equal": bool(np.array_equal(left, right)),
        "max_abs_channel_delta_lsb": int(absolute.max(initial=0)),
        "differing_channel_values": int(np.count_nonzero(delta)),
        "differing_pixels": int(np.count_nonzero(np.any(delta != 0, axis=2))),
    }


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
    preview_evidence: dict[str, Any] = {}
    preview_deviation = False
    for seed in seeds:
        options_by_seed[seed] = {}
        model_init_paths = []
        data_image_hashes = []
        for arm, template in RUN_IDS.items():
            run_id = template.format(seed=seed)
            run_dir = root / run_id
            receipt_path = root / "integrity_receipts" / f"seed{seed}_{arm}.integrity.json"
            receipt = load(receipt_path)
            validate_and_rehash_receipt(receipt, receipt_path, run_dir, seed, arm)
            options = load(run_dir / "training_options.json")
            options_by_seed[seed][arm] = options
            model_init_paths.append(run_dir / "model_init.png")
            data_image_hashes.append(sha256_file(run_dir / "data.png"))
            seed_receipts[f"seed{seed}_{arm}"] = {
                "path": f"integrity_receipts/seed{seed}_{arm}.integrity.json",
                "sha256": sha256_file(receipt_path),
            }

        normalized = [normalized_within_seed(options_by_seed[seed][arm]) for arm in ("A", "B", "C")]
        if not (normalized[0] == normalized[1] == normalized[2]):
            fail(f"seed {seed} A/B/C differ outside gap/LR/run_dir")
        pairwise = [
            preview_difference(model_init_paths[0], path)
            for path in model_init_paths[1:]
        ]
        exact_preview_equality = all(
            item["exact_pixel_values_equal"] for item in pairwise
        )
        preview_deviation = preview_deviation or not exact_preview_equality
        preview_evidence[str(seed)] = {
            "sha256": [sha256_file(path) for path in model_init_paths],
            "exact_file_hashes_equal": len(
                {sha256_file(path) for path in model_init_paths}
            )
            == 1,
            "pairwise_against_A": pairwise,
            "interpretation": "diagnostic generated preview; not a parameter hash",
        }
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
        "status": "adjudication_required" if preview_deviation else "passed",
        "integrity_status": "passed",
        "experiment_id": EXPERIMENT_ID,
        "completed_seeds": seeds,
        "required_arms_per_seed": ["A", "B", "C"],
        "within_seed_allowed_differences": [
            "loss_kwargs.global_gap_scale", "optimizer_kwargs.lr", "run_dir"
        ],
        "within_seed_contract_passed": True,
        "between_seed_allowed_differences": ["seed", "run_dir"],
        "between_seed_contract_passed": seeds == [4, 5],
        "historical_observed_preupdate_parameter_hash": "not_captured",
        "model_init_preview_evidence": preview_evidence,
        "model_init_preview_exact_equality_passed": not preview_deviation,
        "model_init_preview_is_parameter_identity_evidence": False,
        "data_image_pairing_passed": True,
        "per_run_receipts": seed_receipts,
        "launcher_logs": launcher_logs,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if preview_deviation:
        raise SystemExit(4)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        fail(str(exc))
