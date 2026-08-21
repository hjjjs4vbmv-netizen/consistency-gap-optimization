#!/usr/bin/env python3
"""Fail-closed validation for the seed-4/5 Role E evaluation handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HANDOFF = (
    REPO_ROOT
    / "results/gap_lr_seed_replication_role_e_handoff"
    / "role_e_disjoint_5k_handoff.json"
)
BLIND_ROOT = REPO_ROOT / "results/gap_lr_seed_replication_blind_adjudication"
EXPECTED_BLOCKS = (
    ("block_5000_9999", 5000, 9999),
    ("block_10000_14999", 10000, 14999),
    ("block_15000_19999", 15000, 19999),
)
EXPECTED_KEYS = {(seed, arm) for seed in (4, 5) for arm in ("A", "B", "C")}


class HandoffError(ValueError):
    pass


def fail(message: str) -> None:
    raise HandoffError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read JSON {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"JSON root is not an object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def repo_path(relative: str) -> Path:
    path = (REPO_ROOT / relative).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError:
        fail(f"path escapes repository: {relative}")
    return path


def close(left: Any, right: Any) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)


def validate_upstream(handoff: dict[str, Any]) -> None:
    bindings = handoff["upstream_bindings"]
    expected_files = {
        "blind_adjudication_sha256": BLIND_ROOT / "blind_adjudication.json",
        "objective_evidence_sha256": BLIND_ROOT / "blind_evidence.json",
        "initialization_reconstruction_sha256": (
            BLIND_ROOT / "initialization_reconstruction.json"
        ),
    }
    for field, path in expected_files.items():
        require(path.is_file(), f"missing upstream artifact: {path}")
        require(
            sha256_file(path) == bindings[field],
            f"upstream hash mismatch for {field}",
        )

    adjudication = load_json(BLIND_ROOT / "blind_adjudication.json")
    require(
        adjudication.get("verdict") == bindings["machine_adjudication_verdict"],
        "machine adjudication verdict drift",
    )
    require(
        adjudication.get("documented_deviations") == ["D1", "D2", "D3", "D4", "D5"],
        "machine adjudication no longer binds exactly D1-D5",
    )
    decision = adjudication.get("decision", {})
    require(decision.get("protocol_exact") is False, "upstream must remain non-exact")
    require(
        decision.get("quality_evaluation_seed4_seed5_authorized") is False,
        "historical machine candidate must not be rewritten as evaluation authorization",
    )


def validate_blocks(handoff: dict[str, Any]) -> None:
    contract = handoff["evaluation_contract"]
    require(contract.get("nfe") == 1, "evaluation must remain NFE=1")
    require(contract.get("precision") == "fp32", "evaluation must remain FP32")
    require(
        contract.get("metric_names") == ["fid5k_full", "kid5k_full"],
        "metric set drift",
    )
    require(
        contract.get("metric_repeats_per_block") == 1,
        "metric repeats must remain one per block",
    )
    prior = contract.get("prior_block_excluded")
    require(prior == {"start": 0, "end": 4999}, "prior block exclusion drift")
    blocks = contract.get("sample_blocks", [])
    require(len(blocks) == len(EXPECTED_BLOCKS), "expected exactly three new blocks")
    seen: set[int] = set()
    for block, (block_id, start, end) in zip(blocks, EXPECTED_BLOCKS):
        require(block.get("block_id") == block_id, f"block ID drift: {block}")
        require(block.get("start") == start, f"block start drift: {block_id}")
        require(block.get("end") == end, f"block end drift: {block_id}")
        require(block.get("count") == 5000, f"block count drift: {block_id}")
        values = set(range(start, end + 1))
        require(len(values) == 5000, f"block is not 5k: {block_id}")
        require(not seen.intersection(values), f"sample blocks overlap: {block_id}")
        require(not values.intersection(range(0, 5000)), f"prior block reused: {block_id}")
        seen.update(values)
    require(
        contract.get("primary_differences")
        == {"delta_gap": "B-A", "delta_ctrl": "C-B"},
        "primary difference definitions drift",
    )
    forbidden = set(contract.get("forbidden_work", []))
    require("new training" in forbidden, "new training exclusion missing")
    require("seed4/5 FID-50k" in forbidden, "FID-50k exclusion missing")


def validate_endpoint(endpoint: dict[str, Any], experiment_root: Path | None) -> None:
    seed = int(endpoint["training_seed"])
    arm = str(endpoint["arm"])
    expected_run_id = {
        "A": f"arm_a_g1_0_lr_fixed_s{seed}",
        "B": f"arm_b_g1_3_lr_fixed_s{seed}",
        "C": f"arm_c_g1_3_lr_matched_s{seed}",
    }[arm]
    require(endpoint.get("run_id") == expected_run_id, f"run ID mismatch for seed {seed} {arm}")
    expected_relative = f"{expected_run_id}/network-snapshot-000008.pkl"
    require(
        endpoint.get("relative_checkpoint_path") == expected_relative,
        f"checkpoint path mismatch for seed {seed} {arm}",
    )
    require(
        endpoint.get("checkpoint_size_bytes") == 223172916,
        f"checkpoint size contract drift for seed {seed} {arm}",
    )

    receipt_path = repo_path(str(endpoint["public_receipt"]))
    require(receipt_path.is_file(), f"missing public receipt: {receipt_path}")
    require(
        sha256_file(receipt_path) == endpoint["public_receipt_sha256"],
        f"public receipt hash mismatch for seed {seed} {arm}",
    )
    receipt = load_json(receipt_path)
    require(receipt.get("status") == "passed", f"receipt not passed for seed {seed} {arm}")
    require(receipt.get("seed") == seed, f"receipt seed mismatch for seed {seed} {arm}")
    require(receipt.get("arm") == arm, f"receipt arm mismatch for seed {seed} {arm}")
    require(receipt.get("run_id") == expected_run_id, f"receipt run mismatch for seed {seed} {arm}")
    require(close(receipt.get("gap_scale"), endpoint["gap_scale"]), "gap-scale drift")
    require(close(receipt.get("learning_rate"), endpoint["learning_rate"]), "LR drift")
    require(
        receipt.get("completion", {}).get("budget_kimg") == 256,
        f"budget mismatch for seed {seed} {arm}",
    )
    require(
        receipt.get("final_ema_snapshot", {}).get("ema_present") is True
        and receipt.get("final_ema_snapshot", {}).get("ema_finite") is True,
        f"final EMA gate failed for seed {seed} {arm}",
    )
    numbered = receipt.get("artifact_manifest", {}).get("network_snapshot_000008", {})
    require(
        numbered.get("sha256") == endpoint["checkpoint_sha256"],
        f"checkpoint hash drift for seed {seed} {arm}",
    )
    require(
        numbered.get("size_bytes") == endpoint["checkpoint_size_bytes"],
        f"checkpoint size mismatch for seed {seed} {arm}",
    )

    if experiment_root is not None:
        checkpoint = experiment_root / expected_relative
        require(checkpoint.is_file(), f"missing raw checkpoint: {checkpoint}")
        require(
            checkpoint.stat().st_size == endpoint["checkpoint_size_bytes"],
            f"raw checkpoint size mismatch: {checkpoint}",
        )
        require(
            sha256_file(checkpoint) == endpoint["checkpoint_sha256"],
            f"raw checkpoint hash mismatch: {checkpoint}",
        )


def validate_handoff(
    handoff_path: Path = DEFAULT_HANDOFF,
    experiment_root: Path | None = None,
) -> dict[str, Any]:
    handoff = load_json(handoff_path)
    require(handoff.get("schema_version") == 1, "unsupported handoff schema")
    require(
        handoff.get("receipt_type")
        == "gap_lr_seed_replication_role_e_disjoint_evaluation_handoff",
        "wrong handoff receipt type",
    )
    require(
        handoff.get("status") == "ready_for_disjoint_5k_evaluation",
        "handoff is not evaluation-ready",
    )
    authorization = handoff.get("authorization", {})
    require(authorization.get("new_training_authorized") is False, "training must be forbidden")
    require(authorization.get("fid50k_seed4_seed5_authorized") is False, "FID-50k must be forbidden")
    require(
        authorization.get("independent_quality_blind_review_claimed") is False,
        "handoff must not claim an independent blind review",
    )
    require(authorization.get("protocol_exact_claimed") is False, "handoff must remain non-exact")

    validate_upstream(handoff)
    validate_blocks(handoff)

    checkpoint_contract = handoff["checkpoint_contract"]
    require(checkpoint_contract.get("training_seeds") == [4, 5], "seed set drift")
    require(checkpoint_contract.get("arms_per_seed") == ["A", "B", "C"], "arm set drift")
    require(checkpoint_contract.get("budget_kimg") == 256, "budget drift")
    require(
        checkpoint_contract.get("checkpoint_filename") == "network-snapshot-000008.pkl",
        "final numbered checkpoint drift",
    )
    require(checkpoint_contract.get("latest_alias_permitted") is False, "latest alias must be forbidden")
    require(
        checkpoint_contract.get("rehash_before_evaluation_required") is True,
        "pre-evaluation rehash gate missing",
    )

    endpoints = checkpoint_contract.get("endpoints", [])
    keys = {(int(row["training_seed"]), str(row["arm"])) for row in endpoints}
    require(len(endpoints) == 6 and keys == EXPECTED_KEYS, "endpoint matrix is not complete 2x3")
    hashes = {str(row["checkpoint_sha256"]) for row in endpoints}
    require(len(hashes) == 6, "each endpoint must have a distinct checkpoint hash")
    for endpoint in endpoints:
        validate_endpoint(endpoint, experiment_root)

    evidence = load_json(BLIND_ROOT / "blind_evidence.json")
    evidence_ids = [row.get("id") for row in evidence.get("deviations", [])]
    require(evidence_ids == handoff.get("documented_deviations"), "D1-D5 binding drift")

    return {
        "status": "passed",
        "handoff": str(handoff_path.resolve()),
        "handoff_sha256": sha256_file(handoff_path),
        "endpoint_count": len(endpoints),
        "sample_block_count": len(handoff["evaluation_contract"]["sample_blocks"]),
        "raw_checkpoint_rehash": "passed" if experiment_root is not None else "not_run",
        "independent_quality_blind_review_claimed": False,
        "protocol_exact_claimed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        help="optional external seed-4/5 experiment root for raw checkpoint rehash",
    )
    args = parser.parse_args()
    try:
        report = validate_handoff(args.handoff.resolve(), args.experiment_root)
    except (HandoffError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"[validate_role_e_handoff] ERROR: {exc}") from exc
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
