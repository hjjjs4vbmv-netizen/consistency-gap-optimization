#!/usr/bin/env python3
"""Validate the portable structure of the q=128 1024-kimg retrospective package.

This intentionally validates only repository-local consistency.  It does not
claim to authenticate historical checkpoints or reproduce metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path


EXPECTED_METRICS = {"kid50k_full", "fid50k_full"}
EXPECTED_NFE = {"1", "2"}
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    raise SystemExit(f"validation failed: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    package = args.package
    source_manifest = package / "source_run_manifest.json"
    environment_manifest = package / "environment_manifest.json"
    checkpoint_rows = rows(package / "checkpoint_manifest.csv")
    metric_rows = rows(package / "evaluation_results.csv")
    paired_rows = rows(package / "paired_differences.csv")
    job_rows = rows(REPOSITORY_ROOT / "evaluation" / "q128_1024k_retrospective_job_status.csv")

    if len(checkpoint_rows) != 6:
        fail(f"expected 6 checkpoint rows, found {len(checkpoint_rows)}")
    checkpoint_by_id = {row["checkpoint_id"]: row for row in checkpoint_rows}
    if len(checkpoint_by_id) != 6:
        fail("checkpoint IDs are not unique")
    for row in checkpoint_rows:
        if row["budget_kimg"] != "1024" or row["status"] != "incomplete":
            fail(f"invalid checkpoint record for {row['checkpoint_id']}")

    if len(metric_rows) != 24:
        fail(f"expected 24 metric rows, found {len(metric_rows)}")
    combinations = set()
    metric_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in metric_rows:
        if row["evidence_class"] != "retrospective_supplementary":
            fail("metric rows are not retrospective/supplementary")
        if row["phase"] != "retrospective" or row["record_status"] != "reported_unverified":
            fail("metric row phase/status is incorrect")
        if row["integrity_receipt_status"] != "unavailable":
            fail("metric row retains an unsupported integrity receipt status")
        if row["budget_kimg"] != "1024" or row["metric_name"] not in EXPECTED_METRICS:
            fail("metric row has an invalid budget or metric")
        if row["nfe"] not in EXPECTED_NFE or row["checkpoint_id"] not in checkpoint_by_id:
            fail("metric row has an invalid NFE or checkpoint ID")
        if row["checkpoint_sha256"] != checkpoint_by_id[row["checkpoint_id"]]["checkpoint_sha256"]:
            fail("checkpoint SHA does not match checkpoint manifest")
        if row["run_id"].startswith("/") or "/root/" in row["run_id"]:
            fail("metric row contains an absolute run path")
        key = (row["checkpoint_id"], row["nfe"], row["metric_name"])
        combinations.add(key)
        metric_by_key[key] = row
    if len(combinations) != 24:
        fail("metric rows do not cover the complete 6 x 2 x 2 grid")

    if len(paired_rows) != 12:
        fail(f"expected 12 paired rows, found {len(paired_rows)}")
    for row in paired_rows:
        if (row["evidence_class"], row["claim_scope"]) != (
            "retrospective_supplementary", "provisional_within_q128_only"
        ):
            fail("paired record has an invalid evidence scope")
        fixed = metric_by_key.get((row["fixed_checkpoint_id"], row["nfe"], row["metric"]))
        candidate = metric_by_key.get((row["global_only_checkpoint_id"], row["nfe"], row["metric"]))
        if fixed is None or candidate is None:
            fail("paired record cannot be joined to raw metrics")
        if Decimal(row["fixed_value"]) != Decimal(fixed["metric_value"]):
            fail("paired fixed value differs from raw metric")
        if Decimal(row["global_only_value"]) != Decimal(candidate["metric_value"]):
            fail("paired candidate value differs from raw metric")
        reported_delta = Decimal(row["delta"])
        recomputed_delta = Decimal(row["global_only_value"]) - Decimal(row["fixed_value"])
        if abs(reported_delta - recomputed_delta) > Decimal("1e-12"):
            fail("paired delta does not equal candidate minus fixed")

    if len(job_rows) != 12:
        fail(f"expected 12 job rows, found {len(job_rows)}")
    for row in job_rows:
        if row["budget_kimg"] != "1024" or row["record_status"] != "reported_unverified":
            fail("job status is not consistently retrospective")
        if row["integrity_receipt_status"] != "unavailable":
            fail("job status retains an unsupported integrity receipt status")
        if row["run_id"].startswith("/") or "/root/" in row["run_id"]:
            fail("job status contains an absolute run path")

    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    environment = json.loads(environment_manifest.read_text(encoding="utf-8"))
    if source["formal_status"] != "not_formal" or environment["formal_status"] != "not_formal":
        fail("package must not be classified as formal")
    if environment["source_run_manifest_sha256"] != sha256(source_manifest):
        fail("environment manifest source-run SHA mismatch")
    print("pass: 6 checkpoints, 12 reported jobs, and 24 reported raw metric rows are internally consistent; provenance remains incomplete")


if __name__ == "__main__":
    main()
