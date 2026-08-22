import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = (
    ROOT
    / "analysis"
    / "q256_target_weight_factorial"
    / "seed14_18_learning_curve_results"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_manifest_binds_compact_results_and_server_receipt() -> None:
    manifest = json.loads((RESULTS / "REPORT_MANIFEST.json").read_text())
    assert set(manifest) == {
        "SERVER_ARCHIVE_RECEIPT.json",
        "q256_seed14_18_learning_curve_aggregate.csv",
        "q256_seed14_18_learning_curve_audit.json",
        "q256_seed14_18_learning_curve_report.md",
        "q256_seed14_18_learning_curve_results.csv",
    }
    for name, item in manifest.items():
        path = RESULTS / name
        assert path.stat().st_size == item["bytes"]
        assert sha256_file(path) == item["sha256"]


def test_final_audit_and_full_matrix_are_exact() -> None:
    audit = json.loads(
        (RESULTS / "q256_seed14_18_learning_curve_audit.json").read_text()
    )
    assert audit["status"] == "PASS"
    assert audit["classification"] == (
        "post_preregistration_secondary_sensitivity_learning_curve"
    )
    assert audit["training"] == {
        "failure_receipts": 0,
        "milestones_expected": 120,
        "milestones_pass": 120,
        "training_worker_pass": 5,
        "trajectories_expected": 20,
        "trajectories_pass": 20,
    }
    assert audit["evaluation"] == {
        "ab_jobs": 120,
        "cd_jobs": 120,
        "durable_receipts": 240,
        "feature_sha_mismatches": 0,
        "integrity_failures": 0,
        "jobs_expected": 240,
        "jobs_pass": 240,
        "unique_job_keys": 240,
    }
    rows = list(
        csv.DictReader(
            (RESULTS / "q256_seed14_18_learning_curve_results.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    assert len(rows) == 240
    expected = {
        (str(seed), arm, str(budget), str(nfe))
        for seed in range(14, 19)
        for arm in "ABCD"
        for budget in (384, 512, 640, 768, 896, 1024)
        for nfe in (1, 2)
    }
    assert {
        (row["seed"], row["arm"], row["budget_kimg"], row["nfe"])
        for row in rows
    } == expected
    assert all(math.isfinite(float(row["fid50k_full"])) for row in rows)
    assert all(math.isfinite(float(row["kid50k_full"])) for row in rows)


def test_aggregate_endpoints_and_claim_boundary() -> None:
    rows = list(
        csv.DictReader(
            (RESULTS / "q256_seed14_18_learning_curve_aggregate.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    assert len(rows) == 48
    indexed = {
        (row["arm"], int(row["nfe"]), int(row["budget_kimg"])): row
        for row in rows
    }
    assert math.isclose(float(indexed[("D", 1, 1024)]["fid_mean"]), 8.66036393654175)
    assert math.isclose(
        float(indexed[("A", 2, 1024)]["kid_mean"]), 0.0010063388813813827
    )
    report = (RESULTS / "q256_seed14_18_learning_curve_report.md").read_text()
    for phrase in (
        "post-preregistration secondary sensitivity",
        "does not establish a universal arm ranking",
        "240/240 unique FP32 KID/FID-50k jobs",
        "Full receipt-level evidence and checkpoints are retained",
    ):
        assert phrase in report


def test_server_archive_is_double_written_and_hash_bound() -> None:
    receipt = json.loads((RESULTS / "SERVER_ARCHIVE_RECEIPT.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["archive_validation"] == "PASS"
    assert receipt["receipt_count"] == 240
    assert receipt["durable_copy"]["sha256_verified"] is True
    assert receipt["training_copy"]["sha256_verified"] is True
    assert len(receipt["archive_sha256"]) == 64
