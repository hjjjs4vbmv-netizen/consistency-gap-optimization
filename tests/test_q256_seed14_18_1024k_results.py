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
    / "seed14_18_1024k_results"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_report_manifest_binds_every_result_file() -> None:
    manifest = json.loads((RESULTS / "REPORT_MANIFEST.json").read_text(encoding="utf-8"))
    assert set(manifest) == {
        "q256_seed14_18_1024k_fid_kid_report.md",
        "q256_seed14_18_1024k_fid_kid_results.csv",
        "q256_seed14_18_1024k_fid_kid_results.json",
    }
    for name, item in manifest.items():
        path = RESULTS / name
        assert path.stat().st_size == item["bytes"]
        assert sha256_file(path) == item["sha256"]


def test_result_matrix_and_training_endpoints_are_complete() -> None:
    payload = json.loads(
        (RESULTS / "q256_seed14_18_1024k_fid_kid_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status"] == "VERIFIED_PASS_40_OF_40"
    assert payload["protocol"] == {
        "arms": ["A", "B", "C", "D"],
        "metric_seed": 20260730,
        "metrics": ["kid50k_full", "fid50k_full"],
        "nfe": [1, 2],
        "nfe2_mid_t": 0.821,
        "precision": "fp32",
        "sample_count": 50000,
        "sample_seeds": "0-49999",
        "seeds": [14, 15, 16, 17, 18],
        "training_budget_kimg": 1024,
    }
    expected_jobs = {
        (seed, arm, nfe)
        for seed in range(14, 19)
        for arm in "ABCD"
        for nfe in (1, 2)
    }
    assert len(payload["results"]) == 40
    assert {
        (row["seed"], row["arm"], row["nfe"]) for row in payload["results"]
    } == expected_jobs
    assert all(row["status"] == "PASS" for row in payload["results"])

    assert len(payload["training_cells"]) == 20
    for cell in payload["training_cells"]:
        assert cell["attempted_iteration"] == 8000
        assert cell["processed_kimg"] == 1024.0
        assert cell["step_skipped"] == 0
        assert math.isfinite(cell["loss"])
    assert payload["recovery"]["armA_retrained"] is False

    csv_rows = list(
        csv.DictReader(
            (RESULTS / "q256_seed14_18_1024k_fid_kid_results.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    assert len(csv_rows) == 40


def test_primary_result_and_claim_boundary_are_exact() -> None:
    payload = json.loads(
        (RESULTS / "q256_seed14_18_1024k_fid_kid_results.json").read_text(
            encoding="utf-8"
        )
    )
    summaries = {
        (row["nfe"], row["metric"], row["contrast"]): row
        for row in payload["contrast_summaries"]
    }
    primary = summaries[(1, "fid50k_full", "B-A")]
    assert math.isclose(primary["mean"], -0.2738535679993803)
    assert (primary["negative_count"], primary["positive_count"]) == (3, 2)
    secondary_kid = summaries[(2, "kid50k_full", "B-A")]
    assert math.isclose(secondary_kid["mean"], 0.00010773403953952855)
    assert (secondary_kid["negative_count"], secondary_kid["positive_count"]) == (
        0,
        5,
    )

    report = (
        RESULTS / "q256_seed14_18_1024k_fid_kid_report.md"
    ).read_text(encoding="utf-8")
    required = (
        "post-preregistration seed14–18 secondary sensitivity matrix",
        "D has the lowest cohort-mean NFE1 FID",
        "does not support a uniformly best B arm",
        "arm A was not retrained",
        "Coverage: **11/11 checked**",
        "not universal optimizer-mechanism or causal-percentage claims",
    )
    for phrase in required:
        assert phrase in report
