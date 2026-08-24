import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "q256_longitudinal_factorial_seed6_13"
JOB_RE = re.compile(r"^seed\d+-arm[A-D]-k\d+-nfe[12]\.json$")


def test_receipt_coverage_and_frozen_protocol():
    payload = json.loads((RESULTS / "evaluation_receipts.json").read_text())
    receipts = payload["receipts"]
    assert payload["receipt_count"] == 336
    assert len(receipts) == 336
    keys = {
        (row["seed"], row["arm"], row["budget_kimg"], row["nfe"])
        for row in receipts
    }
    assert len(keys) == 336
    assert all(row["status"] == "PASS" for row in receipts)
    assert all(row["sample_count"] == 50_000 for row in receipts)
    assert all(row["metric_seed"] == 20260730 for row in receipts)
    assert all(JOB_RE.match(row["receipt_file"]) for row in receipts)

    expected = {
        *((seed, arm, budget, nfe) for seed in (6, 7) for arm in ("C", "D") for budget in (384, 512, 640, 768, 896, 1024) for nfe in (1, 2)),
        *((seed, arm, budget, nfe) for seed in range(8, 14) for arm in "ABCD" for budget in (384, 512, 640, 768, 896, 1024) for nfe in (1, 2)),
    }
    assert keys == expected


def test_machine_readable_headlines_match_report():
    summary = json.loads((RESULTS / "factorial_summary.json").read_text())
    nfe2 = summary["balanced"]["nfe2"]
    assert abs(nfe2["arms"]["B"]["aulc_mean"] - 18.8516) < 1e-4
    assert abs(nfe2["arms"]["D"]["aulc_mean"] - 21.2206) < 1e-4
    assert abs(nfe2["aulc_contrasts"]["B_minus_A"]["mean"] + 13.8880) < 1e-4
    assert abs(nfe2["aulc_contrasts"]["D_minus_A"]["mean"] + 11.5190) < 1e-4
    assert nfe2["aulc_contrasts"]["D_minus_A"]["negative_seed_count"] == 6
    assert nfe2["arms"]["B"]["sustained_fid10_median_kimg"] == 576


def test_csv_shapes_and_no_generated_arrays():
    expected_rows = {
        "evaluation_results.csv": 336,
        "longitudinal_summary.csv": 48,
        "aulc_per_seed.csv": 48,
        "factorial_contrasts.csv": 12,
    }
    for filename, count in expected_rows.items():
        with (RESULTS / filename).open(newline="") as handle:
            assert len(list(csv.DictReader(handle))) == count
    assert not list(RESULTS.rglob("*.npy"))
    assert not list(RESULTS.rglob("*.pkl"))
