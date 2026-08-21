import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis" / "q256_target_weight_factorial"
EVALUATION = ANALYSIS / "seed14_18_extension_frozen_evaluation"
CSV_PATH = ANALYSIS / "q256_factorial_seed14_18_extension_results.csv"
JSON_PATH = ANALYSIS / "q256_factorial_seed14_18_extension_results.json"
SCOPE_PATH = ANALYSIS / "q256_factorial_seed14_18_extension_report.md"

SEEDS = tuple(range(14, 19))
ARMS = ("A", "B", "C", "D")
NFES = (1, 2)
METRICS = ("kid50k_full", "fid50k_full")
TRAINING_COMMIT = "dcca41b19e7c45512b5fbe98776520396a1bf9ac"
EVALUATOR_COMMIT = "d6aba02fb88e9db0993623895eb2228ed717d810"
DATASET_SHA256 = "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_seed14_18_aggregate_is_the_exact_40_job_matrix() -> None:
    payload = load_json(JSON_PATH)
    assert payload["status"] == "VERIFIED_PASS_40_OF_40"
    assert payload["protocol"] == {
        "arms": list(ARMS),
        "metric_seed": 20260730,
        "metrics": list(METRICS),
        "nfe": list(NFES),
        "nfe2_mid_t": 0.821,
        "precision": "fp32",
        "sample_count": 50000,
        "sample_seeds": "0-49999",
        "seeds": list(SEEDS),
    }
    expected = {(seed, arm, nfe) for seed in SEEDS for arm in ARMS for nfe in NFES}
    results = payload["results"]
    assert len(results) == 40
    observed = {(row["seed"], row["arm"], row["nfe"]) for row in results}
    assert observed == expected
    assert all(row["status"] == "PASS" for row in results)
    assert all(math.isfinite(float(row["fid50k_full"])) for row in results)
    assert all(math.isfinite(float(row["kid50k_full"])) for row in results)

    csv_rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8")))
    assert len(csv_rows) == 40
    json_by_key = {
        (row["seed"], row["arm"], row["nfe"]): row for row in results
    }
    for row in csv_rows:
        key = (int(row["seed"]), row["arm"], int(row["nfe"]))
        aggregate = json_by_key[key]
        assert float(row["fid50k_full"]) == float(aggregate["fid50k_full"])
        assert float(row["kid50k_full"]) == float(aggregate["kid50k_full"])
        assert row["receipt_sha256"] == aggregate["receipt_sha256"]


def test_seed14_18_receipts_bind_raw_metrics_and_feature_identity() -> None:
    aggregate = load_json(JSON_PATH)
    rows = {
        (row["seed"], row["arm"], row["nfe"]): row
        for row in aggregate["results"]
    }
    seen = set()
    for seed in SEEDS:
        seed_root = EVALUATION / f"seed{seed}"
        plan_path = seed_root / "evaluation_plan.json"
        plan = load_json(plan_path)
        completion = load_json(seed_root / "WORKER_PASS.json")
        assert plan["seed"] == seed
        assert plan["physical_gpu_index"] == seed - 14
        assert plan["training_source_commit"] == TRAINING_COMMIT
        assert plan["evaluator_source_commit"] == EVALUATOR_COMMIT
        assert plan["dataset"]["sha256"] == DATASET_SHA256
        assert plan["protocol"]["metrics"] == list(METRICS)
        assert plan["protocol"]["sample_count"] == 50000
        assert plan["protocol"]["sample_seeds"] == "0-49999"
        assert plan["protocol"]["metric_seed"] == 20260730
        assert plan["protocol"]["precision"] == "fp32"
        assert plan["protocol"]["nfe_modes"] == {"1": [], "2": [0.821]}
        assert completion["status"] == "WORKER_PASS"
        assert completion["jobs_completed"] == 8
        assert completion["evaluation_plan_sha256"] == sha256_file(plan_path)
        assert len(completion["job_receipts"]) == 8

        for completion_item in completion["job_receipts"]:
            job_id = completion_item["job_id"]
            receipt_path = seed_root / "receipts" / f"{job_id}.json"
            receipt = load_json(receipt_path)
            assert completion_item["receipt_sha256"] == sha256_file(receipt_path)
            assert receipt["status"] == "PASS"
            job = receipt["job"]
            key = (job["seed"], job["arm"], job["nfe"])
            assert key not in seen
            assert job["job_id"] == job_id
            assert job["mid_t"] == ([] if job["nfe"] == 1 else [0.821])
            metrics = receipt["validation"]["metrics"]
            assert [item["metric"] for item in metrics] == list(METRICS)
            values = {item["metric"]: float(item["value"]) for item in metrics}
            assert float(rows[key]["fid50k_full"]) == values["fid50k_full"]
            assert float(rows[key]["kid50k_full"]) == values["kid50k_full"]

            job_root = seed_root / "jobs" / job_id
            for item in metrics:
                raw_path = job_root / f"metric-{item['metric']}.jsonl"
                assert sha256_file(raw_path) == item["raw_sha256"]
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                assert raw["metric"] == item["metric"]
                assert float(raw["results"][item["metric"]]) == float(item["value"])

            artifacts = receipt["validation"]["artifacts"]
            kid_sha = artifacts["generated-features-kid50k_full-repeat00.npy"]["sha256"]
            fid_sha = artifacts["generated-features-fid50k_full-repeat00.npy"]["sha256"]
            assert kid_sha == fid_sha == rows[key]["generated_feature_sha256"]
            block_path = job_root / "sampling_block_diagnostics_v1.json"
            assert sha256_file(block_path) == receipt["validation"]["sampling_block_diagnostics"]["sha256"]
            block = load_json(block_path)
            assert block["sample_count"] == 50000
            assert block["fixed_block_count"] == 10
            seen.add(key)

    assert len(seen) == 40


def test_seed14_18_report_keeps_the_claim_boundary() -> None:
    text = " ".join(SCOPE_PATH.read_text(encoding="utf-8").split())
    required = (
        "post-preregistration secondary seed/sensitivity extension",
        "does not replace, enlarge, or retrospectively relabel",
        "all observed seeds, descriptive only",
        "not a pooled confirmatory `n=10` study",
        "do not support a universal target-geometry-dominant decomposition",
        "no causal percentage decomposition or universal mechanism claim",
    )
    for phrase in required:
        assert phrase in text
