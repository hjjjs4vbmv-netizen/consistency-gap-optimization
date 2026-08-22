import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis" / "q256_target_weight_factorial"
EVALUATION = ANALYSIS / "seed8_13_secondary_extension_frozen_evaluation"
TRAINING = ANALYSIS / "seed8_13_secondary_extension_receipts"
RESULTS = ANALYSIS / "seed8_13_secondary_extension_results_v1"
SUPPORT = ANALYSIS / "extension_support" / "seed8_13"
COLLECTOR = ANALYSIS / "extension_support" / "collect_q256_seed8_13_fid_kid.py"

SEEDS = tuple(range(8, 14))
ARMS = ("A", "B", "C", "D")
NFES = (1, 2)
METRICS = ("kid50k_full", "fid50k_full")
TRAINING_COMMIT = "dcca41b19e7c45512b5fbe98776520396a1bf9ac"
EVALUATOR_COMMIT = "9d06ccc72545d4189af1b86de7f629f9c09d3f73"
DATASET_SHA256 = "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372"


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_aggregate_is_the_exact_48_job_matrix() -> None:
    payload = load_json(RESULTS / "q256_factorial_seed8_13_secondary_extension_results.json")
    assert payload["status"] == "VERIFIED_PASS_48_OF_48"
    assert payload["classification"] == "secondary_precision_extension_not_original_preregistration"
    assert payload["protocol"] == {
        "arms": list(ARMS),
        "dataset_sha256": DATASET_SHA256,
        "evaluator_commit": EVALUATOR_COMMIT,
        "metric_seed": 20260730,
        "metrics": list(METRICS),
        "nfe": list(NFES),
        "nfe2_mid_t": 0.821,
        "precision": "fp32",
        "sample_count": 50000,
        "sample_seeds": "0-49999",
        "seeds": list(SEEDS),
        "training_commit": TRAINING_COMMIT,
        "transfer_sha256": "4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da",
    }
    expected = {(seed, arm, nfe) for seed in SEEDS for arm in ARMS for nfe in NFES}
    rows = payload["results"]
    assert len(rows) == 48
    assert {(row["seed"], row["arm"], row["nfe"]) for row in rows} == expected
    assert all(row["status"] == "PASS" for row in rows)

    csv_rows = list(
        csv.DictReader(
            (RESULTS / "q256_factorial_seed8_13_secondary_extension_results.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    assert len(csv_rows) == 48
    by_key = {(row["seed"], row["arm"], row["nfe"]): row for row in rows}
    for row in csv_rows:
        key = (int(row["seed"]), row["arm"], int(row["nfe"]))
        aggregate = by_key[key]
        assert float(row["fid50k_full"]) == aggregate["fid50k_full"]
        assert float(row["kid50k_full"]) == aggregate["kid50k_full"]
        assert row["receipt_sha256"] == aggregate["receipt_sha256"]


def test_receipts_bind_metrics_features_and_primary_first_order() -> None:
    plan_path = EVALUATION / "evaluation_plan.json"
    plan = load_json(plan_path)
    completion = load_json(EVALUATION / "evaluation_completion.json")
    expected_ids = [
        f"seed{seed}-arm{arm}-nfe{nfe}"
        for nfe in NFES
        for seed in SEEDS
        for arm in ARMS
    ]
    assert [job["job_id"] for job in plan["jobs"]] == expected_ids
    assert completion["completed_job_ids"] == expected_ids
    assert completion["evaluation_plan_sha256"] == sha256_file(plan_path)
    assert completion["status"] == "PASS"

    aggregate = load_json(RESULTS / "q256_factorial_seed8_13_secondary_extension_results.json")
    rows = {(row["seed"], row["arm"], row["nfe"]): row for row in aggregate["results"]}
    for job in plan["jobs"]:
        job_id = job["job_id"]
        receipt_path = EVALUATION / "receipts" / f"{job_id}.json"
        receipt = load_json(receipt_path)
        key = (receipt["seed"], receipt["arm"], receipt["nfe"])
        assert receipt["status"] == "passed"
        assert receipt["returncode"] == 0
        assert receipt["execution_error"] is None
        assert receipt["dataset_sha256"] == DATASET_SHA256
        assert receipt["evaluator_source_git_head"] == EVALUATOR_COMMIT
        assert receipt["gpu_exclusivity_monitor"]["status"] == "PASS"
        assert receipt["gpu_exclusivity_monitor"]["foreign_process_incident"] is None
        assert rows[key]["receipt_sha256"] == sha256_file(receipt_path)
        values = {}
        for item in receipt["metrics"]:
            raw_path = EVALUATION / "jobs" / job_id / f"metric-{item['metric']}.jsonl"
            assert item["raw_sha256"] == sha256_file(raw_path)
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            values[item["metric"]] = float(raw["results"][item["metric"]])
        assert rows[key]["kid50k_full"] == values["kid50k_full"]
        assert rows[key]["fid50k_full"] == values["fid50k_full"]
        artifacts = receipt["artifacts"]
        kid_sha = artifacts["generated-features-kid50k_full-repeat00.npy"]["sha256"]
        fid_sha = artifacts["generated-features-fid50k_full-repeat00.npy"]["sha256"]
        assert kid_sha == fid_sha == rows[key]["generated_feature_sha256"]
        block_path = EVALUATION / "jobs" / job_id / "sampling_block_diagnostics_v1.json"
        assert receipt["sampling_block_diagnostics"]["sha256"] == sha256_file(block_path)


def test_training_audits_and_matrix_bindings_are_complete() -> None:
    matrix_path = EVALUATION / "matrix_binding" / "extension_matrix_binding.json"
    matrix = load_json(matrix_path)
    assert matrix["status"] == "PASS"
    assert matrix["cell_count"] == 24
    assert matrix["seeds"] == list(SEEDS)
    assert matrix["training_source_git_head"] == TRAINING_COMMIT
    assert matrix["metric_numerical_semantics_changed"] is False
    assert len(matrix["cell_receipts"]) == 24
    for item in matrix["cell_receipts"]:
        path = EVALUATION / "matrix_binding" / "cells" / Path(item["path"]).name
        assert path.stat().st_size == item["bytes"]
        assert sha256_file(path) == item["sha256"]

    assert len(matrix["integrity_audit_receipts"]) == 6
    for item in matrix["integrity_audit_receipts"]:
        path = TRAINING / "integrity" / Path(item["path"]).name
        audit = load_json(path)
        assert sha256_file(path) == item["sha256"]
        assert audit["status"] == "PASS"
        assert audit["four_arm_complete"] is True
        assert audit["denominator_integrity"] is True
        assert audit["common_initial_state_identity"] is True
        assert audit["telemetry_identity_checks"]["all_pass"] is True
        seed = audit["seed"]
        common_states = set()
        for arm in ARMS:
            cell = audit["cells"][arm]
            assert cell["attempts"] == 2000
            assert cell["processed_kimg"] == 256.0
            assert cell["semantic_nonfinite_count"] == 0
            assert cell["raw_grad_skip_mismatch_count"] == 0
            assert cell["nonpositive_denominator_count"] == 0
            initial_path = TRAINING / f"seed{seed}" / f"arm{arm}" / "initial_state_receipt_v1.json"
            expected = cell["artifact_hashes"]["initial_state_receipt_v1.json"]
            assert initial_path.stat().st_size == expected["bytes"]
            assert sha256_file(initial_path) == expected["sha256"]
            common_states.add(load_json(initial_path)["common_initial_state_sha256"])
        assert len(common_states) == 1


def test_claim_boundary_and_support_provenance() -> None:
    report = " ".join(
        (RESULTS / "q256_factorial_seed8_13_secondary_extension_results.md")
        .read_text(encoding="utf-8")
        .split()
    )
    required = (
        "secondary precision extension",
        "not a valid execution of PR #72 Cohort III",
        "started 57 minutes 26 seconds before that freeze",
        "two A100 queues rather than PR #72's fixed five-GPU seed mapping",
        "cannot be relabeled as prospective held-out confirmation",
        "4/6 rather than uniform",
        "Seed13 reverses strongly",
        "must not be described as universal B dominance",
        "Statistical fallacy scan (11/11)",
    )
    for phrase in required:
        assert phrase in report

    payload = load_json(RESULTS / "q256_factorial_seed8_13_secondary_extension_results.json")
    boundary = payload["chronology_and_claim_boundary"]
    assert boundary["secondary_run_started_before_cohort3_freeze"] is True
    assert boundary["is_pr72_cohort3_execution"] is False
    assert boundary["is_prospective_confirmation"] is False
    assert boundary["overlapping_seed_labels"] == list(range(8, 13))

    expected_support_hashes = {
        "audit_q256_seed8_13_extension.py": "35ae39f777a22774b188e2f23d1db58803136acac4e924cd99855a259fb27b90",
        "run_q256_seed8_13_chain.sh": "2ac4c07b4227141403335fdd3023b10990c57a56f9644f7c20fe49d137c4bd7f",
        "run_q256_seed8_13_frozen_evaluation.py": "7cbe0b91ea03a53c012d54be782cf6229853e600a8c61e5597dd7f0cc0af7749",
        "run_q256_seed8_13_frozen_evaluation.sh": "17e372995b913f1bf4f606e612bdb791bbf36eca55d0c2cb5256b5f8dd1d55f7",
        "run_q256_seed8_13_worker.sh": "a08229192bfc3871809514ce8db070c1ebe33a54ac88520f6e604a758c1ce87d",
        "run_q256_direct_frozen_evaluation_v6.py": "7e687c7664fdd204153f658539393c6ef6dc7e4fb1c62d54d37414433f13b67f",
    }
    for name, expected in expected_support_hashes.items():
        assert sha256_file(SUPPORT / name) == expected


def test_collector_rebuild_is_byte_identical(tmp_path: Path) -> None:
    outdir = tmp_path / "rebuilt"
    subprocess.run(
        [
            sys.executable,
            str(COLLECTOR),
            "--eval-root",
            str(EVALUATION),
            "--training-root",
            str(TRAINING),
            "--support-root",
            str(SUPPORT),
            "--outdir",
            str(outdir),
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    for name in (
        "q256_factorial_seed8_13_secondary_extension_results.csv",
        "q256_factorial_seed8_13_secondary_extension_results.json",
        "q256_factorial_seed8_13_secondary_extension_results.md",
    ):
        assert (outdir / name).read_bytes() == (RESULTS / name).read_bytes()
