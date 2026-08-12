#!/usr/bin/env python3
"""Quality-blind formal adjudication for the seed-4/5 replication package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "gap_lr_matched_q128_s45_replication_v1"
EXECUTION_PROTOCOL_COMMIT = "583c2fe0f914fc1191903d747737fd54b4ba1eef"
EXPECTED_RUN_KEYS = {
    "seed4_A",
    "seed4_B",
    "seed4_C",
    "seed5_A",
    "seed5_B",
    "seed5_C",
}
EXPECTED_RUNS = {
    "seed4_A": ("arm_a_g1_0_lr_fixed_s4", 4, "A", 1.0, 0.0001),
    "seed4_B": ("arm_b_g1_3_lr_fixed_s4", 4, "B", 1.3, 0.0001),
    "seed4_C": (
        "arm_c_g1_3_lr_matched_s4",
        4,
        "C",
        1.3,
        0.00012963523762588692,
    ),
    "seed5_A": ("arm_a_g1_0_lr_fixed_s5", 5, "A", 1.0, 0.0001),
    "seed5_B": ("arm_b_g1_3_lr_fixed_s5", 5, "B", 1.3, 0.0001),
    "seed5_C": (
        "arm_c_g1_3_lr_matched_s5",
        5,
        "C",
        1.3,
        0.00012963523762588692,
    ),
}
EXPECTED_RUN_IDS = {item[0] for item in EXPECTED_RUNS.values()}
EXPECTED_DEVIATIONS = {"D1", "D2", "D3", "D4", "D5"}
REQUIRED_EXCLUSIONS = {
    "protocol-exact execution",
    "historically observed bitwise pre-update parameter identity",
    "bitwise training equivalence across devices",
    "throughput, latency, or GPU performance comparisons",
}
FORBIDDEN_PUBLIC_TEXT = (
    "/data/",
    "/Users/",
    "172.16.",
    "ECT001@",
    "GPU-d791",
    "GPU-ef9e",
)
SHA256_KEYS = {
    "sha256",
    "internal_receipt_sha256",
    "verifier_source_sha256",
    "execution_protocol_commit",
    "adjudication_tooling_commit",
    "training_code_commit",
    "source_audit_receipt_sha256",
    "matrix_sha256",
    "dataset_sha256",
    "transfer_checkpoint_sha256",
    "objective_evidence_sha256",
    "initialization_reconstruction_sha256",
    "adjudicator_source_sha256",
    "tool_source_sha256",
    "evidence_builder_source_sha256",
}


def fail(message: str) -> None:
    raise SystemExit("BLIND ADJUDICATION REJECTED: " + message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    value = json.loads(text)
    if not isinstance(value, dict):
        fail(f"expected JSON object: {path}")
    return value


def public_text_is_sanitized(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if any(marker in text for marker in FORBIDDEN_PUBLIC_TEXT):
        return False
    value = json.loads(text)

    def safe(item: Any, key: str | None = None) -> bool:
        if isinstance(item, dict):
            return all(safe(child, str(child_key)) for child_key, child in item.items())
        if isinstance(item, list):
            return all(safe(child, key) for child in item)
        if not isinstance(item, str):
            return True
        if key in SHA256_KEYS or key.endswith("_sha256"):
            return bool(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", item))
        forbidden_patterns = (
            r"(?:^|\s)(?:/|~[/\\]|[A-Za-z]:[\\/])",
            r"(?:^|\s)[^\s/@]+@[^\s/]+",
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            r"\bGPU-[0-9a-fA-F-]{16,}\b",
            r"(?:https?|file|ssh)://",
        )
        return not any(re.search(pattern, item) for pattern in forbidden_patterns)

    return safe(value)


def evaluate(
    evidence: dict[str, Any],
    initialization: dict[str, Any],
    public_receipts: dict[str, dict[str, Any]],
) -> tuple[str, list[str], list[str]]:
    failures: list[str] = []
    affected_runs: set[str] = set()

    if (
        evidence.get("receipt_type")
        != "gap_lr_seed_replication_quality_blind_evidence"
        or evidence.get("status") != "adjudication_ready"
        or evidence.get("experiment_id") != EXPERIMENT_ID
        or evidence.get("bindings", {}).get("execution_protocol_commit")
        != EXECUTION_PROTOCOL_COMMIT
    ):
        failures.append("objective evidence identity/binding failed")
        affected_runs.update(EXPECTED_RUN_KEYS)
    if evidence.get("quality_blind", {}).get(
        "generation_quality_metrics_accessed"
    ) is not False:
        failures.append("quality-blind boundary was not preserved")
        affected_runs.update(EXPECTED_RUN_KEYS)

    integrity = evidence.get("per_run_integrity", {})
    manifest = integrity.get("public_receipts", {})
    if (
        integrity.get("passed_runs") != 6
        or integrity.get("required_runs") != 6
        or integrity.get("all_artifact_hashes_recomputed") is not True
        or set(manifest) != EXPECTED_RUN_KEYS
        or set(public_receipts) != EXPECTED_RUN_KEYS
    ):
        failures.append("six-run artifact integrity gate failed")
        affected_runs.update(EXPECTED_RUN_KEYS)
    for key in EXPECTED_RUN_KEYS & set(public_receipts):
        receipt = public_receipts[key]
        run_id, seed, arm, gap, lr = EXPECTED_RUNS[key]
        entry = manifest.get(key, {})
        if (
            receipt.get("receipt_type")
            != "gap_lr_seed_replication_run_integrity_public"
            or receipt.get("status") != "passed"
            or receipt.get("publication", {}).get("sanitized_for_github")
            is not True
            or receipt.get("experiment_id") != EXPERIMENT_ID
            or receipt.get("run_id") != run_id
            or receipt.get("seed") != seed
            or receipt.get("arm") != arm
            or receipt.get("gap_scale") != gap
            or receipt.get("learning_rate") != lr
            or receipt.get("bindings", {}).get("execution_protocol_commit")
            != EXECUTION_PROTOCOL_COMMIT
            or not isinstance(
                receipt.get("bindings", {}).get("internal_receipt_sha256"), str
            )
            or entry.get("run_id") != run_id
            or entry.get("internal_receipt_sha256")
            != receipt.get("bindings", {}).get("internal_receipt_sha256")
            or receipt.get("completion", {})
            .get("summary", {})
            .get("amp_contract_passed")
            is not True
        ):
            failures.append(f"public per-run receipt failed: {key}")
            affected_runs.add(key)

    config = evidence.get("configuration_contract", {})
    if (
        set(config.get("within_seed_passed", {})) != {"4", "5"}
        or set(config.get("between_seed_passed", {})) != {"A", "B", "C"}
        or not all(config.get("within_seed_passed", {}).values())
        or not all(config.get("between_seed_passed", {}).values())
    ):
        failures.append("allowed-difference configuration contract failed")
        affected_runs.update(EXPECTED_RUN_KEYS)

    init_cross = initialization.get("cross_run", {})
    init_runs = initialization.get("runs", {})
    init_ok = (
        initialization.get("status") == "passed"
        and initialization.get("quality_blind", {}).get(
            "generation_quality_metrics_accessed"
        )
        is False
        and init_cross.get("all_six_reconstructed_net_hashes_equal") is True
        and init_cross.get("all_six_initialization_contract_hashes_equal")
        is True
        and set(init_runs) == EXPECTED_RUN_IDS
        and all(
            row.get("copy_contract", {}).get(
                "all_destination_tensors_covered"
            )
            is True
            and not row.get("copy_contract", {}).get(
                "missing_destination_names"
            )
            and not row.get("copy_contract", {}).get(
                "shape_dtype_mismatches"
            )
            and row.get("net", {}).get("tensor_count") == 424
            and row.get("ema", {}).get("sha256")
            == row.get("net", {}).get("sha256")
            for row in init_runs.values()
        )
    )
    if not init_ok:
        failures.append("expected initialization reconstruction failed")
        affected_runs.update(EXPECTED_RUN_KEYS)

    evidence_init = evidence.get("initialization", {})
    if (
        evidence_init.get("historical_observed_preupdate_parameter_hash")
        != "not_captured"
        or evidence_init.get("reconstructed_expected_initialization", {}).get(
            "historical_process_attestation"
        )
        is not False
    ):
        failures.append("initialization evidence boundary is misstated")
        affected_runs.update({"seed5_A", "seed5_B", "seed5_C"})
    previews = evidence_init.get("model_init_previews", {})
    if set(previews) != {"4", "5"} or any(
        row.get("max_abs_channel_delta_lsb", 2) > 1 for row in previews.values()
    ):
        failures.append("model-init preview drift exceeds adjudication policy")
        affected_runs.update({"seed5_A", "seed5_B", "seed5_C"})

    runtime = evidence.get("runtime", {})
    overlaps = runtime.get("directly_observed_overlaps", [])
    runtime_ok = (
        runtime.get("hardware", {}).get("same_model_driver_and_memory") is True
        and set(runtime.get("runs", {})) == EXPECTED_RUN_IDS
        and len(overlaps) == 2
        and all(item.get("different_logged_gpu_indices") is True for item in overlaps)
        and runtime.get("planned", {}).get("execution_mode") == "fully_serial"
        and {
            tuple(item.get("runs", [])) for item in overlaps
        }
        == {
            ("arm_b_g1_3_lr_fixed_s4", "arm_a_g1_0_lr_fixed_s5"),
            ("arm_c_g1_3_lr_matched_s4", "arm_b_g1_3_lr_fixed_s5"),
        }
    )
    if not runtime_ok:
        failures.append("runtime-deviation acceptance conditions failed")
        affected_runs.update(EXPECTED_RUN_KEYS)

    if {item.get("id") for item in evidence.get("deviations", [])} != EXPECTED_DEVIATIONS:
        failures.append("deviation set is incomplete")
        affected_runs.update(EXPECTED_RUN_KEYS)
    if not REQUIRED_EXCLUSIONS.issubset(set(evidence.get("claim_exclusions", []))):
        failures.append("required claim exclusions are missing")
        affected_runs.update(EXPECTED_RUN_KEYS)

    verdict = "rerun_required" if failures else "machine_recommends_acceptance"
    return verdict, failures, sorted(affected_runs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--initialization-reconstruction", required=True, type=Path)
    parser.add_argument("--public-receipt-dir", required=True, type=Path)
    parser.add_argument("--adjudication-tooling-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.adjudication_tooling_commit):
        fail("adjudication tooling commit must be a full Git SHA")

    evidence = load_json(args.evidence)
    initialization = load_json(args.initialization_reconstruction)
    manifest = evidence.get("per_run_integrity", {}).get("public_receipts", {})
    receipts: dict[str, dict[str, Any]] = {}
    receipt_bindings: dict[str, Any] = {}
    for key in sorted(EXPECTED_RUN_KEYS):
        entry = manifest.get(key, {})
        filename = entry.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            fail(f"unsafe or missing public receipt filename for {key}")
        path = args.public_receipt_dir / filename
        if not path.is_file() or not public_text_is_sanitized(path):
            fail(f"public receipt is missing or unsanitized: {key}")
        digest = file_sha256(path)
        if digest != entry.get("sha256"):
            fail(f"public receipt hash mismatch: {key}")
        receipts[key] = load_json(path)
        receipt_bindings[key] = {"file": filename, "sha256": digest}

    if not public_text_is_sanitized(args.evidence) or not public_text_is_sanitized(
        args.initialization_reconstruction
    ):
        fail("public evidence package contains a forbidden internal identifier")
    if (
        evidence.get("bindings", {}).get("initialization_reconstruction_sha256")
        != file_sha256(args.initialization_reconstruction)
    ):
        fail("initialization reconstruction hash mismatch")
    if (
        evidence.get("bindings", {}).get("adjudication_tooling_commit")
        != args.adjudication_tooling_commit
        or initialization.get("bindings", {}).get("adjudication_tooling_commit")
        != args.adjudication_tooling_commit
    ):
        fail("adjudication tooling commit binding mismatch")

    verdict, failures, affected_runs = evaluate(evidence, initialization, receipts)
    receipt = {
        "schema_version": 1,
        "receipt_type": "gap_lr_seed_replication_blind_adjudication",
        "status": "adjudicated",
        "experiment_id": EXPERIMENT_ID,
        "verdict": verdict,
        "adjudicated_at_utc": datetime.now(timezone.utc).isoformat(),
        "quality_blind": {
            "generation_quality_metrics_accessed": False,
            "decision_frozen_before_quality_evaluation": True,
        },
        "bindings": {
            "execution_protocol_commit": EXECUTION_PROTOCOL_COMMIT,
            "adjudication_tooling_commit": args.adjudication_tooling_commit,
            "objective_evidence_sha256": file_sha256(args.evidence),
            "initialization_reconstruction_sha256": file_sha256(
                args.initialization_reconstruction
            ),
            "public_per_run_receipts": receipt_bindings,
            "adjudicator_source_sha256": file_sha256(Path(__file__)),
        },
        "decision_policy": {
            "accept_if": [
                "all six runs pass strengthened integrity and artifact rehash",
                "only preregistered option differences are present",
                "all destination tensors are covered by the frozen transfer",
                "all six reconstructed expected initialization hashes match",
                "preview drift is no greater than one 8-bit level",
                "overlapping runs use different logged indices on equivalent A100 devices",
                "all deviations and claim exclusions are explicit",
            ],
            "rerun_if": [
                "any integrity/configuration/hash binding fails",
                "transfer coverage or reconstructed initialization equality fails",
                "preview drift exceeds one 8-bit level",
                "overlap occurs on the same logged GPU or non-equivalent devices",
            ],
        },
        "decision": {
            "protocol_exact": False,
            "scientific_replication_use": False,
            "quality_evaluation_seed4_seed5_authorized": False,
            "performance_benchmark_use": False,
            "historical_bitwise_initialization_claim": False,
            "rerun_affected_runs": affected_runs,
            "failed_conditions": failures,
        },
        "documented_deviations": ["D1", "D2", "D3", "D4", "D5"],
        "claim_exclusions": sorted(REQUIRED_EXCLUSIONS),
        "adjudicator": {
            "role": "Collaborator",
            "independent_external_signature_present": False,
            "machine_policy_evaluated": True,
        },
        "next_step": (
            "obtain independent quality-blind review bound to this candidate "
            "receipt before issuing accepted_with_documented_deviation"
            if verdict == "machine_recommends_acceptance"
            else "rerun the identified affected runs before quality evaluation"
        ),
        "publication": {
            "sanitized_for_github": True,
            "raw_training_and_quality_artifacts_committed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if verdict == "rerun_required":
        raise SystemExit(4)


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        fail(str(exc))
