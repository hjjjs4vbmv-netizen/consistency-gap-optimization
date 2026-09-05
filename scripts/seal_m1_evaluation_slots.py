#!/usr/bin/env python3
"""Collapse up to three technical attempts into one result row per M1 slot."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import build_m1_evaluation_slots as slots
from scripts import validate_m1_evaluation_job as validation


FIELDS = (
    "slot_id", "status", "fid_status", "fid50k_full", "kid_status",
    "kid50k_full", "selected_attempt", "reason", "receipt_path",
    "receipt_sha256", "evidence_path", "evidence_sha256",
)


class SealError(RuntimeError):
    pass


def _evidence_path(value: str, parent: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else parent / path).resolve(strict=True)


def load_scientific_terminal_rows(
    path: Path | None,
    manifest_rows: list[dict[str, Any]],
    training: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    path = path.resolve(strict=True)
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if ledger.get("schema") != "ect.m1.scientific-missing-evidence/v1":
        raise SealError("scientific missing evidence schema mismatch")
    receipt_values = ledger.get("receipts")
    if not isinstance(receipt_values, list) or not all(
        isinstance(value, str) and value for value in receipt_values
    ):
        raise SealError("scientific missing evidence must list receipt paths")
    receipt_paths = [_evidence_path(value, path.parent) for value in receipt_values]
    if len(receipt_paths) != len(set(receipt_paths)):
        raise SealError("duplicate scientific missing evidence receipt")

    manifest = {row["slot_id"]: row for row in manifest_rows}
    output: dict[str, dict[str, str]] = {}
    for receipt_path in receipt_paths:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        schema = receipt.get("schema")
        if schema == "ect.m1.training-slot/v1":
            if (
                receipt.get("status") != "COMPLETE_WITH_SCIENTIFIC_FAILURES"
                or receipt.get("training_manifest_sha256")
                != training["training_manifest_sha256"]
            ):
                raise SealError("training scientific failure receipt manifest mismatch")
            roster_slot, seed = receipt.get("roster_slot"), receipt.get("seed")
            branches = receipt.get("branches")
            if not isinstance(branches, dict):
                raise SealError("training scientific failure receipt lacks branches")
            for branch, branch_result in branches.items():
                if branch not in slots.BRANCHES:
                    raise SealError("training scientific failure has unknown branch")
                if not isinstance(branch_result, dict) or branch_result.get("status") != "SCIENTIFIC_FAILURE":
                    continue
                attempt_path = _evidence_path(
                    str(branch_result.get("attempt_receipt", "")), receipt_path.parent
                )
                attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
                if (
                    attempt.get("schema") != "ect.m1.training-attempt/v1"
                    or attempt.get("status") != "SCIENTIFIC_FAILURE"
                    or attempt.get("seed") != seed
                    or attempt.get("branch") != branch
                    or attempt.get("training_manifest_sha256")
                    != training["training_manifest_sha256"]
                ):
                    raise SealError("branch scientific failure attempt receipt mismatch")
                branch_manifest_path = _evidence_path(
                    str(attempt.get("branch_manifest_path", "")),
                    attempt_path.parent,
                )
                log_path = _evidence_path(
                    str(attempt.get("log_path", "")), attempt_path.parent
                )
                branch_manifest = json.loads(
                    branch_manifest_path.read_text(encoding="utf-8")
                )
                source_arm = "B" if branch.endswith("_B") else "A"
                expected_source = training["sources"][(seed, source_arm)]
                if (
                    validation.sha256_file(branch_manifest_path)
                    != attempt.get("branch_manifest_sha256")
                    or validation.sha256_file(log_path) != attempt.get("log_sha256")
                    or branch_manifest.get("seed") != seed
                    or branch_manifest.get("branch") != branch
                    or branch_manifest.get("training_manifest_sha256")
                    != training["training_manifest_sha256"]
                    or branch_manifest.get("implementation_commit")
                    != training["implementation_commit"]
                    or branch_manifest.get("source_state", {}).get("sha256")
                    != expected_source
                    or attempt.get("frozen_source_state_sha256") != expected_source
                ):
                    raise SealError("scientific attempt provenance mismatch")
                affected = [
                    row for row in manifest.values()
                    if row["roster_slot"] == roster_slot
                    and int(row["seed"]) == seed
                    and row["branch"] == branch
                ]
                if len(affected) != 5:
                    raise SealError("scientific branch failure must map to exactly five slots")
                for row in affected:
                    if row["slot_id"] in output:
                        raise SealError("overlapping scientific missing evidence")
                    output[row["slot_id"]] = {
                        "status": "NOT_RUN_NO_ENDPOINT",
                        "reason": attempt.get("reason", "SCIENTIFIC_FAILURE"),
                        "evidence_path": str(attempt_path),
                        "evidence_sha256": validation.sha256_file(attempt_path),
                    }
        elif schema == validation.CLASSIFIER_SCHEMA:
            expected = {
                "status": "SCIENTIFIC_READOUT_INVALID",
                "protocol_id": slots.PROTOCOL_ID,
                "training_manifest_sha256": training["training_manifest_sha256"],
                "implementation_commit": training["implementation_commit"],
                "implementation_checkout": {
                    "head": training["implementation_commit"], "clean": True,
                },
                "source_attempted_iteration": 8_000,
                "source_cur_nimg": 1_024_000,
            }
            if any(receipt.get(key) != value for key, value in expected.items()):
                raise SealError("scientific readout-invalid receipt binding mismatch")
            try:
                current_checkout = validation.verify_implementation_checkout(
                    training["implementation_commit"]
                )
            except validation.ValidationError as exc:
                raise SealError(str(exc)) from exc
            if current_checkout != receipt["implementation_checkout"]:
                raise SealError("readout classifier implementation checkout changed")
            classifications = {
                "NONFINITE_READOUT_STATE",
                "NONFINITE_FIXED_INPUT_OUTPUT",
                "NONFINITE_READOUT_STATE_AND_FIXED_OUTPUT",
            }
            nonfinite_state = receipt.get("nonfinite_state_tensor_paths")
            output_nonfinite = receipt.get("output_nonfinite_count")
            invalid_fields = receipt.get("invalid_fields")
            input_spec = receipt.get("fixed_input_spec")
            executed = receipt.get("fixed_input_executed")
            expected_invalid_fields = (
                [f"state_dict:{name}" for name in nonfinite_state]
                if isinstance(nonfinite_state, list) else []
            ) + (["fixed_input_output"] if isinstance(output_nonfinite, int)
                 and not isinstance(output_nonfinite, bool)
                 and output_nonfinite > 0 else [])
            expected_classification = (
                "NONFINITE_READOUT_STATE_AND_FIXED_OUTPUT"
                if nonfinite_state and output_nonfinite
                else "NONFINITE_READOUT_STATE"
                if nonfinite_state
                else "NONFINITE_FIXED_INPUT_OUTPUT"
                if output_nonfinite
                else "FINITE_READOUT"
            )
            if (
                receipt.get("classification") not in classifications
                or receipt.get("classification") != expected_classification
                or not isinstance(nonfinite_state, list)
                or not isinstance(executed, bool)
                or receipt.get("fixed_input") is not executed
                or (
                    executed and (
                        isinstance(output_nonfinite, bool)
                        or not isinstance(output_nonfinite, int)
                        or output_nonfinite < 0
                    )
                )
                or (
                    not executed and (
                        output_nonfinite is not None
                        or receipt.get("classification")
                        != "NONFINITE_READOUT_STATE"
                        or not nonfinite_state
                        or not isinstance(
                            receipt.get("fixed_input_forward_error"), dict
                        )
                    )
                )
                or not isinstance(invalid_fields, list)
                or not invalid_fields
                or invalid_fields != expected_invalid_fields
                or not isinstance(receipt.get("source_readout_sha256"), str)
                or len(receipt["source_readout_sha256"]) != 64
                or any(character not in "0123456789abcdef"
                       for character in receipt["source_readout_sha256"])
                or receipt.get("snapshot_path") is not None
                or receipt.get("snapshot_sha256") is not None
                or (not nonfinite_state and output_nonfinite == 0)
                or not isinstance(input_spec, dict)
                or input_spec.get("x") != {
                    "shape": [1, 3, 32, 32], "dtype": "float32",
                    "fill_value": 0.0,
                }
                or input_spec.get("sigma") != {
                    "shape": [1], "dtype": "float32", "fill_value": 1.0,
                }
                or input_spec.get("class_labels") is not None
                or input_spec.get("force_fp32") is not True
                or input_spec.get("model_mode") != "eval"
                or input_spec.get("autograd") is not False
                or not str(input_spec.get("device", "")).startswith("cuda")
                or (
                    executed and (
                        receipt.get("output_shape") != [1, 3, 32, 32]
                        or receipt.get("output_dtype") != "float32"
                    )
                )
                or (
                    not executed and (
                        receipt.get("output_shape") is not None
                        or receipt.get("output_dtype") is not None
                    )
                )
            ):
                raise SealError("readout-invalid receipt lacks a valid fixed-input observation")
            seed, branch, readout = (
                receipt.get("seed"), receipt.get("branch"), receipt.get("readout")
            )
            source_arm = "B" if str(branch).endswith("_B") else "A"
            if (
                not isinstance(seed, int)
                or branch not in slots.BRANCHES
                or readout not in slots.READOUT_BLOCKS
                or receipt.get("frozen_source_state_sha256")
                != training["sources"].get((seed, source_arm))
            ):
                raise SealError("readout-invalid seed/branch/source binding mismatch")
            try:
                canonical = validation.validate_canonical_training_milestone(
                    training, seed, branch,
                    Path(str(receipt.get("terminal_state_path", ""))),
                    str(receipt.get("terminal_state_sha256", "")),
                )
            except (OSError, validation.ValidationError) as exc:
                raise SealError(str(exc)) from exc
            if any(receipt.get(key) != value for key, value in canonical.items()):
                raise SealError("readout-invalid canonical milestone mismatch")
            branch_path = None
            for file_key, hash_key in (
                ("terminal_state_path", "terminal_state_sha256"),
                ("branch_manifest_path", "branch_manifest_sha256"),
            ):
                artifact = _evidence_path(str(receipt.get(file_key, "")), receipt_path.parent)
                if validation.sha256_file(artifact) != receipt.get(hash_key):
                    raise SealError(f"readout-invalid artifact mismatch: {file_key}")
                if file_key == "branch_manifest_path":
                    branch_path = artifact
            branch_manifest = json.loads(branch_path.read_text(encoding="utf-8"))
            if (
                branch_manifest.get("seed") != seed
                or branch_manifest.get("branch") != branch
                or branch_manifest.get("training_manifest_sha256")
                != training["training_manifest_sha256"]
                or branch_manifest.get("implementation_commit")
                != training["implementation_commit"]
                or branch_manifest.get("source_state", {}).get("sha256")
                != receipt["frozen_source_state_sha256"]
            ):
                raise SealError("readout-invalid branch manifest binding mismatch")
            affected = [
                row for row in manifest.values()
                if int(row["seed"]) == seed and row["branch"] == branch
                and row["readout"] == readout
            ]
            if len(affected) != len(slots.READOUT_BLOCKS.get(str(readout), ())):
                raise SealError("readout-invalid evidence does not map to a complete readout")
            for row in affected:
                if row["slot_id"] in output:
                    raise SealError("overlapping scientific missing evidence")
                output[row["slot_id"]] = {
                    "status": "SCIENTIFIC_READOUT_INVALID",
                    "reason": str(receipt["classification"]),
                    "evidence_path": str(receipt_path),
                    "evidence_sha256": validation.sha256_file(receipt_path),
                }
        else:
            raise SealError(f"unsupported scientific missing evidence: {receipt_path}")
    return output


def load_attempts(
    receipts_dir: Path,
    slot: Mapping[str, Any],
    training: Mapping[str, Any],
    evaluation_manifest_sha256: str,
) -> list[tuple[Path, dict[str, Any]]]:
    attempts = []
    missing_seen = False
    for attempt in range(3):
        path = receipts_dir / f"{slot['slot_id']}-attempt{attempt:02d}.json"
        if not path.exists():
            missing_seen = True
            continue
        if path.is_symlink() or not path.is_file():
            raise SealError(f"attempt receipt must be a regular file: {path}")
        if missing_seen:
            raise SealError(f"attempt gap for {slot['slot_id']}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != validation.RECEIPT_SCHEMA
            or payload.get("slot_id") != slot["slot_id"]
            or payload.get("attempt") != attempt
        ):
            raise SealError(f"attempt receipt identity mismatch: {path}")
        if payload.get("status") in {"SEALED_PASS", "SEALED_PARTIAL"}:
            try:
                validation.validate_sealed_attempt_provenance(
                    payload, slot, training, evaluation_manifest_sha256
                )
            except validation.ValidationError as exc:
                raise SealError(f"attempt provenance mismatch: {path}: {exc}") from exc
        attempts.append((path, payload))
    return attempts


def collapse_slot(
    slot: Mapping[str, Any],
    attempts: list[tuple[Path, dict[str, Any]]],
    terminal: Mapping[str, str] | None,
) -> dict[str, Any]:
    if terminal is not None:
        if any(
            payload.get("status") != "INCOMPLETE_TECHNICAL"
            or (
                isinstance(payload.get("result_row"), dict)
                and payload["result_row"].get("fid_status") == "SEALED_PASS"
            )
            for _, payload in attempts
        ):
            raise SealError(
                f"scientific terminal status conflicts with a nontechnical attempt: {slot['slot_id']}"
            )
        status = terminal["status"]
        attempt_path = attempts[-1][0] if attempts else None
        return {
            "slot_id": slot["slot_id"], "status": status, "fid_status": status,
            "fid50k_full": "", "kid_status": status, "kid50k_full": "",
            "selected_attempt": len(attempts) - 1 if attempts else "",
            "reason": terminal["reason"],
            "receipt_path": str(attempt_path.resolve()) if attempt_path else "",
            "receipt_sha256": (
                validation.sha256_file(attempt_path) if attempt_path else ""
            ),
            "evidence_path": terminal["evidence_path"],
            "evidence_sha256": terminal["evidence_sha256"],
        }
    if not attempts:
        return {
            "slot_id": slot["slot_id"], "status": "MISSING_RESULT",
            "fid_status": "MISSING_RESULT", "fid50k_full": "",
            "kid_status": "MISSING_RESULT", "kid50k_full": "",
            "selected_attempt": "", "reason": "NO_ATTEMPT_RECEIPT",
            "receipt_path": "", "receipt_sha256": "",
            "evidence_path": "", "evidence_sha256": "",
        }

    successes = []
    for index, (_path, payload) in enumerate(attempts):
        result = payload.get("result_row")
        if isinstance(result, dict) and result.get("fid_status") == "SEALED_PASS":
            if payload.get("status") not in {"SEALED_PASS", "SEALED_PARTIAL"}:
                raise SealError(f"sealed FID has inconsistent attempt status: {_path}")
            expected = {
                "slot_index": int(slot["slot_index"]),
                "seed": int(slot["seed"]),
                "branch": slot["branch"],
                "readout": slot["readout"],
                "block": slot["block"],
                "sample_seed_start": int(slot["sample_seed_start"]),
                "sample_seed_end": int(slot["sample_seed_end"]),
                "sample_count": int(slot["sample_count"]),
                "nfe": int(slot["nfe"]),
                "precision": slot["precision"],
                "evaluator_commit": slot["evaluator_commit"],
            }
            if any(payload.get(key) != value for key, value in expected.items()):
                raise SealError(f"sealed attempt does not match its manifest slot: {_path}")
            successes.append(index)
        elif payload.get("status") != "INCOMPLETE_TECHNICAL":
            raise SealError(f"only technical failures may be retried: {_path}")
    if len(successes) > 1 or (successes and successes[0] != len(attempts) - 1):
        raise SealError(f"attempts continued after a sealed FID result: {slot['slot_id']}")
    selected_index = successes[0] if successes else len(attempts) - 1
    path, payload = attempts[selected_index]
    if successes:
        result = dict(payload["result_row"])
        if result.get("slot_id") != slot["slot_id"]:
            raise SealError(f"result row slot mismatch: {path}")
        return {
            **{field: result.get(field, "") for field in FIELDS[:6]},
            "selected_attempt": selected_index,
            "reason": "FID_SEALED",
            "receipt_path": str(path.resolve()),
            "receipt_sha256": validation.sha256_file(path),
            "evidence_path": "", "evidence_sha256": "",
        }
    return {
        "slot_id": slot["slot_id"], "status": "INCOMPLETE_TECHNICAL",
        "fid_status": "INCOMPLETE_TECHNICAL", "fid50k_full": "",
        "kid_status": "INCOMPLETE_TECHNICAL", "kid50k_full": "",
        "selected_attempt": selected_index,
        "reason": payload.get("validation_error", "NO_FID_SEALED"),
        "receipt_path": str(path.resolve()),
        "receipt_sha256": validation.sha256_file(path),
        "evidence_path": "", "evidence_sha256": "",
    }


def seal_rows(
    manifest_rows: list[dict[str, Any]],
    receipts_dir: Path,
    terminal_rows: Mapping[str, Mapping[str, str]],
    training: Mapping[str, Any],
    evaluation_manifest_sha256: str,
) -> list[dict[str, Any]]:
    try:
        slots.validate_slots(manifest_rows)
    except slots.SlotError as exc:
        raise SealError(str(exc)) from exc
    expected = {row["slot_id"] for row in manifest_rows}
    if not set(terminal_rows).issubset(expected):
        raise SealError("terminal ledger contains an unexpected slot")
    allowed_receipts = {
        f"{slot_id}-attempt{attempt:02d}.json"
        for slot_id in expected
        for attempt in range(3)
    }
    unexpected = [
        path.name
        for path in receipts_dir.glob("*-attempt*.json")
        if path.name not in allowed_receipts
    ]
    if unexpected:
        raise SealError(f"unexpected attempt receipt: {sorted(unexpected)[0]}")
    return [
        collapse_slot(
            slot,
            load_attempts(
                receipts_dir, slot, training, evaluation_manifest_sha256
            ),
            terminal_rows.get(slot["slot_id"]),
        )
        for slot in manifest_rows
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--receipts-dir", type=Path, required=True)
    parser.add_argument("--scientific-evidence-json", type=Path)
    parser.add_argument("--output-results-csv", type=Path, required=True)
    parser.add_argument("--output-seal-json", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest_csv.resolve(strict=True)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    training = slots.load_training_identity(args.training_manifest)
    terminal = load_scientific_terminal_rows(
        args.scientific_evidence_json, manifest, training
    )
    rows = seal_rows(
        manifest,
        args.receipts_dir.resolve(strict=True),
        terminal,
        training,
        validation.sha256_file(manifest_path),
    )
    args.output_results_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_results_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["fid_status"] for row in rows)
    kid_counts = Counter(row["kid_status"] for row in rows)
    if counts["INVALID_IMPLEMENTATION"] or kid_counts["INVALID_IMPLEMENTATION"]:
        status = "INVALID_IMPLEMENTATION"
    elif counts["MISSING_RESULT"] or counts["INCOMPLETE_TECHNICAL"]:
        status = "INCOMPLETE"
    elif kid_counts["MISSING_RESULT"] or kid_counts["INCOMPLETE_TECHNICAL"]:
        status = "SEALED_WITH_KID_MISSINGNESS"
    else:
        status = "SEALED"
    seal = {
        "schema": "ect.m1.evaluation-slot-seal/v1",
        "status": status,
        "slots": 320,
        "fid_status_counts": dict(sorted(counts.items())),
        "kid_status_counts": dict(sorted(kid_counts.items())),
        "manifest_sha256": validation.sha256_file(args.manifest_csv),
        "scientific_evidence": (
            None if args.scientific_evidence_json is None else {
                "path": str(args.scientific_evidence_json.resolve(strict=True)),
                "sha256": validation.sha256_file(
                    args.scientific_evidence_json.resolve(strict=True)
                ),
            }
        ),
        "results_csv": str(args.output_results_csv.resolve()),
        "results_csv_sha256": validation.sha256_file(args.output_results_csv),
    }
    validation.atomic_json(args.output_seal_json.resolve(), seal)
    print(json.dumps(seal, sort_keys=True))
    return 0 if status in {"SEALED", "SEALED_WITH_KID_MISSINGNESS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
