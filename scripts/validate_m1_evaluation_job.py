#!/usr/bin/env python3
"""Validate one M1 evaluation slot without conflating FID and KID status."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import build_m1_evaluation_slots as slots


METRICS = ("kid50k_full", "fid50k_full")
EXPORT_SCHEMA = "ect.m1.readout-export/v1"
CLASSIFIER_SCHEMA = "ect.m1.readout-classification/v1"
RECEIPT_SCHEMA = "ect.m1.evaluation-job/v1"
DATASET_SHA256 = (
    "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372"
)
TRAINING_DATASET_SHA256 = (
    "9818e4b801a52eac437485bc8a69e40b54e9ae9c5d1427467343c91de868f1b3"
)
EVALUATOR_CT_EVAL_SHA256 = (
    "8e17e4cd4e12097e12659a9c8849d42554f24efb25e5255261383d952d878c95"
)
EVALUATOR_ARCHIVE_SHA256 = (
    "7ef8a1b22af9beab106ad3adbac6474608f27e74c43629a95fcc71738dab0a6f"
)
EXPECTED_RUNTIME_PROBE = {
    "python": "3.11.13",
    "torch": "2.6.0+cu124",
    "torch_cuda": "12.4",
    "numpy": "2.1.2",
    "scipy": "1.16.1",
}
IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]


class ValidationError(RuntimeError):
    pass


def verify_implementation_checkout(expected_commit: str) -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=IMPLEMENTATION_ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=IMPLEMENTATION_ROOT, text=True,
    )
    if head != expected_commit or dirty:
        raise ValidationError("implementation checkout does not match the clean frozen commit")
    return {"head": head, "clean": True}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()


def load_slot(manifest_path: Path, slot_id: str) -> dict[str, str]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    try:
        slots.validate_slots(rows)
    except slots.SlotError as exc:
        raise ValidationError(str(exc)) from exc

    matches = [row for row in rows if row["slot_id"] == slot_id]
    if len(matches) != 1:
        raise ValidationError(f"slot_id must identify one manifest row: {slot_id}")
    return matches[0]


def canonical_training_slot_receipt(
    training: Mapping[str, Any], seed: int
) -> tuple[str, Path]:
    matches = [
        row for row in training.get("roster", []) if row.get("seed") == seed
    ]
    if len(matches) != 1:
        raise ValidationError("seed does not identify one frozen training slot")
    roster_slot = matches[0]["roster_slot"]
    path = Path(str(training.get("output_root", ""))) / roster_slot / "training_receipt.json"
    if path.is_symlink():
        raise ValidationError("canonical training slot receipt must not be a symlink")
    return roster_slot, path.resolve(strict=True)


def validate_canonical_training_milestone(
    training: Mapping[str, Any],
    seed: int,
    branch: str,
    state_path: Path,
    state_sha256: str,
) -> dict[str, str]:
    roster_slot, receipt_path = canonical_training_slot_receipt(training, seed)
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValidationError("canonical training slot receipt is not a regular file")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    branch_result = receipt.get("branches", {}).get(branch, {})
    milestone = branch_result.get("milestones", {}).get("1024")
    if (
        receipt.get("schema") != "ect.m1.training-slot/v1"
        or receipt.get("status") not in {"PASS", "COMPLETE_WITH_SCIENTIFIC_FAILURES"}
        or receipt.get("training_manifest_sha256")
        != training.get("training_manifest_sha256")
        or receipt.get("roster_slot") != roster_slot
        or receipt.get("seed") != seed
        or branch_result.get("status") != "PASS"
        or not isinstance(milestone, dict)
        or milestone.get("state_path") != str(state_path.resolve(strict=True))
        or milestone.get("state_sha256") != state_sha256
        or milestone.get("attempted_iteration") != 8_000
        or milestone.get("cur_nimg") != 1_024_000
    ):
        raise ValidationError("terminal state is not the canonical 1024 milestone")
    return {
        "roster_slot": roster_slot,
        "training_slot_receipt_path": str(receipt_path),
        "training_slot_receipt_sha256": sha256_file(receipt_path),
    }


def load_valid_readout_classifier_receipt(
    path: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValidationError("readout classifier receipt must be a regular file")
    receipt_path = path.resolve(strict=True)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "schema": CLASSIFIER_SCHEMA,
        "status": "READOUT_VALID",
        "protocol_id": slots.PROTOCOL_ID,
        "classification": "FINITE_READOUT",
        "source_attempted_iteration": 8_000,
        "source_cur_nimg": 1_024_000,
        **expected,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise ValidationError("readout classifier receipt binding mismatch")
    if (
        payload.get("fixed_input") is not True
        or payload.get("fixed_input_executed") is not True
        or payload.get("nonfinite_state_tensor_paths") != []
        or payload.get("output_nonfinite_count") != 0
        or payload.get("invalid_fields") != []
        or payload.get("fixed_input_forward_error") is not None
        or not isinstance(payload.get("source_readout_sha256"), str)
        or len(payload["source_readout_sha256"]) != 64
        or any(character not in "0123456789abcdef"
               for character in payload["source_readout_sha256"])
    ):
        raise ValidationError("readout classifier did not establish a finite readout")
    input_spec = payload.get("fixed_input_spec")
    if (
        not isinstance(input_spec, dict)
        or input_spec.get("x") != {
            "shape": [1, 3, 32, 32], "dtype": "float32", "fill_value": 0.0,
        }
        or input_spec.get("sigma") != {
            "shape": [1], "dtype": "float32", "fill_value": 1.0,
        }
        or input_spec.get("class_labels") is not None
        or input_spec.get("force_fp32") is not True
        or input_spec.get("model_mode") != "eval"
        or input_spec.get("autograd") is not False
        or not str(input_spec.get("device", "")).startswith("cuda")
        or payload.get("output_shape") != [1, 3, 32, 32]
        or payload.get("output_dtype") != "float32"
    ):
        raise ValidationError("readout classifier fixed-input observation mismatch")
    for file_key, hash_key in (
        ("terminal_state_path", "terminal_state_sha256"),
        ("branch_manifest_path", "branch_manifest_sha256"),
        ("training_slot_receipt_path", "training_slot_receipt_sha256"),
    ):
        artifact_input = Path(str(payload.get(file_key, "")))
        if artifact_input.is_symlink():
            raise ValidationError(f"classifier artifact must not be a symlink: {file_key}")
        artifact = artifact_input.resolve(strict=True)
        if not artifact.is_file() or sha256_file(artifact) != payload.get(hash_key):
            raise ValidationError(f"classifier artifact mismatch: {file_key}")
    training_slot = json.loads(
        Path(payload["training_slot_receipt_path"]).read_text(encoding="utf-8")
    )
    milestone = (
        training_slot.get("branches", {})
        .get(expected.get("branch"), {})
        .get("milestones", {})
        .get("1024")
    )
    if (
        training_slot.get("schema") != "ect.m1.training-slot/v1"
        or training_slot.get("status")
        not in {"PASS", "COMPLETE_WITH_SCIENTIFIC_FAILURES"}
        or training_slot.get("training_manifest_sha256")
        != expected.get("training_manifest_sha256")
        or training_slot.get("roster_slot") != expected.get("roster_slot")
        or training_slot.get("seed") != expected.get("seed")
        or training_slot.get("branches", {}).get(expected.get("branch"), {}).get("status")
        != "PASS"
        or not isinstance(milestone, dict)
        or milestone.get("state_path") != expected.get("terminal_state_path")
        or milestone.get("state_sha256") != expected.get("terminal_state_sha256")
        or milestone.get("attempted_iteration") != 8_000
        or milestone.get("cur_nimg") != 1_024_000
    ):
        raise ValidationError("classifier does not bind the canonical 1024 milestone")
    payload["receipt_path"] = str(receipt_path)
    payload["receipt_sha256"] = sha256_file(receipt_path)
    return payload


def load_snapshot_receipt(
    path: Path,
    slot: Mapping[str, str],
    training: Mapping[str, Any],
    *,
    gate: bool = False,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_arm = "B" if slot["branch"].endswith("_B") else "A"
    expected_source = training["sources"].get((int(slot["seed"]), source_arm))
    expected = {
        "schema": EXPORT_SCHEMA,
        "status": "PASS",
        "protocol_id": slots.PROTOCOL_ID,
        "seed": int(slot["seed"]),
        "branch": slot["branch"],
        "readout": slot["readout"],
        "source_attempted_iteration": 4_032 if gate else 8_000,
        "source_cur_nimg": 516_096 if gate else 1_024_000,
        "quality_eligible": not gate,
        "gate_state": gate,
        "training_manifest_sha256": training["training_manifest_sha256"],
        "implementation_commit": training["implementation_commit"],
        "implementation_checkout": {
            "head": training["implementation_commit"], "clean": True,
        },
        "frozen_source_state_sha256": expected_source,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValidationError(f"readout export receipt mismatch: {key}")
    if (
        slot.get("training_manifest_sha256") != training["training_manifest_sha256"]
        or slot.get("implementation_commit") != training["implementation_commit"]
        or slot.get("frozen_source_state_sha256") != expected_source
    ):
        raise ValidationError("evaluation slot training provenance mismatch")

    terminal_input = Path(payload.get("source_state_path", ""))
    if terminal_input.is_symlink():
        raise ValidationError("terminal state must not be a symlink")
    terminal = terminal_input.resolve(strict=True)
    if not terminal.is_file() or sha256_file(terminal) != payload.get("terminal_state_sha256"):
        raise ValidationError("terminal state SHA256 mismatch")

    branch_input = Path(payload.get("branch_manifest_path", ""))
    if branch_input.is_symlink():
        raise ValidationError("branch manifest must not be a symlink")
    branch_path = branch_input.resolve(strict=True)
    if not branch_path.is_file() or sha256_file(branch_path) != payload.get("branch_manifest_sha256"):
        raise ValidationError("branch manifest SHA256 mismatch")
    branch = json.loads(branch_path.read_text(encoding="utf-8"))
    branch_expected = {
        "experiment_protocol": slots.PROTOCOL_ID,
        "seed": int(slot["seed"]),
        "branch": slot["branch"],
        "training_manifest_sha256": training["training_manifest_sha256"],
        "implementation_commit": training["implementation_commit"],
    }
    if any(branch.get(key) != value for key, value in branch_expected.items()):
        raise ValidationError("branch manifest training provenance mismatch")
    if branch.get("source_state", {}).get("sha256") != expected_source:
        raise ValidationError("branch manifest frozen source SHA256 mismatch")

    if not gate:
        canonical = validate_canonical_training_milestone(
            training, int(slot["seed"]), slot["branch"], terminal,
            payload["terminal_state_sha256"],
        )
        if any(payload.get(key) != value for key, value in canonical.items()):
            raise ValidationError("readout export canonical milestone mismatch")
        classifier_input = Path(payload.get("classifier_receipt_path", ""))
        classifier = load_valid_readout_classifier_receipt(
            classifier_input,
            {
                "training_manifest_sha256": training["training_manifest_sha256"],
                "implementation_commit": training["implementation_commit"],
                "implementation_checkout": {
                    "head": training["implementation_commit"], "clean": True,
                },
                "seed": int(slot["seed"]),
                "branch": slot["branch"],
                "readout": slot["readout"],
                "frozen_source_state_sha256": expected_source,
                "terminal_state_path": str(terminal),
                "terminal_state_sha256": payload["terminal_state_sha256"],
                "branch_manifest_path": str(branch_path),
                "branch_manifest_sha256": payload["branch_manifest_sha256"],
                "source_readout_sha256": payload.get("source_readout_sha256"),
                **canonical,
            },
        )
        if (
            classifier["receipt_path"]
            != str(classifier_input.resolve(strict=True))
            or classifier["receipt_sha256"]
            != payload.get("classifier_receipt_sha256")
        ):
            raise ValidationError("readout classifier receipt SHA256 mismatch")

    snapshot_input = Path(payload.get("snapshot_path", ""))
    if snapshot_input.is_symlink():
        raise ValidationError("readout snapshot must not be a symlink")
    snapshot = snapshot_input.resolve(strict=True)
    if not snapshot.is_file():
        raise ValidationError("readout snapshot must be a regular non-symlink file")
    if sha256_file(snapshot) != payload.get("snapshot_sha256"):
        raise ValidationError("readout snapshot SHA256 mismatch")
    payload["snapshot_path"] = str(snapshot)
    payload["source_state_path"] = str(terminal)
    payload["branch_manifest_path"] = str(branch_path)
    return payload


def validate_sealed_attempt_provenance(
    payload: Mapping[str, Any],
    slot: Mapping[str, str],
    training: Mapping[str, Any],
    evaluation_manifest_sha256: str,
) -> None:
    source_arm = "B" if slot["branch"].endswith("_B") else "A"
    expected_source = training["sources"].get((int(slot["seed"]), source_arm))
    expected = {
        "manifest_sha256": evaluation_manifest_sha256,
        "training_manifest_sha256": training["training_manifest_sha256"],
        "implementation_commit": training["implementation_commit"],
        "implementation_checkout": {
            "head": training["implementation_commit"], "clean": True,
        },
        "frozen_source_state_sha256": expected_source,
        "training_runtime_receipt_sha256": training[
            "training_runtime_receipt_sha256"
        ],
        "process_exit_code": 0,
        "process_hard_timeout": False,
        "log_completion_marker": True,
    }
    slot_expected = {
        "training_manifest_sha256": training["training_manifest_sha256"],
        "implementation_commit": training["implementation_commit"],
        "frozen_source_state_sha256": expected_source,
    }
    if (
        any(payload.get(key) != value for key, value in expected.items())
        or any(slot.get(key) != value for key, value in slot_expected.items())
        or not isinstance(payload.get("live_runtime_probe"), dict)
        or payload["live_runtime_probe"].get("cuda_available") is not True
        or any(
            payload["live_runtime_probe"].get(key) != value
            for key, value in EXPECTED_RUNTIME_PROBE.items()
        )
        or payload["live_runtime_probe"].get("pip_freeze_sha256")
        != payload.get("runtime_pip_freeze_sha256")
    ):
        raise ValidationError("sealed attempt execution/provenance mismatch")
    if verify_implementation_checkout(
        training["implementation_commit"]
    ) != payload["implementation_checkout"]:
        raise ValidationError("sealed attempt implementation checkout changed")
    terminal = Path(payload.get("terminal_state_path", ""))
    branch_path = Path(payload.get("branch_manifest_path", ""))
    if (
        terminal.is_symlink()
        or not terminal.is_file()
        or sha256_file(terminal) != payload.get("terminal_state_sha256")
    ):
        raise ValidationError("sealed attempt terminal state SHA256 mismatch")
    if (
        branch_path.is_symlink()
        or not branch_path.is_file()
        or sha256_file(branch_path) != payload.get("branch_manifest_sha256")
    ):
        raise ValidationError("sealed attempt branch manifest SHA256 mismatch")
    branch = json.loads(branch_path.read_text(encoding="utf-8"))
    branch_expected = {
        "experiment_protocol": slots.PROTOCOL_ID,
        "seed": int(slot["seed"]),
        "branch": slot["branch"],
        "training_manifest_sha256": training["training_manifest_sha256"],
        "implementation_commit": training["implementation_commit"],
    }
    if any(branch.get(key) != value for key, value in branch_expected.items()):
        raise ValidationError("sealed attempt branch manifest provenance mismatch")
    if branch.get("source_state", {}).get("sha256") != expected_source:
        raise ValidationError("sealed attempt source SHA256 mismatch")


def read_metric(path: Path, metric: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        return {"status": "INCOMPLETE_TECHNICAL", "reason": "MISSING_METRIC_ARTIFACT"}
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(rows) != 1 or rows[0].get("metric") != metric or rows[0].get("num_gpus") != 1:
            raise ValueError("metric identity or row-count mismatch")
        value = float(rows[0]["results"][metric])
        if not math.isfinite(value) or (metric == "fid50k_full" and value <= 0):
            raise ValueError("invalid metric value")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "INCOMPLETE_TECHNICAL", "reason": str(exc)}
    return {
        "status": "SEALED_PASS",
        "value": value,
        "artifact_sha256": sha256_file(path),
    }


def validate_output(
    slot: Mapping[str, str],
    snapshot: Mapping[str, Any],
    job_dir: Path,
    dataset: Path,
    *,
    process_exit_code: int = 0,
    process_hard_timeout: bool = False,
) -> dict[str, Any]:
    job_dir = job_dir.resolve(strict=True)
    dataset = dataset.resolve(strict=True)
    common_names = ("log.txt", "training_options.json", "generated-samples.npy")
    for name in common_names:
        path = job_dir / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise ValidationError(f"missing common evaluation artifact: {path}")
    log_complete = "Exiting..." in (job_dir / "log.txt").read_text(errors="replace")

    options = json.loads((job_dir / "training_options.json").read_text(encoding="utf-8"))
    start, end = int(slot["sample_seed_start"]), int(slot["sample_seed_end"])
    expected_options = {
        "sample_seeds": list(range(start, end + 1)),
        "seed": int(slot["metric_seed"]),
        "metrics": slot["metrics"].split(","),
        "metric_repeats": 1,
        "metric_generator_batch": 128,
        "retain_generated_artifacts": True,
        "mid_t": [],
    }
    for key, expected in expected_options.items():
        if options.get(key) != expected:
            raise ValidationError(f"evaluation option mismatch: {key}")
    if int(slot["sample_count"]) != end - start + 1:
        raise ValidationError("manifest sample count does not match its range")
    if slot["nfe"] != "1" or slot["precision"] != "fp32":
        raise ValidationError("M1 requires NFE1 FP32")
    if options.get("network_kwargs", {}).get("use_fp16") is not False:
        raise ValidationError("evaluation did not record FP32 network execution")
    if Path(options.get("resume_pkl", "")).resolve() != Path(snapshot["snapshot_path"]):
        raise ValidationError("evaluation snapshot binding mismatch")
    if Path(options.get("dataset_kwargs", {}).get("path", "")).resolve() != dataset:
        raise ValidationError("evaluation dataset binding mismatch")

    metric_results: dict[str, dict[str, Any]] = {}
    feature_hashes: dict[str, str] = {}
    for metric in METRICS:
        feature_path = job_dir / f"generated-features-{metric}-repeat00.npy"
        metric_result = read_metric(job_dir / f"metric-{metric}.jsonl", metric)
        if (
            feature_path.is_symlink()
            or not feature_path.is_file()
            or feature_path.stat().st_size <= 0
        ):
            metric_result = {
                "status": "INCOMPLETE_TECHNICAL",
                "reason": "MISSING_GENERATED_FEATURES",
            }
        else:
            feature_hashes[metric] = sha256_file(feature_path)
            metric_result["generated_feature_sha256"] = feature_hashes[metric]
        metric_results[metric] = metric_result

    if all(metric in feature_hashes for metric in METRICS):
        shared = feature_hashes["kid50k_full"] == feature_hashes["fid50k_full"]
        if not shared:
            metric_results["kid50k_full"] = {
                "status": "INVALID_IMPLEMENTATION",
                "reason": "KID_FID_FEATURES_DIFFER",
                "generated_feature_sha256": feature_hashes["kid50k_full"],
            }
    else:
        shared = None

    process_ok = (
        process_exit_code == 0
        and process_hard_timeout is False
        and log_complete
    )
    if not process_ok:
        if process_hard_timeout:
            reason = "PROCESS_HARD_TIMEOUT"
        elif process_exit_code != 0:
            reason = f"PROCESS_EXIT_{process_exit_code}"
        else:
            reason = "MISSING_LOG_COMPLETION_MARKER"
        metric_results = {
            metric: {"status": "INCOMPLETE_TECHNICAL", "reason": reason}
            for metric in METRICS
        }

    fid_status = metric_results["fid50k_full"]["status"]
    kid_status = metric_results["kid50k_full"]["status"]
    overall = "SEALED_PASS" if fid_status == kid_status == "SEALED_PASS" else "SEALED_PARTIAL"
    if fid_status != "SEALED_PASS":
        overall = "INCOMPLETE_TECHNICAL"
    return {
        "schema": RECEIPT_SCHEMA,
        "status": overall,
        "protocol_id": slots.PROTOCOL_ID,
        "slot_id": slot["slot_id"],
        "slot_index": int(slot["slot_index"]),
        "seed": int(slot["seed"]),
        "branch": slot["branch"],
        "readout": slot["readout"],
        "block": slot["block"],
        "sample_seed_start": start,
        "sample_seed_end": end,
        "sample_count": int(slot["sample_count"]),
        "nfe": int(slot["nfe"]),
        "precision": slot["precision"],
        "metric_seed": int(slot["metric_seed"]),
        "evaluator_commit": slot["evaluator_commit"],
        "snapshot_path": snapshot["snapshot_path"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "terminal_state_path": snapshot["source_state_path"],
        "terminal_state_sha256": snapshot["terminal_state_sha256"],
        "branch_manifest_path": snapshot["branch_manifest_path"],
        "branch_manifest_sha256": snapshot["branch_manifest_sha256"],
        "training_manifest_sha256": snapshot["training_manifest_sha256"],
        "implementation_commit": snapshot["implementation_commit"],
        "frozen_source_state_sha256": snapshot["frozen_source_state_sha256"],
        "evaluation_dataset": str(dataset),
        "evaluation_dataset_sha256": DATASET_SHA256,
        "metrics": metric_results,
        "kid_fid_shared_features": shared,
        "log_completion_marker": log_complete,
        "process_exit_code": process_exit_code,
        "process_hard_timeout": process_hard_timeout,
        "job_dir": str(job_dir),
        "result_row": {
            "slot_id": slot["slot_id"],
            "status": fid_status,
            "fid_status": fid_status,
            "fid50k_full": metric_results["fid50k_full"].get("value", ""),
            "kid_status": kid_status,
            "kid50k_full": metric_results["kid50k_full"].get("value", ""),
        },
    }


def verify_evaluator(
    evaluator_repo: Path,
    expected_commit: str,
    evaluator_archive: Path | None = None,
) -> dict[str, Any]:
    evaluator_repo = evaluator_repo.resolve(strict=True)
    if sha256_file(evaluator_repo / "ct_eval.py") != EVALUATOR_CT_EVAL_SHA256:
        raise ValidationError("evaluator ct_eval.py identity mismatch")
    if not (evaluator_repo / ".git").exists():
        raise ValidationError(
            "formal M1 evaluation requires an exact clean evaluator git checkout; "
            "an adjacent archive does not bind the executed tree"
        )
    head = subprocess.check_output(
        ["git", "-C", str(evaluator_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(evaluator_repo), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    if head != expected_commit or dirty:
        raise ValidationError("evaluator repository identity mismatch or dirty tree")
    return {"evaluator_binding": "GIT", "evaluator_commit": head}


def verify_runtime(
    runtime_base: Path,
    runtime_environment: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    runtime_base = runtime_base.resolve(strict=True)
    runtime_environment = runtime_environment.resolve(strict=True)
    receipt_path = receipt_path.resolve(strict=True)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    schema = receipt.get("schema")
    if receipt.get("status") != "PASS" or schema not in {
        "ect.q256.training-compatible-evaluation-runtime/v1",
        "ect.m1.rebuilt-training-runtime/v1",
    }:
        raise ValidationError("runtime integrity receipt is not PASS")
    probe = receipt.get("runtime_probe")
    normalized_probe = None if not isinstance(probe, dict) else {
        "python": probe.get("python"),
        "torch": probe.get("torch"),
        "torch_cuda": probe.get("torch_cuda", probe.get("cuda")),
        "numpy": probe.get("numpy"),
        "scipy": probe.get("scipy"),
    }
    if normalized_probe != EXPECTED_RUNTIME_PROBE:
        raise ValidationError("runtime receipt is not Python 3.11 / torch 2.6 CUDA 12.4")
    if schema == "ect.q256.training-compatible-evaluation-runtime/v1":
        archive = Path(receipt.get("archive_path", "")).resolve(strict=True)
        archive_sha256 = sha256_file(archive)
        if receipt.get("archive_sha256") != archive_sha256:
            raise ValidationError("runtime archive SHA256 does not match its receipt")
        freeze_path = Path(str(receipt.get("pip_freeze_path", "")))
        if (
            not freeze_path.is_absolute() or freeze_path.is_symlink()
            or not freeze_path.is_file()
            or sha256_file(freeze_path) != receipt.get("pip_freeze_sha256")
        ):
            raise ValidationError("original runtime pip-freeze SHA256 mismatch")
        runtime_origin = "ORIGINAL_FROZEN_ARCHIVE"
        runtime_artifacts = {
            "runtime_archive": str(archive),
            "runtime_archive_sha256": archive_sha256,
            "runtime_pip_freeze": str(freeze_path),
            "runtime_pip_freeze_sha256": receipt["pip_freeze_sha256"],
        }
    else:
        if (
            receipt.get("runtime_origin") != "REBUILT_NOT_BYTE_IDENTICAL"
            or runtime_base != runtime_environment
        ):
            raise ValidationError("rebuilt runtime must use one explicitly identified prefix")
        freeze = receipt.get("pip_freeze")
        if not isinstance(freeze, dict):
            raise ValidationError("rebuilt runtime lacks pip-freeze binding")
        freeze_path = Path(str(freeze.get("path", "")))
        if (
            not freeze_path.is_absolute() or freeze_path.is_symlink()
            or not freeze_path.is_file()
            or sha256_file(freeze_path) != freeze.get("sha256")
        ):
            raise ValidationError("rebuilt runtime pip-freeze SHA256 mismatch")
        runtime_origin = "REBUILT_NOT_BYTE_IDENTICAL"
        runtime_artifacts = {
            "runtime_pip_freeze": str(freeze_path),
            "runtime_pip_freeze_sha256": freeze["sha256"],
        }
    python_candidates = (
        (
            runtime_environment / "bin/python3.11",
            runtime_environment / "bin/python",
        )
        if schema == "ect.m1.rebuilt-training-runtime/v1"
        else (
            runtime_environment / "bin/python",
            runtime_environment / "bin/python3.11",
        )
    )
    runtime_python = next((
        path for path in python_candidates
        if path.is_file() and os.access(path, os.X_OK)
    ), None)
    if runtime_python is None:
        raise ValidationError("runtime environment lacks executable Python 3.11")
    site_packages = runtime_base / "lib/python3.11/site-packages"
    torch_lib = site_packages / "torch/lib"
    if not torch_lib.is_dir():
        raise ValidationError("runtime base lacks the Python 3.11 torch library directory")
    library_paths = [torch_lib, runtime_base / "lib"]
    nvidia_root = site_packages / "nvidia"
    if nvidia_root.is_dir():
        library_paths.extend(sorted(
            path for path in nvidia_root.glob("*/lib") if path.is_dir()
        ))
    return {
        "runtime_base": str(runtime_base),
        "runtime_environment": str(runtime_environment),
        "runtime_python": str(runtime_python),
        "runtime_origin": runtime_origin,
        **runtime_artifacts,
        "runtime_library_paths": [str(path) for path in library_paths],
        "runtime_probe": normalized_probe,
        "runtime_integrity_receipt": str(receipt_path),
        "runtime_integrity_receipt_sha256": sha256_file(receipt_path),
    }


def verify_evaluation_dataset(path: Path) -> str:
    digest = sha256_file(path.resolve(strict=True))
    if digest == TRAINING_DATASET_SHA256:
        raise ValidationError("training ZIP was supplied where the evaluation ZIP is required")
    if digest != DATASET_SHA256:
        raise ValidationError("evaluation dataset SHA256 mismatch")
    return digest


def probe_live_runtime(
    runtime_python: Path,
    environment: Mapping[str, str],
    expected_pip_freeze_sha256: str | None = None,
) -> dict[str, Any]:
    code = (
        "import json,platform,torch,numpy,scipy;"
        "print(json.dumps({'python':platform.python_version(),"
        "'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
        "'numpy':numpy.__version__,'scipy':scipy.__version__,"
        "'cuda_available':torch.cuda.is_available()}))"
    )
    probe = json.loads(
        subprocess.check_output(
            [str(runtime_python), "-c", code], env=dict(environment), text=True
        )
    )
    if any(probe.get(key) != value for key, value in EXPECTED_RUNTIME_PROBE.items()):
        raise ValidationError("live runtime is not Python 3.11 / torch 2.6 CUDA 12.4")
    if probe.get("cuda_available") is not True:
        raise ValidationError("live runtime cannot access CUDA")
    if expected_pip_freeze_sha256 is not None:
        freeze = subprocess.check_output(
            [str(runtime_python), "-m", "pip", "freeze"],
            env=dict(environment),
        )
        canonical_freeze = b"\n".join(sorted(freeze.splitlines())) + b"\n"
        observed = hashlib.sha256(canonical_freeze).hexdigest()
        if observed != expected_pip_freeze_sha256:
            raise ValidationError("live runtime pip-freeze differs from its receipt")
        probe["pip_freeze_sha256"] = observed
    return probe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--slot-id", required=True)
    parser.add_argument("--snapshot-receipt", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--evaluator-repo", type=Path, required=True)
    parser.add_argument("--evaluator-archive", type=Path)
    parser.add_argument("--evaluation-dataset", type=Path, required=True)
    parser.add_argument("--runtime-base", type=Path, required=True)
    parser.add_argument("--runtime-env", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--attempt", type=int, choices=range(3), required=True)
    parser.add_argument("--process-exit-code", type=int, required=True)
    parser.add_argument("--process-hard-timeout", action="store_true")
    args = parser.parse_args()

    slot = load_slot(args.manifest_csv.resolve(strict=True), args.slot_id)
    if slot["evaluator_commit"] != slots.EVALUATOR_COMMIT:
        raise ValidationError("manifest evaluator commit mismatch")
    training = slots.load_training_identity(args.training_manifest)
    implementation_checkout = verify_implementation_checkout(
        training["implementation_commit"]
    )
    snapshot = load_snapshot_receipt(
        args.snapshot_receipt.resolve(strict=True), slot, training
    )
    evaluator = verify_evaluator(
        args.evaluator_repo, slot["evaluator_commit"], args.evaluator_archive
    )
    verify_evaluation_dataset(args.evaluation_dataset)
    runtime = verify_runtime(args.runtime_base, args.runtime_env, args.runtime_receipt)
    probe_environment = os.environ.copy()
    probe_environment["PYTHONNOUSERSITE"] = "1"
    probe_environment["LD_LIBRARY_PATH"] = ":".join(
        runtime["runtime_library_paths"]
    )
    live_runtime_probe = probe_live_runtime(
        Path(runtime["runtime_python"]), probe_environment,
        runtime["runtime_pip_freeze_sha256"],
    )
    payload = validate_output(
        slot,
        snapshot,
        args.job_dir,
        args.evaluation_dataset,
        process_exit_code=args.process_exit_code,
        process_hard_timeout=args.process_hard_timeout,
    )
    payload.update(runtime)
    payload["training_runtime_receipt_sha256"] = training[
        "training_runtime_receipt_sha256"
    ]
    payload["implementation_checkout"] = implementation_checkout
    payload["live_runtime_probe"] = live_runtime_probe
    payload.update(evaluator)
    payload["attempt"] = args.attempt
    payload["evaluator_source"] = str(args.evaluator_repo.resolve())
    payload["manifest_sha256"] = sha256_file(args.manifest_csv)
    payload["snapshot_receipt_sha256"] = sha256_file(args.snapshot_receipt)
    atomic_json(args.receipt.resolve(), payload)
    print(json.dumps(payload["result_row"], sort_keys=True))
    return 0 if payload["metrics"]["fid50k_full"]["status"] == "SEALED_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
