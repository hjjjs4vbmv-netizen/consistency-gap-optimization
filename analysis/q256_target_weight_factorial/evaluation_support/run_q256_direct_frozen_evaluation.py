#!/usr/bin/env python3
"""Bind direct q256 formal outputs and run the frozen evaluator primary-first.

The formal recovery queue predates the receipt-producing matrix launcher.  This
adapter supplies only the missing provenance/binding layer: it validates and
hash-binds the twelve immutable direct-run artifacts, delegates numerical work
to the frozen evaluator at the exact training commit, and changes job ordering
only so every NFE=1 job precedes every NFE=2 job.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

EXPECTED_HEAD = "dcca41b19e7c45512b5fbe98776520396a1bf9ac"
EXPECTED_PRIOR_COMPLETED = (
    "seed3-armA-nfe1",
    "seed3-armB-nfe1",
    "seed3-armC-nfe1",
    "seed3-armD-nfe1",
    "seed4-armA-nfe1",
    "seed4-armB-nfe1",
    "seed4-armC-nfe1",
    "seed4-armD-nfe1",
)
ARMS = ("A", "B", "C", "D")
SEEDS = (3, 4, 5)
ARM_SCALES = {
    "A": (1.0, 1.0),
    "B": (1.1, 1.1),
    "C": (1.1, 1.0),
    "D": (1.0, 1.1),
}
REQUIRED_FILES = (
    "network-snapshot-latest.pkl",
    "training-state-latest.pt",
    "factorial_training_telemetry_v1.csv",
    "train_summary.csv",
    "training_options.json",
    "initial_state_receipt_v1.json",
    "final.png",
)
SEMANTIC_NONFINITE_FIELDS = (
    "loss_nonfinite_count",
    "sanitized_grad_nonfinite_count",
    "update_nonfinite_count",
    "model_nonfinite_count",
    "ema_nonfinite_count",
    "factor_nonfinite_count",
)


class BindingError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise BindingError(message)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        fail(f"missing regular JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"JSON root is not an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        fail(f"missing regular CSV file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        fail(f"empty CSV file: {path}")
    return rows


def as_int(value: str, label: str) -> int:
    try:
        number = float(value)
    except ValueError as exc:
        fail(f"non-numeric {label}: {value!r}")
    if not number.is_integer():
        fail(f"non-integral {label}: {value!r}")
    return int(number)


def as_float(value: str, label: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        fail(f"non-numeric {label}: {value!r}")


def regular_file_binding(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        fail(f"missing immutable artifact: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_direct_cell(run_dir: Path, seed: int, arm: str) -> dict[str, Any]:
    run_dir = run_dir.resolve(strict=True)
    if run_dir.is_symlink() or not run_dir.is_dir():
        fail(f"invalid direct run directory: {run_dir}")
    expected_suffix = Path(f"seed{seed}") / f"arm{arm}"
    if Path(*run_dir.parts[-2:]) != expected_suffix:
        fail(f"run identity/path mismatch: {run_dir}")
    artifacts = {name: regular_file_binding(run_dir / name) for name in REQUIRED_FILES}

    summary = read_csv(run_dir / "train_summary.csv")
    telemetry = read_csv(run_dir / "factorial_training_telemetry_v1.csv")
    if len(summary) != 2000 or len(telemetry) != 2000:
        fail(f"seed{seed}/{arm} requires exactly 2000 summary/telemetry rows")
    skip_attempts: list[int] = []
    for index, (summary_row, telemetry_row) in enumerate(zip(summary, telemetry), start=1):
        attempt = as_int(summary_row["attempted_iteration"], "summary attempt")
        telemetry_attempt = as_int(
            telemetry_row["attempted_iteration"], "telemetry attempt"
        )
        if attempt != index or telemetry_attempt != index:
            fail(f"seed{seed}/{arm} attempt sequence mismatch at row {index}")
        summary_skip = as_int(summary_row["step_skipped"], "summary step_skipped")
        telemetry_skip = as_int(
            telemetry_row["step_skipped"], "telemetry step_skipped"
        )
        if summary_skip not in (0, 1) or telemetry_skip != summary_skip:
            fail(f"seed{seed}/{arm} AMP skip mismatch at attempt {index}")
        raw_nonfinite = as_int(
            telemetry_row["raw_grad_nonfinite_count"], "raw grad nonfinite count"
        )
        if (raw_nonfinite > 0) != bool(summary_skip):
            fail(f"seed{seed}/{arm} raw-gradient/AMP mismatch at attempt {index}")
        if summary_skip:
            skip_attempts.append(index)
        for field in SEMANTIC_NONFINITE_FIELDS:
            if as_int(telemetry_row[field], field) != 0:
                fail(f"seed{seed}/{arm} has nonzero {field} at attempt {index}")
        if as_int(telemetry_row["nonpositive_denominator_count"], "denominator count") != 0:
            fail(f"seed{seed}/{arm} has a nonpositive denominator at attempt {index}")

    final_summary = summary[-1]
    final_telemetry = telemetry[-1]
    final_steps = as_int(final_summary["successful_optimizer_steps"], "final updates")
    exact_final = {
        "attempted_iteration": 2000,
        "processed_nimg": 256000,
    }
    for field, expected in exact_final.items():
        if as_int(final_summary[field], f"final summary {field}") != expected:
            fail(f"seed{seed}/{arm} final {field} mismatch")
        if as_int(final_telemetry[field], f"final telemetry {field}") != expected:
            fail(f"seed{seed}/{arm} telemetry final {field} mismatch")
    if abs(as_float(final_summary["processed_kimg"], "final kimg") - 256.0) > 1e-9:
        fail(f"seed{seed}/{arm} did not finish at 256.000 kimg")
    if final_steps != 2000 - len(skip_attempts):
        fail(f"seed{seed}/{arm} successful-update count disagrees with AMP skips")
    if as_int(final_telemetry["successful_optimizer_steps"], "telemetry updates") != final_steps:
        fail(f"seed{seed}/{arm} summary/telemetry final update mismatch")

    options = load_json(run_dir / "training_options.json")
    initial = load_json(run_dir / "initial_state_receipt_v1.json")
    target_scale, denominator_scale = ARM_SCALES[arm]
    exact_options = {
        "total_kimg": 256,
        "batch_size": 128,
        "batch_gpu": 16,
        "ema_beta": 0.9993,
        "cudnn_benchmark": False,
        "enable_tf32": False,
        "enable_amp": True,
        "seed": seed,
    }
    for field, expected in exact_options.items():
        if options.get(field) != expected:
            fail(f"seed{seed}/{arm} training option {field} mismatch")
    loss = options.get("loss_kwargs")
    if not isinstance(loss, dict) or (
        loss.get("factorial_protocol"),
        loss.get("target_gap_scale"),
        loss.get("denominator_gap_scale"),
    ) != ("q256_target_weight_v1", target_scale, denominator_scale):
        fail(f"seed{seed}/{arm} factorial training options mismatch")
    if initial.get("schema") != "ect.q256.target-weight-initial-state/v1":
        fail(f"seed{seed}/{arm} initial receipt schema mismatch")
    factorial = initial.get("factorial")
    if not isinstance(factorial, dict) or (
        factorial.get("arm"),
        factorial.get("protocol"),
        factorial.get("target_gap_scale"),
        factorial.get("denominator_gap_scale"),
    ) != (arm, "q256_target_weight_v1", target_scale, denominator_scale):
        fail(f"seed{seed}/{arm} initial factorial identity mismatch")
    if initial.get("seed") != seed:
        fail(f"seed{seed}/{arm} initial seed mismatch")
    common_state = initial.get("common_initial_state_sha256")
    if not isinstance(common_state, str) or len(common_state) != 64:
        fail(f"seed{seed}/{arm} invalid common initial-state digest")

    return {
        "schema": "ect.q256.direct-formal-cell-binding/v1",
        "status": "PASS",
        "verified_utc": utc_now(),
        "seed": seed,
        "arm": arm,
        "run_dir": str(run_dir),
        "checkpoint": str(run_dir / "network-snapshot-latest.pkl"),
        "checkpoint_sha256": artifacts["network-snapshot-latest.pkl"]["sha256"],
        "checkpoint_bytes": artifacts["network-snapshot-latest.pkl"]["bytes"],
        "attempted_iterations": 2000,
        "processed_nimg": 256000,
        "processed_kimg": 256.0,
        "successful_optimizer_steps": final_steps,
        "amp_skip_attempts": skip_attempts,
        "semantic_nonfinite_count": 0,
        "nonpositive_denominator_count": 0,
        "raw_gradient_amp_skip_mismatch_count": 0,
        "initial_common_state_sha256": common_state,
        "artifacts": artifacts,
    }


def write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o640)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def create_binding(matrix_dir: Path, formal_root: Path, evaluator: Any) -> dict[str, Any]:
    if matrix_dir.exists():
        fail(f"refuse to reuse matrix binding directory: {matrix_dir}")
    if not matrix_dir.parent.is_dir():
        fail(f"matrix binding parent is missing: {matrix_dir.parent}")
    source = evaluator.source_snapshot(require_clean=True)
    if source.get("git_head") != EXPECTED_HEAD:
        fail(f"evaluator/training source mismatch: {source.get('git_head')} != {EXPECTED_HEAD}")
    if source.get("git_branch") != evaluator.EXPECTED_BRANCH:
        fail(f"wrong frozen source branch: {source.get('git_branch')}")
    formal_root = formal_root.resolve(strict=True)
    cells = [
        verify_direct_cell(formal_root / f"seed{seed}" / f"arm{arm}", seed, arm)
        for seed in SEEDS
        for arm in ARMS
    ]
    for seed in SEEDS:
        common = {
            cell["initial_common_state_sha256"]
            for cell in cells
            if cell["seed"] == seed
        }
        if len(common) != 1:
            fail(f"seed {seed} arms do not share one common initial state")

    matrix_dir.mkdir(mode=0o750)
    (matrix_dir / "cells").mkdir()
    cell_receipts = []
    for cell in cells:
        path = matrix_dir / "cells" / f"seed{cell['seed']}-arm{cell['arm']}.json"
        write_exclusive_json(path, cell)
        cell_receipts.append(
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    adapter_path = Path(__file__).resolve(strict=True)
    preregistration = evaluator.REPO_ROOT / "analysis/q256_target_weight_factorial/preregistration.json"
    payload = {
        "schema": "ect.q256.direct-formal-matrix-binding/v1",
        "status": "PASS",
        "created_utc": utc_now(),
        "selection_policy": "all_12_final_256kimg_checkpoints_no_intermediate_selection",
        "formal_root": str(formal_root),
        "cell_count": 12,
        "cell_receipts": cell_receipts,
        "training_source_git_head": EXPECTED_HEAD,
        "training_source_content_sha256": source["content_sha256"],
        "preregistration_path": "analysis/q256_target_weight_factorial/preregistration.json",
        "preregistration_sha256": sha256_file(preregistration),
        "adapter": {
            "path": str(adapter_path),
            "bytes": adapter_path.stat().st_size,
            "sha256": sha256_file(adapter_path),
            "scope": "provenance binding and NFE-primary job ordering only",
            "checkpoint_mutation": False,
            "metric_numerical_semantics_changed": False,
        },
    }
    payload["cell_receipts_tree_sha256"] = canonical_sha256(cell_receipts)
    write_exclusive_json(matrix_dir / "direct_matrix_binding.json", payload)
    return payload


def load_bound_matrix(matrix_dir: Path, evaluator: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matrix_dir = matrix_dir.resolve(strict=True)
    binding_path = matrix_dir / "direct_matrix_binding.json"
    binding = load_json(binding_path)
    if binding.get("schema") != "ect.q256.direct-formal-matrix-binding/v1" or binding.get("status") != "PASS":
        fail("direct matrix binding is not PASS")
    if binding.get("training_source_git_head") != EXPECTED_HEAD:
        fail("direct matrix binding has the wrong source commit")
    adapter = binding.get("adapter")
    if not isinstance(adapter, dict):
        fail("direct matrix binding has no adapter receipt")
    recorded_adapter = Path(str(adapter.get("path", ""))).resolve(strict=True)
    if adapter.get("sha256") != sha256_file(recorded_adapter):
        fail("direct matrix binding adapter changed after authorization")
    receipts = binding.get("cell_receipts")
    if not isinstance(receipts, list) or len(receipts) != 12:
        fail("direct matrix binding must contain 12 cell receipts")
    if binding.get("cell_receipts_tree_sha256") != canonical_sha256(receipts):
        fail("direct matrix receipt tree binding is stale")
    cells = []
    for receipt in receipts:
        path = Path(str(receipt.get("path", ""))).resolve(strict=True)
        if path.parent != matrix_dir / "cells":
            fail(f"cell binding escapes matrix directory: {path}")
        if receipt.get("bytes") != path.stat().st_size or receipt.get("sha256") != sha256_file(path):
            fail(f"cell binding receipt changed: {path}")
        recorded = load_json(path)
        current = verify_direct_cell(
            Path(recorded["run_dir"]), int(recorded["seed"]), str(recorded["arm"])
        )
        comparable_keys = set(recorded) - {"verified_utc"}
        if {key: current[key] for key in comparable_keys} != {
            key: recorded[key] for key in comparable_keys
        }:
            fail(f"immutable direct training evidence changed: {path}")
        cells.append(
            {
                "arm": recorded["arm"],
                "seed": recorded["seed"],
                "run_dir": recorded["run_dir"],
                "checkpoint": recorded["checkpoint"],
                "checkpoint_sha256": recorded["checkpoint_sha256"],
                "checkpoint_bytes": recorded["checkpoint_bytes"],
                "training_validation_receipt": str(path),
                "training_validation_receipt_sha256": receipt["sha256"],
                "training_hash_receipt": str(path),
                "training_hash_receipt_sha256": receipt["sha256"],
                "training_source_git_head": EXPECTED_HEAD,
                "training_source_content_sha256": binding["training_source_content_sha256"],
                "preregistration_path": binding["preregistration_path"],
                "preregistration_sha256": binding["preregistration_sha256"],
                "initial_common_state_sha256": recorded["initial_common_state_sha256"],
                "amp_skip_attempts": recorded["amp_skip_attempts"],
                "successful_optimizer_steps": recorded["successful_optimizer_steps"],
                "amp_skip_signature_expected_value_enforced": False,
                "production_verifier_receipts": {
                    "kind": "direct-formal-provenance-binding",
                    "receipt": str(path),
                    "sha256": receipt["sha256"],
                },
            }
        )
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if {(cell["seed"], cell["arm"]) for cell in cells} != expected:
        fail("direct matrix cell identity is incomplete")
    matrix = {
        "matrix_dir": str(matrix_dir),
        "direct_matrix_binding": str(binding_path),
        "direct_matrix_binding_sha256": sha256_file(binding_path),
        "training_source_git_head": EXPECTED_HEAD,
        "training_source_content_sha256": binding["training_source_content_sha256"],
        "preregistration_path": binding["preregistration_path"],
        "preregistration_sha256": binding["preregistration_sha256"],
        "cell_count": 12,
        "expected_amp_skip_attempts": None,
        "selection_policy": binding["selection_policy"],
        "provenance_adapter": binding["adapter"],
        "evaluation_repair_adapter": regular_file_binding(Path(__file__).resolve()),
    }
    return sorted(cells, key=lambda cell: (cell["seed"], ARMS.index(cell["arm"]))), matrix


def validate_prior_evaluation(
    prior_root: Path, evaluator: Any
) -> tuple[set[str], dict[str, Any]]:
    prior_root = prior_root.resolve(strict=True)
    if prior_root.is_symlink() or not prior_root.is_dir():
        fail(f"invalid prior evaluation root: {prior_root}")
    plan_path = prior_root / "evaluation_plan.json"
    completion_path = prior_root / "evaluation_completion.json"
    plan = load_json(plan_path)
    completion = load_json(completion_path)
    if plan.get("schema") != evaluator.PLAN_SCHEMA:
        fail("prior evaluation has the wrong plan schema")
    if completion.get("schema") != evaluator.COMPLETION_SCHEMA:
        fail("prior evaluation has the wrong completion schema")
    if completion.get("status") != "STOPPED_FOR_AUDIT":
        fail("continuation requires one fail-closed prior evaluation")
    if completion.get("failed_job_id") != "seed5-armA-nfe1":
        fail("prior evaluation did not stop at the authorized seed5/A boundary")
    if completion.get("evaluation_plan_sha256") != sha256_file(plan_path):
        fail("prior evaluation completion does not bind its plan")
    completed = completion.get("completed_job_ids")
    if tuple(completed or ()) != EXPECTED_PRIOR_COMPLETED:
        fail(f"unexpected prior completed job prefix: {completed}")
    if (
        plan.get("precision") != "fp32"
        or plan.get("sample_count_per_job") != 50_000
        or plan.get("sample_seed_range") != "0-49999"
        or plan.get("metric_seed") != 20_260_730
        or tuple(plan.get("metrics_per_job", ())) != ("kid50k_full", "fid50k_full")
        or plan.get("nfe_modes") != {"1": [], "2": [0.821]}
    ):
        fail("prior evaluation numerical contract is not frozen")

    prior_source = plan.get("evaluator_source")
    current_source = evaluator.source_snapshot(require_clean=True)
    if not isinstance(prior_source, dict):
        fail("prior evaluation has no evaluator source receipt")
    prior_files = {
        item["path"]: item["sha256"]
        for item in prior_source.get("files", [])
        if isinstance(item, dict) and "path" in item and "sha256" in item
    }
    current_files = {
        item["path"]: item["sha256"]
        for item in current_source.get("files", [])
        if isinstance(item, dict) and "path" in item and "sha256" in item
    }
    scheduling_file = "scripts/run_q256_target_weight_evaluation.py"
    if not prior_files or set(prior_files) != set(current_files):
        fail("prior/current evaluator source file sets differ")
    for relative in prior_files:
        if relative != scheduling_file and prior_files[relative] != current_files[relative]:
            fail(f"numerical evaluator source changed since prior PASS: {relative}")

    jobs = {str(job["job_id"]): job for job in plan.get("jobs", [])}
    carry_receipts = []
    for job_id in EXPECTED_PRIOR_COMPLETED:
        job = jobs.get(job_id)
        if not isinstance(job, dict):
            fail(f"prior plan lacks completed job {job_id}")
        receipt_path = prior_root / "receipts" / f"{job_id}.json"
        receipt = load_json(receipt_path)
        if (
            receipt.get("schema") != evaluator.JOB_RECEIPT_SCHEMA
            or receipt.get("status") != "passed"
            or receipt.get("job_id") != job_id
            or receipt.get("checkpoint_sha256") != job.get("checkpoint_sha256")
            or receipt.get("dataset_sha256") != plan["dataset"]["sha256"]
        ):
            fail(f"prior PASS receipt identity mismatch: {job_id}")
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, dict):
            fail(f"prior PASS receipt has no artifact tree: {job_id}")
        observed = evaluator.hash_regular_tree(Path(job["output_directory"]))
        if observed != artifacts or receipt.get("artifacts_tree_sha256") != evaluator.canonical_sha256(observed):
            fail(f"prior PASS artifacts changed: {job_id}")
        kid = artifacts.get("generated-features-kid50k_full-repeat00.npy", {})
        fid = artifacts.get("generated-features-fid50k_full-repeat00.npy", {})
        if kid.get("sha256") != fid.get("sha256"):
            fail(f"prior PASS feature identity changed: {job_id}")
        carry_receipts.append(
            {
                "job_id": job_id,
                "receipt": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
                "artifacts_tree_sha256": receipt["artifacts_tree_sha256"],
            }
        )
    record = {
        "schema": "ect.q256.formal-evaluation-continuation/v1",
        "status": "PASS",
        "prior_root": str(prior_root),
        "prior_plan": str(plan_path),
        "prior_plan_sha256": sha256_file(plan_path),
        "prior_completion": str(completion_path),
        "prior_completion_sha256": sha256_file(completion_path),
        "carried_forward_job_count": len(carry_receipts),
        "carried_forward_receipts": carry_receipts,
        "failed_job_restarted_fresh": "seed5-armA-nfe1",
        "metric_numerical_semantics_changed": False,
    }
    return set(EXPECTED_PRIOR_COMPLETED), record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--reuse-bound-matrix", action="store_true")
    parser.add_argument("--prior-evaluation-root", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--base-port", type=int, default=31_800)
    parser.add_argument("--lock-root", type=Path, default=Path("/data/temp/ECT001-q256-evaluation-locks"))
    args = parser.parse_args(argv)

    repo = args.repo.resolve(strict=True)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from scripts import run_q256_target_weight_evaluation as evaluator

    if evaluator.REPO_ROOT.resolve() != repo:
        fail(f"loaded evaluator from unexpected repository: {evaluator.REPO_ROOT}")
    if args.reuse_bound_matrix:
        if not args.matrix_dir.resolve().is_dir():
            fail(f"bound matrix directory is missing: {args.matrix_dir}")
    else:
        create_binding(args.matrix_dir.resolve(), args.formal_root, evaluator)

    original_build_jobs = evaluator.build_jobs
    carried_forward_ids: set[str] = set()
    continuation_record = None
    if args.prior_evaluation_root is not None:
        carried_forward_ids, continuation_record = validate_prior_evaluation(
            args.prior_evaluation_root, evaluator
        )

    def primary_first_jobs(cells: Sequence[Mapping[str, Any]], output_root: Path, base_port: int) -> list[dict[str, Any]]:
        jobs = original_build_jobs(cells, output_root, base_port)
        ordered = sorted(jobs, key=lambda job: (int(job["nfe"]), int(job["seed"]), ARMS.index(str(job["arm"]))))
        return [job for job in ordered if str(job["job_id"]) not in carried_forward_ids]

    def direct_revalidation(
        run_dir: Path,
        *,
        phase: str,
        arm: str,
        seed: int,
        expected_skip_attempts: list[int] | None,
        runtime_command: Sequence[str] | None,
        process_env: Mapping[str, str] | None,
    ) -> dict[str, Any]:
        if phase != "formal" or expected_skip_attempts is not None:
            fail("direct revalidation received a non-frozen contract")
        record = verify_direct_cell(run_dir, seed, arm)
        return {
            "status": "PASS",
            "kind": "direct-formal-read-only-revalidation",
            "checkpoint_sha256": record["checkpoint_sha256"],
            "cell_binding_sha256": canonical_sha256(
                {key: value for key, value in record.items() if key != "verified_utc"}
            ),
            "artifact_count": len(record["artifacts"]),
            "checkpoint_mutation": False,
        }

    evaluator.load_training_matrix = lambda matrix_dir: load_bound_matrix(matrix_dir, evaluator)
    evaluator.build_jobs = primary_first_jobs
    evaluator.training_launcher.deep_revalidate_existing_arm = direct_revalidation

    # A single host-side nvidia-smi probe exceeded the frozen 0.4 s timeout in
    # the first launch even though the immediately following idle audit passed
    # and no foreign process existed.  Preserve the one-second monitor cadence
    # and every exclusivity check, but allow one explicitly recorded bounded
    # bounded retry per job for host-tool timeouts only.  Foreign processes,
    # cadence failures, and a second timeout from either probe still stop the
    # job for audit.
    original_stream_process = evaluator.training_launcher.stream_process
    original_gpu_query = evaluator.training_launcher.query_gpu_compute_processes
    original_process_tree = evaluator.training_launcher.process_tree_pids

    def stream_with_one_bounded_gpu_probe_retry(*stream_args: Any, **stream_kwargs: Any) -> int:
        gpu_retries_used = 0
        process_tree_retries_used = 0

        def bounded_gpu_query(gpu_uuid: str, *, timeout_seconds: float = 5.0) -> list[dict[str, object]]:
            nonlocal gpu_retries_used
            try:
                return original_gpu_query(gpu_uuid, timeout_seconds=timeout_seconds)
            except evaluator.training_launcher.LaunchError as exc:
                if gpu_retries_used == 0 and "timed out after" in str(exc):
                    gpu_retries_used = 1
                    return original_gpu_query(gpu_uuid, timeout_seconds=0.8)
                raise

        def bounded_process_tree(root_pid: int, *, timeout_seconds: float = 5.0) -> set[int]:
            nonlocal process_tree_retries_used
            try:
                return original_process_tree(root_pid, timeout_seconds=timeout_seconds)
            except evaluator.training_launcher.LaunchError as exc:
                if process_tree_retries_used == 0 and "timed out after" in str(exc):
                    process_tree_retries_used = 1
                    return original_process_tree(root_pid, timeout_seconds=0.8)
                raise

        evaluator.training_launcher.query_gpu_compute_processes = bounded_gpu_query
        evaluator.training_launcher.process_tree_pids = bounded_process_tree
        try:
            return original_stream_process(*stream_args, **stream_kwargs)
        finally:
            monitor = stream_kwargs.get("gpu_monitor_record")
            if isinstance(monitor, dict):
                monitor["gpu_audit_probe_timeout_recovery"] = {
                    "schema": "ect.q256.gpu-audit-probe-timeout-recovery/v1",
                    "base_timeout_seconds": 0.4,
                    "bounded_retry_timeout_seconds": 0.8,
                    "maximum_retries_per_probe_per_job": 1,
                    "nvidia_smi_retries_used": gpu_retries_used,
                    "process_tree_ps_retries_used": process_tree_retries_used,
                    "trigger": "nvidia-smi or process-tree ps timeout only",
                    "foreign_process_tolerance_changed": False,
                    "metric_numerical_semantics_changed": False,
                }
            evaluator.training_launcher.query_gpu_compute_processes = original_gpu_query
            evaluator.training_launcher.process_tree_pids = original_process_tree

    evaluator.training_launcher.stream_process = stream_with_one_bounded_gpu_probe_retry
    if continuation_record is not None:
        original_build_plan = evaluator.build_plan

        def build_continuation_plan(**plan_kwargs: Any) -> dict[str, Any]:
            plan = original_build_plan(**plan_kwargs)
            plan["continuation"] = continuation_record
            return plan

        evaluator.build_plan = build_continuation_plan
    data = args.data or evaluator.DEFAULT_DATASET
    execute_args = argparse.Namespace(
        matrix_dir=args.matrix_dir,
        data=data,
        outdir=args.outdir,
        gpu=args.gpu,
        base_port=args.base_port,
        lock_root=args.lock_root,
        evaluator_repair_base_git_head=EXPECTED_HEAD,
    )
    return evaluator.execute(execute_args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BindingError as exc:
        raise SystemExit(f"[q256-direct-evaluation-adapter] ERROR: {exc}") from exc
