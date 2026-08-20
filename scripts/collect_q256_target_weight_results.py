#!/usr/bin/env python3
"""Validate and summarize the frozen q256 target/weight formal evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_q256_target_weight_evaluation as evaluator


RESULT_SCHEMA = "ect.q256.target-weight-factorial-results/v1"
COLLECTION_SCHEMA = "ect.q256.target-weight-factorial-collection/v1"
PREREGISTRATION = (
    REPO_ROOT / "analysis" / "q256_target_weight_factorial" / "preregistration.json"
)
CLAIM_BOUNDARY = (
    REPO_ROOT / "analysis" / "q256_target_weight_factorial" / "CLAIM_BOUNDARY.md"
)

CONTRASTS = (
    ("target_at_baseline_weight", "Y_C-Y_A", "C", "A"),
    ("target_at_g_weight", "Y_B-Y_D", "B", "D"),
    ("weight_at_baseline_target", "Y_D-Y_A", "D", "A"),
    ("weight_at_g_target", "Y_B-Y_C", "B", "C"),
)
INTERACTION = ("target_x_weight", "Y_B-Y_C-Y_D+Y_A")
QUANTITIES = evaluator.ARMS + tuple(item[0] for item in CONTRASTS) + (INTERACTION[0],)


class CollectionError(RuntimeError):
    """Formal evaluation artifacts are incomplete, changed, or inconsistent."""


def fail(message: str) -> None:
    raise CollectionError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        fail(f"cannot hash {path}: {exc}")
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {label} {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must contain one JSON object: {path}")
    return value


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        fail(f"refuse to overwrite immutable collection output: {path}")


def write_csv_exclusive(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    try:
        handle = path.open("x", newline="", encoding="utf-8")
    except FileExistsError:
        fail(f"refuse to overwrite immutable collection output: {path}")
    with handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def require_exact_file(path: Path, binding: Mapping[str, Any], label: str) -> None:
    if path.is_symlink() or not path.is_file():
        fail(f"{label} is missing or a symlink: {path}")
    expected_sha = binding.get("sha256")
    expected_bytes = binding.get("bytes")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        fail(f"{label} has an invalid SHA256 binding")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes <= 0:
        fail(f"{label} has an invalid byte binding")
    if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha:
        fail(f"{label} changed after its PASS receipt: {path}")


def read_raw_metric(path: Path, metric: str, checkpoint: Path) -> float:
    if path.is_symlink() or not path.is_file():
        fail(f"missing raw metric result: {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        fail(f"raw metric result must contain exactly one line: {path}")
    try:
        payload = json.loads(lines[0])
        value = float(payload["results"][metric])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        fail(f"malformed raw metric result {path}: {exc}")
    if payload.get("metric") != metric or payload.get("num_gpus") != 1:
        fail(f"raw metric identity mismatch: {path}")
    if not math.isfinite(value) or (metric.startswith("fid") and value < 0):
        fail(f"invalid raw metric value in {path}: {value}")
    raw_snapshot = payload.get("snapshot_pkl")
    if not isinstance(raw_snapshot, str):
        fail(f"raw metric result lacks checkpoint provenance: {path}")
    if (path.parent / raw_snapshot).resolve() != checkpoint.resolve(strict=True):
        fail(f"raw metric result points to the wrong checkpoint: {path}")
    return value


def _validate_preregistration(plan: Mapping[str, Any]) -> dict[str, Any]:
    training_matrix = plan.get("training_matrix")
    if not isinstance(training_matrix, dict):
        fail("evaluation plan lacks the training-matrix provenance")
    expected_sha = training_matrix.get("preregistration_sha256")
    if sha256_file(PREREGISTRATION) != expected_sha:
        fail("frozen preregistration changed after formal training")
    preregistration = load_json(PREREGISTRATION, "frozen preregistration")
    if preregistration.get("schema") != "ect.q256.target-weight-factorial-preregistration/v1":
        fail("wrong frozen preregistration schema")
    if preregistration.get("independent_unit", {}).get("values") != list(evaluator.SEEDS):
        fail("preregistration training seeds differ from the evaluated matrix")
    if preregistration.get("evaluation", {}).get("sample_count") != evaluator.SAMPLE_COUNT:
        fail("preregistration sample count differs from the evaluated matrix")
    return preregistration


def _validate_training_cells(plan: Mapping[str, Any]) -> None:
    training_matrix = plan.get("training_matrix")
    if (
        not isinstance(training_matrix, dict)
        or "expected_amp_skip_attempts" not in training_matrix
    ):
        fail("evaluation plan lacks the matrix AMP skip contract")
    expected_skip_attempts = training_matrix["expected_amp_skip_attempts"]
    records = plan.get("training_cells")
    if not isinstance(records, list) or len(records) != 12:
        fail("evaluation plan must bind exactly 12 training runs")
    expected = {(seed, arm) for seed in evaluator.SEEDS for arm in evaluator.ARMS}
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            fail("evaluation plan training-cell record is not an object")
        key = (record.get("seed"), record.get("arm"))
        if key not in expected or key in seen:
            fail(f"unexpected or duplicate training cell in evaluation plan: {key}")
        seen.add(key)
        current = evaluator.validate_training_run(
            Path(record["run_dir"]),
            key[1],
            key[0],
            expected_skip_attempts=expected_skip_attempts,
        )
        exact_fields = (
            "checkpoint_sha256",
            "training_validation_receipt_sha256",
            "training_hash_receipt_sha256",
            "training_source_git_head",
            "training_source_content_sha256",
            "preregistration_sha256",
        )
        for field in exact_fields:
            if current[field] != record.get(field):
                fail(f"training cell {key} changed field {field} after evaluation authorization")
    if seen != expected:
        fail(f"evaluation plan has incomplete training runs: {sorted(expected - seen)}")


def _validate_training_revalidation(plan: Mapping[str, Any]) -> None:
    records = plan.get("training_arm_revalidation")
    cells = plan.get("training_cells")
    training_matrix = plan.get("training_matrix")
    runtime = plan.get("runtime")
    if not isinstance(records, list) or len(records) != 12:
        fail("evaluation plan lacks exactly 12 fresh arm-revalidation records")
    if not isinstance(cells, list) or not isinstance(training_matrix, dict):
        fail("evaluation plan lacks training scope for arm revalidation")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("python_executable"), str):
        fail("evaluation plan lacks the revalidation Python runtime binding")
    expected_skip = training_matrix.get("expected_amp_skip_attempts")
    cells_by_key = {
        (cell.get("seed"), cell.get("arm")): cell
        for cell in cells
        if isinstance(cell, dict)
    }
    expected_keys = {(seed, arm) for seed in evaluator.SEEDS for arm in evaluator.ARMS}
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            fail("evaluation arm-revalidation record is not an object")
        key = (record.get("seed"), record.get("arm"))
        if key not in expected_keys or key in seen or key not in cells_by_key:
            fail(f"unexpected or duplicate arm-revalidation record: {key}")
        if record.get("status") != "PASS":
            fail(f"arm revalidation was not PASS: {key}")
        cell = cells_by_key[key]
        run_dir = Path(cell["run_dir"]).resolve(strict=True)
        expected_command = [
            runtime["python_executable"],
            str(evaluator.REPO_ROOT / "scripts" / "verify_q256_target_weight_arm.py"),
            "--run-dir",
            str(run_dir),
            "--arm",
            str(key[1]),
            "--seed",
            str(key[0]),
            "--mode",
            "formal",
            "--check-only",
        ]
        if expected_skip is not None:
            expected_command += [
                "--expected-skip-attempts",
                json.dumps(expected_skip, separators=(",", ":")),
            ]
        if record.get("command_argv") != expected_command:
            fail(f"arm revalidation command changed for {key}")
        validation = load_json(
            run_dir / evaluator.VALIDATION_FILENAME,
            f"arm revalidation source receipt {key}",
        )
        if record.get("report_sha256") != evaluator.canonical_sha256(validation):
            fail(f"arm revalidation report hash changed for {key}")
        seen.add(key)
    if seen != expected_keys:
        fail(f"evaluation plan has incomplete arm revalidation: {sorted(expected_keys - seen)}")


def validate_and_collect(eval_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    eval_root = eval_root.expanduser().resolve(strict=True)
    plan_path = eval_root / "evaluation_plan.json"
    completion_path = eval_root / "evaluation_completion.json"
    plan = load_json(plan_path, "evaluation plan")
    completion = load_json(completion_path, "evaluation completion")
    if plan.get("schema") != evaluator.PLAN_SCHEMA or plan.get("protocol") != evaluator.PROTOCOL:
        fail("wrong formal evaluation plan schema/protocol")
    planned_source = plan.get("evaluator_source")
    if not isinstance(planned_source, dict):
        fail("evaluation plan lacks the frozen evaluator/collector source")
    try:
        current_source = evaluator.source_snapshot(require_clean=True)
    except evaluator.EvaluationError as exc:
        fail(f"collector source is not clean/frozen: {exc}")
    for field in ("git_head", "content_sha256"):
        if current_source.get(field) != planned_source.get(field):
            fail(f"collector source differs from evaluation plan: {field}")
    if plan.get("status") != "authorized_exact_matrix" or plan.get("job_count") != 24:
        fail("evaluation plan is not the exact authorized 24-job matrix")
    if plan.get("selection_policy") != "all_12_final_256kimg_checkpoints_no_intermediate_selection":
        fail("evaluation plan does not prohibit intermediate selection")
    if plan.get("dataset", {}).get("sha256") != evaluator.DATASET_SHA256:
        fail("evaluation plan used a noncanonical dataset")
    planned_dataset = plan.get("dataset")
    if not isinstance(planned_dataset, dict) or not isinstance(planned_dataset.get("path"), str):
        fail("evaluation plan lacks the canonical dataset path binding")
    current_dataset = evaluator.verify_dataset(Path(planned_dataset["path"]))
    for field in ("path", "sha256", "bytes"):
        if current_dataset[field] != planned_dataset.get(field):
            fail(f"canonical dataset changed after evaluation authorization: {field}")
    if plan.get("sample_count_per_job") != evaluator.SAMPLE_COUNT:
        fail("evaluation plan sample count mismatch")
    if plan.get("sample_seed_range") != evaluator.SAMPLE_SEEDS:
        fail("evaluation plan generation seed range mismatch")
    if plan.get("metric_seed") != evaluator.METRIC_SEED:
        fail("evaluation plan metric seed mismatch")
    if plan.get("metrics_per_job") != list(evaluator.METRICS):
        fail("evaluation plan raw metric set mismatch")
    if plan.get("nfe_modes") != {"1": [], "2": [0.821]}:
        fail("evaluation plan NFE settings mismatch")
    planned_gpu = plan.get("gpu")
    if (
        not isinstance(planned_gpu, dict)
        or not isinstance(planned_gpu.get("uuid"), str)
    ):
        fail("evaluation plan lacks the selected GPU UUID")
    if completion.get("schema") != evaluator.COMPLETION_SCHEMA:
        fail("wrong formal evaluation completion schema")
    if completion.get("status") != "PASS" or completion.get("job_count") != 24:
        fail("formal evaluation did not complete with an exact PASS")
    if completion.get("evaluation_plan_sha256") != sha256_file(plan_path):
        fail("formal evaluation completion does not bind the current plan")
    preregistration = _validate_preregistration(plan)
    _validate_training_cells(plan)
    _validate_training_revalidation(plan)

    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 24:
        fail("evaluation plan must contain exactly 24 job records")
    expected_jobs = {
        (seed, arm, nfe)
        for seed in evaluator.SEEDS
        for arm in evaluator.ARMS
        for nfe in evaluator.NFE_SETTINGS
    }
    completed_ids = completion.get("completed_job_ids")
    if not isinstance(completed_ids, list) or len(completed_ids) != 24 or len(set(completed_ids)) != 24:
        fail("evaluation completion job list is not exact")

    endpoint_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    seen_jobs = set()
    source_binding = None
    cache_binding: dict[str, Any] | None = None
    for job in jobs:
        if not isinstance(job, dict):
            fail("evaluation job record is not an object")
        key = (job.get("seed"), job.get("arm"), job.get("nfe"))
        if key not in expected_jobs or key in seen_jobs:
            fail(f"unexpected or duplicate evaluation job: {key}")
        seen_jobs.add(key)
        job_id = job.get("job_id")
        if job_id not in completed_ids:
            fail(f"PASS completion omits evaluation job {job_id}")
        expected_mid_t = evaluator.NFE_SETTINGS[key[2]]
        exact_job = {
            "mid_t": expected_mid_t,
            "sample_count": evaluator.SAMPLE_COUNT,
            "sample_seeds": evaluator.SAMPLE_SEEDS,
            "metric_seed": evaluator.METRIC_SEED,
            "metrics": list(evaluator.METRICS),
            "precision": "fp32",
        }
        for field, value in exact_job.items():
            if job.get(field) != value:
                fail(f"evaluation job {job_id} field {field} mismatch")
        checkpoint = Path(str(job["checkpoint"])).resolve(strict=True)
        if sha256_file(checkpoint) != job.get("checkpoint_sha256"):
            fail(f"evaluation checkpoint changed: {checkpoint}")

        receipt_path = eval_root / "receipts" / f"{job_id}.json"
        receipt = load_json(receipt_path, "evaluation job receipt")
        if receipt.get("schema") != evaluator.JOB_RECEIPT_SCHEMA:
            fail(f"wrong evaluation job receipt schema: {receipt_path}")
        receipt_exact = {
            "status": "passed",
            "protocol": evaluator.PROTOCOL,
            "job_id": job_id,
            "seed": key[0],
            "arm": key[1],
            "nfe": key[2],
            "mid_t": expected_mid_t,
            "checkpoint_sha256": job["checkpoint_sha256"],
            "dataset_sha256": evaluator.DATASET_SHA256,
            "sample_count": evaluator.SAMPLE_COUNT,
            "sample_seed_range": evaluator.SAMPLE_SEEDS,
            "metric_seed": evaluator.METRIC_SEED,
            "precision": "fp32",
            "returncode": 0,
        }
        for field, value in receipt_exact.items():
            if receipt.get(field) != value:
                fail(f"evaluation receipt {job_id} field {field} mismatch")
        if receipt.get("execution_error") is not None:
            fail(f"evaluation receipt {job_id} records an execution error")
        try:
            evaluator.training_launcher.validate_gpu_monitor_record(
                receipt.get("gpu_exclusivity_monitor"),
                label=f"evaluation receipt {job_id}",
                expected_gpu_uuid=planned_gpu["uuid"],
            )
            evaluator.training_launcher.validate_gpu_idle_record(
                receipt.get("post_job_gpu_idle_check"),
                label=f"evaluation receipt {job_id} post-job",
                expected_gpu_uuid=planned_gpu["uuid"],
            )
        except evaluator.training_launcher.LaunchError as exc:
            fail(f"evaluation receipt {job_id} GPU evidence failed: {exc}")
        launch_path = Path(str(receipt.get("launch_manifest", "")))
        process_log = Path(str(receipt.get("process_log", "")))
        if (
            not launch_path.is_file()
            or sha256_file(launch_path) != receipt.get("launch_manifest_sha256")
            or not process_log.is_file()
            or sha256_file(process_log) != receipt.get("process_log_sha256")
        ):
            fail(f"evaluation receipt provenance changed: {job_id}")
        launch = load_json(launch_path, "evaluation job launch manifest")
        if launch.get("schema") != evaluator.JOB_LAUNCH_SCHEMA:
            fail(f"wrong evaluation launch schema: {launch_path}")
        if launch.get("evaluation_plan_sha256") != sha256_file(plan_path):
            fail(f"evaluation launch does not bind the plan: {job_id}")
        if launch.get("job") != job:
            fail(f"evaluation launch job contract differs from plan: {job_id}")
        if launch.get("gpu") != planned_gpu:
            fail(f"evaluation launch selected another GPU: {job_id}")
        if launch.get("gpu_exclusivity_monitor_contract") != {
            "schema": evaluator.training_launcher.GPU_MONITOR_SCHEMA,
            "gpu_uuid": planned_gpu["uuid"],
            "poll_interval_seconds": 1.0,
            "fail_closed": True,
        }:
            fail(f"evaluation launch has a stale GPU monitor contract: {job_id}")

        current_source = (
            receipt.get("evaluator_source_git_head"),
            receipt.get("evaluator_source_content_sha256"),
        )
        planned_source = (
            plan.get("evaluator_source", {}).get("git_head"),
            plan.get("evaluator_source", {}).get("content_sha256"),
        )
        if current_source != planned_source:
            fail(f"evaluation source binding mismatch: {job_id}")
        if source_binding is None:
            source_binding = current_source
        elif source_binding != current_source:
            fail("evaluation jobs do not share one exact evaluator source")

        target = Path(str(job["output_directory"])).resolve(strict=True)
        evaluator.validate_evaluation_options(
            target / "training_options.json",
            job=job,
            dataset=Path(current_dataset["path"]),
            checkpoint=checkpoint,
        )
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            fail(f"evaluation receipt has no artifact hashes: {job_id}")
        required_artifacts = {
            "training_options.json",
            "log.txt",
            "metric-kid50k_full.jsonl",
            "metric-fid50k_full.jsonl",
            "generated-features-kid50k_full-repeat00.npy",
            "generated-features-fid50k_full-repeat00.npy",
            "generated-samples.npy",
            "sampling_block_diagnostics_v1.json",
        }
        if not required_artifacts.issubset(artifacts):
            fail(
                f"evaluation receipt {job_id} is missing required artifacts: "
                f"{sorted(required_artifacts - set(artifacts))}"
            )
        for relative, binding in artifacts.items():
            if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
                fail(f"evaluation artifact path escapes job {job_id}: {relative!r}")
            if not isinstance(binding, dict):
                fail(f"invalid evaluation artifact binding: {job_id}/{relative}")
            require_exact_file(target / relative, binding, f"evaluation artifact {job_id}/{relative}")
        if canonical_sha256(artifacts) != receipt.get("artifacts_tree_sha256"):
            fail(f"evaluation artifact tree digest mismatch: {job_id}")

        receipt_metrics = receipt.get("metrics")
        if not isinstance(receipt_metrics, list) or len(receipt_metrics) != 2:
            fail(f"evaluation receipt lacks both raw metrics: {job_id}")
        embedded = {item.get("metric"): item for item in receipt_metrics if isinstance(item, dict)}
        if set(embedded) != set(evaluator.METRICS):
            fail(f"evaluation receipt metric identities differ: {job_id}")
        for metric in evaluator.METRICS:
            raw_path = target / f"metric-{metric}.jsonl"
            value = read_raw_metric(raw_path, metric, checkpoint)
            if embedded[metric].get("value") != value:
                fail(f"raw and receipted metric values differ: {job_id}/{metric}")
            if embedded[metric].get("raw_sha256") != sha256_file(raw_path):
                fail(f"raw metric hash differs from receipt: {job_id}/{metric}")
            endpoint_rows.append(
                {
                    "endpoint_class": (
                        "primary" if metric == "fid50k_full" and key[2] == 1 else "secondary"
                    ),
                    "training_seed": key[0],
                    "arm": key[1],
                    "metric": metric,
                    "nfe": key[2],
                    "mid_t": json.dumps(expected_mid_t, separators=(",", ":")),
                    "value": value,
                    "checkpoint_sha256": job["checkpoint_sha256"],
                    "raw_metric_path": str(raw_path),
                    "raw_metric_sha256": sha256_file(raw_path),
                    "evaluation_job_id": job_id,
                    "independent_unit": "training_seed",
                    "independent_n_contribution": 1,
                    "reporting_label": "observed_result",
                }
            )

        diagnostic_record = receipt.get("sampling_block_diagnostics")
        if not isinstance(diagnostic_record, dict):
            fail(f"evaluation receipt lacks sampling-block diagnostics: {job_id}")
        diagnostic_path = Path(str(diagnostic_record.get("path", "")))
        if (
            not diagnostic_path.is_file()
            or sha256_file(diagnostic_path) != diagnostic_record.get("sha256")
        ):
            fail(f"sampling-block diagnostic changed: {job_id}")
        diagnostic = load_json(diagnostic_path, "sampling-block diagnostic")
        expected_diagnostic = {
            "schema": evaluator.BLOCK_SCHEMA,
            "status": "descriptive_variation_only",
            "sample_seed_range": evaluator.SAMPLE_SEEDS,
            "sample_count": evaluator.SAMPLE_COUNT,
            "fixed_block_size": evaluator.BLOCK_SIZE,
            "fixed_block_count": evaluator.BLOCK_COUNT,
            "independent_training_replicate_contribution": 0,
            "quality_endpoint": False,
            "selection_criterion": False,
        }
        for field, value in expected_diagnostic.items():
            if diagnostic.get(field) != value:
                fail(f"sampling-block diagnostic {job_id} field {field} mismatch")
        blocks = diagnostic.get("blocks")
        if not isinstance(blocks, list) or len(blocks) != evaluator.BLOCK_COUNT:
            fail(f"sampling-block diagnostic {job_id} does not contain ten blocks")
        for index, block in enumerate(blocks):
            expected_start = index * evaluator.BLOCK_SIZE
            expected_stop = expected_start + evaluator.BLOCK_SIZE - 1
            if not isinstance(block, dict):
                fail(f"sampling block is not an object: {job_id}/{index}")
            if (
                block.get("block_index") != index
                or block.get("sample_seed_start") != expected_start
                or block.get("sample_seed_end") != expected_stop
                or block.get("sample_count") != evaluator.BLOCK_SIZE
            ):
                fail(f"sampling block identity mismatch: {job_id}/{index}")
            centroid = float(block.get("feature_mean_l2_distance_from_full"))
            variance = float(block.get("feature_variance_trace"))
            if not math.isfinite(centroid) or centroid < 0 or not math.isfinite(variance) or variance < 0:
                fail(f"invalid sampling-block feature diagnostic: {job_id}/{index}")
            block_rows.append(
                {
                    "training_seed": key[0],
                    "arm": key[1],
                    "nfe": key[2],
                    "block_index": index,
                    "sample_seed_start": expected_start,
                    "sample_seed_end": expected_stop,
                    "sample_count": evaluator.BLOCK_SIZE,
                    "feature_mean_l2_distance_from_full": centroid,
                    "feature_variance_trace": variance,
                    "independent_n_contribution": 0,
                    "quality_endpoint": False,
                    "reporting_label": "exploratory_observation",
                }
            )
        cache = receipt.get("cache")
        if not isinstance(cache, dict) or cache.get("inception_detector_url") != evaluator.INCEPTION_URL:
            fail(f"evaluation receipt lacks exact detector/cache binding: {job_id}")
        cache_root = Path(str(cache.get("root", ""))).resolve()
        expected_cache_root = (eval_root / "evaluator_cache").resolve(strict=True)
        if cache_root != expected_cache_root:
            fail(f"evaluation cache root mismatch: {job_id}")
        cache_artifacts = cache.get("artifacts")
        if not isinstance(cache_artifacts, dict) or not cache_artifacts:
            fail(f"evaluation cache binding is empty: {job_id}")
        if cache.get("artifact_count") != len(cache_artifacts):
            fail(f"evaluation cache artifact count mismatch: {job_id}")
        current_cache = cache.get("tree_sha256")
        if current_cache != evaluator.canonical_sha256(cache_artifacts):
            fail(f"evaluation cache tree digest mismatch: {job_id}")
        if cache_binding is None:
            for relative, binding in cache_artifacts.items():
                if (
                    not isinstance(relative, str)
                    or Path(relative).is_absolute()
                    or ".." in Path(relative).parts
                    or not isinstance(binding, dict)
                ):
                    fail(f"evaluation cache artifact binding is invalid: {relative!r}")
                require_exact_file(
                    cache_root / relative,
                    binding,
                    f"evaluation cache artifact {relative}",
                )
            cache_binding = cache
        elif cache_binding != cache:
            fail("evaluation jobs do not share one immutable reference/detector cache")

    if seen_jobs != expected_jobs:
        fail(f"formal evaluation is missing jobs: {sorted(expected_jobs - seen_jobs)}")
    if len(endpoint_rows) != 48 or len(block_rows) != 240:
        fail(
            f"formal result cardinality mismatch: endpoints={len(endpoint_rows)}, "
            f"blocks={len(block_rows)}"
        )
    if completion.get("cache_tree_sha256") != cache_binding["tree_sha256"]:
        fail("evaluation completion does not bind the immutable detector/reference cache")
    provenance = {
        "evaluation_root": str(eval_root),
        "evaluation_plan": str(plan_path),
        "evaluation_plan_sha256": sha256_file(plan_path),
        "evaluation_completion": str(completion_path),
        "evaluation_completion_sha256": sha256_file(completion_path),
        "evaluator_source_git_head": source_binding[0],
        "evaluator_source_content_sha256": source_binding[1],
        "cache_tree_sha256": cache_binding["tree_sha256"],
        "dataset_sha256": evaluator.DATASET_SHA256,
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "claim_boundary_sha256": sha256_file(CLAIM_BOUNDARY),
        "preregistration": preregistration,
    }
    return endpoint_rows, block_rows, provenance


def endpoint_order(metric: str, nfe: int) -> tuple[int, int]:
    return (0 if nfe == 1 else 1, 0 if metric == "fid50k_full" else 1)


def build_factorial_tables(
    endpoint_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed: dict[tuple[int, str, int, str], float] = {}
    for row in endpoint_rows:
        key = (int(row["training_seed"]), str(row["metric"]), int(row["nfe"]), str(row["arm"]))
        if key in indexed:
            fail(f"duplicate observed endpoint row: {key}")
        indexed[key] = float(row["value"])
    expected = {
        (seed, metric, nfe, arm)
        for seed in evaluator.SEEDS
        for metric in evaluator.METRICS
        for nfe in evaluator.NFE_SETTINGS
        for arm in evaluator.ARMS
    }
    if set(indexed) != expected:
        fail(f"incomplete factorial endpoint table: missing={sorted(expected - set(indexed))}")

    seed_rows = []
    endpoint_keys = sorted(
        {(metric, nfe) for _seed, metric, nfe, _arm in indexed},
        key=lambda item: endpoint_order(item[0], item[1]),
    )
    for metric, nfe in endpoint_keys:
        for seed in evaluator.SEEDS:
            arms = {arm: indexed[(seed, metric, nfe, arm)] for arm in evaluator.ARMS}
            row: dict[str, Any] = {
                "endpoint_class": "primary" if metric == "fid50k_full" and nfe == 1 else "secondary",
                "metric": metric,
                "nfe": nfe,
                "mid_t": json.dumps(evaluator.NFE_SETTINGS[nfe], separators=(",", ":")),
                "training_seed": seed,
                **arms,
            }
            for name, _formula, left, right in CONTRASTS:
                row[name] = arms[left] - arms[right]
            row[INTERACTION[0]] = arms["B"] - arms["C"] - arms["D"] + arms["A"]
            row.update(
                {
                    "contrast_direction": "lower_is_better_negative_favors_first_term",
                    "independent_unit": "training_seed",
                    "independent_n_contribution": 1,
                    "reporting_label": "observed_result_under_preregistered_contrast",
                }
            )
            seed_rows.append(row)

    summaries = []
    for metric, nfe in endpoint_keys:
        group = [row for row in seed_rows if row["metric"] == metric and row["nfe"] == nfe]
        if [row["training_seed"] for row in group] != list(evaluator.SEEDS):
            fail(f"cross-seed summary lacks exact seeds for {metric}/nfe{nfe}")
        for quantity in QUANTITIES:
            values = [float(row[quantity]) for row in group]
            summaries.append(
                {
                    "endpoint_class": "primary" if metric == "fid50k_full" and nfe == 1 else "secondary",
                    "metric": metric,
                    "nfe": nfe,
                    "quantity": quantity,
                    "independent_unit": "training_seed",
                    "independent_n": 3,
                    "training_seeds": "3,4,5",
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "minimum": min(values),
                    "maximum": max(values),
                    "range": max(values) - min(values),
                    "reporting_label": (
                        "observed_result" if quantity in evaluator.ARMS
                        else "observed_result_under_preregistered_contrast"
                    ),
                }
            )
    return seed_rows, summaries


def summarize_block_variation(block_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, int], list[Mapping[str, Any]]] = {}
    for row in block_rows:
        key = (int(row["training_seed"]), str(row["arm"]), int(row["nfe"]))
        grouped.setdefault(key, []).append(row)
    expected = {
        (seed, arm, nfe)
        for seed in evaluator.SEEDS
        for arm in evaluator.ARMS
        for nfe in evaluator.NFE_SETTINGS
    }
    if set(grouped) != expected:
        fail("sampling-block variation groups do not match the 24 evaluation jobs")
    summaries = []
    for key in sorted(grouped, key=lambda item: (item[0], evaluator.ARMS.index(item[1]), item[2])):
        rows = sorted(grouped[key], key=lambda item: int(item["block_index"]))
        if len(rows) != evaluator.BLOCK_COUNT:
            fail(f"sampling-block group {key} does not contain ten fixed blocks")
        centroid = [float(row["feature_mean_l2_distance_from_full"]) for row in rows]
        variance = [float(row["feature_variance_trace"]) for row in rows]
        summaries.append(
            {
                "training_seed": key[0],
                "arm": key[1],
                "nfe": key[2],
                "sampling_blocks": evaluator.BLOCK_COUNT,
                "block_size": evaluator.BLOCK_SIZE,
                "centroid_distance_minimum": min(centroid),
                "centroid_distance_median": statistics.median(centroid),
                "centroid_distance_maximum": max(centroid),
                "centroid_distance_range": max(centroid) - min(centroid),
                "feature_variance_trace_minimum": min(variance),
                "feature_variance_trace_median": statistics.median(variance),
                "feature_variance_trace_maximum": max(variance),
                "feature_variance_trace_range": max(variance) - min(variance),
                "independent_n_contribution": 0,
                "quality_endpoint": False,
                "reporting_label": "exploratory_observation",
            }
        )
    return summaries


def build_result(
    endpoint_rows: list[dict[str, Any]], block_rows: list[dict[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    seed_rows, cross_seed = build_factorial_tables(endpoint_rows)
    block_summary = summarize_block_variation(block_rows)
    preregistration = provenance["preregistration"]
    return {
        "schema": RESULT_SCHEMA,
        "status": "ANALYZED",
        "experiment_id": evaluator.EXPERIMENT_ID,
        "protocol": evaluator.PROTOCOL,
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "validate",
            "verification_status": "ANALYZED",
            "version_label": "q256_target_weight_factorial_results_v1",
        },
        "independent_unit": {
            "name": "training_seed",
            "values": list(evaluator.SEEDS),
            "n": 3,
            "sampling_blocks_are_repeated_measurements": True,
            "sampling_block_independent_n_contribution": 0,
        },
        "endpoint_definitions": preregistration["evaluation"],
        "contrast_definitions": [
            {"name": name, "formula": formula} for name, formula, _left, _right in CONTRASTS
        ],
        "interaction_definition": {
            "name": INTERACTION[0],
            "formula": INTERACTION[1],
        },
        "direction": "lower_is_better_negative_contrast_favors_first_term",
        "raw_endpoint_rows": endpoint_rows,
        "seed_level_factorial": seed_rows,
        "cross_seed_summaries": cross_seed,
        "sampling_block_variation": {
            "status": "exploratory_variation_only",
            "quality_endpoint": False,
            "selection_criterion": False,
            "independent_n_contribution": 0,
            "diagnostic": (
                "fixed 5k seed-block Inception-feature moment variation; not blockwise FID/KID"
            ),
            "rows": block_summary,
        },
        "reporting_boundaries": {
            "observed_result": (
                "raw FID/KID values, preregistered seed-level contrasts, interaction, "
                "and cross-seed mean/median/range"
            ),
            "preregistered_interpretation": preregistration["interpretation_branches"],
            "exploratory_observation": (
                "fixed sampling-block feature variation only; it neither replaces an endpoint "
                "nor increases the independent n"
            ),
            "unsupported_claims": [
                "optimizer history causes, explains, or mediates a quality change",
                "a contrast is a percentage of a total effect explained",
                "sampling blocks, minibatches, checkpoints, or NFE modes increase n above three",
                "an exploratory checkpoint, metric, NFE, or seed block replaces the frozen primary endpoint",
                "a failed fresh B-versus-A reproduction may be hidden by a secondary result",
            ],
            "automatic_interpretation_branch_selection": False,
            "reason": "approximately has no post-hoc numerical margin; complete seed values must be reviewed descriptively",
        },
        "provenance": {key: value for key, value in provenance.items() if key != "preregistration"},
    }


def format_number(value: Any) -> str:
    return f"{float(value):.9g}"


def render_markdown(result: Mapping[str, Any]) -> str:
    seed_rows = result["seed_level_factorial"]
    cross_seed = result["cross_seed_summaries"]
    lines = [
        "# q256 target geometry × loss weighting: formal results",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        "- Verification Status: ANALYZED",
        "- Version Label: q256_target_weight_factorial_results_v1",
        "",
        "## Observed results",
        "",
        "The independent unit is the training seed (`n=3`: seeds 3, 4, and 5). "
        "Lower FID/KID is better; a negative preregistered contrast favors its first term.",
        "",
        "| Class | Metric | NFE | Seed | A | B | C | D | C−A | B−D | D−A | B−C | Interaction |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in seed_rows:
        lines.append(
            f"| {row['endpoint_class']} | {row['metric']} | {row['nfe']} | {row['training_seed']} | "
            f"{format_number(row['A'])} | {format_number(row['B'])} | {format_number(row['C'])} | "
            f"{format_number(row['D'])} | {format_number(row['target_at_baseline_weight'])} | "
            f"{format_number(row['target_at_g_weight'])} | {format_number(row['weight_at_baseline_target'])} | "
            f"{format_number(row['weight_at_g_target'])} | {format_number(row['target_x_weight'])} |"
        )
    lines.extend(
        [
            "",
            "## Cross-seed descriptive summaries",
            "",
            "Every row uses the same three independent training seeds. No minibatch, metric repeat, "
            "NFE mode, or sampling block is counted as another replicate.",
            "",
            "| Class | Metric | NFE | Quantity | n | Mean | Median | Min | Max | Range |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in cross_seed:
        lines.append(
            f"| {row['endpoint_class']} | {row['metric']} | {row['nfe']} | {row['quantity']} | 3 | "
            f"{format_number(row['mean'])} | {format_number(row['median'])} | "
            f"{format_number(row['minimum'])} | {format_number(row['maximum'])} | "
            f"{format_number(row['range'])} |"
        )
    lines.extend(
        [
            "",
            "## Preregistered interpretation boundary",
            "",
            "The table above is an **observed result** under frozen endpoints and contrasts. "
            "The preregistered interpretation branches remain labels for descriptive review; this "
            "collector does not automatically choose one because no post-hoc numerical meaning of "
            "“approximately” is allowed.",
            "",
            "## Exploratory sampling-block variation",
            "",
            "The retained 50k Inception features are partitioned into the ten fixed contiguous 5k "
            "generation-seed blocks only to describe feature-moment variation. These diagnostics are "
            "not blockwise FID/KID, are not quality endpoints, are never selection criteria, and add "
            "zero independent replicates (`n` remains 3).",
            "",
            "## Unsupported claims",
            "",
            "These results do not establish optimizer-history causation or mediation, do not estimate "
            "a percentage explained, and do not permit an exploratory checkpoint, NFE, metric, or "
            "sampling block to replace the frozen primary FID-50k/NFE=1 endpoint.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(outdir: Path, result: Mapping[str, Any], block_rows: Sequence[Mapping[str, Any]]) -> None:
    outdir = outdir.expanduser().resolve()
    if outdir.exists():
        fail(f"refuse to reuse or overwrite collection output: {outdir}")
    if not outdir.parent.is_dir():
        fail(f"collection output parent does not exist: {outdir.parent}")
    outdir.mkdir(mode=0o750)
    endpoint_rows = result["raw_endpoint_rows"]
    seed_rows = result["seed_level_factorial"]
    cross_seed = result["cross_seed_summaries"]
    block_summary = result["sampling_block_variation"]["rows"]
    write_csv_exclusive(outdir / "endpoint_values.csv", endpoint_rows, list(endpoint_rows[0]))
    write_csv_exclusive(outdir / "seed_level_factorial.csv", seed_rows, list(seed_rows[0]))
    write_csv_exclusive(outdir / "cross_seed_summary.csv", cross_seed, list(cross_seed[0]))
    write_csv_exclusive(outdir / "sampling_block_diagnostics.csv", block_rows, list(block_rows[0]))
    write_csv_exclusive(outdir / "sampling_block_variation.csv", block_summary, list(block_summary[0]))
    write_json_exclusive(outdir / "q256_target_weight_results.json", result)
    markdown_path = outdir / "q256_target_weight_results.md"
    try:
        with markdown_path.open("x", encoding="utf-8") as handle:
            handle.write(render_markdown(result))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        fail(f"refuse to overwrite immutable collection output: {markdown_path}")
    output_bindings = {}
    for path in sorted(outdir.iterdir()):
        if path.name == "collection_receipt.json":
            continue
        if path.is_symlink() or not path.is_file():
            fail(f"unexpected non-file collection artifact: {path}")
        output_bindings[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    receipt = {
        "schema": COLLECTION_SCHEMA,
        "status": "PASS",
        "experiment_id": evaluator.EXPERIMENT_ID,
        "protocol": evaluator.PROTOCOL,
        "independent_training_seed_n": 3,
        "sampling_block_independent_n_contribution": 0,
        "input_provenance": result["provenance"],
        "outputs": output_bindings,
        "outputs_tree_sha256": canonical_sha256(output_bindings),
    }
    write_json_exclusive(outdir / "collection_receipt.json", receipt)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        endpoints, blocks, provenance = validate_and_collect(args.evaluation_root)
        result = build_result(endpoints, blocks, provenance)
        write_outputs(args.outdir, result, blocks)
    except (CollectionError, evaluator.EvaluationError) as exc:
        print(f"[collect-q256-target-weight] ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Validated 12 runs, 24 jobs, and 48 raw endpoints: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
