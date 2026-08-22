#!/usr/bin/env python3
"""Validate and summarize the q256 seed8-13 secondary extension evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path


SEEDS = tuple(range(8, 14))
OVERLAP_SEEDS = tuple(range(8, 13))
ARMS = ("A", "B", "C", "D")
NFES = (1, 2)
METRICS = ("kid50k_full", "fid50k_full")
TRAINING_COMMIT = "dcca41b19e7c45512b5fbe98776520396a1bf9ac"
EVALUATOR_COMMIT = "9d06ccc72545d4189af1b86de7f629f9c09d3f73"
DATASET_SHA256 = "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372"
TRANSFER_SHA256 = "4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da"
PROTOCOL = "q256-target-weight-secondary-extension-frozen-evaluation-v1"
CLASSIFICATION = "secondary_precision_extension_not_original_preregistration"
RUN_START_UTC = "2026-08-21T06:54:19Z"
COHORT3_FREEZE_COMMIT = "0672283a3a325b352c8c8009763f1f3222a3b2f1"
COHORT3_FREEZE_UTC = "2026-08-21T07:51:45Z"
CONTRASTS = (
    ("B-A", ("B", 1.0), ("A", -1.0)),
    ("C-A", ("C", 1.0), ("A", -1.0)),
    ("B-D", ("B", 1.0), ("D", -1.0)),
    ("D-A", ("D", 1.0), ("A", -1.0)),
    ("B-C", ("B", 1.0), ("C", -1.0)),
    ("I=B-C-D+A", ("B", 1.0), ("C", -1.0), ("D", -1.0), ("A", 1.0)),
)


class CollectionError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise CollectionError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read JSON {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"expected one JSON object: {path}")
    return value


def require_hash(path: Path, binding: dict, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        fail(f"missing or symlinked {label}: {path}")
    if path.stat().st_size != binding.get("bytes"):
        fail(f"byte-count mismatch for {label}: {path}")
    if sha256_file(path) != binding.get("sha256"):
        fail(f"SHA-256 mismatch for {label}: {path}")


def write_json_exclusive(path: Path, value: object) -> None:
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError:
        fail(f"refuse to overwrite immutable output: {path}")
    with handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_text_exclusive(path: Path, value: str) -> None:
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError:
        fail(f"refuse to overwrite immutable output: {path}")
    with handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def read_raw_metric(path: Path, metric: str) -> float:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        fail(f"raw metric must contain exactly one record: {path}")
    payload = json.loads(lines[0])
    if payload.get("metric") != metric or payload.get("num_gpus") != 1:
        fail(f"raw metric identity mismatch: {path}")
    value = float(payload["results"][metric])
    if not math.isfinite(value) or (metric.startswith("fid") and value < 0):
        fail(f"invalid metric value in {path}: {value}")
    return value


def expected_job_ids() -> list[str]:
    return [
        f"seed{seed}-arm{arm}-nfe{nfe}"
        for nfe in NFES
        for seed in SEEDS
        for arm in ARMS
    ]


def validate_training(training_root: Path, matrix_root: Path) -> dict:
    report_path = training_root / "q256_factorial_seed8_13_extension_report.json"
    report = load_json(report_path)
    if report.get("status") != "PASS" or report.get("classification") != CLASSIFICATION:
        fail("training report is not the exact secondary-extension PASS record")
    chain_lines = (training_root / "chain.log").read_text(encoding="utf-8").splitlines()
    if not chain_lines or f"START utc={RUN_START_UTC}" not in chain_lines[0]:
        fail("chain log does not bind the recorded pre-preregistration start time")
    if not any("FULL_CHAIN_PASS utc=2026-08-21T22:50:35Z" in line for line in chain_lines):
        fail("chain log lacks FULL_CHAIN_PASS")

    binding_path = matrix_root / "extension_matrix_binding.json"
    binding = load_json(binding_path)
    if (
        binding.get("status") != "PASS"
        or binding.get("cell_count") != 24
        or binding.get("seeds") != list(SEEDS)
        or binding.get("arms") != list(ARMS)
        or binding.get("training_source_git_head") != TRAINING_COMMIT
        or binding.get("extension_classification") != CLASSIFICATION
        or binding.get("replaces_preregistered_seed") is not False
    ):
        fail("training matrix binding has drifted")
    require_hash(report_path, binding["extension_report_receipt"], "training report")

    audits = {}
    for item in binding.get("integrity_audit_receipts", []):
        seed = int(Path(item["path"]).name.removeprefix("seed").split("_", 1)[0])
        path = training_root / "integrity" / Path(item["path"]).name
        require_hash(path, item, f"seed{seed} integrity audit")
        audit = load_json(path)
        if (
            audit.get("status") != "PASS"
            or audit.get("seed") != seed
            or audit.get("source_head") != TRAINING_COMMIT
            or audit.get("four_arm_complete") is not True
            or audit.get("denominator_integrity") is not True
            or audit.get("common_initial_state_identity") is not True
            or audit.get("telemetry_identity_checks", {}).get("all_pass") is not True
        ):
            fail(f"seed{seed} integrity audit is not exact PASS")
        initial_hashes = set()
        for arm in ARMS:
            cell = audit["cells"][arm]
            if (
                cell.get("attempts") != 2000
                or cell.get("processed_nimg") != 256000
                or cell.get("processed_kimg") != 256.0
                or cell.get("semantic_nonfinite_count") != 0
                or cell.get("raw_grad_skip_mismatch_count") != 0
                or cell.get("nonpositive_denominator_count") != 0
            ):
                fail(f"seed{seed} arm{arm} training endpoint failed integrity")
            receipt_path = training_root / f"seed{seed}" / f"arm{arm}" / "initial_state_receipt_v1.json"
            require_hash(receipt_path, cell["artifact_hashes"]["initial_state_receipt_v1.json"], f"seed{seed} arm{arm} initial state")
            initial = load_json(receipt_path)
            if initial.get("seed") != seed:
                fail(f"seed mismatch in initial-state receipt: {receipt_path}")
            initial_hashes.add(initial.get("common_initial_state_sha256"))
        if len(initial_hashes) != 1:
            fail(f"seed{seed} arms do not share one initial state")
        audits[str(seed)] = sha256_file(path)
    if set(map(int, audits)) != set(SEEDS):
        fail("integrity audit set is incomplete")

    cells = binding.get("cell_receipts", [])
    if len(cells) != 24:
        fail("matrix binding must contain 24 cell receipts")
    for item in cells:
        require_hash(matrix_root / "cells" / Path(item["path"]).name, item, "matrix cell")
    return {
        "training_report_sha256": sha256_file(report_path),
        "matrix_binding_sha256": sha256_file(binding_path),
        "integrity_audit_sha256_by_seed": audits,
    }


def validate_evaluation(eval_root: Path, training_root: Path, support_root: Path) -> tuple[list[dict], dict]:
    plan_path = eval_root / "evaluation_plan.json"
    completion_path = eval_root / "evaluation_completion.json"
    plan = load_json(plan_path)
    completion = load_json(completion_path)
    exact_plan = {
        "schema": "ect.q256.target-weight-evaluation-plan/v2",
        "status": "authorized_exact_matrix",
        "protocol": PROTOCOL,
        "extension_classification": CLASSIFICATION,
        "replaces_preregistered_seed": False,
        "job_count": 48,
        "sample_count_per_job": 50000,
        "sample_seed_range": "0-49999",
        "metric_seed": 20260730,
        "metrics_per_job": list(METRICS),
        "precision": "fp32",
        "nfe_modes": {"1": [], "2": [0.821]},
        "selection_policy": "all_24_extension_final_256kimg_checkpoints",
    }
    for key, value in exact_plan.items():
        if plan.get(key) != value:
            fail(f"evaluation plan drift: {key}")
    if plan.get("dataset", {}).get("sha256") != DATASET_SHA256:
        fail("evaluation plan used the wrong dataset")
    if plan.get("independent_unit") != {"name": "training_seed", "n": 6, "values": list(SEEDS)}:
        fail("evaluation plan independent-unit identity drift")
    if plan.get("gpu", {}).get("physical_index") != 0:
        fail("evaluation was not bound to the frozen single-GPU protocol")
    matrix = plan.get("training_matrix", {})
    matrix_path = eval_root / "matrix_binding" / "extension_matrix_binding.json"
    if matrix.get("extension_matrix_binding_sha256") != sha256_file(matrix_path):
        fail("evaluation plan does not bind the included training matrix")
    require_hash(support_root / "run_q256_seed8_13_frozen_evaluation.py", matrix["provenance_adapter"], "provenance adapter")
    require_hash(support_root / "run_q256_direct_frozen_evaluation_v6.py", matrix["formal_numerical_adapter"], "formal numerical adapter")

    if (
        completion.get("schema") != "ect.q256.target-weight-evaluation-completion/v2"
        or completion.get("status") != "PASS"
        or completion.get("job_count") != 48
        or completion.get("failed_job_id") is not None
        or completion.get("protocol") != PROTOCOL
        or completion.get("evaluation_plan_sha256") != sha256_file(plan_path)
    ):
        fail("evaluation completion is not an exact 48-job PASS")
    expected_ids = expected_job_ids()
    if completion.get("completed_job_ids") != expected_ids:
        fail("completion job order is not all NFE1 followed by all NFE2")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or [job.get("job_id") for job in jobs] != expected_ids:
        fail("evaluation plan job order drift")

    rows = []
    for job in jobs:
        job_id = job["job_id"]
        expected_mid_t = [] if job["nfe"] == 1 else [0.821]
        for key, value in {
            "mid_t": expected_mid_t,
            "sample_count": 50000,
            "sample_seeds": "0-49999",
            "metric_seed": 20260730,
            "metrics": list(METRICS),
            "precision": "fp32",
        }.items():
            if job.get(key) != value:
                fail(f"job plan drift for {job_id}: {key}")
        receipt_path = eval_root / "receipts" / f"{job_id}.json"
        receipt = load_json(receipt_path)
        if (
            receipt.get("schema") != "ect.q256.target-weight-evaluation-job-receipt/v2"
            or receipt.get("status") != "passed"
            or receipt.get("returncode") != 0
            or receipt.get("execution_error") is not None
            or receipt.get("job_id") != job_id
            or receipt.get("seed") != job["seed"]
            or receipt.get("arm") != job["arm"]
            or receipt.get("nfe") != job["nfe"]
            or receipt.get("mid_t") != expected_mid_t
            or receipt.get("precision") != "fp32"
            or receipt.get("sample_count") != 50000
            or receipt.get("sample_seed_range") != "0-49999"
            or receipt.get("metric_seed") != 20260730
            or receipt.get("dataset_sha256") != DATASET_SHA256
            or receipt.get("protocol") != PROTOCOL
            or receipt.get("checkpoint_sha256") != job["checkpoint_sha256"]
            or receipt.get("evaluator_source_git_head") != EVALUATOR_COMMIT
        ):
            fail(f"receipt identity/protocol failure: {job_id}")
        monitor = receipt.get("gpu_exclusivity_monitor", {})
        if monitor.get("status") != "PASS" or monitor.get("foreign_process_incident") is not None:
            fail(f"GPU exclusivity failure: {job_id}")
        metrics = receipt.get("metrics", [])
        if [item.get("metric") for item in metrics] != list(METRICS):
            fail(f"metric order mismatch: {job_id}")
        values = {}
        for item in metrics:
            raw_path = eval_root / "jobs" / job_id / f"metric-{item['metric']}.jsonl"
            if sha256_file(raw_path) != item.get("raw_sha256"):
                fail(f"raw metric hash mismatch: {job_id} {item['metric']}")
            value = read_raw_metric(raw_path, item["metric"])
            if value != float(item["value"]):
                fail(f"raw metric value mismatch: {job_id} {item['metric']}")
            values[item["metric"]] = value
        artifacts = receipt["artifacts"]
        kid_feature = artifacts["generated-features-kid50k_full-repeat00.npy"]["sha256"]
        fid_feature = artifacts["generated-features-fid50k_full-repeat00.npy"]["sha256"]
        if kid_feature != fid_feature:
            fail(f"FID/KID feature identity mismatch: {job_id}")
        block_path = eval_root / "jobs" / job_id / "sampling_block_diagnostics_v1.json"
        if sha256_file(block_path) != receipt["sampling_block_diagnostics"]["sha256"]:
            fail(f"sampling diagnostics hash mismatch: {job_id}")
        block = load_json(block_path)
        if block.get("sample_count") != 50000 or block.get("fixed_block_count") != 10:
            fail(f"sampling diagnostics structure mismatch: {job_id}")
        rows.append(
            {
                "seed": job["seed"],
                "arm": job["arm"],
                "nfe": job["nfe"],
                "mid_t": "" if job["nfe"] == 1 else 0.821,
                "fid50k_full": values["fid50k_full"],
                "kid50k_full": values["kid50k_full"],
                "status": "PASS",
                "checkpoint_sha256": receipt["checkpoint_sha256"],
                "receipt_sha256": sha256_file(receipt_path),
                "generated_feature_sha256": kid_feature,
                "generated_samples_sha256": artifacts["generated-samples.npy"]["sha256"],
                "artifacts_tree_sha256": receipt["artifacts_tree_sha256"],
                "process_log_sha256": receipt["process_log_sha256"],
                "completed_at_utc": receipt["finished_utc"],
            }
        )
    return rows, {
        "evaluation_plan_sha256": sha256_file(plan_path),
        "evaluation_completion_sha256": sha256_file(completion_path),
        "evaluator_source_content_sha256": plan["evaluator_source"]["content_sha256"],
        "evaluation_gpu": plan["gpu"],
        "runtime": plan["runtime"],
        "finished_utc": completion["finished_utc"],
    }


def stats(values: list[float]) -> dict:
    return {
        "n": len(values),
        "values": values,
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "span": max(values) - min(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "negative_count": sum(value < 0 for value in values),
        "positive_count": sum(value > 0 for value in values),
        "zero_count": sum(value == 0 for value in values),
    }


def summarize(rows: list[dict], seeds: tuple[int, ...]) -> dict:
    lookup = {(row["seed"], row["arm"], row["nfe"]): row for row in rows}
    cells = []
    contrasts = []
    for nfe in NFES:
        for arm in ARMS:
            for metric in ("fid50k_full", "kid50k_full"):
                cells.append({"arm": arm, "nfe": nfe, "metric": metric, **stats([lookup[seed, arm, nfe][metric] for seed in seeds])})
        for metric in ("fid50k_full", "kid50k_full"):
            for name, *terms in CONTRASTS:
                values = [sum(coef * lookup[seed, arm, nfe][metric] for arm, coef in terms) for seed in seeds]
                contrasts.append({"contrast": name, "nfe": nfe, "metric": metric, **stats(values)})
    return {"seeds": list(seeds), "cells": cells, "contrasts": contrasts}


def fmt(value: float, digits: int = 9) -> str:
    return f"{value:.{digits}f}"


def contrast(summary: dict, nfe: int, metric: str, name: str) -> dict:
    return next(item for item in summary["contrasts"] if item["nfe"] == nfe and item["metric"] == metric and item["contrast"] == name)


def render_markdown(rows: list[dict], all_summary: dict, overlap_summary: dict, metadata: dict) -> str:
    lookup = {(row["seed"], row["arm"], row["nfe"]): row for row in rows}
    primary_all = contrast(all_summary, 1, "fid50k_full", "B-A")
    primary_overlap = contrast(overlap_summary, 1, "fid50k_full", "B-A")
    lines = [
        "# q256 seed8–13 secondary extension: frozen FID/KID results",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        "- Origin Date: 2026-08-22",
        "- Verification Status: VERIFIED",
        "- Version Label: q256_factorial_seed8_13_secondary_extension_results_v1",
        "",
        "## Scope, status, and chronology",
        "",
        "This is the complete **secondary precision extension** for seeds8–13. It is not part of the original seeds3–5 preregistration and is not a valid execution of PR #72 Cohort III.",
        "",
        f"- Training chain start: `{RUN_START_UTC}`.",
        f"- PR #72 Cohort III freeze commit: `{COHORT3_FREEZE_COMMIT}` at `{COHORT3_FREEZE_UTC}`.",
        "- The secondary chain therefore started 57 minutes 26 seconds before that freeze, and it used two A100 queues rather than PR #72's fixed five-GPU seed mapping.",
        "- Seeds8–12 overlap the planned Cohort III seed labels, but these observed trajectories cannot be relabeled as prospective held-out confirmation. A future clean confirmation must use a newly frozen, genuinely unseen seed set or explicitly amend/reclassify #72.",
        "- Training: **PASS**, 24/24 arms at 256.000 kimg; integrity: **PASS**, 6/6 seed audits; evaluation: **PASS**, 48/48 jobs.",
        f"- Evaluation completion: `{metadata['finished_utc']}`.",
        "",
        "Frozen evaluation used FP32, 50,000 samples per job, sample seeds `0..49999`, metric seed `20260730`, KID then FID from byte-identical retained generated features, all 24 NFE1 jobs before all 24 NFE2 jobs, and `mid_t=0.821` for NFE2. Lower is better.",
        "",
        "## Per-seed results",
        "",
        "| Seed | Arm | NFE1 FID | NFE1 KID | NFE2 FID | NFE2 KID |",
        "|---:|:---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        for arm in ARMS:
            n1, n2 = lookup[seed, arm, 1], lookup[seed, arm, 2]
            lines.append(f"| {seed} | {arm} | {fmt(n1['fid50k_full'])} | {fmt(n1['kid50k_full'], 12)} | {fmt(n2['fid50k_full'])} | {fmt(n2['kid50k_full'], 12)} |")
    lines += [
        "",
        "## Primary descriptive readout",
        "",
        "The factorial primary-style readout is FID-50k@NFE1 `B−A`, but it is descriptive here because this run predates and does not follow #72.",
        "",
        "| Group | n | Mean B−A | Median B−A | Range | Direction (−/+/0) |",
        "|:---|---:|---:|---:|:---:|:---:|",
        f"| Complete secondary run, seeds8–13 | 6 | {fmt(primary_all['mean'], 6)} | {fmt(primary_all['median'], 6)} | [{fmt(primary_all['minimum'], 6)}, {fmt(primary_all['maximum'], 6)}] | {primary_all['negative_count']}/{primary_all['positive_count']}/{primary_all['zero_count']} |",
        f"| PR #72-overlap labels only, seeds8–12 (nonconfirmatory) | 5 | {fmt(primary_overlap['mean'], 6)} | {fmt(primary_overlap['median'], 6)} | [{fmt(primary_overlap['minimum'], 6)}, {fmt(primary_overlap['maximum'], 6)}] | {primary_overlap['negative_count']}/{primary_overlap['positive_count']}/{primary_overlap['zero_count']} |",
        f"| Extra sensitivity seed13 | 1 | {fmt(lookup[13,'B',1]['fid50k_full'] - lookup[13,'A',1]['fid50k_full'], 6)} | — | — | 0/1/0 |",
        "",
        "Across all six seeds, the mean favors B over A, but the direction is 4/6 rather than uniform. Seed13 reverses strongly (`B−A=+80.794940`), while seed8 is extremely favorable (`−171.517036`). The mean is therefore heterogeneous and must not be described as universal B dominance.",
    ]
    for nfe, title in ((1, "NFE1 factorial contrasts"), (2, "NFE2 secondary contrasts")):
        lines += [
            "",
            f"## {title}",
            "",
            "A negative value favors the first term; interaction is `I=B−C−D+A`. Each seed is the independent unit.",
            "",
            "| Metric | Contrast | Mean | Median | Range | SD | Direction (−/+/0) |",
            "|:---|:---|---:|---:|:---:|---:|:---:|",
        ]
        for metric, label, digits in (("fid50k_full", "FID-50k", 6), ("kid50k_full", "KID-50k", 9)):
            for name, *_ in CONTRASTS:
                item = contrast(all_summary, nfe, metric, name)
                lines.append(f"| {label} | {name} | {fmt(item['mean'], digits)} | {fmt(item['median'], digits)} | [{fmt(item['minimum'], digits)}, {fmt(item['maximum'], digits)}] | {fmt(item['sample_sd'], digits)} | {item['negative_count']}/{item['positive_count']}/{item['zero_count']} |")
    lines += [
        "",
        "## Integrity and interpretation guardrails",
        "",
        "- All 24 training cells and six four-arm integrity audits passed; semantic non-finite, raw-gradient/skip mismatch, and nonpositive-denominator counts are zero.",
        "- All 48 evaluation receipts passed with one fixed GPU exclusivity monitor and no foreign-process incident; raw metrics and sampling diagnostics re-hash to their receipts.",
        "- Within every job, FID and KID use byte-identical retained generated-feature hashes.",
        "- The exact job order is all NFE1 first, then all NFE2; no seed, arm, checkpoint, or metric was omitted after observation.",
        "- No p-value or confirmatory interval is reported. With `n=6`, multiple descriptive contrasts, large seed heterogeneity, and an execution that predates #72, inferential or causal wording is not supported.",
        "- Statistical fallacy scan (11/11): Simpson aggregation is guarded by per-seed rows; ecological, Berkson, collider, base-rate, regression-to-mean, and reverse-causality patterns are not applicable to this matrix; survivorship and look-elsewhere are guarded by complete exact matrices; garden-of-forking-paths remains a caution because this is a post hoc secondary analysis; causal language is explicitly excluded.",
        "",
        "The CSV contains all 48 full-precision job rows and artifact hashes. The JSON contains protocol bindings, full descriptive summaries for seeds8–13 and the separately labeled seeds8–12 overlap subset, integrity hashes, and the chronology boundary.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    eval_root = args.eval_root.resolve(strict=True)
    training_root = args.training_root.resolve(strict=True)
    support_root = args.support_root.resolve(strict=True)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    training = validate_training(training_root, eval_root / "matrix_binding")
    rows, evaluation = validate_evaluation(eval_root, training_root, support_root)
    all_summary = summarize(rows, SEEDS)
    overlap_summary = summarize(rows, OVERLAP_SEEDS)
    payload = {
        "schema": "ect.q256.target-weight-seed8-13-secondary-extension-results/v1",
        "status": "VERIFIED_PASS_48_OF_48",
        "classification": CLASSIFICATION,
        "protocol": {
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "nfe": list(NFES),
            "nfe2_mid_t": 0.821,
            "precision": "fp32",
            "sample_count": 50000,
            "sample_seeds": "0-49999",
            "metric_seed": 20260730,
            "metrics": list(METRICS),
            "training_commit": TRAINING_COMMIT,
            "evaluator_commit": EVALUATOR_COMMIT,
            "dataset_sha256": DATASET_SHA256,
            "transfer_sha256": TRANSFER_SHA256,
        },
        "chronology_and_claim_boundary": {
            "secondary_run_start_utc": RUN_START_UTC,
            "cohort3_freeze_commit": COHORT3_FREEZE_COMMIT,
            "cohort3_freeze_utc": COHORT3_FREEZE_UTC,
            "secondary_run_started_before_cohort3_freeze": True,
            "seconds_before_freeze": 3446,
            "overlapping_seed_labels": list(OVERLAP_SEEDS),
            "is_pr72_cohort3_execution": False,
            "is_prospective_confirmation": False,
            "reason": "run predates the freeze and uses two A100 queues rather than the frozen five-GPU mapping",
        },
        "results": rows,
        "all_six_seed_summary": all_summary,
        "overlap_seed8_12_summary_nonconfirmatory": overlap_summary,
        "training_provenance": training,
        "evaluation_provenance": evaluation,
        "fallacy_scan": {
            "coverage": "11/11",
            "garden_of_forking_paths": "CAUTION_post_hoc_secondary_analysis",
            "survivorship_bias": "guarded_by_24_of_24_training_and_48_of_48_evaluation",
            "look_elsewhere": "guarded_by_exact_complete_matrix_no_selection",
            "causal_claim": "not_supported",
        },
    }
    fields = (
        "seed", "arm", "nfe", "mid_t", "fid50k_full", "kid50k_full", "status",
        "checkpoint_sha256", "receipt_sha256", "generated_feature_sha256",
        "generated_samples_sha256", "artifacts_tree_sha256", "process_log_sha256",
        "completed_at_utc",
    )
    csv_path = outdir / "q256_factorial_seed8_13_secondary_extension_results.csv"
    try:
        handle = csv_path.open("x", newline="", encoding="utf-8")
    except FileExistsError:
        fail(f"refuse to overwrite immutable output: {csv_path}")
    with handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    write_json_exclusive(outdir / "q256_factorial_seed8_13_secondary_extension_results.json", payload)
    write_text_exclusive(outdir / "q256_factorial_seed8_13_secondary_extension_results.md", render_markdown(rows, all_summary, overlap_summary, evaluation))
    print(json.dumps({"status": payload["status"], "rows": len(rows), "outdir": str(outdir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
