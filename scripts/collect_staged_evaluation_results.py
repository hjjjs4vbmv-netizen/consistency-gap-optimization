#!/usr/bin/env python3
"""Validate a staged evaluation run and emit a unified result table/statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

PROTOCOL_ID = "staged-checkpoint-evaluation-v1"


def fail(message: str) -> None:
    raise SystemExit(f"[collect_staged_evaluation_results] ERROR: {message}")


def load_json(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {label} {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"{label} must contain a JSON object: {path}")
    return payload


def read_metric(path: Path, metric_name: str) -> float:
    if not path.is_file():
        fail(f"missing metric result: {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        fail(f"expected exactly one result in {path}, found {len(lines)}")
    try:
        payload = json.loads(lines[0])
        if payload["metric"] != metric_name:
            fail(f"metric name mismatch in {path}: {payload.get('metric')} != {metric_name}")
        value = float(payload["results"][metric_name])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        fail(f"malformed metric result {path}: {exc}")
    if not math.isfinite(value):
        fail(f"non-finite metric result in {path}: {value}")
    return value


def validate_manifest(manifest: dict) -> None:
    if manifest.get("protocol") != PROTOCOL_ID:
        fail(f"run manifest protocol must be {PROTOCOL_ID!r}")
    if manifest.get("status") != "completed":
        fail(f"run is not complete: {manifest.get('status')}")
    if manifest.get("precision") != "fp32":
        fail("staged evaluation requires FP32")
    if manifest.get("metric_repeats") != 1:
        fail("staged evaluation requires exactly one metric result per cell")
    if manifest.get("phase") not in ("smoke", "quick", "formal"):
        fail(f"unknown staged evaluation phase: {manifest.get('phase')}")
    if not isinstance(manifest.get("jobs"), list) or not manifest["jobs"]:
        fail("run manifest contains no jobs")


def collect(eval_root: Path) -> tuple[list[dict], dict]:
    manifest = load_json(eval_root / "run_manifest.json", "run manifest")
    validate_manifest(manifest)
    rows = []
    expected_metrics = list(manifest["metric_names"])
    for job in manifest["jobs"]:
        if job.get("status") != "completed":
            fail(f"job is not complete: {job.get('checkpoint_id')} nfe={job.get('nfe')}")
        if job.get("metric_names") != expected_metrics:
            fail("jobs use inconsistent metric sets")
        if job.get("sample_count") != manifest["sample_count"]:
            fail("job and manifest sample counts differ")
        if job.get("sample_seeds") != manifest["sample_seeds"]:
            fail("job and manifest generation seed ranges differ")
        if job.get("metric_seed") != manifest["metric_seed"]:
            fail("job and manifest metric seeds differ")
        if job.get("evidence_class") != manifest["evidence_class"]:
            fail("job and manifest evidence classes differ")
        receipt = job.get("integrity_receipt", {})
        if manifest["phase"] == "formal" and receipt.get("status") != "passed":
            fail(f"formal job lacks a passed integrity receipt: {job.get('checkpoint_id')}")
        for metric_name in expected_metrics:
            value = read_metric(
                Path(job["output_directory"]) / f"metric-{metric_name}.jsonl",
                metric_name,
            )
            rows.append({
                "evidence_class": manifest["evidence_class"],
                "phase": manifest["phase"],
                "checkpoint_id": job["checkpoint_id"],
                "method": job["method"],
                "training_seed": int(job["training_seed"]),
                "budget_kimg": int(job["budget_kimg"]),
                "checkpoint_sha256": job["checkpoint_sha256"],
                "integrity_receipt_status": receipt.get("status"),
                "nfe": int(job["nfe"]),
                "mid_t": json.dumps(job["mid_t"]),
                "metric_name": metric_name,
                "metric_value": value,
                "generated_sample_count": int(job["sample_count"]),
                "generation_seed_range": job["sample_seeds"],
                "metric_seed": int(job["metric_seed"]),
                "dataset_sha256": manifest["dataset_sha256"],
                "evaluation_git_commit": manifest["evaluation_git_commit"],
                "run_path": job["output_directory"],
                "completion_status": job["status"],
            })
    return rows, build_statistics(rows, manifest)


def build_statistics(rows: list[dict], manifest: dict) -> dict:
    grouped: dict[tuple[str, str, int, str], list[float]] = {}
    for row in rows:
        key = (row["evidence_class"], row["metric_name"], row["nfe"], row["method"])
        grouped.setdefault(key, []).append(row["metric_value"])
    summary_rows = []
    for key in sorted(grouped):
        evidence_class, metric_name, nfe, method = key
        values = grouped[key]
        summary_rows.append({
            "evidence_class": evidence_class,
            "metric_name": metric_name,
            "nfe": nfe,
            "method": method,
            "count": len(values),
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "minimum": min(values),
            "maximum": max(values),
        })
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_ID,
        "phase": manifest["phase"],
        "evidence_class": manifest["evidence_class"],
        "row_count": len(rows),
        "statistics_grouping": ["evidence_class", "metric_name", "nfe", "method"],
        "pairwise_statistics": "not computed; provide an explicit pairing contract before calculating deltas",
        "statistics": summary_rows,
    }


def write_outputs(outdir: Path, rows: list[dict], summary: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "evaluation_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (outdir / "evaluation_statistics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Staged evaluation statistics",
        "",
        f"Evidence class: `{summary['evidence_class']}`; phase: `{summary['phase']}`.",
        "",
        "| Evidence | Metric | NFE | Method | Count | Mean | Sample SD | Min | Max |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["statistics"]:
        sample_sd = "—" if row["sample_sd"] is None else f"{row['sample_sd']:.9f}"
        lines.append(
            f"| {row['evidence_class']} | {row['metric_name']} | {row['nfe']} | "
            f"{row['method']} | {row['count']} | {row['mean']:.9f} | {sample_sd} | "
            f"{row['minimum']:.9f} | {row['maximum']:.9f} |"
        )
    lines.extend([
        "",
        "Statistics are segregated by evidence class, metric, NFE, and method. "
        "No pairwise delta is emitted without an explicit pairing contract.",
        "",
    ])
    (outdir / "evaluation_statistics.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)
    rows, summary = collect(args.eval_root.resolve())
    write_outputs(args.outdir.resolve(), rows, summary)
    print(f"Validated {len(rows)} metric rows; output: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
