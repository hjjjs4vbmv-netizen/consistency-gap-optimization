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
PAIRING_KEY = ("training_seed", "budget_kimg", "nfe", "metric_name")


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
        "pairwise_statistics": build_pairwise_statistics(rows, manifest.get("comparison")),
        "statistics": summary_rows,
    }


def build_pairwise_statistics(rows: list[dict], comparison: dict | None) -> dict:
    if comparison is None:
        return {"status": "not_computed", "reason": "no explicit pairing contract"}
    required = ("pairing_key", "baseline_method", "candidate_method", "delta_direction")
    missing = [field for field in required if field not in comparison]
    if missing:
        fail(f"pairing contract is incomplete: {missing}")
    pairing_key = comparison["pairing_key"]
    if not isinstance(pairing_key, list) or not pairing_key or not all(
        isinstance(field, str) and field for field in pairing_key
    ):
        fail("pairing_key must be a non-empty list of row field names")
    pairing_key = tuple("metric_name" if field == "metric" else field for field in pairing_key)
    if pairing_key != PAIRING_KEY:
        fail(
            "fixed/global paired analysis requires pairing_key "
            f"{list(PAIRING_KEY)!r}, got {list(pairing_key)!r}"
        )
    unknown = [field for field in pairing_key if any(field not in row for row in rows)]
    if unknown:
        fail(f"pairing_key fields are absent from result rows: {unknown}")
    baseline = comparison["baseline_method"]
    candidate = comparison["candidate_method"]
    if baseline != "fixed" or candidate != "global110":
        fail("fixed/global paired analysis requires baseline_method='fixed' and candidate_method='global110'")
    candidate_label = comparison.get("candidate_label", "global_only")
    if candidate_label != "global_only":
        fail("fixed/global paired analysis requires candidate_label='global_only'")
    direction = comparison["delta_direction"]
    expected_direction = "global_only - fixed"
    if direction != expected_direction:
        fail(f"delta_direction must be exactly {expected_direction!r}")

    grouped: dict[str, dict[tuple, dict]] = {baseline: {}, candidate: {}}
    for row in rows:
        if row["method"] not in (baseline, candidate):
            continue
        pair = tuple(row[field] for field in pairing_key)
        if pair in grouped[row["method"]]:
            fail(f"duplicate {row['method']} row for pairing key {pair}")
        grouped[row["method"]][pair] = row

    if not grouped[baseline] and not grouped[candidate]:
        fail("pairing contract methods do not appear in result rows")
    baseline_pairs = set(grouped[baseline])
    candidate_pairs = set(grouped[candidate])
    if baseline_pairs != candidate_pairs:
        fail(
            "unpaired fixed/global results: "
            f"fixed_only={sorted(baseline_pairs - candidate_pairs)}, "
            f"global_only={sorted(candidate_pairs - baseline_pairs)}"
        )

    differences = []
    for pair in sorted(baseline_pairs):
        fixed_row = grouped[baseline][pair]
        global_row = grouped[candidate][pair]
        fixed_value = fixed_row["metric_value"]
        global_value = global_row["metric_value"]
        delta = global_value - fixed_value
        winner = "global_only" if delta < 0 else "fixed" if delta > 0 else "tie"
        differences.append({
            "training_seed": fixed_row["training_seed"],
            "budget_kimg": fixed_row["budget_kimg"],
            "nfe": fixed_row["nfe"],
            "metric": fixed_row["metric_name"],
            "fixed_checkpoint_id": fixed_row["checkpoint_id"],
            "global_only_checkpoint_id": global_row["checkpoint_id"],
            "fixed_checkpoint_sha256": fixed_row["checkpoint_sha256"],
            "global_only_checkpoint_sha256": global_row["checkpoint_sha256"],
            "fixed_value": fixed_value,
            "global_only_value": global_value,
            "delta": delta,
            "winner": winner,
        })

    statistics_groups: dict[tuple[str, int, int], list[dict]] = {}
    for difference in differences:
        key = (difference["metric"], difference["budget_kimg"], difference["nfe"])
        statistics_groups.setdefault(key, []).append(difference)
    statistics_rows = []
    for (metric_name, budget_kimg, nfe), group in sorted(statistics_groups.items()):
        deltas = [item["delta"] for item in group]
        statistics_rows.append({
            "metric_name": metric_name,
            "budget_kimg": budget_kimg,
            "nfe": nfe,
            "pair_count": len(group),
            "mean_delta": statistics.mean(deltas),
            "sample_sd_delta": statistics.stdev(deltas) if len(deltas) > 1 else None,
            "minimum_delta": min(deltas),
            "maximum_delta": max(deltas),
            "global_wins": sum(item["winner"] == "global_only" for item in group),
            "fixed_wins": sum(item["winner"] == "fixed" for item in group),
            "ties": sum(item["winner"] == "tie" for item in group),
        })
    return {
        "status": "computed",
        "schema_version": 1,
        "pairing_key": ["training_seed", "budget_kimg", "nfe", "metric"],
        "baseline_method": baseline,
        "candidate_method": candidate,
        "candidate_label": candidate_label,
        "delta_direction": direction,
        "paired_differences": differences,
        "statistics": statistics_rows,
    }


def write_paired_outputs(outdir: Path, pairwise: dict) -> None:
    """Write per-seed fixed/global evidence as standalone reviewable files."""
    if pairwise["status"] != "computed":
        return
    differences = pairwise["paired_differences"]
    with (outdir / "paired_differences.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(differences[0]))
        writer.writeheader()
        writer.writerows(differences)
    paired_statistics = {
        key: value for key, value in pairwise.items() if key != "paired_differences"
    }
    (outdir / "paired_statistics.json").write_text(
        json.dumps(paired_statistics, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Fixed vs global-only paired statistics",
        "",
        "Pairing key: `training_seed + budget_kimg + nfe + metric`.",
        "Delta: `global_only - fixed`; negative values favor global-only.",
        "",
        "| Metric | Budget (kimg) | NFE | Pairs | Mean delta | Sample SD | Global wins | Fixed wins | Ties |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pairwise["statistics"]:
        sample_sd = "—" if row["sample_sd_delta"] is None else f"{row['sample_sd_delta']:.9f}"
        lines.append(
            f"| {row['metric_name']} | {row['budget_kimg']} | {row['nfe']} | "
            f"{row['pair_count']} | {row['mean_delta']:.9f} | {sample_sd} | "
            f"{row['global_wins']} | {row['fixed_wins']} | {row['ties']} |"
        )
    lines.append("")
    (outdir / "paired_statistics.md").write_text("\n".join(lines), encoding="utf-8")


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
    pairwise = summary["pairwise_statistics"]
    lines.extend(["", "Statistics are segregated by evidence class, metric, NFE, and method."])
    if pairwise["status"] == "computed":
        lines.extend([
            "",
            "## Paired deltas",
            "",
            "Pairing key: `" + ", ".join(pairwise["pairing_key"]) + "`; "
            "delta: `" + pairwise["delta_direction"] + "`.",
            "",
            "| Metric | Budget | NFE | Pairs | Mean delta | Sample SD | Global wins | Fixed wins | Ties |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in pairwise["statistics"]:
            sample_sd = "—" if row["sample_sd_delta"] is None else f"{row['sample_sd_delta']:.9f}"
            lines.append(
                f"| {row['metric_name']} | {row['budget_kimg']} | {row['nfe']} | {row['pair_count']} | "
                f"{row['mean_delta']:.9f} | {sample_sd} | {row['global_wins']} | "
                f"{row['fixed_wins']} | {row['ties']} |"
            )
    else:
        lines.extend(["", "No pairwise delta is emitted: " + pairwise["reason"] + "."])
    lines.append("")
    (outdir / "evaluation_statistics.md").write_text("\n".join(lines), encoding="utf-8")
    write_paired_outputs(outdir, pairwise)


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
