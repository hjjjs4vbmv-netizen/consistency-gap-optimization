#!/usr/bin/env python3
"""Validate a staged evaluation run and emit a unified result table/statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

PROTOCOL_ID = "staged-checkpoint-evaluation-v1"
PAIRING_KEY = ("training_seed", "budget_kimg", "nfe", "metric_name")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_731


def fail(message: str) -> None:
    raise SystemExit(f"[collect_staged_evaluation_results] ERROR: {message}")


def percentile(values: list[float], probability: float) -> float:
    """Return a linearly interpolated percentile of a non-empty sample."""
    if not values:
        fail("cannot calculate a percentile of an empty sample")
    if not 0 <= probability <= 1:
        fail(f"percentile probability must lie in [0, 1], got {probability}")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(values: list[float], seed: int) -> dict | None:
    """Percentile bootstrap CI for a seed-level mean; descriptive only."""
    if not values:
        return None
    generator = random.Random(seed)
    sample_size = len(values)
    replicates = [
        statistics.mean(generator.choice(values) for _ in range(sample_size))
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    return {
        "method": "nonparametric_seed_bootstrap_percentile",
        "replicates": BOOTSTRAP_REPLICATES,
        "confidence_level": 0.95,
        "lower": percentile(replicates, 0.025),
        "upper": percentile(replicates, 0.975),
        "interpretation": "descriptive resampling interval; resampled seeds are not additional independent observations",
    }


def exact_two_sided_sign_test(
    positive_signs: int, negative_signs: int, ties: int = 0,
    positive_label: str = "global_only_better", negative_label: str = "fixed_better",
) -> dict | None:
    """Exact two-sided binomial sign test, excluding tied paired effects."""
    non_ties = positive_signs + negative_signs
    if non_ties == 0:
        return None
    less_extreme = min(positive_signs, negative_signs)
    tail = sum(math.comb(non_ties, value) for value in range(less_extreme + 1))
    p_value = min(1.0, 2 * tail / (2 ** non_ties))
    return {
        "test": "exact_two_sided_sign_test",
        "non_tied_pairs": non_ties,
        "positive_signs": positive_signs,
        "negative_signs": negative_signs,
        "positive_label": positive_label,
        "negative_label": negative_label,
        "ties_excluded": ties,
        "p_value": p_value,
        "interpretation": "low-resolution descriptive check with one sign per independent training seed; not a basis for a strong significance claim",
    }


def average_ranks(values: list[float]) -> list[float]:
    """Ranks with average tie handling; lower metric values receive lower ranks."""
    ranks = [0.0] * len(values)
    ordered = sorted((value, index) for index, value in enumerate(values))
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][0] == ordered[cursor][0]:
            end += 1
        rank = (cursor + 1 + end) / 2
        for _, original_index in ordered[cursor:end]:
            ranks[original_index] = rank
        cursor = end
    return ranks


def spearman_rank_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    left_center = statistics.mean(left_ranks)
    right_center = statistics.mean(right_ranks)
    numerator = sum((a - left_center) * (b - right_center) for a, b in zip(left_ranks, right_ranks))
    left_sum_squares = sum((a - left_center) ** 2 for a in left_ranks)
    right_sum_squares = sum((b - right_center) ** 2 for b in right_ranks)
    if left_sum_squares == 0 or right_sum_squares == 0:
        return None
    return numerator / math.sqrt(left_sum_squares * right_sum_squares)


def enrich_paired_differences(differences: list[dict]) -> None:
    """Add scale-free seed-level effects to lower-is-better paired metrics."""
    for difference in differences:
        fixed_value = difference["fixed_value"]
        global_value = difference["global_only_value"]
        if fixed_value <= 0 or global_value <= 0:
            fail(
                "relative and geometric paired effects require strictly positive "
                f"metric values, got fixed={fixed_value}, global_only={global_value}"
            )
        ratio = global_value / fixed_value
        difference["global_only_to_fixed_ratio"] = ratio
        difference["relative_improvement_pct"] = (1 - ratio) * 100


def summarize_group(group: list[dict], bootstrap_seed: int) -> dict:
    """Compute robust seed-level summaries for one metric/budget/NFE stratum."""
    deltas = [item["delta"] for item in group]
    improvements = [item["relative_improvement_pct"] for item in group]
    ratios = [item["global_only_to_fixed_ratio"] for item in group]
    fixed_values = [item["fixed_value"] for item in group]
    global_values = [item["global_only_value"] for item in group]
    global_wins = sum(item["winner"] == "global_only" for item in group)
    fixed_wins = sum(item["winner"] == "fixed" for item in group)
    ties = sum(item["winner"] == "tie" for item in group)
    mean_improvement = statistics.mean(improvements)
    rank_correlation = spearman_rank_correlation(fixed_values, global_values)
    fixed_ranks = average_ranks(fixed_values)
    global_ranks = average_ranks(global_values)
    leave_one_seed_out = []
    for omitted in sorted(group, key=lambda item: item["training_seed"]):
        retained = [item for item in group if item["training_seed"] != omitted["training_seed"]]
        retained_improvements = [item["relative_improvement_pct"] for item in retained]
        leave_one_seed_out.append({
            "omitted_training_seed": omitted["training_seed"],
            "retained_pair_count": len(retained),
            "mean_relative_improvement_pct": statistics.mean(retained_improvements),
            "median_delta": statistics.median(item["delta"] for item in retained),
            "global_wins": sum(item["winner"] == "global_only" for item in retained),
            "fixed_wins": sum(item["winner"] == "fixed" for item in retained),
            "ties": sum(item["winner"] == "tie" for item in retained),
        })
    return {
        "pair_count": len(group),
        "mean_delta": statistics.mean(deltas),
        "median_delta": statistics.median(deltas),
        "sample_sd_delta": statistics.stdev(deltas) if len(deltas) > 1 else None,
        "minimum_delta": min(deltas),
        "maximum_delta": max(deltas),
        "mean_relative_improvement_pct": mean_improvement,
        "geometric_mean_relative_improvement_pct": (1 - math.exp(statistics.mean(math.log(value) for value in ratios))) * 100,
        "relative_improvement_sample_sd_pct": statistics.stdev(improvements) if len(improvements) > 1 else None,
        "relative_improvement_cv_pct": (
            statistics.stdev(improvements) / mean_improvement * 100
            if len(improvements) > 1 and mean_improvement != 0 else None
        ),
        "worst_case_relative_improvement_pct": min(improvements),
        "best_case_relative_improvement_pct": max(improvements),
        "rank_consistency_spearman": rank_correlation,
        "rank_order_exact_match": fixed_ranks == global_ranks,
        "global_wins": global_wins,
        "fixed_wins": fixed_wins,
        "ties": ties,
        "exact_sign_test": exact_two_sided_sign_test(global_wins, fixed_wins, ties),
        "bootstrap_mean_relative_improvement_pct_95ci": bootstrap_mean_ci(improvements, bootstrap_seed),
        "leave_one_seed_out": leave_one_seed_out,
    }


def summarize_nfe_effect_heterogeneity(statistics_groups: dict[tuple[str, int, int], list[dict]]) -> list[dict]:
    """Compare NFE=2 and NFE=1 effects within the same training seeds."""
    by_metric_budget: dict[tuple[str, int], dict[int, list[dict]]] = {}
    for (metric_name, budget_kimg, nfe), group in statistics_groups.items():
        by_metric_budget.setdefault((metric_name, budget_kimg), {})[nfe] = group
    summaries = []
    for (metric_name, budget_kimg), by_nfe in sorted(by_metric_budget.items()):
        if set(by_nfe) != {1, 2}:
            continue
        nfe1 = {item["training_seed"]: item for item in by_nfe[1]}
        nfe2 = {item["training_seed"]: item for item in by_nfe[2]}
        if set(nfe1) != set(nfe2):
            fail(f"NFE effect heterogeneity requires identical seed sets for {metric_name}")
        changes = []
        for training_seed in sorted(nfe1):
            change = nfe2[training_seed]["relative_improvement_pct"] - nfe1[training_seed]["relative_improvement_pct"]
            changes.append({
                "training_seed": training_seed,
                "nfe2_minus_nfe1_relative_improvement_pct_points": change,
            })
        values = [item["nfe2_minus_nfe1_relative_improvement_pct_points"] for item in changes]
        nfe2_larger = sum(value > 0 for value in values)
        nfe1_larger = sum(value < 0 for value in values)
        summaries.append({
            "metric_name": metric_name,
            "budget_kimg": budget_kimg,
            "pair_count": len(values),
            "effect_measure": "per-seed relative improvement percentage; NFE=2 minus NFE=1",
            "mean_change_percentage_points": statistics.mean(values),
            "median_change_percentage_points": statistics.median(values),
            "sample_sd_change_percentage_points": statistics.stdev(values) if len(values) > 1 else None,
            "minimum_change_percentage_points": min(values),
            "maximum_change_percentage_points": max(values),
            "nfe2_larger_effect_seeds": nfe2_larger,
            "nfe1_larger_effect_seeds": nfe1_larger,
            "ties": sum(value == 0 for value in values),
            "exact_sign_test": exact_two_sided_sign_test(
                nfe2_larger, nfe1_larger, sum(value == 0 for value in values),
                "NFE2_larger_effect", "NFE1_larger_effect",
            ),
            "bootstrap_mean_change_percentage_points_95ci": bootstrap_mean_ci(values, BOOTSTRAP_SEED + len(summaries)),
            "per_seed_changes": changes,
        })
    return summaries


def build_pairwise_summary(
    differences: list[dict], baseline: str, candidate: str, candidate_label: str, direction: str
) -> dict:
    """Summarize a complete list of fixed/global paired differences."""
    enrich_paired_differences(differences)
    statistics_groups: dict[tuple[str, int, int], list[dict]] = {}
    for difference in differences:
        key = (difference["metric"], difference["budget_kimg"], difference["nfe"])
        statistics_groups.setdefault(key, []).append(difference)
    statistics_rows = []
    for index, ((metric_name, budget_kimg, nfe), group) in enumerate(sorted(statistics_groups.items())):
        statistic = summarize_group(group, BOOTSTRAP_SEED + index)
        statistic.update({
            "metric_name": metric_name,
            "budget_kimg": budget_kimg,
            "nfe": nfe,
        })
        statistics_rows.append(statistic)
    return {
        "status": "computed",
        "schema_version": 2,
        "pairing_key": ["training_seed", "budget_kimg", "nfe", "metric"],
        "baseline_method": baseline,
        "candidate_method": candidate,
        "candidate_label": candidate_label,
        "delta_direction": direction,
        "effect_definitions": {
            "relative_improvement_pct": "100 * (fixed - global_only) / fixed; positive favors global-only",
            "geometric_mean_relative_improvement_pct": "100 * (1 - geometric_mean(global_only / fixed))",
            "rank_consistency_spearman": "Spearman correlation of lower-is-better seed ranks between fixed and global-only",
            "nfe_effect_heterogeneity": "per-seed relative-improvement difference, NFE=2 minus NFE=1, in percentage points",
        },
        "inference_note": "The independent units are the three training seeds. Exact sign tests and bootstrap intervals are descriptive sensitivity summaries, not a basis for strong significance claims.",
        "paired_differences": differences,
        "statistics": statistics_rows,
        "nfe_effect_heterogeneity": summarize_nfe_effect_heterogeneity(statistics_groups),
    }


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
    # Smoke runs intentionally exercise one checkpoint rather than the entire
    # fixed/global matrix.  They must never be interpreted as evidence for the
    # paired comparison (or fail merely because the companion arm was not run).
    # Quick and formal runs keep the strict missing/duplicate-pair failure.
    if manifest["phase"] == "smoke":
        pairwise_statistics = {
            "status": "not_computed",
            "reason": "smoke phase does not form the predeclared comparison matrix",
        }
    else:
        pairwise_statistics = build_pairwise_statistics(rows, manifest.get("comparison"))

    return {
        "schema_version": 1,
        "protocol": PROTOCOL_ID,
        "phase": manifest["phase"],
        "evidence_class": manifest["evidence_class"],
        "row_count": len(rows),
        "statistics_grouping": ["evidence_class", "metric_name", "nfe", "method"],
        "pairwise_statistics": pairwise_statistics,
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

    return build_pairwise_summary(differences, baseline, candidate, candidate_label, direction)


def write_paired_outputs(outdir: Path, pairwise: dict) -> None:
    """Write per-seed fixed/global evidence as standalone reviewable files."""
    if pairwise["status"] != "computed":
        return
    differences = pairwise["paired_differences"]
    with (outdir / "paired_differences.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(differences[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(differences)
    paired_statistics = {
        key: value for key, value in pairwise.items() if key != "paired_differences"
    }
    (outdir / "paired_statistics.json").write_text(
        json.dumps(paired_statistics, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Fixed vs global-only paired robustness statistics",
        "",
        "Pairing key: `training_seed + budget_kimg + nfe + metric`.",
        "Delta: `global_only - fixed`; negative values favor global-only.",
        "Relative improvement: `100 × (fixed - global_only) / fixed`; positive values favor global-only.",
        "Independent units are training seeds; the pair count is reported for each metric/NFE stratum.",
        "",
        "The exact two-sided sign test is reported only as a low-resolution directional check. "
        "Bootstrap intervals resample these same seeds and are descriptive sensitivity intervals, not additional independent-sample inference.",
        "",
        "## Paired effect summary",
        "",
        "| Metric | Budget (kimg) | NFE | Pairs | Arithmetic relative improvement | Geometric relative improvement | Median delta | Worst-case improvement | Seed CV | Rank consistency (Spearman) | Wins | Exact sign p (two-sided) | Bootstrap 95% CI |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pairwise["statistics"]:
        sign_test = row["exact_sign_test"]
        bootstrap = row["bootstrap_mean_relative_improvement_pct_95ci"]
        sign_p = "—" if sign_test is None else f"{sign_test['p_value']:.6f}"
        bootstrap_text = "—" if bootstrap is None else f"[{bootstrap['lower']:.6f}, {bootstrap['upper']:.6f}]%"
        rank = "—" if row["rank_consistency_spearman"] is None else f"{row['rank_consistency_spearman']:.6f}"
        cv = "—" if row["relative_improvement_cv_pct"] is None else f"{row['relative_improvement_cv_pct']:.6f}%"
        lines.append(
            f"| {row['metric_name']} | {row['budget_kimg']} | {row['nfe']} | "
            f"{row['pair_count']} | {row['mean_relative_improvement_pct']:.6f}% | "
            f"{row['geometric_mean_relative_improvement_pct']:.6f}% | {row['median_delta']:.9f} | "
            f"{row['worst_case_relative_improvement_pct']:.6f}% | {cv} | {rank} | "
            f"{row['global_wins']}/{row['fixed_wins']}/{row['ties']} | {sign_p} | {bootstrap_text} |"
        )
    lines.extend([
        "",
        "## Leave-one-seed-out arithmetic relative improvement",
        "",
        "| Metric | NFE | Omitted seed | Retained pairs | Mean relative improvement | Global/fixed/tie wins |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in pairwise["statistics"]:
        for loo in row["leave_one_seed_out"]:
            lines.append(
                f"| {row['metric_name']} | {row['nfe']} | {loo['omitted_training_seed']} | "
                f"{loo['retained_pair_count']} | {loo['mean_relative_improvement_pct']:.6f}% | "
                f"{loo['global_wins']}/{loo['fixed_wins']}/{loo['ties']} |"
            )
    lines.extend([
        "",
        "## NFE effect heterogeneity",
        "",
        "Effect change is the per-seed relative improvement at NFE=2 minus that at NFE=1, in percentage points. Positive values indicate a larger global-only advantage at NFE=2.",
        "",
        "| Metric | Pairs | Mean change | Median change | Range | NFE=2 larger / NFE=1 larger / ties | Exact sign p (two-sided) | Bootstrap 95% CI |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in pairwise["nfe_effect_heterogeneity"]:
        sign_test = row["exact_sign_test"]
        bootstrap = row["bootstrap_mean_change_percentage_points_95ci"]
        sign_p = "—" if sign_test is None else f"{sign_test['p_value']:.6f}"
        bootstrap_text = "—" if bootstrap is None else f"[{bootstrap['lower']:.6f}, {bootstrap['upper']:.6f}] pp"
        lines.append(
            f"| {row['metric_name']} | {row['pair_count']} | {row['mean_change_percentage_points']:.6f} pp | "
            f"{row['median_change_percentage_points']:.6f} pp | "
            f"[{row['minimum_change_percentage_points']:.6f}, {row['maximum_change_percentage_points']:.6f}] pp | "
            f"{row['nfe2_larger_effect_seeds']} / {row['nfe1_larger_effect_seeds']} / {row['ties']} | {sign_p} | {bootstrap_text} |"
        )
    lines.append("")
    (outdir / "paired_statistics.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(outdir: Path, rows: list[dict], summary: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "evaluation_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
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


def read_paired_differences(path: Path) -> list[dict]:
    """Load a previously emitted paired-difference table for a reproducible refresh."""
    if not path.is_file():
        fail(f"missing paired differences CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "training_seed", "budget_kimg", "nfe", "metric", "fixed_checkpoint_id",
            "global_only_checkpoint_id", "fixed_checkpoint_sha256", "global_only_checkpoint_sha256",
            "fixed_value", "global_only_value", "delta", "winner",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            fail(f"paired differences CSV is missing fields: {sorted(missing)}")
        differences = []
        seen = set()
        for row in reader:
            try:
                difference = {
                    "training_seed": int(row["training_seed"]),
                    "budget_kimg": int(row["budget_kimg"]),
                    "nfe": int(row["nfe"]),
                    "metric": row["metric"],
                    "fixed_checkpoint_id": row["fixed_checkpoint_id"],
                    "global_only_checkpoint_id": row["global_only_checkpoint_id"],
                    "fixed_checkpoint_sha256": row["fixed_checkpoint_sha256"],
                    "global_only_checkpoint_sha256": row["global_only_checkpoint_sha256"],
                    "fixed_value": float(row["fixed_value"]),
                    "global_only_value": float(row["global_only_value"]),
                    "delta": float(row["delta"]),
                    "winner": row["winner"],
                }
            except (TypeError, ValueError) as exc:
                fail(f"malformed paired difference in {path}: {exc}")
            key = (difference["training_seed"], difference["budget_kimg"], difference["nfe"], difference["metric"])
            if key in seen:
                fail(f"duplicate paired difference for {key}")
            seen.add(key)
            expected_delta = difference["global_only_value"] - difference["fixed_value"]
            if not math.isclose(difference["delta"], expected_delta, rel_tol=0, abs_tol=1e-12):
                fail(f"delta does not match absolute values for {key}")
            expected_winner = "global_only" if expected_delta < 0 else "fixed" if expected_delta > 0 else "tie"
            if difference["winner"] != expected_winner:
                fail(f"winner does not match delta for {key}")
            differences.append(difference)
    if not differences:
        fail(f"paired differences CSV has no rows: {path}")
    return sorted(differences, key=lambda item: (item["training_seed"], item["budget_kimg"], item["nfe"], item["metric"]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--eval-root", type=Path)
    input_group.add_argument("--paired-differences", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)
    outdir = args.outdir.resolve()
    if args.paired_differences is not None:
        differences = read_paired_differences(args.paired_differences.resolve())
        summary = build_pairwise_summary(
            differences, "fixed", "global110", "global_only", "global_only - fixed"
        )
        outdir.mkdir(parents=True, exist_ok=True)
        write_paired_outputs(outdir, summary)
        print(f"Refreshed {len(differences)} paired differences; output: {outdir}")
        return
    rows, summary = collect(args.eval_root.resolve())
    write_outputs(outdir, rows, summary)
    print(f"Validated {len(rows)} metric rows; output: {outdir}")


if __name__ == "__main__":
    main()
