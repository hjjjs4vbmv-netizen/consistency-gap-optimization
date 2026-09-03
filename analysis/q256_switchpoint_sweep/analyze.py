#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.q256_switchpoint_sweep import result_conversion

SEEDS, SWITCH_POINTS, ENDPOINTS = result_conversion.SEEDS, result_conversion.SWITCH_POINTS, result_conversion.ENDPOINTS
ORDERED_VERDICT = "ORDERED_PREFIX_DEPENDENCE"
T_CRITICAL_975 = (None, 12.7062047364, 4.3026527297, 3.1824463053, 2.7764451052, 2.5705818356,
    2.4469118488, 2.3646242510, 2.3060041350, 2.2621571629, 2.2281388520, 2.2009851601,
)


def midranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = ((start + 1) + stop) / 2
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def exact_page_test(g_rows: list[list[float]]) -> dict[str, float | int | str]:
    if not g_rows:
        raise ValueError("Page test requires at least one complete seed")
    ranked = [midranks([-value for value in row]) for row in g_rows]
    observed_l2 = sum(
        (j + 1) * int(round(2 * rank))
        for ranks in ranked
        for j, rank in enumerate(ranks)
    )
    distribution = Counter({0: 1})
    for ranks in ranked:
        ranks2 = [int(round(2 * rank)) for rank in ranks]
        block = Counter(
            sum((j + 1) * rank for j, rank in enumerate(permutation))
            for permutation in itertools.permutations(ranks2)
        )
        updated: Counter[int] = Counter()
        for total, count in distribution.items():
            for block_total, block_count in block.items():
                updated[total + block_total] += count * block_count
        distribution = updated
    tail = sum(count for score, count in distribution.items() if score >= observed_l2)
    denominator = 24 ** len(g_rows)
    return {
        "direction": "G_128 >= G_256 >= G_384 >= G_512",
        "L_observed": observed_l2 / 2,
        "p_exact": tail / denominator,
    }


def descriptive_summary(values: list[float]) -> dict[str, object]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "sample_sd": None,
            "mean_t_ci95": None,
            "sign_counts": {"negative": 0, "zero": 0, "positive": 0},
        }
    mean = statistics.mean(values)
    sample_sd = statistics.stdev(values) if len(values) > 1 else None
    ci = None
    if sample_sd is not None:
        half_width = T_CRITICAL_975[len(values) - 1] * sample_sd / math.sqrt(len(values))
        ci = [mean - half_width, mean + half_width]
    return {
        "n": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "sample_sd": sample_sd,
        "mean_t_ci95": ci,
        "sign_counts": {
            "negative": sum(value < 0 for value in values),
            "zero": sum(value == 0 for value in values),
            "positive": sum(value > 0 for value in values),
        },
    }


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.6g}"


def analyze(rows: list[dict[str, object]]) -> dict:
    by_seed = {
        seed: sorted((row for row in rows if row["seed"] == seed), key=lambda row: row["s_kimg"])
        for seed in SEEDS
    }
    complete: list[int] = []
    incomplete: list[int] = []
    root_causes: dict[str, str] = {}
    missing_cells: dict[str, list[str]] = {}
    g_by_seed: dict[int, list[float]] = {}
    for seed, seed_rows in by_seed.items():
        missing = []
        for row in seed_rows:
            switch = row["s_kimg"]
            if not row["ba_valid"]:
                missing.append(f"BA_{switch}@{ENDPOINTS[switch]}")
            if not row["ctrl_valid"]:
                missing.append(f"CTRL@{ENDPOINTS[switch]}")
        causes = {str(row["root_cause"]) for row in seed_rows if row["root_cause"]}
        if missing:
            incomplete.append(seed)
            missing_cells[str(seed)] = missing
            if len(causes) != 1:
                raise ValueError(f"incomplete seed {seed} must have one root cause")
            root_causes[str(seed)] = causes.pop()
            continue
        complete.append(seed)
        g_by_seed[seed] = [
            math.log(row["ba_fid50k"]) - math.log(row["ctrl_fid50k"]) for row in seed_rows
        ]
    g_rows = [g_by_seed[seed] for seed in complete]
    point_summaries = {f"G_{switch}": descriptive_summary([row[j] for row in g_rows])
                       for j, switch in enumerate(SWITCH_POINTS)}
    adjacent_summaries = {
        f"G_{later}_minus_G_{earlier}": descriptive_summary(
            [row[j + 1] - row[j] for row in g_rows]
        )
        for j, (earlier, later) in enumerate(zip(SWITCH_POINTS, SWITCH_POINTS[1:]))
    }
    arm_counts = Counter()
    for missing in missing_cells.values():
        for switch in SWITCH_POINTS:
            if any(cell.startswith(f"BA_{switch}@") for cell in missing):
                arm_counts[f"BA_{switch}"] += 1
    concentrated = sorted(arm for arm, count in arm_counts.items() if count >= 3)
    eligible = len(complete) >= 9
    page = exact_page_test(g_rows) if eligible else None
    if page is not None:
        ordered = (
            page["p_exact"] <= 0.05
            and point_summaries["G_512"]["mean"] < point_summaries["G_128"]["mean"]
        )
        page.update({
            "alpha": 0.05,
            "verdict": ORDERED_VERDICT if ordered else "ORDERING_NOT_RESOLVED",
        })
    return {
        "protocol_id": "q256_switchpoint_sweep_v1",
        "metric": {"name": "fid50k_full", "nfe": 1, "transform": "natural_log"},
        "complete_seeds": complete,
        "incomplete_seeds": incomplete,
        "n_complete": len(complete),
        "primary_status": "ANALYZED" if eligible else "ABORTED_INCOMPLETE",
        "page_test": page,
        "point_summaries": point_summaries,
        "adjacent_paired_differences": adjacent_summaries,
        "missingness": {
            "root_cause_by_seed": root_causes,
            "missing_cells_by_seed": missing_cells,
            "arm_concentration_flag": concentrated,
        },
        "allowed_wording_key": page["verdict"] if page else "DESCRIPTIVE_ONLY",
        "g_by_seed": {str(seed): values for seed, values in g_by_seed.items()},
    }


def summarize_common_endpoint(h_values: dict[int, dict[int, float]]) -> dict:
    points = {}
    for switch in SWITCH_POINTS:
        values = h_values[switch]
        points[f"H_{switch}"] = {
            "available_seeds": sorted(values),
            "missing_seeds": sorted(set(SEEDS) - set(values)),
            "seed_values": {str(seed): values[seed] for seed in sorted(values)},
            "summary": descriptive_summary(list(values.values())),
        }
    return {
        "protocol_id": "q256_switchpoint_sweep_v1",
        "estimand": "H_s = ln FID(BA(s)@1024) - ln FID(CTRL@1024)",
        "inferential_role": "NONE",
        "points": points,
    }


def write_outputs(result: dict, output_dir: Path, common: dict | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fixed_chase_primary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# TASK 2 fixed-chase analysis",
        "",
        f"Primary status: `{result['primary_status']}`; complete seeds: {result['n_complete']}/12.",
        "",
        "| s (kimg) | mean G | median | sample SD | mean 95% t-CI | signs −/0/+ |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for switch in SWITCH_POINTS:
        summary = result["point_summaries"][f"G_{switch}"]
        ci = summary["mean_t_ci95"]
        signs = summary["sign_counts"]
        ci_text = "—" if ci is None else f"[{ci[0]:.6g}, {ci[1]:.6g}]"
        lines.append(
            f"| {switch} | {_number(summary['mean'])} | {_number(summary['median'])} | "
            f"{_number(summary['sample_sd'])} | {ci_text} | {signs['negative']}/{signs['zero']}/{signs['positive']} |"
        )
    lines.extend(["", "| adjacent contrast | mean | median | sample SD | mean 95% t-CI | signs −/0/+ |", "|---|---:|---:|---:|---:|---:|"])
    for name, summary in result["adjacent_paired_differences"].items():
        ci = summary["mean_t_ci95"]
        signs = summary["sign_counts"]
        ci_text = "—" if ci is None else f"[{ci[0]:.6g}, {ci[1]:.6g}]"
        lines.append(
            f"| {name} | {_number(summary['mean'])} | {_number(summary['median'])} | "
            f"{_number(summary['sample_sd'])} | {ci_text} | {signs['negative']}/{signs['zero']}/{signs['positive']} |"
        )
    page = result["page_test"]
    if page:
        lines.extend(["", f"Page L={page['L_observed']:.6g}, exact one-sided p={page['p_exact']:.6g}; verdict: `{page['verdict']}`."])
        if page["verdict"] == ORDERED_VERDICT:
            lines.extend(["", "With a fixed 512-kimg A chase, the endpoint contrast showed the prespecified ordered dependence on B-prefix duration along the schedules tested here."])
    if result["incomplete_seeds"]:
        lines.extend(["", f"Incomplete seeds: {result['incomplete_seeds']}; root causes: {result['missingness']['root_cause_by_seed']}."])
    if common is not None:
        (output_dir / "common_endpoint_descriptive.json").write_text(
            json.dumps(common, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        lines.extend([
            "", "## Common 1024-kimg endpoint H (descriptive only)", "",
            "| s (kimg) | n | mean H | median | sample SD | mean 95% t-CI | signs −/0/+ |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for switch in SWITCH_POINTS:
            summary = common["points"][f"H_{switch}"]["summary"]
            ci, signs = summary["mean_t_ci95"], summary["sign_counts"]
            ci_text = "—" if ci is None else f"[{ci[0]:.6g}, {ci[1]:.6g}]"
            lines.append(
                f"| {switch} | {summary['n']} | {_number(summary['mean'])} | "
                f"{_number(summary['median'])} | {_number(summary['sample_sd'])} | {ci_text} | "
                f"{signs['negative']}/{signs['zero']}/{signs['positive']} |"
            )
        lines.extend(["", "These common-endpoint contrasts are descriptive and combine differences in B-prefix and subsequent A-chase duration; no test or shape label is assigned."])
    (output_dir / "FINAL_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--decoded-results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    common = None
    if args.decoded_results:
        decoded = json.loads(args.decoded_results.read_text(encoding="utf-8"))
        rows, h_values = result_conversion.convert_decoded(decoded)
        result_conversion.write_rows(rows, args.output_dir / "fixed_chase_seed_results.csv")
        common = summarize_common_endpoint(h_values)
    else:
        rows = result_conversion.read_rows(args.input)
    write_outputs(analyze(rows), args.output_dir, common)


if __name__ == "__main__":
    main()
