#!/usr/bin/env python3
"""Collect a complete two-method, multi-budget paired metric matrix.

Required input columns:
method,training_seed,budget_kimg,nfe,metric_name,metric_value

Optional columns:
training_time_hours,quality_target,checkpoint_sha256

The collector validates every observed method/seed/budget/NFE/metric cell,
then emits seed-level, aggregate, time-to-quality, table, and figure artifacts.
Metrics are treated as lower-is-better.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ect-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REQUIRED = ("method", "training_seed", "budget_kimg", "nfe", "metric_name", "metric_value")
METHOD_COLORS = ("#2563EB", "#C98212")
SEED_COLORS = ("#2563EB", "#C98212", "#C2416C", "#5B8C5A", "#7C5AC8", "#64748B")
INK = "#1F2937"
GRID = "#D1D5DB"


def fail(message: str) -> None:
    raise SystemExit("[collect_multibudget_results] ERROR: " + message)


def optional_float(value: str, field: str, row_number: int) -> float | None:
    if not value.strip():
        return None
    try:
        result = float(value)
    except ValueError as exc:
        fail("row {}: {} must be numeric".format(row_number, field))
        raise exc
    if not math.isfinite(result):
        fail("row {}: {} must be finite".format(row_number, field))
    return result


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        fail("input CSV does not exist: {}".format(path))
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        missing = set(REQUIRED) - set(fields)
        if missing:
            fail("missing required columns: {}".format(sorted(missing)))
        raw_rows = list(reader)
    if not raw_rows:
        fail("input CSV has no rows")
    rows = []
    for row_number, raw in enumerate(raw_rows, start=2):
        try:
            row = {
                "method": raw["method"].strip(),
                "training_seed": int(raw["training_seed"]),
                "budget_kimg": float(raw["budget_kimg"]),
                "nfe": int(raw["nfe"]),
                "metric_name": raw["metric_name"].strip(),
                "metric_value": float(raw["metric_value"]),
                "training_time_hours": optional_float(raw.get("training_time_hours", ""), "training_time_hours", row_number),
                "quality_target": optional_float(raw.get("quality_target", ""), "quality_target", row_number),
                "checkpoint_sha256": raw.get("checkpoint_sha256", "").strip(),
            }
        except (TypeError, ValueError) as exc:
            fail("row {}: malformed required value".format(row_number))
            raise exc
        if (not row["method"] or not row["metric_name"] or row["training_seed"] < 0
                or row["budget_kimg"] <= 0 or row["nfe"] < 1
                or not math.isfinite(row["metric_value"])):
            fail("row {}: invalid method, metric, seed, budget, NFE, or metric value".format(row_number))
        if row["training_time_hours"] is not None and row["training_time_hours"] < 0:
            fail("row {}: training_time_hours must be non-negative".format(row_number))
        rows.append(row)
    return rows


def validate(rows: list[dict], baseline: str, candidate: str) -> dict:
    if baseline == candidate:
        fail("baseline and candidate methods must differ")
    methods = {row["method"] for row in rows}
    if methods != {baseline, candidate}:
        fail("methods must be exactly {}, got {}".format([baseline, candidate], sorted(methods)))
    seeds = sorted({row["training_seed"] for row in rows})
    budgets = sorted({row["budget_kimg"] for row in rows})
    nfes = sorted({row["nfe"] for row in rows})
    metrics = sorted({row["metric_name"] for row in rows})
    index = {}
    for row in rows:
        key = (row["method"], row["training_seed"], row["budget_kimg"], row["nfe"], row["metric_name"])
        if key in index:
            fail("duplicate matrix cell: {}".format(key))
        index[key] = row
    expected = {
        (method, seed, budget, nfe, metric)
        for method in (baseline, candidate)
        for seed in seeds for budget in budgets for nfe in nfes for metric in metrics
    }
    missing = expected - set(index)
    extra = set(index) - expected
    if missing or extra:
        fail("matrix incomplete; missing={}, extra={}".format(sorted(missing), sorted(extra)))
    for metric in metrics:
        for nfe in nfes:
            targets = {
                row["quality_target"] for row in rows
                if row["metric_name"] == metric and row["nfe"] == nfe
                and row["quality_target"] is not None
            }
            if len(targets) > 1:
                fail("quality_target must be consistent for metric={} NFE={}".format(metric, nfe))
    return {
        "methods": [baseline, candidate], "seeds": seeds, "budgets": budgets,
        "nfes": nfes, "metrics": metrics, "index": index,
    }


def paired_rows(matrix: dict) -> list[dict]:
    baseline, candidate = matrix["methods"]
    index = matrix["index"]
    output = []
    for metric in matrix["metrics"]:
        for nfe in matrix["nfes"]:
            for budget in matrix["budgets"]:
                for seed in matrix["seeds"]:
                    fixed = index[(baseline, seed, budget, nfe, metric)]
                    tested = index[(candidate, seed, budget, nfe, metric)]
                    delta = tested["metric_value"] - fixed["metric_value"]
                    output.append({
                        "metric_name": metric, "nfe": nfe, "budget_kimg": budget,
                        "training_seed": seed, "baseline_method": baseline,
                        "candidate_method": candidate, "baseline_value": fixed["metric_value"],
                        "candidate_value": tested["metric_value"],
                        "delta_candidate_minus_baseline": delta,
                        "relative_improvement_pct": (
                            100 * (fixed["metric_value"] - tested["metric_value"]) / fixed["metric_value"]
                            if fixed["metric_value"] > 0 else None
                        ),
                        "winner": candidate if delta < 0 else baseline if delta > 0 else "tie",
                    })
    return output


def aggregate_rows(rows: list[dict], paired: list[dict], matrix: dict) -> tuple[list[dict], list[dict]]:
    curves, summaries = [], []
    baseline, candidate = matrix["methods"]
    for metric in matrix["metrics"]:
        for nfe in matrix["nfes"]:
            for budget in matrix["budgets"]:
                selected_pairs = [
                    row for row in paired
                    if (row["metric_name"], row["nfe"], row["budget_kimg"]) == (metric, nfe, budget)
                ]
                for method in matrix["methods"]:
                    values = [
                        row["metric_value"] for row in rows
                        if (row["method"], row["metric_name"], row["nfe"], row["budget_kimg"])
                        == (method, metric, nfe, budget)
                    ]
                    curves.append({
                        "metric_name": metric, "nfe": nfe, "budget_kimg": budget,
                        "method": method, "mean_metric_value": statistics.mean(values),
                        "sample_sd_metric_value": statistics.stdev(values) if len(values) > 1 else None,
                        "seed_count": len(values),
                    })
                deltas = [row["delta_candidate_minus_baseline"] for row in selected_pairs]
                summaries.append({
                    "metric_name": metric, "nfe": nfe, "budget_kimg": budget,
                    "seed_count": len(deltas), "mean_paired_delta": statistics.mean(deltas),
                    "median_paired_delta": statistics.median(deltas),
                    "sample_sd_paired_delta": statistics.stdev(deltas) if len(deltas) > 1 else None,
                    "candidate_wins": sum(row["winner"] == candidate for row in selected_pairs),
                    "baseline_wins": sum(row["winner"] == baseline for row in selected_pairs),
                    "ties": sum(row["winner"] == "tie" for row in selected_pairs),
                })
    return curves, summaries


def time_to_quality(rows: list[dict], matrix: dict) -> list[dict]:
    output = []
    for metric in matrix["metrics"]:
        for nfe in matrix["nfes"]:
            for method in matrix["methods"]:
                for seed in matrix["seeds"]:
                    trajectory = sorted(
                        (row for row in rows if (row["metric_name"], row["nfe"], row["method"], row["training_seed"])
                         == (metric, nfe, method, seed)),
                        key=lambda row: row["budget_kimg"],
                    )
                    targets = {row["quality_target"] for row in trajectory if row["quality_target"] is not None}
                    target = next(iter(targets)) if targets else None
                    reached = next(
                        (row for row in trajectory if target is not None and row["metric_value"] <= target),
                        None,
                    )
                    times_present = all(row["training_time_hours"] is not None for row in trajectory)
                    status = (
                        "target_missing" if target is None else
                        "training_time_missing" if not times_present else
                        "not_reached" if reached is None else "reached"
                    )
                    output.append({
                        "metric_name": metric, "nfe": nfe, "method": method,
                        "training_seed": seed, "quality_target": target, "status": status,
                        "budget_kimg_at_quality": reached["budget_kimg"] if reached else "",
                        "training_time_hours_at_quality": (
                            reached["training_time_hours"] if reached and times_present else ""
                        ),
                        "best_observed_value": min(row["metric_value"] for row in trajectory),
                        "max_observed_budget_kimg": trajectory[-1]["budget_kimg"],
                    })
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        fail("cannot write empty CSV: {}".format(path))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_tables(summary: list[dict], outdir: Path) -> None:
    markdown = [
        "# Multi-budget paired summary", "",
        "Delta is candidate minus baseline; negative values favor the candidate.", "",
        "| Metric | NFE | Budget (kimg) | Mean paired delta | Median delta | Sample SD | Candidate/baseline/tie |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    latex = [
        "\\begin{tabular}{llrrrr}",
        "\\hline",
        "Metric & NFE & Budget (kimg) & Mean paired delta & Sample SD & Candidate/baseline/tie \\\\",
        "\\hline",
    ]
    for row in summary:
        wins = "{}/{}/{}".format(row["candidate_wins"], row["baseline_wins"], row["ties"])
        markdown.append(
            "| {} | {} | {:g} | {:.8g} | {:.8g} | {:.8g} | {} |".format(
                row["metric_name"], row["nfe"], row["budget_kimg"],
                row["mean_paired_delta"], row["median_paired_delta"],
                row["sample_sd_paired_delta"], wins,
            )
        )
        latex.append(
            "{} & {} & {:g} & {:.8g} & {:.8g} & {} \\\\".format(
                row["metric_name"].replace("_", "\\_"), row["nfe"], row["budget_kimg"],
                row["mean_paired_delta"], row["sample_sd_paired_delta"], wins,
            )
        )
    latex.extend(["\\hline", "\\end{tabular}", ""])
    (outdir / "summary_table.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (outdir / "summary_table.tex").write_text("\n".join(latex), encoding="utf-8")


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#9CA3AF")
    axis.spines["bottom"].set_color("#9CA3AF")
    axis.tick_params(colors=INK, labelsize=9)


def save_figure(figure: plt.Figure, figure_dir: Path, stem: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        figure.savefig(
            figure_dir / "{}.{}".format(stem, extension),
            dpi=220, bbox_inches="tight", facecolor="white",
        )
    plt.close(figure)


def make_panels(matrix: dict, title: str) -> tuple[plt.Figure, object]:
    figure, axes = plt.subplots(
        len(matrix["metrics"]), len(matrix["nfes"]),
        figsize=(5.1 * len(matrix["nfes"]), 3.7 * len(matrix["metrics"])),
        squeeze=False,
    )
    figure.suptitle(title, x=0.02, y=0.99, ha="left", fontsize=15, fontweight="bold", color=INK)
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.10, top=0.85, hspace=0.34, wspace=0.23)
    return figure, axes


def plot_budget_curves(curves: list[dict], matrix: dict, figure_dir: Path) -> None:
    figure, axes = make_panels(matrix, "Budget curves by metric and NFE")
    for metric_index, metric in enumerate(matrix["metrics"]):
        for nfe_index, nfe in enumerate(matrix["nfes"]):
            axis = axes[metric_index][nfe_index]
            for method_index, method in enumerate(matrix["methods"]):
                selected = sorted(
                    (row for row in curves if (row["metric_name"], row["nfe"], row["method"]) == (metric, nfe, method)),
                    key=lambda row: row["budget_kimg"],
                )
                axis.errorbar(
                    [row["budget_kimg"] for row in selected],
                    [row["mean_metric_value"] for row in selected],
                    yerr=[row["sample_sd_metric_value"] or 0 for row in selected],
                    color=METHOD_COLORS[method_index], marker="o", linewidth=2, capsize=3, label=method,
                )
            axis.set_title("{} · NFE={}".format(metric, nfe), loc="left", fontsize=11, fontweight="bold", color=INK)
            axis.set_xlabel("Training budget (kimg)")
            axis.set_ylabel(metric)
            axis.set_xticks(matrix["budgets"])
            style_axis(axis)
    figure.legend(
        handles=[Line2D([0], [0], color=METHOD_COLORS[index], marker="o", label=method)
                 for index, method in enumerate(matrix["methods"])],
        loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.60, 0.965),
    )
    save_figure(figure, figure_dir, "budget_curves")


def plot_trajectories(rows: list[dict], matrix: dict, figure_dir: Path) -> None:
    figure, axes = make_panels(matrix, "Per-seed trajectories across training budget")
    colors = {seed: SEED_COLORS[index % len(SEED_COLORS)] for index, seed in enumerate(matrix["seeds"])}
    for metric_index, metric in enumerate(matrix["metrics"]):
        for nfe_index, nfe in enumerate(matrix["nfes"]):
            axis = axes[metric_index][nfe_index]
            for seed in matrix["seeds"]:
                for method_index, method in enumerate(matrix["methods"]):
                    selected = sorted(
                        (row for row in rows if (row["metric_name"], row["nfe"], row["training_seed"], row["method"])
                         == (metric, nfe, seed, method)),
                        key=lambda row: row["budget_kimg"],
                    )
                    axis.plot(
                        [row["budget_kimg"] for row in selected],
                        [row["metric_value"] for row in selected],
                        color=colors[seed], linestyle="-" if method_index else "--",
                        marker="o", markerfacecolor=colors[seed] if method_index else "white",
                        linewidth=1.8, alpha=0.9,
                    )
            axis.set_title("{} · NFE={}".format(metric, nfe), loc="left", fontsize=11, fontweight="bold", color=INK)
            axis.set_xlabel("Training budget (kimg)")
            axis.set_ylabel(metric)
            axis.set_xticks(matrix["budgets"])
            style_axis(axis)
    legend = [Line2D([0], [0], color=colors[seed], marker="o", label="Seed {}".format(seed))
              for seed in matrix["seeds"]]
    legend += [
        Line2D([0], [0], color=INK, linestyle="--", label="{} (open)".format(matrix["methods"][0])),
        Line2D([0], [0], color=INK, linestyle="-", label="{} (filled)".format(matrix["methods"][1])),
    ]
    figure.legend(handles=legend, loc="upper center", ncol=min(5, len(legend)),
                  frameon=False, bbox_to_anchor=(0.56, 0.965), fontsize=8.5)
    save_figure(figure, figure_dir, "per_seed_trajectories")


def plot_paired_deltas(paired: list[dict], matrix: dict, figure_dir: Path) -> None:
    figure, axes = make_panels(matrix, "Paired deltas across training budget")
    colors = {seed: SEED_COLORS[index % len(SEED_COLORS)] for index, seed in enumerate(matrix["seeds"])}
    for metric_index, metric in enumerate(matrix["metrics"]):
        for nfe_index, nfe in enumerate(matrix["nfes"]):
            axis = axes[metric_index][nfe_index]
            means, deviations = [], []
            for budget in matrix["budgets"]:
                values = [
                    row["delta_candidate_minus_baseline"] for row in paired
                    if (row["metric_name"], row["nfe"], row["budget_kimg"]) == (metric, nfe, budget)
                ]
                means.append(statistics.mean(values))
                deviations.append(statistics.stdev(values) if len(values) > 1 else 0)
            for seed in matrix["seeds"]:
                selected = sorted(
                    (row for row in paired if (row["metric_name"], row["nfe"], row["training_seed"]) == (metric, nfe, seed)),
                    key=lambda row: row["budget_kimg"],
                )
                axis.plot([row["budget_kimg"] for row in selected],
                          [row["delta_candidate_minus_baseline"] for row in selected],
                          color=colors[seed], marker="o", linewidth=1.8)
            axis.errorbar(matrix["budgets"], means, yerr=deviations, color="#111827",
                          marker="D", linewidth=1.7, capsize=3, zorder=4)
            axis.axhline(0, color="#6B7280", linewidth=1, linestyle=(0, (4, 3)))
            axis.set_title("{} · NFE={}".format(metric, nfe), loc="left", fontsize=11, fontweight="bold", color=INK)
            axis.set_xlabel("Training budget (kimg)")
            axis.set_ylabel("Candidate − baseline")
            axis.set_xticks(matrix["budgets"])
            style_axis(axis)
    legend = [Line2D([0], [0], color=colors[seed], marker="o", label="Seed {}".format(seed))
              for seed in matrix["seeds"]]
    legend.append(Line2D([0], [0], color="#111827", marker="D", label="Mean ± sample SD"))
    figure.legend(handles=legend, loc="upper center", ncol=min(5, len(legend)),
                  frameon=False, bbox_to_anchor=(0.56, 0.965), fontsize=8.5)
    save_figure(figure, figure_dir, "paired_deltas")


def plot_time_to_quality(records: list[dict], matrix: dict, figure_dir: Path) -> bool:
    if not any(row["status"] == "reached" for row in records):
        return False
    figure, axes = make_panels(matrix, "Time to pre-specified quality target")
    colors = {seed: SEED_COLORS[index % len(SEED_COLORS)] for index, seed in enumerate(matrix["seeds"])}
    for metric_index, metric in enumerate(matrix["metrics"]):
        for nfe_index, nfe in enumerate(matrix["nfes"]):
            axis = axes[metric_index][nfe_index]
            selected = [
                row for row in records if (row["metric_name"], row["nfe"]) == (metric, nfe)
            ]
            for seed in matrix["seeds"]:
                by_method = {
                    row["method"]: row for row in selected
                    if row["training_seed"] == seed and row["status"] == "reached"
                }
                if len(by_method) == 2:
                    axis.plot(
                        [0, 1],
                        [by_method[method]["training_time_hours_at_quality"] for method in matrix["methods"]],
                        color=colors[seed], marker="o", linewidth=1.8,
                    )
            target = next((row["quality_target"] for row in selected if row["quality_target"] is not None), None)
            axis.set_xticks([0, 1], matrix["methods"])
            axis.set_ylabel("Training time to target (hours)")
            suffix = " · target ≤ {:g}".format(target) if target is not None else ""
            axis.set_title("{} · NFE={}{}".format(metric, nfe, suffix), loc="left",
                           fontsize=10.5, fontweight="bold", color=INK)
            style_axis(axis)
    figure.text(0.02, 0.02, "Only reached trajectories are plotted; unreached and missing-time cases remain in time_to_quality.csv.", fontsize=9, color=INK)
    save_figure(figure, figure_dir, "time_to_quality")
    return True


def write_readme(outdir: Path, matrix: dict, time_plot_written: bool) -> None:
    text = """# Multi-budget collector output

The input was validated as a complete paired matrix with two methods, {} training seeds, {} budgets, {} NFE settings, and {} metrics.

| Artifact | Contents |
| --- | --- |
| normalized_metrics.csv | Validated long-form input rows. |
| budget_curves.csv | Method mean and sample SD by budget, metric, and NFE. |
| per_seed_trajectories.csv | Seed-level, figure-ready metric trajectories. |
| paired_deltas.csv | Paired candidate-minus-baseline rows. |
| paired_summary.csv | Descriptive paired mean, median, SD, and win counts. |
| time_to_quality.csv | First target crossing, including transparent missing and unreached statuses. |
| summary_table.md and summary_table.tex | Markdown and LaTeX paired summary tables. |
| figures | SVG and PNG budget, trajectory, paired-delta{} figures. |

All deltas are candidate minus baseline; negative values favor the candidate. Sample SD is descriptive across training seeds, not a confidence interval.
""".format(
        len(matrix["seeds"]), len(matrix["budgets"]), len(matrix["nfes"]), len(matrix["metrics"]),
        ", and time-to-quality" if time_plot_written else "",
    )
    (outdir / "README.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--baseline-method", default="fixed")
    parser.add_argument("--candidate-method", default="global110")
    args = parser.parse_args(argv)

    rows = read_rows(args.input_csv.resolve())
    matrix = validate(rows, args.baseline_method, args.candidate_method)
    paired = paired_rows(matrix)
    curves, summary = aggregate_rows(rows, paired, matrix)
    times = time_to_quality(rows, matrix)
    outdir = args.outdir.resolve()
    figure_dir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    write_csv(outdir / "normalized_metrics.csv", rows)
    write_csv(outdir / "per_seed_trajectories.csv", rows)
    write_csv(outdir / "paired_deltas.csv", paired)
    write_csv(outdir / "paired_summary.csv", summary)
    write_csv(outdir / "budget_curves.csv", curves)
    write_csv(outdir / "time_to_quality.csv", times)
    write_csv(outdir / "figure_ready_budget_curves.csv", curves)
    write_csv(outdir / "figure_ready_per_seed_trajectories.csv", rows)
    write_csv(outdir / "figure_ready_paired_deltas.csv", paired)
    write_csv(outdir / "figure_ready_time_to_quality.csv", times)
    write_tables(summary, outdir)
    plot_budget_curves(curves, matrix, figure_dir)
    plot_trajectories(rows, matrix, figure_dir)
    plot_paired_deltas(paired, matrix, figure_dir)
    time_plot_written = plot_time_to_quality(times, matrix, figure_dir)
    write_readme(outdir, matrix, time_plot_written)
    manifest = {key: value for key, value in matrix.items() if key != "index"}
    manifest.update({
        "input_csv": str(args.input_csv.resolve()),
        "row_count": len(rows),
        "time_to_quality_plot_written": time_plot_written,
    })
    (outdir / "collector_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Validated {} metric rows; output={}".format(len(rows), outdir))


if __name__ == "__main__":
    main()
