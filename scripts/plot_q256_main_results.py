#!/usr/bin/env python3
"""Render the two main-text figures for the q=256 formal paired results.

The script reads only the versioned seed-level paired-difference table and
writes SVG and PNG copies of both figures.  It does not recalculate metrics or
perform inferential tests.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ect-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


INK = "#1F2937"
GRID = "#D1D5DB"
FIXED = "#FFFFFF"
MEAN = "#111827"
SEED_COLORS = {3: "#2563EB", 4: "#C0841A", 5: "#C2416C"}
METRIC_LABELS = {"kid50k_full": "KID-50k", "fid50k_full": "FID-50k"}


def read_pairs(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"training_seed", "nfe", "metric", "fixed_value", "global_only_value", "delta"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"paired-differences CSV is missing required fields: {path}")
    parsed = []
    for row in rows:
        parsed.append({
            "training_seed": int(row["training_seed"]),
            "nfe": int(row["nfe"]),
            "metric": row["metric"],
            "fixed_value": float(row["fixed_value"]),
            "global_only_value": float(row["global_only_value"]),
            "delta": float(row["delta"]),
        })
    expected = {(metric, nfe, seed) for metric in METRIC_LABELS for nfe in (1, 2) for seed in (3, 4, 5)}
    observed = {(row["metric"], row["nfe"], row["training_seed"]) for row in parsed}
    if observed != expected:
        raise SystemExit(f"expected complete q=256 2-metric × 2-NFE × 3-seed matrix; got {sorted(observed)}")
    return parsed


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("#FFFFFF")
    axis.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#9CA3AF")
    axis.spines["bottom"].set_color("#9CA3AF")
    axis.tick_params(colors=INK, labelsize=10)
    axis.yaxis.label.set_color(INK)


def data_index(rows: list[dict]) -> dict[tuple[str, int], list[dict]]:
    indexed: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        indexed[(row["metric"], row["nfe"])].append(row)
    for group in indexed.values():
        group.sort(key=lambda item: item["training_seed"])
    return indexed


def padded_limits(values: list[float]) -> tuple[float, float]:
    low, high = min(values), max(values)
    span = high - low
    pad = span * 0.18 if span else max(abs(low) * 0.12, 0.02)
    return low - pad, high + pad


def render_figure1(indexed: dict[tuple[str, int], list[dict]], outdir: Path) -> None:
    """Four slope panels: fixed to global-only for every seed and endpoint."""
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 8.4))
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.08, top=0.835, hspace=0.22, wspace=0.15)
    panels = [("kid50k_full", 1), ("fid50k_full", 1), ("kid50k_full", 2), ("fid50k_full", 2)]
    for axis, (metric, nfe) in zip(axes.flat, panels):
        group = indexed[(metric, nfe)]
        all_values = [value for row in group for value in (row["fixed_value"], row["global_only_value"])]
        for row in group:
            color = SEED_COLORS[row["training_seed"]]
            axis.plot([0, 1], [row["fixed_value"], row["global_only_value"]], color=color, linewidth=2.2, zorder=2)
            axis.scatter(0, row["fixed_value"], s=70, facecolor=FIXED, edgecolor=color, linewidth=2.1, zorder=3)
            axis.scatter(1, row["global_only_value"], s=70, facecolor=color, edgecolor=color, linewidth=1.2, zorder=3)
        axis.set_xlim(-0.23, 1.23)
        axis.set_ylim(*padded_limits(all_values))
        axis.set_xticks([0, 1], ["Fixed", "Global-only\n(g=1.10)"])
        axis.set_ylabel(METRIC_LABELS[metric], fontsize=11)
        axis.set_title(f"{METRIC_LABELS[metric]} · NFE={nfe}", loc="left", fontsize=12, fontweight="bold", color=INK, pad=9)
        axis.text(0.02, 0.04, "3/3 paired comparisons favor global-only", transform=axis.transAxes, fontsize=9, color=INK)
        style_axis(axis)
    legend = [
        Line2D([0], [0], color=SEED_COLORS[seed], marker="o", markersize=7, linewidth=2, label=f"Training seed {seed}")
        for seed in (3, 4, 5)
    ]
    figure.legend(handles=legend, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.58, 0.972), fontsize=10)
    figure.suptitle("Figure 1. Per-seed paired comparison at 256 kimg", x=0.02, y=0.99, ha="left", fontsize=16, fontweight="bold", color=INK)
    figure.text(0.02, 0.902, "50k generated samples per checkpoint; lower values are better. Open markers: fixed; filled markers: global-only. Panel-specific y-scales.", fontsize=10, color=INK)
    save_figure(figure, outdir, "figure1_per_seed_paired_comparison")


def render_figure2(indexed: dict[tuple[str, int], list[dict]], outdir: Path) -> None:
    """Seed-level deltas with mean and SD, emphasizing NFE=2 heterogeneity."""
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 5.7))
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.12, top=0.80, wspace=0.17)
    for axis, metric in zip(axes, ("kid50k_full", "fid50k_full")):
        by_seed = {seed: {} for seed in (3, 4, 5)}
        for nfe in (1, 2):
            for row in indexed[(metric, nfe)]:
                by_seed[row["training_seed"]][nfe] = row["delta"]
        all_deltas = []
        for seed, values in by_seed.items():
            color = SEED_COLORS[seed]
            points = [values[1], values[2]]
            all_deltas.extend(points)
            axis.plot([1, 2], points, color=color, linewidth=1.8, alpha=0.82, zorder=2)
            axis.scatter([1, 2], points, s=65, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        means = []
        sds = []
        for nfe in (1, 2):
            deltas = [row["delta"] for row in indexed[(metric, nfe)]]
            means.append(statistics.mean(deltas))
            sds.append(statistics.stdev(deltas))
        axis.errorbar([1, 2], means, yerr=sds, fmt="D", color=MEAN, markersize=7, capsize=5, linewidth=1.6, zorder=4, label="Mean ± sample SD")
        axis.axhline(0, color="#6B7280", linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
        axis.set_xlim(0.66, 2.34)
        axis.set_ylim(*padded_limits(all_deltas + [0]))
        axis.set_xticks([1, 2], ["NFE=1", "NFE=2"])
        axis.set_ylabel(f"Δ {METRIC_LABELS[metric]} (global-only − fixed)", fontsize=10.5)
        axis.set_title(f"{METRIC_LABELS[metric]} paired deltas", loc="left", fontsize=12, fontweight="bold", color=INK, pad=9)
        near_flat = by_seed[5][2]
        annotation = f"Seed 5\n{near_flat:.3g}"
        offset = 0.10 * (axis.get_ylim()[1] - axis.get_ylim()[0])
        axis.annotate(annotation, xy=(2, near_flat), xytext=(2.12, near_flat + offset), fontsize=9.5, color=INK, ha="left", va="bottom", arrowprops={"arrowstyle": "-", "color": "#6B7280", "lw": 1.0})
        style_axis(axis)
    seed_legend = [
        Line2D([0], [0], color=SEED_COLORS[seed], marker="o", markersize=7, linewidth=1.8, label=f"Seed {seed}")
        for seed in (3, 4, 5)
    ]
    seed_legend.append(Line2D([0], [0], color=MEAN, marker="D", markersize=7, linewidth=1.6, label="Mean ± sample SD"))
    figure.legend(handles=seed_legend, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.60, 0.972), fontsize=10)
    figure.suptitle("Figure 2. Mean paired delta and between-seed variation", x=0.02, y=0.99, ha="left", fontsize=16, fontweight="bold", color=INK)
    figure.text(0.02, 0.902, "Points are independent training-seed deltas; negative values favor global-only. Whiskers are sample SD, not confidence intervals. The NFE=2 effect is heterogeneous because seed 5 is near flat.", fontsize=9.7, color=INK)
    save_figure(figure, outdir, "figure2_mean_delta_seed_variation")


def save_figure(figure: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        figure.savefig(outdir / f"{stem}.{extension}", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("results/q256_256k_formal/paired_differences.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results/q256_256k_formal/figures"))
    args = parser.parse_args()
    indexed = data_index(read_pairs(args.pairs.resolve()))
    render_figure1(indexed, args.outdir.resolve())
    render_figure2(indexed, args.outdir.resolve())
    print(f"Wrote Figure 1 and Figure 2 to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
