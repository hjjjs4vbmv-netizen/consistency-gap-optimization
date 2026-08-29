#!/usr/bin/env python3
"""Build the ImageNet-64 per-seed and cross-dataset paper figures.

The ImageNet source table is a versioned snapshot of the frozen 120-cell
evaluation.  The cross-dataset figure additionally reads the repository's
sealed q256 replay matrix and prospective q128 A/B matrix.  No inferential
statistics are performed; all summaries are deterministic and descriptive.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ect-mpl-imagenet91"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGENET = ROOT / "results/imagenet91_paper_summary/imagenet_per_trajectory_source.csv"
DEFAULT_Q256 = ROOT / "results/q256_target_weight_replay_curve_seed3_5/fidkid50k-final-20260823/evaluation_results.csv"
DEFAULT_Q128 = ROOT / "results/second_q_q128_ab_v2/final/paired_results.csv"
DEFAULT_OUTDIR = ROOT / "figures/main"
DEFAULT_SUMMARY = ROOT / "results/imagenet91_paper_summary/contraction_per_seed.csv"

SEEDS_IMAGENET = (101, 102, 103)
CHECKPOINTS_IMAGENET = tuple(range(1280, 12801, 1280))
EARLY_KIMG = 6400
LATE_KIMG = 12800

INK = "#1F2937"
MUTED = "#667085"
GRID = "#D0D5DD"
IA = "#0077BB"
IB = "#EE7733"
SEED_COLORS = {101: "#0077BB", 102: "#009988", 103: "#CC3311"}
UNSTABLE_FILL = "#FDECEC"
SENSITIVITY_FILL = "#EEF6FF"


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_imagenet(path: Path) -> list[dict]:
    rows = read_csv(path)
    required = {"kimg", "seed", "method", "nfe", "fid50k", "kid50k"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"ImageNet source is missing required columns: {path}")
    parsed = [
        {
            "kimg": int(row["kimg"]),
            "seed": int(row["seed"]),
            "method": row["method"],
            "nfe": int(row["nfe"]),
            "fid50k": float(row["fid50k"]),
            "kid50k": float(row["kid50k"]),
        }
        for row in rows
    ]
    keys = {(r["kimg"], r["seed"], r["method"], r["nfe"]) for r in parsed}
    expected = {
        (kimg, seed, method, nfe)
        for kimg in CHECKPOINTS_IMAGENET
        for seed in SEEDS_IMAGENET
        for method in ("IA", "IB")
        for nfe in (1, 2)
    }
    if len(parsed) != 120 or keys != expected:
        raise ValueError(f"expected the frozen 120-cell ImageNet matrix; got {len(parsed)} rows/{len(keys)} keys")
    if not all(math.isfinite(r["fid50k"]) and math.isfinite(r["kid50k"]) for r in parsed):
        raise ValueError("ImageNet source contains a non-finite metric")
    return parsed


def imagenet_index(rows: list[dict]) -> dict[tuple[int, int, str], dict[int, float]]:
    out: dict[tuple[int, int, str], dict[int, float]] = defaultdict(dict)
    for row in rows:
        out[(row["seed"], row["nfe"], row["method"])][row["kimg"]] = row["fid50k"]
    return out


def paired_imagenet(rows: list[dict]) -> dict[tuple[int, int], dict[int, float]]:
    idx = imagenet_index(rows)
    return {
        (seed, nfe): {
            kimg: idx[(seed, nfe, "IA")][kimg] - idx[(seed, nfe, "IB")][kimg]
            for kimg in CHECKPOINTS_IMAGENET
        }
        for seed in SEEDS_IMAGENET
        for nfe in (1, 2)
    }


def write_contraction(rows: list[dict], path: Path) -> None:
    deltas = paired_imagenet(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "nfe",
        "kearly_kimg",
        "klate_kimg",
        "delta_early_fid_ia_minus_ib",
        "abs_delta_early_fid",
        "delta_late_fid_ia_minus_ib",
        "abs_delta_late_fid",
        "contraction_ratio",
        "interpretation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for seed in SEEDS_IMAGENET:
            for nfe in (1, 2):
                early = deltas[(seed, nfe)][EARLY_KIMG]
                late = deltas[(seed, nfe)][LATE_KIMG]
                writer.writerow(
                    {
                        "seed": seed,
                        "nfe": nfe,
                        "kearly_kimg": EARLY_KIMG,
                        "klate_kimg": LATE_KIMG,
                        "delta_early_fid_ia_minus_ib": f"{early:.12g}",
                        "abs_delta_early_fid": f"{abs(early):.12g}",
                        "delta_late_fid_ia_minus_ib": f"{late:.12g}",
                        "abs_delta_late_fid": f"{abs(late):.12g}",
                        "contraction_ratio": f"{abs(late) / abs(early):.12g}",
                        "interpretation": (
                            "not interpretable after trajectory instability"
                            if seed == 103
                            else "descriptive contraction on stable trajectory"
                        ),
                    }
                )


def style_trajectory_axis(axis: plt.Axes, seed: int, nfe: int) -> None:
    axis.set_yscale("log")
    axis.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.8)
    axis.tick_params(colors=INK, length=2.5)
    axis.spines["left"].set_color("#98A2B3")
    axis.spines["bottom"].set_color("#98A2B3")
    axis.text(0.02, 0.90, f"seed{seed}", transform=axis.transAxes, va="top", ha="left", weight="bold", color=INK)
    axis.set_xlim(1050, 13030)
    axis.set_xticks((1280, 6400, 12800))
    if nfe == 1:
        axis.set_ylim(4.5, 430)
    else:
        axis.set_ylim(2.7, 430)
    if seed != 103:
        axis.set_xticklabels([])
    else:
        axis.set_xlabel("Training budget K (kimg)")


def plot_trajectory_panel(parent, idx, nfe: int, label: str, title: str, rows: list[dict]) -> None:
    sub = parent[idx].subgridspec(3, 1, hspace=0.08)
    data = imagenet_index(rows)
    for row_i, seed in enumerate(SEEDS_IMAGENET):
        axis = plt.subplot(sub[row_i, 0])
        for method, color, marker, linestyle in (
            ("IA", IA, "o", "-"),
            ("IB", IB, "s", (0, (4, 2))),
        ):
            y = [data[(seed, nfe, method)][k] for k in CHECKPOINTS_IMAGENET]
            axis.plot(
                CHECKPOINTS_IMAGENET,
                y,
                color=color,
                marker=marker,
                markersize=2.9,
                linewidth=1.45,
                linestyle=linestyle,
                label=method,
            )
        if seed == 103:
            axis.axvspan(7680, 12800, color=UNSTABLE_FILL, alpha=0.95, zorder=-2)
            axis.text(
                0.985,
                0.88,
                "instability region",
                transform=axis.transAxes,
                ha="right",
                va="top",
                color="#A61B1B",
                fontsize=7.1,
            )
        style_trajectory_axis(axis, seed, nfe)
        if row_i == 1:
            axis.set_ylabel("FID-50k (log scale; lower is better)")
        if row_i == 0:
            axis.set_title(f"{label}  {title}", loc="left", weight="bold", color=INK, pad=4)
            axis.legend(loc="upper right", frameon=False, ncol=2, handlelength=2.2, borderaxespad=0.2)


def plot_delta_panel(parent, idx, rows: list[dict]) -> None:
    sub = parent[idx].subgridspec(2, 1, hspace=0.18)
    deltas = paired_imagenet(rows)
    for row_i, nfe in enumerate((1, 2)):
        axis = plt.subplot(sub[row_i, 0])
        for seed in SEEDS_IMAGENET:
            axis.plot(
                CHECKPOINTS_IMAGENET,
                [deltas[(seed, nfe)][k] for k in CHECKPOINTS_IMAGENET],
                color=SEED_COLORS[seed],
                marker="o",
                markersize=2.7,
                linewidth=1.4,
                label=f"seed{seed}",
            )
        axis.axhline(0, color="#475467", linewidth=0.8, linestyle=(0, (4, 3)))
        axis.axvspan(7680, 12800, color=UNSTABLE_FILL, alpha=0.85, zorder=-3)
        axis.set_yscale("symlog", linthresh=0.10, linscale=0.75)
        axis.set_xlim(1050, 13030)
        axis.set_xticks((1280, 6400, 12800))
        axis.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.8)
        axis.spines["left"].set_color("#98A2B3")
        axis.spines["bottom"].set_color("#98A2B3")
        axis.text(0.02, 0.91, f"NFE={nfe}", transform=axis.transAxes, va="top", weight="bold", color=INK)
        axis.set_ylabel(r"$\Delta$ FID (IA $-$ IB)")
        if row_i == 0:
            axis.set_title("C  Paired delta trajectories", loc="left", weight="bold", color=INK, pad=4)
            axis.set_xticklabels([])
            axis.legend(loc="lower left", frameon=False, ncol=3, handlelength=1.8, columnspacing=0.8)
        else:
            axis.set_xlabel("Training budget K (kimg)")
            axis.text(
                0.985,
                0.08,
                "negative favors IA",
                transform=axis.transAxes,
                ha="right",
                color=MUTED,
                fontsize=7.2,
            )


def plot_endpoint_table(parent, idx, rows: list[dict]) -> None:
    axis = plt.subplot(parent[idx])
    axis.axis("off")
    axis.set_title("D  Late endpoint (12,800 kimg)", loc="left", weight="bold", color=INK, pad=7)
    data = imagenet_index(rows)
    columns = ["NFE", "Seed", "IA", "IB", r"$\Delta$"]
    frozen = []
    for nfe in (1, 2):
        for seed in SEEDS_IMAGENET:
            ia = data[(seed, nfe, "IA")][LATE_KIMG]
            ib = data[(seed, nfe, "IB")][LATE_KIMG]
            frozen.append([str(nfe), str(seed), f"{ia:.3f}", f"{ib:.3f}", f"{ia - ib:+.3f}"])
    stable = []
    for nfe in (1, 2):
        ia = statistics.mean(data[(seed, nfe, "IA")][LATE_KIMG] for seed in (101, 102))
        ib = statistics.mean(data[(seed, nfe, "IB")][LATE_KIMG] for seed in (101, 102))
        stable.append([str(nfe), "101/102", f"{ia:.3f}", f"{ib:.3f}", f"{ia - ib:+.3f}"])

    axis.text(0.01, 0.94, "Frozen analysis — all three seeds (no exclusions)", transform=axis.transAxes, weight="bold", color=INK)
    t1 = axis.table(cellText=frozen, colLabels=columns, cellLoc="center", colLoc="center", bbox=(0.01, 0.43, 0.98, 0.47))
    t1.auto_set_font_size(False)
    t1.set_fontsize(7.5)
    for (r, c), cell in t1.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        cell.set_linewidth(0.55)
        if r == 0:
            cell.set_facecolor("#F2F4F7")
            cell.set_text_props(weight="bold", color=INK)
        elif frozen[r - 1][1] == "103":
            cell.set_facecolor(UNSTABLE_FILL)

    box = FancyBboxPatch(
        (0.005, 0.055),
        0.99,
        0.31,
        boxstyle="round,pad=0.006,rounding_size=0.008",
        transform=axis.transAxes,
        facecolor=SENSITIVITY_FILL,
        edgecolor="#84ADFF",
        linewidth=0.8,
        zorder=-1,
    )
    axis.add_patch(box)
    axis.text(0.02, 0.33, "Descriptive sensitivity — stable seeds101/102 only", transform=axis.transAxes, weight="bold", color="#1849A9")
    t2 = axis.table(cellText=stable, colLabels=columns, cellLoc="center", colLoc="center", bbox=(0.02, 0.10, 0.96, 0.20))
    t2.auto_set_font_size(False)
    t2.set_fontsize(7.5)
    for (r, c), cell in t2.get_celld().items():
        cell.set_edgecolor("#B2CCFF")
        cell.set_linewidth(0.55)
        cell.set_facecolor("#FFFFFF" if r else "#D1E9FF")
        if r == 0:
            cell.set_text_props(weight="bold", color="#1849A9")
    axis.text(0.02, 0.068, "Sensitivity values are two-seed means; they are not the frozen three-seed estimand.", transform=axis.transAxes, fontsize=7.1, color="#1849A9")


def render_imagenet_main(rows: list[dict], path: Path) -> None:
    figure = plt.figure(figsize=(13.8, 9.7), constrained_layout=False)
    outer = figure.add_gridspec(2, 2, left=0.055, right=0.985, bottom=0.105, top=0.92, wspace=0.16, hspace=0.23)
    plot_trajectory_panel(outer, (0, 0), 1, "A", "NFE=1 per-seed trajectories", rows)
    plot_trajectory_panel(outer, (0, 1), 2, "B", "NFE=2 per-seed trajectories", rows)
    plot_delta_panel(outer, (1, 0), rows)
    plot_endpoint_table(outer, (1, 1), rows)
    figure.suptitle("ImageNet-64 paired quality trajectories: early separation, late heterogeneity", x=0.055, y=0.975, ha="left", fontsize=15, weight="bold", color=INK)
    figure.text(
        0.055,
        0.947,
        "Each line is one frozen training seed. No mean trajectory is shown. Shading marks the post-7,680 region in which seed103 becomes unstable.",
        ha="left",
        fontsize=9,
        color=MUTED,
    )
    figure.text(
        0.055,
        0.035,
        "Note. FID-50k uses 50,000 fixed-seed samples; lower is better. Panels A/B use log y-scales. Panel D keeps the frozen all-seed endpoint analysis visually separate from the post hoc seeds101/102 descriptive sensitivity summary. Seed103 is retained in the frozen display; its post-instability contraction ratio is not treatment-interpretable.",
        ha="left",
        va="bottom",
        fontsize=8.1,
        color=INK,
        wrap=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def read_q256(path: Path) -> tuple[dict[tuple[int, int], float], dict[int, float], float]:
    rows = read_csv(path)
    values = {
        (int(r["seed"]), int(r["budget_kimg"]), int(r["nfe"]), r["arm"]): float(r["fid50k_full"])
        for r in rows
    }
    budgets = (256, 384, 512, 640, 768, 896, 1024)
    seed_delta = {
        (seed, kimg): values[(seed, kimg, 2, "B")] - values[(seed, kimg, 2, "A")]
        for seed in (3, 4, 5)
        for kimg in budgets
    }
    mean_delta = {k: statistics.mean(seed_delta[(s, k)] for s in (3, 4, 5)) for k in budgets}
    aulc = []
    for seed in (3, 4, 5):
        arm_area = {}
        for arm in ("A", "B"):
            y = [math.log(values[(seed, k, 2, arm)]) for k in budgets]
            arm_area[arm] = sum((budgets[i + 1] - budgets[i]) * (y[i + 1] + y[i]) / 2 for i in range(6)) / (budgets[-1] - budgets[0])
        aulc.append(arm_area["B"] - arm_area["A"])
    return seed_delta, mean_delta, statistics.mean(aulc)


def read_q128(path: Path) -> tuple[dict[tuple[int, int], float], dict[int, float]]:
    rows = read_csv(path)
    selected = [r for r in rows if int(r["nfe"]) == 1]
    seed_delta = {(int(r["seed"]), int(r["budget_kimg"])): float(r["fid_delta_B_minus_A"]) for r in selected}
    budgets = (512, 640, 768, 896, 1024)
    mean_delta = {k: statistics.mean(seed_delta[(s, k)] for s in (3, 4, 5)) for k in budgets}
    return seed_delta, mean_delta


def style_synthesis_axis(axis: plt.Axes, title: str, subtitle: str, linthresh: float) -> None:
    axis.axhline(0, color="#475467", linewidth=0.9, linestyle=(0, (4, 3)), zorder=0)
    axis.set_yscale("symlog", linthresh=linthresh, linscale=0.8)
    axis.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.85)
    axis.spines["left"].set_color("#98A2B3")
    axis.spines["bottom"].set_color("#98A2B3")
    axis.set_title(title, loc="left", weight="bold", color=INK, pad=17)
    axis.text(0, 1.015, subtitle, transform=axis.transAxes, color=MUTED, fontsize=7.7, va="bottom")


def render_cross_dataset(imagenet_rows: list[dict], q256_path: Path, q128_path: Path, path: Path) -> None:
    q256_seed, q256_mean, q256_aulc = read_q256(q256_path)
    q128_seed, q128_mean = read_q128(q128_path)
    imagenet_delta = paired_imagenet(imagenet_rows)
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.9))
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.19, top=0.78, wspace=0.22)

    budgets = (256, 384, 512, 640, 768, 896, 1024)
    for seed, color in zip((3, 4, 5), ("#84ADFF", "#6CE9A6", "#FDA29B")):
        axes[0].plot(budgets, [q256_seed[(seed, k)] for k in budgets], color=color, linewidth=1.1, marker="o", markersize=2.5, alpha=0.9, label=f"seed{seed}")
    axes[0].plot(budgets, [q256_mean[k] for k in budgets], color=INK, linewidth=2.4, marker="D", markersize=4.0, label="3-seed mean")
    style_synthesis_axis(axes[0], "A  q256", "NFE=2; paired difference B − A", 0.1)
    axes[0].set_xticks((256, 640, 1024))
    axes[0].set_xlabel("Training budget (kimg)")
    axes[0].set_ylabel("Paired FID-50k difference\n(negative favors first-named arm)")
    axes[0].legend(loc="upper right", frameon=False, fontsize=7.0, handlelength=1.6, ncol=2, columnspacing=0.7)
    axes[0].text(0.04, 0.06, f"large full-curve gap\nmean log-FID AULC Δ = {q256_aulc:+.3f}\nsmall endpoint gap = {q256_mean[1024]:+.3f}", transform=axes[0].transAxes, color=INK, fontsize=8.1, bbox={"facecolor": "white", "edgecolor": "#D0D5DD", "boxstyle": "round,pad=0.35"})

    budgets128 = (512, 640, 768, 896, 1024)
    for seed, color in zip((3, 4, 5), ("#84ADFF", "#6CE9A6", "#FDA29B")):
        axes[1].plot(budgets128, [q128_seed[(seed, k)] for k in budgets128], color=color, linewidth=1.1, marker="o", markersize=2.5, alpha=0.9, label=f"seed{seed}")
    axes[1].plot(budgets128, [q128_mean[k] for k in budgets128], color=INK, linewidth=2.4, marker="D", markersize=4.0, label="3-seed mean")
    style_synthesis_axis(axes[1], "B  q128", "NFE=1; paired difference B − A", 0.05)
    axes[1].set_xticks((512, 640, 1024))
    axes[1].set_xlabel("Training budget (kimg)")
    axes[1].legend(loc="upper right", frameon=False, fontsize=7.0, handlelength=1.6, ncol=2, columnspacing=0.7)
    axes[1].text(0.04, 0.06, f"early harmful {q128_mean[512]:+.2f}\nneutral at 640 {q128_mean[640]:+.3f}\nsmall late effect {q128_mean[1024]:+.3f}", transform=axes[1].transAxes, color=INK, fontsize=8.1, bbox={"facecolor": "white", "edgecolor": "#D0D5DD", "boxstyle": "round,pad=0.35"})

    stable_mean = {k: statistics.mean(imagenet_delta[(s, 2)][k] for s in (101, 102)) for k in CHECKPOINTS_IMAGENET}
    for seed in SEEDS_IMAGENET:
        axes[2].plot(
            CHECKPOINTS_IMAGENET,
            [imagenet_delta[(seed, 2)][k] for k in CHECKPOINTS_IMAGENET],
            color=SEED_COLORS[seed],
            linewidth=1.25,
            marker="o",
            markersize=2.5,
            linestyle=(0, (3, 2)) if seed == 103 else "-",
            alpha=0.92,
            label=f"seed{seed}",
        )
    axes[2].plot(CHECKPOINTS_IMAGENET, [stable_mean[k] for k in CHECKPOINTS_IMAGENET], color=INK, linewidth=2.4, marker="D", markersize=4.0, label="101/102 descriptive mean")
    axes[2].axvspan(7680, 12800, color=UNSTABLE_FILL, alpha=0.85, zorder=-3)
    style_synthesis_axis(axes[2], "C  ImageNet-64", "NFE=2; paired difference IA − IB", 0.02)
    axes[2].set_xticks((1280, 6400, 12800))
    axes[2].set_xlabel("Training budget (kimg)")
    axes[2].text(0.04, 0.06, f"uniform early IA advantage\nstable late near-equivalence {stable_mean[12800]:+.4f}\nseed103 unstable: {imagenet_delta[(103, 2)][12800]:+.1f}", transform=axes[2].transAxes, color=INK, fontsize=8.1, bbox={"facecolor": "white", "edgecolor": "#D0D5DD", "boxstyle": "round,pad=0.35"})
    axes[2].legend(loc="upper right", frameon=False, fontsize=7.1, handlelength=1.8)

    figure.suptitle("Cross-dataset quality emergence: finite-budget effects converge, reverse, or destabilize", x=0.065, y=0.95, ha="left", fontsize=14.5, weight="bold", color=INK)
    figure.text(0.065, 0.875, "Thin colored lines are individual training seeds; the dark line is the stated descriptive mean. Symlog y-scales preserve both near-zero endpoints and large early/unstable excursions.", color=MUTED, fontsize=8.5)
    figure.text(0.065, 0.045, "Note. Panels use different arm contrasts and NFE settings, stated above each axis; the figure synthesizes trajectory shape rather than a pooled effect. q256 AULC is a deterministic normalized trapezoidal area under log FID. The ImageNet 101/102 mean is a post hoc sensitivity summary; seed103 remains displayed and is not interpreted after instability.", color=INK, fontsize=8.0, wrap=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagenet", type=Path, default=DEFAULT_IMAGENET)
    parser.add_argument("--q256", type=Path, default=DEFAULT_Q256)
    parser.add_argument("--q128", type=Path, default=DEFAULT_Q128)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--contraction", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    configure_style()
    imagenet_rows = read_imagenet(args.imagenet.resolve())
    write_contraction(imagenet_rows, args.contraction.resolve())
    render_imagenet_main(imagenet_rows, args.outdir.resolve() / "imagenet_per_seed_trajectories.pdf")
    render_cross_dataset(imagenet_rows, args.q256.resolve(), args.q128.resolve(), args.outdir.resolve() / "cross_dataset_quality_emergence.pdf")
    print(f"Wrote figures to {args.outdir.resolve()} and contraction table to {args.contraction.resolve()}")


if __name__ == "__main__":
    main()
