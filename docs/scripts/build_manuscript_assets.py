#!/usr/bin/env python3
"""Build traceable source tables and manuscript figures from audited CSVs.

The script does not interpolate between checkpoints.  A time-to-quality event
is the first observed budget at which FID-50k@NFE1 is at most 10.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ect_manuscript_matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "docs" / "figure_source_data"
FIGURE_DIR = ROOT / "docs" / "figures"

INPUTS = {
    "formal": ROOT
    / "results/q256_target_weight_replay_curve_seed3_5/"
    "fidkid50k-final-20260823/evaluation_results.csv",
    "secondary_ab": ROOT
    / "analysis/q256_target_weight_factorial/"
    "seed6_7_ab_128k_streaming_results/reports/"
    "learning_curve_seed6_7_ab_nfe1_nfe2_128k.csv",
    "secondary_factorial": ROOT
    / "analysis/q256_target_weight_factorial/"
    "seed14_18_learning_curve_results/q256_seed14_18_learning_curve_results.csv",
}

GROUP_LABELS = {
    "formal": "formal replay (seeds 3--5)",
    "secondary_ab": "secondary A/B (seeds 6--7)",
    "secondary_factorial": "secondary factorial (seeds 14--18)",
}

PROVENANCE = {
    "formal": {
        "evidence_status": "formal deterministic replay",
        "source_pr": 79,
        "source_commit": "deddffa61fc6010c12f8ccfe59ed9a1d87dd80e3",
        "source_file": str(INPUTS["formal"].relative_to(ROOT)),
    },
    "secondary_ab": {
        "evidence_status": "descriptive secondary precision extension",
        "source_pr": 78,
        "source_commit": "a6ed00df6f3ba7c5ebc9c6105a84bb693ecf9371",
        "source_file": str(INPUTS["secondary_ab"].relative_to(ROOT)),
    },
    "secondary_factorial": {
        "evidence_status": "post-preregistration secondary sensitivity evidence",
        "source_pr": 80,
        "source_commit": "2ed14cb1829ad2cda6c8db7ca5485a7bd055e075",
        "source_file": str(INPUTS["secondary_factorial"].relative_to(ROOT)),
    },
}

COLORS = {
    "A": "#5B6770",
    "B": "#0072B2",
    "earlier": "#0072B2",
    "tie": "#6C757D",
    "later": "#D55E00",
    "censored": "#8E44AD",
}


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_all(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(FIGURE_DIR / f"{stem}.{suffix}", **kwargs)


def load_unified() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    keep = [
        "evidence_group",
        "evidence_status",
        "source_pr",
        "source_commit",
        "source_file",
        "seed",
        "arm",
        "budget_kimg",
        "nfe",
        "fid50k_full",
        "kid50k_full",
    ]
    for group, path in INPUTS.items():
        frame = pd.read_csv(path)
        required = {
            "seed",
            "arm",
            "budget_kimg",
            "nfe",
            "fid50k_full",
            "kid50k_full",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame = frame.assign(evidence_group=group, **PROVENANCE[group])[keep]
        frames.append(frame)

    unified = pd.concat(frames, ignore_index=True)
    keys = ["evidence_group", "seed", "arm", "budget_kimg", "nfe"]
    if unified.duplicated(keys).any():
        raise ValueError("Duplicate evidence rows after source concatenation")
    if not (unified["fid50k_full"].notna().all() and unified["kid50k_full"].notna().all()):
        raise ValueError("Missing FID/KID values in the unified evidence table")
    unified = unified.sort_values(keys).reset_index(drop=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    unified.to_csv(SOURCE_DIR / "q256_unified_learning_curve_source.csv", index=False)
    return unified


def build_time_to_quality(unified: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ab = unified[(unified["nfe"] == 1) & unified["arm"].isin(["A", "B"])]
    for (group, seed), pair in ab.groupby(["evidence_group", "seed"], sort=True):
        record: dict[str, object] = {
            "evidence_group": group,
            "seed": int(seed),
            "threshold_fid50k_nfe1": threshold,
        }
        for arm in ("A", "B"):
            arm_rows = pair[pair["arm"] == arm].sort_values("budget_kimg")
            observed = arm_rows[arm_rows["fid50k_full"] <= threshold]
            record[f"tau_{arm}_kimg"] = (
                pd.NA if observed.empty else int(observed.iloc[0]["budget_kimg"])
            )
            record[f"last_{arm}_budget_kimg"] = int(arm_rows.iloc[-1]["budget_kimg"])
            record[f"last_{arm}_fid50k_nfe1"] = float(arm_rows.iloc[-1]["fid50k_full"])

        tau_a = record["tau_A_kimg"]
        tau_b = record["tau_B_kimg"]
        if pd.isna(tau_a) or pd.isna(tau_b):
            classification = "censored"
        elif tau_b < tau_a:
            classification = "B earlier"
        elif tau_b == tau_a:
            classification = "tie"
        else:
            classification = "B later"
        record["classification"] = classification
        rows.append(record)

    result = pd.DataFrame(rows).sort_values(["evidence_group", "seed"]).reset_index(drop=True)
    result.to_csv(SOURCE_DIR / "figure3_time_to_quality_source.csv", index=False)
    return result


def rounded_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float,
                text: str, facecolor: str, edgecolor: str = "#30343B",
                fontsize: float = 8.2) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=0.9,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
          color: str = "#30343B", style: str = "-") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            linestyle=style,
            color=color,
        )
    )


def figure1_composite_intervention() -> None:
    fig, ax = plt.subplots(figsize=(7.05, 2.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rounded_box(ax, (0.01, 0.35), 0.15, 0.29,
                "Pair spacing\n$\\Delta_1 \\rightarrow \\Delta_g$\nprobe: $g=1.10$", "#E9ECEF", fontsize=7.8)
    rounded_box(ax, (0.21, 0.66), 0.21, 0.22,
                "Detached target endpoint\n$r_1 \\rightarrow r_g$", "#D9EDF7", "#0072B2")
    rounded_box(ax, (0.21, 0.12), 0.21, 0.22,
                "Explicit denominator\n$1/\\Delta_1 \\rightarrow 1/\\Delta_g$", "#FCE8D5", "#D55E00")
    rounded_box(ax, (0.48, 0.37), 0.17, 0.24,
                "Exact matched-state\nfactorization", "#E8F5E9", "#2E7D32")
    rounded_box(ax, (0.70, 0.66), 0.13, 0.22,
                "Trajectory A\n$\\theta^A_k,z^A_k$", "#F5F5F5", fontsize=7.7)
    rounded_box(ax, (0.70, 0.12), 0.13, 0.22,
                "Trajectory B\n$\\theta^B_k,z^B_k$", "#F5F5F5", fontsize=7.7)
    rounded_box(ax, (0.87, 0.31), 0.12, 0.38,
                "Time-to-quality\nand horizon-\ndependent ranking", "#FDEDEC", "#8B1E1E", fontsize=7.3)

    arrow(ax, (0.16, 0.49), (0.21, 0.77), "#0072B2")
    arrow(ax, (0.16, 0.49), (0.21, 0.23), "#D55E00")
    arrow(ax, (0.42, 0.77), (0.48, 0.52), "#0072B2")
    arrow(ax, (0.42, 0.23), (0.48, 0.46), "#D55E00")
    arrow(ax, (0.65, 0.50), (0.70, 0.77), "#30343B")
    arrow(ax, (0.65, 0.48), (0.70, 0.23), "#30343B")
    arrow(ax, (0.83, 0.77), (0.87, 0.57), "#30343B")
    arrow(ax, (0.83, 0.23), (0.87, 0.43), "#30343B")

    ax.text(0.565, 0.78, "same $\\theta$, batch, RNG", ha="center", fontsize=6.9,
            color="#2E7D32")
    ax.text(0.565, 0.18, "separate finite training", ha="center", fontsize=6.9,
            color="#30343B")
    ax.set_title("Pair spacing is a composite local intervention, not a global training equivalence",
                 loc="left", fontweight="bold", pad=3)
    save_all(fig, "figure1_composite_intervention")
    plt.close(fig)


def figure3_time_to_quality(ttq: pd.DataFrame) -> None:
    ordered = ttq.copy()
    group_order = ["formal", "secondary_ab", "secondary_factorial"]
    ordered["group_rank"] = ordered["evidence_group"].map({g: i for i, g in enumerate(group_order)})
    ordered = ordered.sort_values(["group_rank", "seed"]).reset_index(drop=True)
    y = list(range(len(ordered)))[::-1]

    fig, ax = plt.subplots(figsize=(7.05, 3.4))
    for idx, row in ordered.iterrows():
        yy = y[idx]
        tau_a, tau_b = row["tau_A_kimg"], row["tau_B_kimg"]
        if not pd.isna(tau_a) and not pd.isna(tau_b):
            ax.plot([tau_a, tau_b], [yy, yy], color="#C8CDD2", lw=2.0, zorder=1)
        for arm, marker in (("A", "o"), ("B", "s")):
            tau = row[f"tau_{arm}_kimg"]
            if pd.isna(tau):
                x = row[f"last_{arm}_budget_kimg"]
                ax.scatter(x, yy, marker=marker, s=42, facecolors="none",
                           edgecolors=COLORS[arm], linewidths=1.3, zorder=3)
                ax.annotate("$>$", (x, yy), xytext=(5, -1), textcoords="offset points",
                            color=COLORS[arm], fontsize=8)
            else:
                ax.scatter(tau, yy, marker=marker, s=38, color=COLORS[arm], zorder=3)
        class_key = str(row["classification"]).replace("B ", "")
        ax.text(1048, yy, str(row["classification"]), va="center", ha="left",
                fontsize=7.2, color=COLORS[class_key])

    labels = [f"seed {int(seed)}" for seed in ordered["seed"]]
    ax.set_yticks(y, labels)
    ax.set_xlim(330, 1160)
    ax.set_xticks([384, 512, 640, 768, 896, 1024])
    ax.set_xlabel("First observed training budget with FID-50k@NFE1 $\\leq 10$ (kimg)")
    ax.grid(axis="x", color="#E2E5E8", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    boundaries = []
    cursor = 0
    for group in group_order:
        n = int((ordered["evidence_group"] == group).sum())
        if cursor > 0:
            boundaries.append(len(ordered) - cursor - 0.5)
        cursor += n
    for boundary in boundaries:
        ax.axhline(boundary, color="#B8BEC4", linewidth=0.8)

    ax.scatter([], [], marker="o", s=38, color=COLORS["A"], label="A: baseline spacing")
    ax.scatter([], [], marker="s", s=38, color=COLORS["B"], label="B: enlarged spacing")
    ax.scatter([], [], marker="o", s=42, facecolors="none", edgecolors="#333333",
               label="open marker: threshold not reached")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3, frameon=False)

    group_centers = {}
    for group in group_order:
        positions = [y[i] for i in ordered.index[ordered["evidence_group"] == group]]
        group_centers[group] = sum(positions) / len(positions)
    for group, center in group_centers.items():
        ax.text(340, center + 0.38, GROUP_LABELS[group], fontsize=6.8,
                color="#5A6268", ha="left", va="bottom")

    ax.set_title("Larger spacing sometimes advances time-to-quality, but not for every seed",
                 loc="left", fontweight="bold", pad=20)
    save_all(fig, "figure3_time_to_quality_seed_resolved")
    plt.close(fig)


def main() -> None:
    configure_plotting()
    unified = load_unified()
    ttq = build_time_to_quality(unified)
    figure1_composite_intervention()
    figure3_time_to_quality(ttq)

    counts = ttq["classification"].value_counts().to_dict()
    expected = {"B earlier": 5, "tie": 2, "B later": 2, "censored": 1}
    if counts != expected:
        raise AssertionError(f"Unexpected time-to-quality classification: {counts}")
    print(f"wrote {len(unified)} unified metric rows")
    print(f"time-to-quality classifications: {counts}")


if __name__ == "__main__":
    main()
