#!/usr/bin/env python3
"""Render paper Asset C: seed-resolved four-arm FID dispersion contraction.

For each training seed ``s`` and checkpoint budget ``K``, the renderer computes

    S_s(K) = max_a FID_{s,a}(K) - min_a FID_{s,a}(K),  a in {A, B, C, D}.

Every seed remains a visible thin line.  Mean and median, when shown, are
explicit summaries over those seed-level values rather than replacements for
them.
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

try:  # Supports both ``python scripts/...`` and package-level tests.
    from .paper_asset_data import complete_matrix, fail, load_trajectory, sha256
except ImportError:
    from paper_asset_data import complete_matrix, fail, load_trajectory, sha256


PREFIX = "render_paper_asset_c"
SEED_COLORS = ("#2563EB", "#C2416C", "#5B8C5A", "#7C5AC8", "#C98212", "#64748B")
SEED_MARKERS = ("o", "s", "^", "D", "P", "X")
INK = "#1F2937"
GRID = "#D1D5DB"


def read_config(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(PREFIX, "cannot read frozen arm config {}: {}".format(path, exc))
        raise exc
    required = {
        "schema_version", "asset", "metric_name", "nfe", "evaluation_contract",
        "analysis_track", "budgets_kimg", "arms",
    }
    missing = required - set(payload)
    if missing:
        fail(PREFIX, "frozen arm config is missing {}".format(sorted(missing)))
    if payload["schema_version"] != 1 or payload["asset"] != "C":
        fail(PREFIX, "arm config must declare schema_version=1 and asset='C'")
    if not isinstance(payload["metric_name"], str) or not payload["metric_name"].strip():
        fail(PREFIX, "metric_name must be a non-empty string")
    if not isinstance(payload["nfe"], int) or payload["nfe"] < 1:
        fail(PREFIX, "nfe must be a positive integer")
    if (not isinstance(payload["evaluation_contract"], str) or not payload["evaluation_contract"].strip()
            or not isinstance(payload["analysis_track"], str) or not payload["analysis_track"].strip()):
        fail(PREFIX, "evaluation_contract and analysis_track must be non-empty strings")
    if not isinstance(payload["budgets_kimg"], list) or not payload["budgets_kimg"]:
        fail(PREFIX, "budgets_kimg must be a non-empty ascending list")
    try:
        budgets = tuple(int(value) for value in payload["budgets_kimg"])
    except (TypeError, ValueError) as exc:
        fail(PREFIX, "budgets_kimg must contain integers")
        raise exc
    if tuple(sorted(budgets)) != budgets or len(set(budgets)) != len(budgets) or any(value <= 0 for value in budgets):
        fail(PREFIX, "budgets_kimg must contain unique, strictly ascending positive values")
    payload["budgets_kimg"] = list(budgets)
    arms = payload["arms"]
    if (not isinstance(arms, dict) or set(arms) != {"A", "B", "C", "D"}
            or any(not isinstance(value, str) or not value.strip() for value in arms.values())):
        fail(PREFIX, "arms must map exactly A, B, C, and D to non-empty method names")
    if len(set(arms.values())) != 4:
        fail(PREFIX, "each of the four arms must map to a distinct method")
    return payload


def save_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def format_value(value: float | None) -> str:
    return "" if value is None else "{:.12g}".format(value)


def calculate(index: dict, config: dict, seeds: list[int]) -> tuple[list[dict], list[dict], list[dict]]:
    budgets = tuple(config["budgets_kimg"])
    per_seed, summary, contraction = [], [], []
    for budget in budgets:
        values = []
        for seed in seeds:
            arm_values = {
                arm: index[(method, seed, float(budget))]["metric_value"]
                for arm, method in config["arms"].items()
            }
            min_arm = min(arm_values, key=arm_values.get)
            max_arm = max(arm_values, key=arm_values.get)
            dispersion = arm_values[max_arm] - arm_values[min_arm]
            values.append(dispersion)
            per_seed.append({
                "training_seed": seed,
                "budget_kimg": budget,
                "dispersion_fid": format_value(dispersion),
                "min_fid": format_value(arm_values[min_arm]),
                "max_fid": format_value(arm_values[max_arm]),
                "min_arm": min_arm,
                "max_arm": max_arm,
                "metric_name": config["metric_name"],
                "nfe": config["nfe"],
            })
        summary.append({
            "budget_kimg": budget,
            "seed_count": len(values),
            "mean_dispersion_fid": format_value(statistics.mean(values)),
            "median_dispersion_fid": format_value(statistics.median(values)),
            "min_seed_dispersion_fid": format_value(min(values)),
            "max_seed_dispersion_fid": format_value(max(values)),
            "metric_name": config["metric_name"],
            "nfe": config["nfe"],
        })
    per_seed_index = {(row["training_seed"], row["budget_kimg"]): row for row in per_seed}
    start, final = budgets[0], budgets[-1]
    for seed in seeds:
        start_value = float(per_seed_index[(seed, start)]["dispersion_fid"])
        final_value = float(per_seed_index[(seed, final)]["dispersion_fid"])
        change = final_value - start_value
        contraction.append({
            "training_seed": seed,
            "start_budget_kimg": start,
            "end_budget_kimg": final,
            "start_dispersion_fid": format_value(start_value),
            "end_dispersion_fid": format_value(final_value),
            "change_end_minus_start_fid": format_value(change),
            "relative_change_pct": format_value(100 * change / start_value if start_value else None),
            "contracts": str(final_value < start_value).lower(),
        })
    return per_seed, summary, contraction


def render(per_seed: list[dict], summary: list[dict], config: dict, outdir: Path) -> list[Path]:
    budgets = list(config["budgets_kimg"])
    seeds = sorted({row["training_seed"] for row in per_seed})
    figure, axis = plt.subplots(figsize=(7.0, 4.75))
    median = [float(row["median_dispersion_fid"]) for row in summary]
    mean = [float(row["mean_dispersion_fid"]) for row in summary]
    # Draw summaries first.  This keeps the seed-level marks on top even when
    # a median happens to coincide exactly with one or more seed trajectories.
    axis.plot(budgets, median, color="#111827", linewidth=2.2, marker="D", markersize=4.2,
              label="Median (summary)", zorder=1, alpha=0.72)
    axis.plot(budgets, mean, color="#4B5563", linewidth=1.3, linestyle=(0, (4, 3)), marker="s", markersize=3.5,
              markerfacecolor="white", label="Mean (summary)", zorder=1, alpha=0.80)
    for index, seed in enumerate(seeds):
        selected = sorted((row for row in per_seed if row["training_seed"] == seed), key=lambda row: row["budget_kimg"])
        color = SEED_COLORS[index % len(SEED_COLORS)]
        axis.plot(
            [row["budget_kimg"] for row in selected],
            [float(row["dispersion_fid"]) for row in selected],
            color=color, marker=SEED_MARKERS[index % len(SEED_MARKERS)], linewidth=1.45,
            markersize=5.3, markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.3,
            alpha=0.95, label="Seed {}".format(seed), zorder=3,
        )
    axis.set_xlabel("Training budget (kimg)", color=INK)
    axis.set_ylabel(r"Arm dispersion $S_s(K)$ (FID range)", color=INK)
    axis.set_xticks(budgets)
    axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#6B7280")
    axis.tick_params(colors=INK)
    figure.suptitle("Asset C · Arm dispersion contraction", x=0.11, y=0.98, ha="left", fontsize=14, fontweight="bold", color=INK)
    figure.text(0.11, 0.90, r"$S_s(K)=\max_{a\in\{A,B,C,D\}}\mathrm{FID}_{s,a}(K)-\min_{a\in\{A,B,C,D\}}\mathrm{FID}_{s,a}(K)$", fontsize=9.5, color=INK)
    figure.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=min(4, len(seeds) + 2), frameon=False, fontsize=8.5)
    figure.subplots_adjust(left=0.13, right=0.98, top=0.83, bottom=0.22)
    outputs = []
    for extension in ("svg", "png", "pdf"):
        path = outdir / "asset_c_arm_dispersion.{}".format(extension)
        figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    plt.close(figure)
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--arm-config", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)

    config_path = args.arm_config.resolve()
    config = read_config(config_path)
    source = args.input_csv.resolve()
    outdir = args.outdir.resolve()
    rows, protocol = load_trajectory(
        source, config["metric_name"], config["nfe"], tuple(config["budgets_kimg"]),
        config["analysis_track"], PREFIX,
    )
    if protocol["evaluation_contract"] != config["evaluation_contract"]:
        fail(PREFIX, "config evaluation_contract={!r} does not match data={!r}".format(config["evaluation_contract"], protocol["evaluation_contract"]))
    methods = tuple(config["arms"][arm] for arm in ("A", "B", "C", "D"))
    seeds, index = complete_matrix(rows, methods, tuple(config["budgets_kimg"]), PREFIX)
    if len(seeds) < 2:
        fail(PREFIX, "Asset C requires at least two seeds; a mean-only arm-dispersion chart is not allowed")
    per_seed, summary, contraction = calculate(index, config, seeds)
    outdir.mkdir(parents=True, exist_ok=True)
    per_seed_path = outdir / "asset_c_dispersion_by_seed.csv"
    summary_path = outdir / "asset_c_dispersion_summary.csv"
    contraction_path = outdir / "asset_c_contraction_by_seed.csv"
    save_csv(per_seed_path, per_seed)
    save_csv(summary_path, summary)
    save_csv(contraction_path, contraction)
    figures = render(per_seed, summary, config, outdir)
    manifest = {
        "asset": "C",
        "title": "Arm dispersion contraction",
        "source_csv": str(source),
        "source_sha256": sha256(source),
        "arm_config": str(config_path),
        "arm_config_sha256": sha256(config_path),
        "config": config,
        "protocol": protocol,
        "seeds": seeds,
        "all_seed_dispersions_contract": all(row["contracts"] == "true" for row in contraction),
        "outputs": {path.name: sha256(path) for path in [per_seed_path, summary_path, contraction_path] + figures},
    }
    (outdir / "asset_c_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Rendered Asset C with {} seeds to {}".format(len(seeds), outdir))


if __name__ == "__main__":
    main()
