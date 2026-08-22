#!/usr/bin/env python3
"""Render the q256 two-budget, four-arm FID-50k endpoint comparison.

This deliberately is *not* Asset A.  It compares two protocol-matched
endpoints (256 and 1024 kimg) without presenting them as a complete learning
curve.  Every seed and all four factorial arms remain visible.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ect-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from .paper_asset_data import (
        PAPER_PREVIEW_DPI, command_text, complete_matrix, fail, load_trajectory, sha256,
        write_publication_sidecars,
    )
except ImportError:
    from paper_asset_data import (
        PAPER_PREVIEW_DPI, command_text, complete_matrix, fail, load_trajectory, sha256,
        write_publication_sidecars,
    )


PREFIX = "render_paper_asset_endpoint"
ARM_COLORS = {"A": "#2563EB", "B": "#C2416C", "C": "#5B8C5A", "D": "#C98212"}
ARM_MARKERS = {"A": "o", "B": "s", "C": "^", "D": "D"}
INK = "#1F2937"
GRID = "#D1D5DB"


def read_config(path: Path) -> dict:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(PREFIX, "cannot read endpoint config {}: {}".format(path, exc))
        raise exc
    required = {
        "schema_version", "asset", "metric_name", "nfe", "evaluation_contract",
        "analysis_track", "budgets_kimg", "arms",
    }
    missing = required - set(config)
    if missing:
        fail(PREFIX, "endpoint config is missing {}".format(sorted(missing)))
    if config["schema_version"] != 1 or config["asset"] != "two_budget_endpoint":
        fail(PREFIX, "endpoint config must declare schema_version=1 and asset='two_budget_endpoint'")
    if config["metric_name"] != "fid50k_full" or config["analysis_track"] != "two_budget_endpoint":
        fail(PREFIX, "the endpoint comparison is restricted to the FID-50k two_budget_endpoint track")
    if not isinstance(config["nfe"], int) or config["nfe"] < 1:
        fail(PREFIX, "nfe must be a positive integer")
    try:
        budgets = [int(value) for value in config["budgets_kimg"]]
    except (TypeError, ValueError) as exc:
        fail(PREFIX, "budgets_kimg must contain integers")
        raise exc
    if len(budgets) != 2 or budgets != sorted(budgets) or len(set(budgets)) != 2 or any(value <= 0 for value in budgets):
        fail(PREFIX, "budgets_kimg must contain exactly two strictly increasing positive endpoints")
    config["budgets_kimg"] = budgets
    if not isinstance(config["arms"], dict) or set(config["arms"]) != {"A", "B", "C", "D"}:
        fail(PREFIX, "arms must map exactly A, B, C, and D")
    if len(set(config["arms"].values())) != 4 or any(not isinstance(value, str) or not value for value in config["arms"].values()):
        fail(PREFIX, "each arm must map to a distinct non-empty method")
    if not isinstance(config["evaluation_contract"], str) or not config["evaluation_contract"].strip():
        fail(PREFIX, "evaluation_contract must be a non-empty string")
    return config


def save_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def build_records(index: dict, config: dict, seeds: list[int]) -> list[dict]:
    records = []
    for seed in seeds:
        for arm in ("A", "B", "C", "D"):
            method = config["arms"][arm]
            for budget in config["budgets_kimg"]:
                row = index[(method, seed, float(budget))]
                records.append({
                    "training_seed": seed,
                    "arm": arm,
                    "method": method,
                    "budget_kimg": budget,
                    "fid50k": "{:.12g}".format(row["metric_value"]),
                    "nfe": config["nfe"],
                    "sample_count": row["sample_count"],
                    "generation_seed_range": row["generation_seed_range"],
                    "metric_seed": row["metric_seed"],
                    "evaluation_contract": row["evaluation_contract"],
                })
    return records


def render(records: list[dict], config: dict, outdir: Path) -> list[Path]:
    seeds = sorted({row["training_seed"] for row in records})
    figure, axes = plt.subplots(1, len(seeds), figsize=(4.1 * len(seeds), 4.25), sharey=True, squeeze=False)
    budgets = config["budgets_kimg"]
    for axis, seed in zip(axes[0], seeds):
        for arm in ("A", "B", "C", "D"):
            selected = sorted(
                (row for row in records if row["training_seed"] == seed and row["arm"] == arm),
                key=lambda row: row["budget_kimg"],
            )
            is_primary = arm in {"A", "B"}
            axis.plot(
                [row["budget_kimg"] for row in selected], [float(row["fid50k"]) for row in selected],
                color=ARM_COLORS[arm], marker=ARM_MARKERS[arm], linewidth=2.0 if is_primary else 1.25,
                markersize=5.8, markerfacecolor="white", markeredgewidth=1.4,
                alpha=1.0 if is_primary else 0.65, zorder=3 if is_primary else 2,
            )
        axis.set_title("Seed {}".format(seed), loc="left", fontsize=11, fontweight="bold", color=INK)
        axis.set_xticks(budgets)
        axis.set_xlabel("Training budget (kimg)", color=INK)
        axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#6B7280")
        axis.tick_params(colors=INK)
    axes[0][0].set_ylabel("FID-50k (lower is better)", color=INK)
    figure.suptitle("Two protocol-matched FID-50k endpoints", x=0.08, y=0.98, ha="left", fontsize=14, fontweight="bold", color=INK)
    figure.text(0.08, 0.90, "256 and 1024 kimg only — endpoint comparison, not a complete learning curve", fontsize=9.5, color="#4B5563")
    figure.legend(
        handles=[
            Line2D([0], [0], color=ARM_COLORS[arm], marker=ARM_MARKERS[arm],
                   label="Arm {} ({}){}".format(arm, config["arms"][arm], " · primary" if arm in {"A", "B"} else " · context"))
            for arm in ("A", "B", "C", "D")
        ],
        loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False, fontsize=8.1,
    )
    figure.subplots_adjust(left=0.08, right=0.99, top=0.80, bottom=0.24, wspace=0.20)
    outputs = []
    for extension in ("svg", "png", "pdf"):
        path = outdir / "asset_endpoint_two_budget.{}".format(extension)
        figure.savefig(path, dpi=PAPER_PREVIEW_DPI, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    plt.close(figure)
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--endpoint-config", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)

    config_path = args.endpoint_config.resolve()
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
        fail(PREFIX, "two-budget endpoint comparison requires at least two visible seeds")
    records = build_records(index, config, seeds)
    outdir.mkdir(parents=True, exist_ok=True)
    source_csv = outdir / "asset_endpoint_two_budget.csv"
    save_csv(source_csv, records)
    figures = render(records, config, outdir)
    png_path = next(path for path in figures if path.suffix == ".png")
    command = command_text([
        "python", "scripts/render_paper_asset_endpoint.py", "--input-csv", source,
        "--endpoint-config", config_path, "--outdir", outdir,
    ])
    sidecars = write_publication_sidecars(
        outdir,
        "asset_endpoint_two_budget",
        png_path,
        "Figure. Matched-seed, four-arm FID-50k values at 256 and 1024 kimg. Arms A and B are the "
        "primary comparison; C and D provide factorization context. Each panel preserves a single training seed.",
        "This is a two-budget endpoint comparison under one FID-50k protocol. It quantifies observed endpoint "
        "differences and contraction, but is not a complete learning curve and does not interpolate unobserved budgets.",
        command,
    )
    manifest = {
        "asset": "two_budget_endpoint",
        "title": "Two protocol-matched FID-50k endpoints",
        "source_csv": str(source),
        "source_sha256": sha256(source),
        "endpoint_config": str(config_path),
        "endpoint_config_sha256": sha256(config_path),
        "config": config,
        "protocol": protocol,
        "seeds": seeds,
        "not_a_complete_learning_curve": True,
        "publication_qa": sidecars,
        "outputs": {
            path.name: sha256(path)
            for path in [source_csv] + figures + [
                outdir / sidecars["caption"], outdir / sidecars["interpretation_boundary"],
                outdir / sidecars["render_command"], outdir / sidecars["grayscale_preview"],
                outdir / sidecars["grayscale_qa"],
            ]
        },
    }
    (outdir / "asset_endpoint_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Rendered two-budget endpoint asset with {} seeds to {}".format(len(seeds), outdir))


if __name__ == "__main__":
    main()
