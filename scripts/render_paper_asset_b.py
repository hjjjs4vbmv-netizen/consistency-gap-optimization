#!/usr/bin/env python3
"""Render paper Asset B: seed-resolved compute-to-quality.

Asset B is controlled by a committed frozen threshold JSON.  It reports either
the first observed checkpoint crossing or a clearly labelled descriptive linear
interpolation.  A figure never mixes the two crossing definitions.

Example threshold config::

    {
      "schema_version": 1,
      "asset": "B",
      "threshold_id": "fid5k-eta-100",
      "metric_name": "fid5k_full",
      "nfe": 2,
      "threshold": 100.0,
      "crossing_mode": "first_observed",
      "evaluation_contract": "q256-common-5k-v1",
      "analysis_track": "budget_curve",
      "budgets_kimg": [256, 512, 768, 1024],
      "arms": {"A": "fixed", "B": "global110"}
    }
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
    from .paper_asset_data import (
        PAPER_PREVIEW_DPI, command_text, complete_matrix, fail, load_trajectory, sha256,
        write_publication_sidecars,
    )
except ImportError:
    from paper_asset_data import (
        PAPER_PREVIEW_DPI, command_text, complete_matrix, fail, load_trajectory, sha256,
        write_publication_sidecars,
    )


PREFIX = "render_paper_asset_b"
MODE_FIRST_OBSERVED = "first_observed"
MODE_LINEAR = "linear_interpolation_descriptive"
ARM_COLORS = {"A": "#2563EB", "B": "#C2416C"}
INK = "#1F2937"
GRID = "#D1D5DB"


def read_config(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(PREFIX, "cannot read frozen threshold config {}: {}".format(path, exc))
        raise exc
    required = {
        "schema_version", "asset", "threshold_id", "metric_name", "nfe", "threshold",
        "crossing_mode", "evaluation_contract", "analysis_track", "budgets_kimg", "arms",
    }
    missing = required - set(payload)
    if missing:
        fail(PREFIX, "frozen threshold config is missing {}".format(sorted(missing)))
    if payload["schema_version"] != 1 or payload["asset"] != "B":
        fail(PREFIX, "threshold config must declare schema_version=1 and asset='B'")
    if not isinstance(payload["threshold_id"], str) or not payload["threshold_id"].strip():
        fail(PREFIX, "threshold_id must be a non-empty string")
    if not isinstance(payload["metric_name"], str) or not payload["metric_name"].strip():
        fail(PREFIX, "metric_name must be a non-empty string")
    if not isinstance(payload["nfe"], int) or payload["nfe"] < 1:
        fail(PREFIX, "nfe must be a positive integer")
    try:
        threshold = float(payload["threshold"])
    except (TypeError, ValueError) as exc:
        fail(PREFIX, "threshold must be finite")
        raise exc
    if not math.isfinite(threshold):
        fail(PREFIX, "threshold must be finite")
    payload["threshold"] = threshold
    if payload["crossing_mode"] not in {MODE_FIRST_OBSERVED, MODE_LINEAR}:
        fail(PREFIX, "crossing_mode must be {!r} or {!r}".format(MODE_FIRST_OBSERVED, MODE_LINEAR))
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
    if (not isinstance(payload["evaluation_contract"], str) or not payload["evaluation_contract"].strip()
            or not isinstance(payload["analysis_track"], str) or not payload["analysis_track"].strip()):
        fail(PREFIX, "evaluation_contract and analysis_track must be non-empty strings")
    arms = payload["arms"]
    if not isinstance(arms, dict) or set(arms) != {"A", "B"} or any(not isinstance(value, str) or not value for value in arms.values()):
        fail(PREFIX, "arms must map exactly A and B to non-empty method names")
    if arms["A"] == arms["B"]:
        fail(PREFIX, "arms A and B must name different methods")
    return payload


def crossing(curve: list[dict], threshold: float, mode: str) -> tuple[float | None, str]:
    """Compute one consistently defined threshold crossing from discrete checkpoints."""
    first = next((index for index, row in enumerate(curve) if row["metric_value"] <= threshold), None)
    if first is None:
        return None, "not_reached"
    if mode == MODE_FIRST_OBSERVED:
        return curve[first]["budget_kimg"], MODE_FIRST_OBSERVED
    if first == 0:
        fail(
            PREFIX,
            "linear interpolation is impossible when the first observed checkpoint already meets the threshold; "
            "choose first_observed or a threshold bracketed by checkpoints",
        )
    previous, current = curve[first - 1], curve[first]
    if previous["metric_value"] <= threshold or current["metric_value"] > threshold:
        fail(PREFIX, "linear interpolation requires a strictly bracketed first crossing")
    denominator = current["metric_value"] - previous["metric_value"]
    if denominator == 0:
        fail(PREFIX, "linear interpolation is undefined on a zero-slope crossing segment")
    estimate = previous["budget_kimg"] + (
        (threshold - previous["metric_value"])
        * (current["budget_kimg"] - previous["budget_kimg"]) / denominator
    )
    return estimate, MODE_LINEAR


def format_value(value: float | None) -> str:
    return "" if value is None else "{:.12g}".format(value)


def build_records(index: dict, config: dict, seeds: list[int]) -> tuple[list[dict], list[dict]]:
    budgets = tuple(config["budgets_kimg"])
    per_arm, per_seed = [], []
    by_seed: dict[int, dict[str, tuple[float | None, str]]] = {}
    for seed in seeds:
        by_seed[seed] = {}
        for arm in ("A", "B"):
            method = config["arms"][arm]
            curve = [index[(method, seed, float(budget))] for budget in budgets]
            tau, status = crossing(curve, config["threshold"], config["crossing_mode"])
            by_seed[seed][arm] = (tau, status)
            per_arm.append({
                "training_seed": seed,
                "arm": arm,
                "method": method,
                "tau_kimg": format_value(tau),
                "crossing_status": status,
                "threshold_id": config["threshold_id"],
                "threshold": format_value(config["threshold"]),
                "crossing_mode": config["crossing_mode"],
            })
        tau_a, status_a = by_seed[seed]["A"]
        tau_b, status_b = by_seed[seed]["B"]
        delta = tau_b - tau_a if tau_a is not None and tau_b is not None else None
        per_seed.append({
            "training_seed": seed,
            "tau_A_kimg": format_value(tau_a),
            "tau_B_kimg": format_value(tau_b),
            "delta_tau_B_minus_A_kimg": format_value(delta),
            "A_crossing_status": status_a,
            "B_crossing_status": status_b,
            "paired_status": "paired" if delta is not None else "unpaired_not_reached",
            "direction": "B_earlier" if delta is not None and delta < 0 else (
                "A_earlier" if delta is not None and delta > 0 else "tie" if delta == 0 else "not_estimable"
            ),
            "threshold_id": config["threshold_id"],
            "threshold": format_value(config["threshold"]),
            "crossing_mode": config["crossing_mode"],
        })
    return per_arm, per_seed


def summary_values(records: list[dict], field: str) -> dict:
    values = [float(row[field]) for row in records if row[field] != ""]
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
    }


def save_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def render(per_arm: list[dict], per_seed: list[dict], config: dict, outdir: Path) -> list[Path]:
    seeds = [row["training_seed"] for row in per_seed]
    positions = list(range(len(seeds)))
    by_seed_arm = {(row["training_seed"], row["arm"]): row for row in per_arm}
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})
    left, right = axes
    for position, seed in zip(positions, seeds):
        points = []
        for arm in ("A", "B"):
            record = by_seed_arm[(seed, arm)]
            if record["tau_kimg"]:
                points.append((float(record["tau_kimg"]), arm))
        if len(points) == 2:
            left.plot([point[0] for point in points], [position, position], color="#9CA3AF", linewidth=1.1, zorder=1)
        for tau, arm in points:
            left.scatter(tau, position, color=ARM_COLORS[arm], s=40, zorder=3)
    left.set_yticks(positions, ["Seed {}".format(seed) for seed in seeds])
    left.set_xlabel("Budget to reach threshold (kimg)", color=INK)
    left.set_title("Per-seed threshold crossing", loc="left", fontsize=11, fontweight="bold", color=INK)

    delta_records = [row for row in per_seed if row["delta_tau_B_minus_A_kimg"]]
    for position, seed in zip(positions, seeds):
        record = next(row for row in per_seed if row["training_seed"] == seed)
        if record["delta_tau_B_minus_A_kimg"]:
            delta = float(record["delta_tau_B_minus_A_kimg"])
            right.plot([0, delta], [position, position], color="#9CA3AF", linewidth=1.1, zorder=1)
            right.scatter(delta, position, color="#16A34A" if delta < 0 else "#C2416C", s=40, zorder=3)
        else:
            right.text(0.02, position, "not reached", transform=right.get_yaxis_transform(), va="center", color="#6B7280", fontsize=8)
    right.axvline(0, color="#4B5563", linewidth=1, linestyle=(0, (4, 3)))
    right.set_yticks(positions, ["Seed {}".format(seed) for seed in seeds])
    right.set_xlabel(r"$\Delta\tau_s = \tau_{B,s} - \tau_{A,s}$ (kimg)", color=INK)
    right.set_title("Paired compute difference", loc="left", fontsize=11, fontweight="bold", color=INK)

    tau_summary_a = summary_values([row for row in per_arm if row["arm"] == "A"], "tau_kimg")
    tau_summary_b = summary_values([row for row in per_arm if row["arm"] == "B"], "tau_kimg")
    delta_summary = summary_values(per_seed, "delta_tau_B_minus_A_kimg")
    for arm, summary in (("A", tau_summary_a), ("B", tau_summary_b)):
        if summary["median"] is not None:
            left.axvline(summary["median"], color=ARM_COLORS[arm], linewidth=1, linestyle=(0, (3, 2)), alpha=0.9)
    if delta_summary["median"] is not None:
        right.axvline(delta_summary["median"], color="#111827", linewidth=1.2, linestyle=(0, (3, 2)), alpha=0.9)

    label = (
        "first observed checkpoint crossing"
        if config["crossing_mode"] == MODE_FIRST_OBSERVED
        else "descriptive linear interpolation between adjacent checkpoints"
    )
    figure.suptitle("Asset B · Compute-to-quality", x=0.08, y=0.99, ha="left", fontsize=14, fontweight="bold", color=INK)
    figure.text(0.08, 0.89, r"{}: FID $\leq$ {} · {}".format(config["threshold_id"], config["threshold"], label), fontsize=9, color=INK)
    figure.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=ARM_COLORS[arm], label="Arm {} ({})".format(arm, config["arms"][arm]), markersize=7)
            for arm in ("A", "B")
        ] + [
            Line2D([0], [0], color="#111827", linestyle=(0, (3, 2)), label="Median summary"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False, fontsize=8.5,
    )
    for axis in axes:
        axis.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#6B7280")
        axis.tick_params(colors=INK)
    figure.subplots_adjust(left=0.12, right=0.98, top=0.80, bottom=0.20, wspace=0.28)
    outputs = []
    for extension in ("svg", "png", "pdf"):
        path = outdir / "asset_b_compute_to_quality.{}".format(extension)
        figure.savefig(path, dpi=PAPER_PREVIEW_DPI, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    plt.close(figure)
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--threshold-config", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)

    config_path = args.threshold_config.resolve()
    config = read_config(config_path)
    source = args.input_csv.resolve()
    outdir = args.outdir.resolve()
    rows, protocol = load_trajectory(
        source, config["metric_name"], config["nfe"], tuple(config["budgets_kimg"]),
        config["analysis_track"], PREFIX,
    )
    if protocol["evaluation_contract"] != config["evaluation_contract"]:
        fail(PREFIX, "config evaluation_contract={!r} does not match data={!r}".format(config["evaluation_contract"], protocol["evaluation_contract"]))
    methods = (config["arms"]["A"], config["arms"]["B"])
    seeds, index = complete_matrix(rows, methods, tuple(config["budgets_kimg"]), PREFIX)
    per_arm, per_seed = build_records(index, config, seeds)
    outdir.mkdir(parents=True, exist_ok=True)
    arm_path = outdir / "asset_b_tau_by_seed.csv"
    delta_path = outdir / "asset_b_delta_tau_by_seed.csv"
    save_csv(arm_path, per_arm)
    save_csv(delta_path, per_seed)
    figures = render(per_arm, per_seed, config, outdir)
    png_path = next(path for path in figures if path.suffix == ".png")
    command = command_text([
        "python", "scripts/render_paper_asset_b.py", "--input-csv", source,
        "--threshold-config", config_path, "--outdir", outdir,
    ])
    crossing_label = (
        "the first observed evaluation checkpoint" if config["crossing_mode"] == MODE_FIRST_OBSERVED
        else "a descriptive linear interpolation between adjacent evaluation checkpoints"
    )
    sidecars = write_publication_sidecars(
        outdir,
        "asset_b_compute_to_quality",
        png_path,
        "Figure. Per-seed compute required for arms A and B to reach FID <= {}. "
        "Each paired connector retains the matched training seed; the dashed summaries are medians. "
        "Crossing times use {}.".format(config["threshold"], crossing_label),
        "The figure reports the frozen threshold and one frozen crossing definition. "
        "When interpolation is selected, each crossing is a descriptive between-checkpoint estimate, "
        "not an observed crossing or a population-level guarantee.",
        command,
    )
    paired_deltas = [float(row["delta_tau_B_minus_A_kimg"]) for row in per_seed if row["delta_tau_B_minus_A_kimg"]]
    manifest = {
        "asset": "B",
        "title": "Compute-to-quality",
        "source_csv": str(source),
        "source_sha256": sha256(source),
        "threshold_config": str(config_path),
        "threshold_config_sha256": sha256(config_path),
        "threshold": config,
        "protocol": protocol,
        "seeds": seeds,
        "paired_seed_count": len(paired_deltas),
        "all_paired_deltas_negative": bool(paired_deltas) and len(paired_deltas) == len(seeds) and all(value < 0 for value in paired_deltas),
        "tau_summary": {
            "A": summary_values([row for row in per_arm if row["arm"] == "A"], "tau_kimg"),
            "B": summary_values([row for row in per_arm if row["arm"] == "B"], "tau_kimg"),
            "delta_B_minus_A": summary_values(per_seed, "delta_tau_B_minus_A_kimg"),
        },
        "publication_qa": sidecars,
        "outputs": {
            path.name: sha256(path)
            for path in [arm_path, delta_path] + figures + [
                outdir / sidecars["caption"], outdir / sidecars["interpretation_boundary"],
                outdir / sidecars["render_command"], outdir / sidecars["grayscale_preview"],
                outdir / sidecars["grayscale_qa"],
            ]
        },
    }
    (outdir / "asset_b_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Rendered Asset B with {} paired seeds to {}".format(len(paired_deltas), outdir))


if __name__ == "__main__":
    main()
