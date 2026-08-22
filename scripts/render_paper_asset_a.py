#!/usr/bin/env python3
"""Render paper Asset A: per-seed FID versus training budget.

The input uses the long-form result schema consumed by
``collect_multibudget_results.py``.  Asset A is intentionally strict:

* one FID metric, one NFE, and one explicit sampling/evaluation contract;
* every method/seed trajectory has the same requested budget checkpoints; and
* 5k and 50k FID are never connected in one curve.

By default, the first two seeds are emphasized (the A/B curves); remaining
seeds are rendered as contextual, faded curves.  ``--secondary-mode panel``
instead places the contextual seeds in a second panel.

Example:
    python scripts/render_paper_asset_a.py \
        --input-csv results/q256_learning_curve/evaluation_results.csv \
        --outdir paper_assets/asset_a \
        --metric-name fid5k_full --nfe 2 \
        --analysis-track budget_curve --primary-seeds 3,4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ect-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:  # Supports both ``python scripts/...`` and package-level tests.
    from .collect_multibudget_results import read_rows
except ImportError:
    from collect_multibudget_results import read_rows


DEFAULT_BUDGETS = (256, 512, 768, 1024)
METHOD_COLORS = ("#2563EB", "#C2416C")
SEED_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<")
INK = "#1F2937"
GRID = "#D1D5DB"


def fail(message: str) -> None:
    raise SystemExit("[render_paper_asset_a] ERROR: " + message)


def parse_csv_values(raw: str, kind: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        fail("{} must be a comma-separated list of integers".format(kind))
        raise exc
    if not values or len(values) != len(set(values)):
        fail("{} must contain one or more unique values".format(kind))
    return values


def parse_seed_labels(raw: str) -> dict[int, str]:
    """Parse ``3=A,4=B`` while preserving a readable default elsewhere."""
    if not raw.strip():
        return {}
    labels: dict[int, str] = {}
    for item in raw.split(","):
        if "=" not in item:
            fail("seed-labels entries must use SEED=LABEL, for example 3=A")
        seed_raw, label = item.split("=", 1)
        try:
            seed = int(seed_raw.strip())
        except ValueError as exc:
            fail("seed-labels seed must be an integer: {}".format(seed_raw.strip()))
            raise exc
        label = label.strip()
        if not label or seed in labels:
            fail("seed-labels must use unique, non-empty labels")
        labels[seed] = label
    return labels


def metric_display_name(metric_name: str) -> str:
    names = {
        "fid5k_full": "FID-5k",
        "fid50k_full": "FID-50k",
    }
    return names.get(metric_name, metric_name.replace("_", " ").upper())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_one_protocol(rows: list[dict], analysis_track: str) -> dict:
    """Reject curves whose points cannot be shown as one measurement series."""
    tracks = {row["analysis_track"] for row in rows}
    if tracks != {analysis_track}:
        fail("selected rows must all have analysis_track={!r}; got {}".format(analysis_track, sorted(tracks)))
    contracts = {row["evaluation_contract"] for row in rows}
    protocol = {
        (row["sample_count"], row["generation_seed_range"], row["metric_seed"])
        for row in rows
    }
    if (len(contracts) != 1 or "" in contracts or len(protocol) != 1
            or None in next(iter(protocol)) or "" in next(iter(protocol))):
        fail(
            "Asset A requires one explicit evaluation_contract and one complete "
            "(sample_count, generation_seed_range, metric_seed) protocol"
        )
    sample_count, generation_seed_range, metric_seed = next(iter(protocol))
    return {
        "evaluation_contract": next(iter(contracts)),
        "sample_count": sample_count,
        "generation_seed_range": generation_seed_range,
        "metric_seed": metric_seed,
        "evidence_classes": sorted({row["evidence_class"] for row in rows}),
    }


def select_rows(
    rows: list[dict], metric_name: str, nfe: int, budgets: tuple[int, ...], analysis_track: str,
) -> tuple[list[dict], list[str], list[int], dict]:
    selected = [
        row for row in rows
        if row["metric_name"] == metric_name and row["nfe"] == nfe
    ]
    if not selected:
        available = sorted({(row["metric_name"], row["nfe"]) for row in rows})
        fail("no rows for metric_name={!r}, nfe={}; available={}".format(metric_name, nfe, available))
    selected_budgets = {row["budget_kimg"] for row in selected}
    requested = {float(budget) for budget in budgets}
    if selected_budgets != requested:
        fail(
            "requested budgets {} do not match the selected metric/NFE budgets {}. "
            "Do not silently truncate a paper learning curve.".format(
                list(budgets), sorted(selected_budgets),
            )
        )
    protocol = require_one_protocol(selected, analysis_track)
    methods = sorted({row["method"] for row in selected})
    seeds = sorted({row["training_seed"] for row in selected})
    if len(methods) != 2:
        fail("Asset A currently requires exactly two paired methods; got {}".format(methods))
    index: dict[tuple[str, int, float], dict] = {}
    for row in selected:
        key = (row["method"], row["training_seed"], row["budget_kimg"])
        if key in index:
            fail("duplicate Asset A point: {}".format(key))
        if not math.isfinite(row["metric_value"]):
            fail("non-finite FID value at {}".format(key))
        index[key] = row
    expected = {
        (method, seed, float(budget))
        for method in methods for seed in seeds for budget in budgets
    }
    missing = expected - set(index)
    extra = set(index) - expected
    if missing or extra:
        fail("Asset A matrix incomplete; missing={}, extra={}".format(sorted(missing), sorted(extra)))
    ordered = [
        index[(method, seed, float(budget))]
        for method in methods for seed in seeds for budget in budgets
    ]
    return ordered, methods, seeds, protocol


def style_axis(axis: plt.Axes, budgets: tuple[int, ...], ylabel: str) -> None:
    axis.set_xlabel("Training budget (kimg)", color=INK)
    axis.set_ylabel(ylabel, color=INK)
    axis.set_xticks(budgets)
    axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#6B7280")
    axis.tick_params(colors=INK)


def render(
    rows: list[dict], methods: list[str], seeds: list[int], budgets: tuple[int, ...],
    primary_seeds: tuple[int, ...], seed_labels: dict[int, str], secondary_mode: str,
    metric_name: str, outdir: Path,
) -> list[Path]:
    by_curve: dict[tuple[str, int], list[dict]] = {}
    for method in methods:
        for seed in seeds:
            by_curve[(method, seed)] = sorted(
                (row for row in rows if row["method"] == method and row["training_seed"] == seed),
                key=lambda row: row["budget_kimg"],
            )
    contextual = [seed for seed in seeds if seed not in primary_seeds]
    if secondary_mode == "panel" and contextual:
        figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.25), sharey=True)
        axes = list(axes)
        panel_specs = [
            (axes[0], list(primary_seeds), "Primary seeds"),
            (axes[1], contextual, "Context seeds"),
        ]
    else:
        figure, axis = plt.subplots(figsize=(6.7, 4.65))
        axes = [axis]
        panel_specs = [(axis, seeds, "Per-seed trajectories")]

    ylabel = "{} (lower is better)".format(metric_display_name(metric_name))
    method_colors = {method: METHOD_COLORS[index] for index, method in enumerate(methods)}
    seed_markers = {seed: SEED_MARKERS[index % len(SEED_MARKERS)] for index, seed in enumerate(seeds)}
    for axis, panel_seeds, panel_title in panel_specs:
        for seed in panel_seeds:
            is_primary = seed in primary_seeds
            for method in methods:
                curve = by_curve[(method, seed)]
                axis.plot(
                    [row["budget_kimg"] for row in curve],
                    [row["metric_value"] for row in curve],
                    color=method_colors[method], marker=seed_markers[seed],
                    markeredgecolor="white" if not is_primary else method_colors[method],
                    markeredgewidth=0.8, markersize=5.5 if is_primary else 4.6,
                    linewidth=2.3 if is_primary else 1.25,
                    alpha=1.0 if is_primary else 0.28,
                    zorder=3 if is_primary else 2,
                )
        axis.set_title(panel_title, loc="left", fontsize=11, fontweight="bold", color=INK)
        style_axis(axis, budgets, ylabel)

    figure.suptitle("Asset A · FID vs training budget", x=0.09, y=0.98,
                     ha="left", fontsize=14, fontweight="bold", color=INK)
    method_handles = [
        Line2D([0], [0], color=method_colors[method], linewidth=2.3, marker="o", label=method)
        for method in methods
    ]
    seed_handles = [
        Line2D([0], [0], color=INK, marker=seed_markers[seed], linestyle="None",
               label="{}{}".format(
                   seed_labels.get(seed, "Seed {}".format(seed)),
                   " (primary)" if seed in primary_seeds else " (context)",
               ))
        for seed in seeds
    ]
    figure.legend(
        handles=method_handles + seed_handles,
        loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=min(4, len(method_handles + seed_handles)),
        frameon=False, fontsize=8.8,
    )
    figure.subplots_adjust(left=0.12, right=0.98, top=0.84, bottom=0.22, wspace=0.16)
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ("svg", "png", "pdf"):
        path = outdir / "asset_a_fid_vs_budget.{}".format(extension)
        figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    plt.close(figure)
    return outputs


def write_csv(path: Path, rows: list[dict], primary_seeds: tuple[int, ...], seed_labels: dict[int, str]) -> None:
    normalized = []
    for row in rows:
        normalized.append({
            "method": row["method"],
            "training_seed": row["training_seed"],
            "seed_label": seed_labels.get(row["training_seed"], "Seed {}".format(row["training_seed"])),
            "visibility": "primary" if row["training_seed"] in primary_seeds else "context",
            "budget_kimg": "{:g}".format(row["budget_kimg"]),
            "nfe": row["nfe"],
            "metric_name": row["metric_name"],
            "fid": "{:.12g}".format(row["metric_value"]),
            "sample_count": row["sample_count"],
            "generation_seed_range": row["generation_seed_range"],
            "metric_seed": row["metric_seed"],
            "evidence_class": row["evidence_class"],
            "evaluation_contract": row["evaluation_contract"],
            "analysis_track": row["analysis_track"],
            "checkpoint_sha256": row["checkpoint_sha256"],
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(normalized[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(normalized)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--metric-name", default="fid5k_full")
    parser.add_argument("--nfe", type=int, required=True)
    parser.add_argument("--analysis-track", default="budget_curve")
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    parser.add_argument("--primary-seeds", default="", help="Comma-separated seed IDs; default: first two.")
    parser.add_argument("--seed-labels", default="", help="Optional labels, e.g. 3=A,4=B,5=C,6=D.")
    parser.add_argument("--secondary-mode", choices=("faded", "panel"), default="faded")
    args = parser.parse_args(argv)

    budgets = parse_csv_values(args.budgets, "budgets")
    if tuple(sorted(budgets)) != budgets:
        fail("budgets must be strictly ascending")
    source = args.input_csv.resolve()
    output = args.outdir.resolve()
    rows, methods, seeds, protocol = select_rows(
        read_rows(source), args.metric_name, args.nfe, budgets, args.analysis_track,
    )
    requested_primary = parse_csv_values(args.primary_seeds, "primary-seeds") if args.primary_seeds else tuple(seeds[:2])
    if len(requested_primary) != 2:
        fail("choose exactly two primary seeds for the A/B main curves")
    if set(requested_primary) - set(seeds):
        fail("primary-seeds must exist in the selected data; available={}".format(seeds))
    seed_labels = parse_seed_labels(args.seed_labels)
    if set(seed_labels) - set(seeds):
        fail("seed-labels includes an unobserved seed; available={}".format(seeds))
    if output.exists() and not output.is_dir():
        fail("outdir exists and is not a directory: {}".format(output))
    output.mkdir(parents=True, exist_ok=True)

    files = render(
        rows, methods, seeds, budgets, requested_primary, seed_labels,
        args.secondary_mode, args.metric_name, output,
    )
    figure_data = output / "asset_a_fid_vs_budget.csv"
    write_csv(figure_data, rows, requested_primary, seed_labels)
    manifest = {
        "asset": "A",
        "title": "FID vs training budget",
        "x_axis": "Training budget (kimg)",
        "y_axis": "{} (lower is better)".format(metric_display_name(args.metric_name)),
        "source_csv": str(source),
        "source_sha256": sha256(source),
        "metric_name": args.metric_name,
        "nfe": args.nfe,
        "budgets_kimg": list(budgets),
        "methods": methods,
        "seeds": seeds,
        "primary_seeds": list(requested_primary),
        "seed_labels": {str(seed): seed_labels.get(seed, "Seed {}".format(seed)) for seed in seeds},
        "secondary_mode": args.secondary_mode,
        "row_count": len(rows),
        "protocol": protocol,
        "outputs": {path.name: sha256(path) for path in [figure_data] + files},
    }
    (output / "asset_a_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print("Rendered Asset A with {} points to {}".format(len(rows), output))


if __name__ == "__main__":
    main()
