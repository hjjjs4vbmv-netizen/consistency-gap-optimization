#!/usr/bin/env python3
"""Validate the seed-0 global-gap sweep and select the confirmation scale.

The selection rule is intentionally narrow: among the four non-unit
global-only candidates, choose the lowest NFE=1 KID-5k result.  FID and
NFE=2 are reported for diagnosis but never participate in selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


CELLS = (
    {
        "arm": "fixed",
        "global_scale": 1.0,
        "global_scale_text": "1",
        "label": "fixed-g1p0000-seed0-256k",
        "expected_schedule": "sigmoid",
        "eligible_for_selection": False,
    },
    {
        "arm": "global",
        "global_scale": 0.97,
        "global_scale_text": "0.97",
        "label": "global-g0p9700-seed0-256k",
        "expected_schedule": "global_sigmoid",
        "eligible_for_selection": True,
    },
    {
        "arm": "global",
        "global_scale": 1.032,
        "global_scale_text": "1.032",
        "label": "global-g1p0320-seed0-256k",
        "expected_schedule": "global_sigmoid",
        "eligible_for_selection": True,
    },
    {
        "arm": "global",
        "global_scale": 1.06,
        "global_scale_text": "1.06",
        "label": "global-g1p0600-seed0-256k",
        "expected_schedule": "global_sigmoid",
        "eligible_for_selection": True,
    },
    {
        "arm": "global",
        "global_scale": 1.10,
        "global_scale_text": "1.10",
        "label": "global-g1p1000-seed0-256k",
        "expected_schedule": "global_sigmoid",
        "eligible_for_selection": True,
    },
)
NFES = (1, 2)
METRICS = ("kid5k_full", "fid5k_full")
TIE_TARGET = 1.0317


def fail(message: str) -> None:
    raise SystemExit(f"[select_gap_scale] ERROR: {message}")


def load_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        fail(f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read {description} {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{description} must contain a JSON object: {path}")
    return value


def require_training_validation(run_dir: Path, cell: dict[str, Any]) -> Path:
    path = run_dir / "validation.json"
    payload = load_json_object(path, "training validation")
    if payload.get("status") != "passed":
        fail(
            f"training validation did not pass for {cell['label']}: "
            f"status={payload.get('status')!r}"
        )
    if payload.get("expected_schedule") != cell["expected_schedule"]:
        fail(
            f"training validation schedule mismatch for {cell['label']}: "
            f"expected {cell['expected_schedule']!r}, "
            f"got {payload.get('expected_schedule')!r}"
        )
    return path.resolve()


def read_single_metric(path: Path, metric: str) -> float:
    if not path.is_file():
        fail(f"missing metric result: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        fail(f"cannot read metric result {path}: {exc}")
    if len(lines) != 1 or not lines[0].strip():
        fail(f"expected exactly one result line in {path}, found {len(lines)}")
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        fail(f"malformed metric JSON in {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"metric result must be a JSON object: {path}")
    if payload.get("metric") != metric:
        fail(
            f"metric name mismatch in {path}: "
            f"{payload.get('metric')!r} != {metric!r}"
        )
    results = payload.get("results")
    if not isinstance(results, dict) or set(results) != {metric}:
        fail(f"expected exactly one {metric!r} result in {path}")
    raw_value = results[metric]
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        fail(f"metric result is not numeric in {path}: {raw_value!r}")
    value = float(raw_value)
    if not math.isfinite(value):
        fail(f"metric result is non-finite in {path}: {value}")
    return value


def require_evaluation_completion(eval_dir: Path) -> None:
    meta_path = eval_dir / "experiment_meta.env"
    if not meta_path.is_file():
        fail(f"missing evaluation metadata: {meta_path}")
    lines = meta_path.read_text(encoding="utf-8").splitlines()
    if lines.count("exit_code=0") != 1:
        fail(f"evaluation did not record exactly one exit_code=0: {meta_path}")


def load_response_curve(runs_root: Path, eval_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in CELLS:
        cell = dict(spec)
        run_dir = runs_root / cell["label"]
        validation_path = require_training_validation(run_dir, cell)
        cell["training_validation_path"] = str(validation_path)
        for nfe in NFES:
            eval_dir = eval_root / cell["label"] / f"nfe{nfe}"
            require_evaluation_completion(eval_dir)
            for metric in METRICS:
                cell[f"nfe{nfe}_{metric}"] = read_single_metric(
                    eval_dir / f"metric-{metric}.jsonl", metric
                )
        rows.append(cell)
    return rows


def select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if row["eligible_for_selection"]]
    if len(candidates) != 4:
        fail(f"expected four non-unit candidates, found {len(candidates)}")
    # Only NFE=1 KID appears before the tie-breakers.  Do not add FID or NFE=2
    # here: those quantities are intentionally diagnostic-only.
    return min(
        candidates,
        key=lambda row: (
            row["nfe1_kid5k_full"],
            abs(row["global_scale"] - TIE_TARGET),
            abs(row["global_scale"] - 1.0),
            row["global_scale"],
        ),
    )


def add_comparisons(
    rows: list[dict[str, Any]], selected: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fixed_rows = [row for row in rows if row["global_scale"] == 1.0]
    if len(fixed_rows) != 1:
        fail(f"expected exactly one unit-scale fixed row, found {len(fixed_rows)}")
    fixed = fixed_rows[0]
    enriched: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        for nfe in NFES:
            for metric in METRICS:
                key = f"nfe{nfe}_{metric}"
                row[f"{key}_delta_vs_fixed"] = row[key] - fixed[key]
        row["is_selected"] = row["label"] == selected["label"]
        enriched.append(row)
    selected_enriched = next(row for row in enriched if row["is_selected"])
    fixed_enriched = next(row for row in enriched if row["global_scale"] == 1.0)
    return fixed_enriched, enriched


def csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "arm",
        "global_scale",
        "global_scale_text",
        "label",
        "eligible_for_selection",
        "is_selected",
        "nfe1_kid5k_full",
        "nfe1_kid5k_full_delta_vs_fixed",
        "nfe1_fid5k_full",
        "nfe1_fid5k_full_delta_vs_fixed",
        "nfe2_kid5k_full",
        "nfe2_kid5k_full_delta_vs_fixed",
        "nfe2_fid5k_full",
        "nfe2_fid5k_full_delta_vs_fixed",
        "training_validation_path",
    ]
    return [{field: row[field] for field in fields} for row in rows]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    output_rows = csv_rows(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(output_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_rows)


def format_metric(value: float) -> str:
    return f"{value:.9g}"


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    fixed: dict[str, Any],
    selected: dict[str, Any],
) -> None:
    beats = selected["nfe1_kid5k_full"] < fixed["nfe1_kid5k_full"]
    lines = [
        "# Global-gap response curve: seed 0 at 256 kimg",
        "",
        "> These KID-5k and FID-5k values are 5k-sample proxies, not standard "
        "FID-50k benchmark results.",
        "",
        "Lower is better. The frozen selection rule uses **only NFE=1 "
        "KID-5k** among the four non-unit global-only candidates. The fixed "
        "`g=1` row is a reference and is not eligible for selection. Exact "
        f"ties are resolved by distance to `g={TIE_TARGET}`, then distance "
        "to `g=1`.",
        "",
        "| Arm | g | NFE=1 KID | Δ vs fixed | NFE=1 FID | "
        "NFE=2 KID | NFE=2 FID | Selected |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['global_scale_text']} | "
            f"{format_metric(row['nfe1_kid5k_full'])} | "
            f"{format_metric(row['nfe1_kid5k_full_delta_vs_fixed'])} | "
            f"{format_metric(row['nfe1_fid5k_full'])} | "
            f"{format_metric(row['nfe2_kid5k_full'])} | "
            f"{format_metric(row['nfe2_fid5k_full'])} | "
            f"{'yes' if row['is_selected'] else ''} |"
        )
    comparison = "does beat" if beats else "does not beat"
    lines.extend(
        [
            "",
            "## Frozen exploratory selection",
            "",
            f"Selected `g*={selected['global_scale_text']}` with NFE=1 "
            f"KID-5k `{format_metric(selected['nfe1_kid5k_full'])}`. It "
            f"{comparison} the fixed seed-0 reference "
            f"(`{format_metric(fixed['nfe1_kid5k_full'])}`); "
            f"Δ = `{format_metric(selected['nfe1_kid5k_full_delta_vs_fixed'])}`.",
            "",
            "## Interpretation limits",
            "",
            "- This is a single-training-seed response curve using noisy "
            "5k-sample proxy metrics.",
            "- Selecting the best value on these same seed-0 measurements "
            "creates selection bias (the winner's-curse effect). The chosen "
            "scale is a hypothesis for held-out multi-seed confirmation, "
            "not evidence of a general improvement.",
            "- NFE=1 FID and all NFE=2 values are diagnostic only and did "
            "not influence the selected scale.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = load_response_curve(args.runs_root.resolve(), args.eval_root.resolve())
    selected_raw = select_candidate(rows)
    fixed, enriched = add_comparisons(rows, selected_raw)
    selected = next(row for row in enriched if row["is_selected"])
    beats_fixed = selected["nfe1_kid5k_full"] < fixed["nfe1_kid5k_full"]

    caveats = [
        "KID-5k and FID-5k are 5k-sample proxies, not standard FID-50k benchmarks.",
        "The response curve uses one training seed (seed 0).",
        "Selecting and reporting the minimum on the same seed-0 sweep creates "
        "selection bias; g* requires held-out multi-seed confirmation.",
        "NFE=1 FID and all NFE=2 metrics are diagnostic and were not used "
        "to select g*.",
    ]
    report = {
        "schema_version": 1,
        "status": "passed",
        "selection_rule": {
            "eligible_arms": ["global"],
            "eligible_global_scales": [0.97, 1.032, 1.06, 1.10],
            "primary_metric": "nfe1_kid5k_full",
            "direction": "lower_is_better",
            "fixed_g1_excluded_from_selection": True,
            "exact_tie_breakers": [
                f"closest_global_scale_to_{TIE_TARGET}",
                "closest_global_scale_to_1.0",
                "lower_global_scale",
            ],
            "diagnostic_metrics_not_used_for_selection": [
                "nfe1_fid5k_full",
                "nfe2_kid5k_full",
                "nfe2_fid5k_full",
            ],
        },
        "fixed": fixed,
        "selected": selected,
        "selected_global_scale": selected["global_scale"],
        "selected_global_scale_text": selected["global_scale_text"],
        "selected_label": selected["label"],
        "selected_beats_fixed_nfe1_kid": beats_fixed,
        "nfe1_kid_delta_vs_fixed": selected[
            "nfe1_kid5k_full_delta_vs_fixed"
        ],
        "response_curve": enriched,
        "caveats": caveats,
    }

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "response_curve.csv", enriched)
    (outdir / "selection.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (outdir / "selected_g.txt").write_text(
        selected["global_scale_text"] + "\n", encoding="utf-8"
    )
    write_markdown(outdir / "response_curve.md", enriched, fixed, selected)
    print(
        f"Selected g*={selected['global_scale_text']} "
        f"(NFE=1 KID={selected['nfe1_kid5k_full']:.9g}, "
        f"beats_fixed={str(beats_fixed).lower()}); output={outdir}"
    )


if __name__ == "__main__":
    main()
