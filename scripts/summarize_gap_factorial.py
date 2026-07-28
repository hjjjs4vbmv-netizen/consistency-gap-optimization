#!/usr/bin/env python3
"""Fail-closed summary for the 3-seed global/local gap factorial study.

This script consumes the global-scale selection artifact plus the completed
training and evaluation trees.  It deliberately validates every formal cell
before writing any output, and reads each unique metric JSONL file exactly
once.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


TRAINING_SEEDS = (0, 1, 2)
HELDOUT_SEEDS = (1, 2)
NFES = (1, 2)
METRICS = ("kid5k_full", "fid5k_full")
PROFILES = ("conservative", "aggressive")
HEADLINE_ARMS = (
    "global",
    "local-conservative",
    "combined-conservative",
    "local-aggressive",
    "combined-aggressive",
)
T_CRITICAL_DF2_95 = 4.3026527
OUTPUT_NAMES = (
    "per_cell_metrics.csv",
    "per_seed_effects.csv",
    "heldout_headlines.csv",
    "factorial_summary.csv",
    "factorial_summary.json",
    "factorial_summary.md",
)

EFFECT_DEFINITIONS = {
    "global_at_local0": "global - fixed",
    "global_at_local1": "combined - local",
    "local_at_global0": "local - fixed",
    "local_at_global1": "combined - global",
    "combined_vs_fixed": "combined - fixed",
    "additive_interaction": "combined - global - local + fixed",
    "global_main_effect": "0.5 * [(global - fixed) + (combined - local)]",
    "local_main_effect": "0.5 * [(local - fixed) + (combined - global)]",
}

LOG_EFFECT_DEFINITIONS = {
    "global_at_local0": "log(global / fixed)",
    "global_at_local1": "log(combined / local)",
    "local_at_global0": "log(local / fixed)",
    "local_at_global1": "log(combined / global)",
    "combined_vs_fixed": "log(combined / fixed)",
    "additive_interaction": "log(combined * fixed / (global * local))",
    "global_main_effect": (
        "0.5 * [log(global / fixed) + log(combined / local)]"
    ),
    "local_main_effect": (
        "0.5 * [log(local / fixed) + log(combined / global)]"
    ),
}


def fail(message: str) -> None:
    raise SystemExit(f"[summarize_gap_factorial] ERROR: {message}")


def finite_number(value: object, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        fail(f"{context} is not numeric: {exc}")
    if not math.isfinite(result):
        fail(f"{context} is non-finite: {result}")
    return result


def scale_slug(scale: float) -> str:
    return f"{scale:.4f}".replace(".", "p")


def parse_env(path: Path) -> Dict[str, str]:
    if not path.is_file():
        fail(f"missing experiment metadata: {path}")
    values: Dict[str, str] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8", errors="strict").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in values:
            fail(f"duplicate key {key!r} in {path}:{line_number}")
        values[key] = value
    return values


def read_json_object(path: Path, description: str) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"missing or empty {description}: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"malformed {description} {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"{description} must be a JSON object: {path}")
    return payload


def load_selection(path: Path) -> Tuple[dict, float]:
    selection = read_json_object(path, "selection artifact")
    if selection.get("status") != "passed":
        fail(f"selection status is not 'passed': {path}")
    if selection.get("schema_version") != 1:
        fail(
            f"unsupported selection schema_version "
            f"{selection.get('schema_version')!r}: {path}"
        )

    scale = finite_number(
        selection.get("selected_global_scale"), "selected_global_scale"
    )
    if scale <= 0:
        fail(f"selected_global_scale must be > 0, got {scale}")
    scale_text = selection.get("selected_global_scale_text")
    if not isinstance(scale_text, str) or not scale_text:
        fail("selection is missing selected_global_scale_text")
    text_scale = finite_number(scale_text, "selected_global_scale_text")
    if not math.isclose(scale, text_scale, rel_tol=0.0, abs_tol=1e-12):
        fail(
            "selected_global_scale and selected_global_scale_text disagree: "
            f"{scale!r} vs {scale_text!r}"
        )

    expected_label = f"global-g{scale_slug(scale)}-seed0-256k"
    if selection.get("selected_label") != expected_label:
        fail(
            f"selected_label mismatch: expected {expected_label!r}, "
            f"got {selection.get('selected_label')!r}"
        )
    selected = selection.get("selected")
    if not isinstance(selected, dict):
        fail("selection is missing the selected row object")
    if selected.get("label") != expected_label:
        fail("selected row label disagrees with selected_label")
    selected_row_scale = finite_number(
        selected.get("global_scale"), "selected.global_scale"
    )
    if not math.isclose(scale, selected_row_scale, rel_tol=0.0, abs_tol=1e-12):
        fail("selected row global_scale disagrees with selected_global_scale")
    return selection, scale


def arm_specs(selected_scale: float) -> List[dict]:
    return [
        {
            "arm": "fixed",
            "profile": "shared",
            "global_level": 0,
            "local_level": 0,
            "scale": 1.0,
            "schedule": "sigmoid",
            "metadata_profile": "none",
        },
        {
            "arm": "global",
            "profile": "shared",
            "global_level": 1,
            "local_level": 0,
            "scale": selected_scale,
            "schedule": "global_sigmoid",
            "metadata_profile": "none",
        },
        {
            "arm": "local-conservative",
            "profile": "conservative",
            "global_level": 0,
            "local_level": 1,
            "scale": 1.0,
            "schedule": "local_tbin_v2",
            "metadata_profile": "conservative",
        },
        {
            "arm": "combined-conservative",
            "profile": "conservative",
            "global_level": 1,
            "local_level": 1,
            "scale": selected_scale,
            "schedule": "local_tbin_v3",
            "metadata_profile": "conservative",
        },
        {
            "arm": "local-aggressive",
            "profile": "aggressive",
            "global_level": 0,
            "local_level": 1,
            "scale": 1.0,
            "schedule": "local_tbin_v2",
            "metadata_profile": "aggressive",
        },
        {
            "arm": "combined-aggressive",
            "profile": "aggressive",
            "global_level": 1,
            "local_level": 1,
            "scale": selected_scale,
            "schedule": "local_tbin_v3",
            "metadata_profile": "aggressive",
        },
    ]


def run_label(spec: Mapping[str, object], seed: int) -> str:
    return (
        f"{spec['arm']}-g{scale_slug(float(spec['scale']))}"
        f"-seed{seed}-256k"
    )


def validate_training(
    run_dir: Path, spec: Mapping[str, object], seed: int
) -> Tuple[Path, Path]:
    validation_path = run_dir / "validation.json"
    validation = read_json_object(validation_path, "training validation")
    if validation.get("status") != "passed":
        fail(f"training validation did not pass: {validation_path}")
    if validation.get("expected_schedule") != spec["schedule"]:
        fail(
            f"schedule mismatch in {validation_path}: expected "
            f"{spec['schedule']!r}, got {validation.get('expected_schedule')!r}"
        )
    final_kimg = finite_number(
        validation.get("final_processed_kimg"),
        f"{validation_path}: final_processed_kimg",
    )
    if not 256.0 <= final_kimg < 256.128:
        fail(f"formal run is not a complete 256 kimg run: {validation_path}")

    meta_path = run_dir / "experiment_meta.env"
    meta = parse_env(meta_path)
    expected = {
        "arm": str(spec["arm"]),
        "schedule": str(spec["schedule"]),
        "local_profile": str(spec["metadata_profile"]),
        "seed": str(seed),
    }
    for key, expected_value in expected.items():
        if meta.get(key) != expected_value:
            fail(
                f"{meta_path}: expected {key}={expected_value!r}, "
                f"got {meta.get(key)!r}"
            )
    recorded_scale = finite_number(
        meta.get("global_gap_scale"), f"{meta_path}: global_gap_scale"
    )
    if not math.isclose(
        recorded_scale, float(spec["scale"]), rel_tol=0.0, abs_tol=1e-12
    ):
        fail(
            f"{meta_path}: global gap scale mismatch "
            f"{recorded_scale} != {spec['scale']}"
        )
    duration = finite_number(
        meta.get("duration_mimg"), f"{meta_path}: duration_mimg"
    )
    if not math.isclose(duration, 0.256, rel_tol=0.0, abs_tol=1e-12):
        fail(f"{meta_path}: expected duration_mimg=0.256, got {duration}")
    if meta.get("exit_code") != "0":
        fail(f"{meta_path}: training exit_code is not 0")
    return validation_path, meta_path


def read_metric_once(path: Path, metric: str) -> float:
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"missing or empty metric output: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        fail(f"cannot read metric output {path}: {exc}")
    lines = [line for line in raw.splitlines() if line.strip()]
    if len(lines) != 1:
        fail(
            f"expected exactly one JSONL result in {path}, found {len(lines)}"
        )
    try:
        payload = json.loads(lines[0])
        if not isinstance(payload, dict):
            raise TypeError("top-level JSON value is not an object")
        if payload.get("metric") != metric:
            fail(
                f"metric name mismatch in {path}: "
                f"{payload.get('metric')!r} != {metric!r}"
            )
        results = payload["results"]
        if not isinstance(results, dict):
            raise TypeError("results is not an object")
        value = float(results[metric])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        fail(f"malformed metric output {path}: {exc}")
    if not math.isfinite(value):
        fail(f"non-finite metric value in {path}: {value}")
    if value <= 0:
        fail(
            f"metric value must be > 0 for log contrasts in {path}: {value}"
        )
    return value


def validate_eval_metadata(eval_dir: Path, expected_label: str, nfe: int) -> None:
    meta_path = eval_dir / "experiment_meta.env"
    meta = parse_env(meta_path)
    expected = {
        "label": expected_label,
        "nfe": str(nfe),
        "exit_code": "0",
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            fail(
                f"{meta_path}: expected {key}={value!r}, "
                f"got {meta.get(key)!r}"
            )


def load_cells(
    runs_root: Path, eval_root: Path, selected_scale: float
) -> List[dict]:
    rows: List[dict] = []
    seen_metric_paths = set()
    for spec in arm_specs(selected_scale):
        for seed in TRAINING_SEEDS:
            label = run_label(spec, seed)
            run_dir = runs_root / label
            if not run_dir.is_dir():
                fail(f"missing formal training cell: {run_dir}")
            validation_path, _ = validate_training(run_dir, spec, seed)
            for nfe in NFES:
                eval_dir = eval_root / label / f"nfe{nfe}"
                if not eval_dir.is_dir():
                    fail(f"missing formal evaluation cell: {eval_dir}")
                validate_eval_metadata(eval_dir, label, nfe)
                metric_values = {}
                metric_paths = {}
                for metric in METRICS:
                    path = eval_dir / f"metric-{metric}.jsonl"
                    resolved = path.resolve()
                    if resolved in seen_metric_paths:
                        fail(f"metric file would be read more than once: {path}")
                    seen_metric_paths.add(resolved)
                    metric_values[metric] = read_metric_once(path, metric)
                    metric_paths[metric] = str(resolved)
                rows.append(
                    {
                        "arm": spec["arm"],
                        "profile": spec["profile"],
                        "global_level": spec["global_level"],
                        "local_level": spec["local_level"],
                        "global_gap_scale": spec["scale"],
                        "training_seed": seed,
                        "nfe": nfe,
                        **metric_values,
                        "run_label": label,
                        "run_dir": str(run_dir.resolve()),
                        "eval_dir": str(eval_dir.resolve()),
                        "training_validation_path": str(
                            validation_path.resolve()
                        ),
                        "kid_metric_path": metric_paths["kid5k_full"],
                        "fid_metric_path": metric_paths["fid5k_full"],
                    }
                )

    expected_rows = len(arm_specs(selected_scale)) * len(TRAINING_SEEDS) * len(NFES)
    if len(rows) != expected_rows:
        fail(f"internal matrix error: expected {expected_rows} rows, got {len(rows)}")
    expected_metric_files = expected_rows * len(METRICS)
    if len(seen_metric_paths) != expected_metric_files:
        fail(
            f"internal metric-file error: expected {expected_metric_files}, "
            f"got {len(seen_metric_paths)}"
        )
    return rows


def compute_contrasts(
    fixed: float, global_only: float, local_only: float, combined: float
) -> Tuple[Dict[str, float], Dict[str, float]]:
    deltas = {
        "global_at_local0": global_only - fixed,
        "global_at_local1": combined - local_only,
        "local_at_global0": local_only - fixed,
        "local_at_global1": combined - global_only,
        "combined_vs_fixed": combined - fixed,
        "additive_interaction": (
            combined - global_only - local_only + fixed
        ),
    }
    deltas["global_main_effect"] = 0.5 * (
        deltas["global_at_local0"] + deltas["global_at_local1"]
    )
    deltas["local_main_effect"] = 0.5 * (
        deltas["local_at_global0"] + deltas["local_at_global1"]
    )

    log_fixed = math.log(fixed)
    log_global = math.log(global_only)
    log_local = math.log(local_only)
    log_combined = math.log(combined)
    logs = {
        "global_at_local0": log_global - log_fixed,
        "global_at_local1": log_combined - log_local,
        "local_at_global0": log_local - log_fixed,
        "local_at_global1": log_combined - log_global,
        "combined_vs_fixed": log_combined - log_fixed,
        "additive_interaction": (
            log_combined - log_global - log_local + log_fixed
        ),
    }
    logs["global_main_effect"] = 0.5 * (
        logs["global_at_local0"] + logs["global_at_local1"]
    )
    logs["local_main_effect"] = 0.5 * (
        logs["local_at_global0"] + logs["local_at_global1"]
    )
    return deltas, logs


def build_per_seed_effects(cell_rows: Sequence[dict]) -> List[dict]:
    index = {
        (
            str(row["arm"]),
            int(row["training_seed"]),
            int(row["nfe"]),
            metric,
        ): float(row[metric])
        for row in cell_rows
        for metric in METRICS
    }
    rows: List[dict] = []
    for profile in PROFILES:
        for seed in TRAINING_SEEDS:
            for nfe in NFES:
                for metric in METRICS:
                    fixed = index[("fixed", seed, nfe, metric)]
                    global_only = index[("global", seed, nfe, metric)]
                    local_only = index[
                        (f"local-{profile}", seed, nfe, metric)
                    ]
                    combined = index[
                        (f"combined-{profile}", seed, nfe, metric)
                    ]
                    deltas, logs = compute_contrasts(
                        fixed, global_only, local_only, combined
                    )
                    row = {
                        "profile": profile,
                        "training_seed": seed,
                        "nfe": nfe,
                        "metric": metric,
                        "fixed": fixed,
                        "global": global_only,
                        "local": local_only,
                        "combined": combined,
                    }
                    for effect in EFFECT_DEFINITIONS:
                        row[f"{effect}_delta"] = deltas[effect]
                        row[f"{effect}_log_contrast"] = logs[effect]
                        row[f"{effect}_relative_percent"] = (
                            100.0 * math.expm1(logs[effect])
                        )
                    row["multiplicative_interaction_log"] = logs[
                        "additive_interaction"
                    ]
                    row["multiplicative_interaction_percent"] = (
                        100.0
                        * math.expm1(logs["additive_interaction"])
                    )
                    rows.append(row)
    return rows


def mean_sd_ci(values: Sequence[float]) -> Tuple[float, float, float, float]:
    if len(values) != 3:
        fail(f"expected exactly three paired seeds, got {len(values)}")
    mean = statistics.mean(values)
    sample_sd = statistics.stdev(values)
    half_width = T_CRITICAL_DF2_95 * sample_sd / math.sqrt(len(values))
    return mean, sample_sd, mean - half_width, mean + half_width


def build_summary(per_seed_rows: Sequence[dict]) -> List[dict]:
    rows: List[dict] = []
    for profile in PROFILES:
        for nfe in NFES:
            for metric in METRICS:
                selected = [
                    row
                    for row in per_seed_rows
                    if row["profile"] == profile
                    and row["nfe"] == nfe
                    and row["metric"] == metric
                ]
                if [row["training_seed"] for row in selected] != list(
                    TRAINING_SEEDS
                ):
                    fail(
                        f"incomplete paired seed order for "
                        f"{profile}/NFE={nfe}/{metric}"
                    )
                for effect in EFFECT_DEFINITIONS:
                    deltas = [
                        float(row[f"{effect}_delta"]) for row in selected
                    ]
                    logs = [
                        float(row[f"{effect}_log_contrast"])
                        for row in selected
                    ]
                    mean_delta, sd_delta, low_delta, high_delta = mean_sd_ci(
                        deltas
                    )
                    mean_log, sd_log, low_log, high_log = mean_sd_ci(logs)
                    wins = sum(value < 0 for value in deltas)
                    losses = sum(value > 0 for value in deltas)
                    ties = sum(value == 0 for value in deltas)
                    rows.append(
                        {
                            "profile": profile,
                            "nfe": nfe,
                            "metric": metric,
                            "effect": effect,
                            "arithmetic_definition": EFFECT_DEFINITIONS[
                                effect
                            ],
                            "log_definition": LOG_EFFECT_DEFINITIONS[effect],
                            "n": 3,
                            "per_seed_deltas_json": json.dumps(
                                deltas, separators=(",", ":")
                            ),
                            "mean_delta": mean_delta,
                            "sample_sd_delta": sd_delta,
                            "ci95_low_delta": low_delta,
                            "ci95_high_delta": high_delta,
                            "negative_wins": wins,
                            "positive_losses": losses,
                            "ties": ties,
                            "wins_of_3": f"{wins}/3",
                            "per_seed_log_contrasts_json": json.dumps(
                                logs, separators=(",", ":")
                            ),
                            "mean_log_contrast": mean_log,
                            "sample_sd_log_contrast": sd_log,
                            "ci95_low_log_contrast": low_log,
                            "ci95_high_log_contrast": high_log,
                            "geometric_relative_percent": (
                                100.0 * math.expm1(mean_log)
                            ),
                            "geometric_relative_ci95_low_percent": (
                                100.0 * math.expm1(low_log)
                            ),
                            "geometric_relative_ci95_high_percent": (
                                100.0 * math.expm1(high_log)
                            ),
                        }
                    )
    return rows


def build_heldout_headlines(cell_rows: Sequence[dict]) -> List[dict]:
    """Compare arithmetic metric means over seeds 1/2 against fixed sigmoid."""
    index = {
        (
            str(row["arm"]),
            int(row["training_seed"]),
            int(row["nfe"]),
            metric,
        ): float(row[metric])
        for row in cell_rows
        for metric in METRICS
    }
    rows = []
    for arm in HEADLINE_ARMS:
        for nfe in NFES:
            for metric in METRICS:
                fixed = [
                    index[("fixed", seed, nfe, metric)]
                    for seed in HELDOUT_SEEDS
                ]
                candidate = [
                    index[(arm, seed, nfe, metric)]
                    for seed in HELDOUT_SEEDS
                ]
                deltas = [
                    candidate[index] - fixed[index]
                    for index in range(len(HELDOUT_SEEDS))
                ]
                per_seed_relative = [
                    100.0 * deltas[index] / fixed[index]
                    for index in range(len(HELDOUT_SEEDS))
                ]
                fixed_mean = statistics.fmean(fixed)
                candidate_mean = statistics.fmean(candidate)
                absolute_change_total = sum(abs(value) for value in deltas)
                rows.append(
                    {
                        "arm": arm,
                        "nfe": nfe,
                        "metric": metric,
                        "heldout_seeds": "1,2",
                        "fixed_mean": fixed_mean,
                        "arm_mean": candidate_mean,
                        "headline_relative_percent": (
                            100.0 * (candidate_mean / fixed_mean - 1.0)
                        ),
                        "seed1_delta": deltas[0],
                        "seed2_delta": deltas[1],
                        "seed1_relative_percent": per_seed_relative[0],
                        "seed2_relative_percent": per_seed_relative[1],
                        "seed1_absolute_change_share_percent": (
                            100.0 * abs(deltas[0]) / absolute_change_total
                            if absolute_change_total > 0
                            else 0.0
                        ),
                    }
                )
    return rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        fail(f"refusing to write empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.6g}"


def build_markdown(
    selected_scale: float,
    selection: Mapping[str, object],
    cell_rows: Sequence[dict],
    heldout_rows: Sequence[dict],
    summary_rows: Sequence[dict],
) -> str:
    lines = [
        "# Global/local gap factorial summary",
        "",
        "> KID-5k and FID-5k are 5,000-sample proxy metrics, not standard "
        "50,000-sample benchmarks.",
        "",
        f"Selected global gap scale: **g\\* = {selected_scale:.6g}**. "
        "Lower is better for both metrics.",
        "",
        "This is a paired three-training-seed (`n=3`) descriptive analysis. "
        "The 95% intervals use the two-sided Student-t critical value "
        f"`t(df=2)={T_CRITICAL_DF2_95}`; they are not evidence for broad "
        "population-level significance.",
        "",
        "Seed 0 was used to select g\\* in the response-curve stage and its "
        "fixed/global observations are reused in this formal matrix. Therefore "
        "seed 0 has selection/evaluation overlap; interpret selected-g\\* "
        "effects as selection-aware descriptive estimates.",
        "",
        "## Held-out seed 1/2 headline calculation",
        "",
        "For arm `A` and metric `M`, the reported headline is",
        "",
        "`100 × (mean(M_A,seed1, M_A,seed2) / "
        "mean(M_fixed,seed1, M_fixed,seed2) - 1)`.",
        "",
        "It is the percentage difference between the two arithmetic metric "
        "means. It is **not** the mean of the two per-seed percentage changes.",
        "",
        "| Arm | NFE | Metric | Fixed mean | Arm mean | Headline % | "
        "Seed 1 % | Seed 2 % |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in heldout_rows:
        lines.append(
            f"| {row['arm']} | {row['nfe']} | {row['metric']} | "
            f"{fmt(float(row['fixed_mean']))} | "
            f"{fmt(float(row['arm_mean']))} | "
            f"{fmt(float(row['headline_relative_percent']))}% | "
            f"{fmt(float(row['seed1_relative_percent']))}% | "
            f"{fmt(float(row['seed2_relative_percent']))}% |"
        )
    lines.extend(
        [
            "",
            "At NFE=2, both held-out seeds improve directionally for "
            "`global` and `combined-aggressive`, but seed 1 has a much larger "
            "effect. Seed 1 accounts for the following share of the total "
            "absolute two-seed metric decrease:",
            "",
            "| Arm | Metric | Seed 1 share |",
            "| --- | --- | ---: |",
        ]
    )
    for row in heldout_rows:
        if (
            int(row["nfe"]) == 2
            and row["arm"] in ("global", "combined-aggressive")
        ):
            lines.append(
                f"| {row['arm']} | {row['metric']} | "
                f"{fmt(float(row['seed1_absolute_change_share_percent']))}% |"
            )
    lines.extend(
        [
            "",
            "A “win” means a negative paired contrast because lower is better. "
            "For interaction rows, negative means the combination is better than "
            "the corresponding additive prediction on the raw scale. The "
            "geometric relative percentage for that row is the multiplicative "
            "interaction `combined × fixed / (global × local) - 1`.",
            "",
            "## Validated matrix",
            "",
            f"- Unique training cells: {len(cell_rows) // len(NFES)} "
            "(fixed/global are shared across profiles)",
            f"- Evaluated training-seed × NFE cells: {len(cell_rows)}",
            f"- Scalar metric files read exactly once: "
            f"{len(cell_rows) * len(METRICS)}",
            f"- Selection artifact status: `{selection.get('status')}`",
            "",
            "## Effect definitions",
            "",
            "| Effect | Raw-scale paired contrast | Log-scale contrast |",
            "| --- | --- | --- |",
        ]
    )
    for effect in EFFECT_DEFINITIONS:
        log_label = LOG_EFFECT_DEFINITIONS[effect]
        if effect == "additive_interaction":
            log_label += " (multiplicative interaction)"
        lines.append(
            f"| `{effect}` | `{EFFECT_DEFINITIONS[effect]}` | "
            f"`{log_label}` |"
        )

    lines.extend(
        [
            "",
            "## Three-seed summaries",
            "",
            "| Profile | NFE | Metric | Effect | Mean Δ | Sample SD | "
            "95% t CI | Wins/3 | Geometric relative % | Relative 95% t CI |",
            "| --- | ---: | --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['profile']} | {row['nfe']} | {row['metric']} | "
            f"`{row['effect']}` | {fmt(float(row['mean_delta']))} | "
            f"{fmt(float(row['sample_sd_delta']))} | "
            f"[{fmt(float(row['ci95_low_delta']))}, "
            f"{fmt(float(row['ci95_high_delta']))}] | "
            f"{row['wins_of_3']} | "
            f"{fmt(float(row['geometric_relative_percent']))}% | "
            f"[{fmt(float(row['geometric_relative_ci95_low_percent']))}%, "
            f"{fmt(float(row['geometric_relative_ci95_high_percent']))}%] |"
        )
    lines.extend(
        [
            "",
            "The CSV files contain every per-cell value and every per-seed "
            "raw/log contrast used above.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)

    runs_root = args.runs_root.resolve()
    eval_root = args.eval_root.resolve()
    if not runs_root.is_dir():
        fail(f"runs root does not exist: {runs_root}")
    if not eval_root.is_dir():
        fail(f"evaluation root does not exist: {eval_root}")

    selection, selected_scale = load_selection(
        args.selection_json.resolve()
    )
    cell_rows = load_cells(runs_root, eval_root, selected_scale)
    per_seed_rows = build_per_seed_effects(cell_rows)
    heldout_rows = build_heldout_headlines(cell_rows)
    summary_rows = build_summary(per_seed_rows)

    expected_per_seed = (
        len(PROFILES) * len(TRAINING_SEEDS) * len(NFES) * len(METRICS)
    )
    expected_summary = (
        len(PROFILES) * len(NFES) * len(METRICS) * len(EFFECT_DEFINITIONS)
    )
    if len(per_seed_rows) != expected_per_seed:
        fail(
            f"internal per-seed row error: expected {expected_per_seed}, "
            f"got {len(per_seed_rows)}"
        )
    if len(summary_rows) != expected_summary:
        fail(
            f"internal summary row error: expected {expected_summary}, "
            f"got {len(summary_rows)}"
        )

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "per_cell_metrics.csv", cell_rows)
    write_csv(outdir / "per_seed_effects.csv", per_seed_rows)
    write_csv(outdir / "heldout_headlines.csv", heldout_rows)
    write_csv(outdir / "factorial_summary.csv", summary_rows)

    summary_json = {
        "schema_version": 2,
        "status": "passed",
        "evaluation_label": (
            "KID-5k and FID-5k 5,000-sample proxies; not standard "
            "50,000-sample benchmarks"
        ),
        "selected_global_scale": selected_scale,
        "selected_global_scale_text": selection[
            "selected_global_scale_text"
        ],
        "selected_label": selection["selected_label"],
        "training_seeds": list(TRAINING_SEEDS),
        "nfes": list(NFES),
        "metrics": list(METRICS),
        "profiles": list(PROFILES),
        "matrix": {
            "unique_training_cells": len(cell_rows) // len(NFES),
            "evaluated_training_seed_nfe_cells": len(cell_rows),
            "unique_metric_files_read_once": len(cell_rows) * len(METRICS),
            "fixed_and_global_cells_shared_across_profiles": True,
        },
        "statistics": {
            "design": "paired by training seed",
            "n": 3,
            "sample_sd_ddof": 1,
            "ci": "two-sided 95% Student-t interval around the paired mean",
            "degrees_of_freedom": 2,
            "t_critical": T_CRITICAL_DF2_95,
            "wins_definition": (
                "number of negative paired contrasts out of 3; lower is better"
            ),
            "inference_scope": (
                "descriptive only; n=3 is insufficient for broad "
                "significance claims"
            ),
        },
        "selection_overlap": {
            "present": True,
            "training_seed": 0,
            "description": (
                "seed 0 selected g* in the response curve and is reused in "
                "the fixed/global formal cells"
            ),
        },
        "heldout_headline": {
            "seeds": list(HELDOUT_SEEDS),
            "definition": (
                "100 * (arithmetic mean of arm metrics over seeds 1/2 / "
                "arithmetic mean of fixed metrics over seeds 1/2 - 1)"
            ),
            "not_equal_to": "mean of per-seed percentage changes",
            "rows": heldout_rows,
        },
        "effect_definitions": EFFECT_DEFINITIONS,
        "log_effect_definitions": LOG_EFFECT_DEFINITIONS,
        "interaction_note": (
            "The additive_interaction raw contrast is C-G-L+F. Its paired "
            "log contrast is the multiplicative interaction "
            "log(C*F/(G*L))."
        ),
        "summaries": summary_rows,
        "outputs": list(OUTPUT_NAMES),
    }
    (outdir / "factorial_summary.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (outdir / "factorial_summary.md").write_text(
        build_markdown(
            selected_scale,
            selection,
            cell_rows,
            heldout_rows,
            summary_rows,
        ),
        encoding="utf-8",
    )
    print(
        "Validated 18 unique training cells, 36 NFE cells, and 72 metric "
        f"files; selected g*={selected_scale:.6g}; output={outdir}"
    )


if __name__ == "__main__":
    main()
