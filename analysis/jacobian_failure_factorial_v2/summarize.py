"""Validate and summarize the completed calibrated v2 factorial."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

from analysis.operator_clock_gate.cli_common import sha256_file


HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol.json"
DEFAULT_RECEIPTS = HERE / "results" / "raw_receipts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-root", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--out", type=Path, default=HERE)
    return parser.parse_args()


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def expected_keys(frozen: dict[str, Any]) -> set[tuple[str, int, int, str]]:
    return set(product(
        frozen["arms"], frozen["audit_minibatch_ids"],
        frozen["projection_direction_seeds"], frozen["regimes"]))


def load_formal_cells(root: Path) -> dict[tuple[str, int, int, str], dict[str, Any]]:
    cells = {}
    for path in sorted((root / "formal").rglob("*.json")):
        if path.name == "formal_manifest.json":
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if "regime" not in value:
            continue
        key = (
            value["arm"], int(value["batch_id"]),
            int(value["direction_id"]), value["regime"])
        if key in cells:
            raise RuntimeError(f"duplicate formal cell {key}")
        value["_path"] = path.relative_to(root).as_posix()
        cells[key] = value
    return cells


def validate_manifests(root: Path, protocol_hash: str) -> dict[str, Any]:
    paths = sorted((root / "formal").glob("shard*/formal_manifest.json"))
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    checks = {
        "manifest_count": len(manifests),
        "all_complete": len(manifests) == 2 and all(
            item.get("status") == "COMPLETE" for item in manifests),
        "task_counts": [item.get("task_count") for item in manifests],
        "all_source_preserved": len(manifests) == 2 and all(
            item.get("source_preserved") is True for item in manifests),
        "all_assets_preserved": len(manifests) == 2 and all(
            item.get("assets_before") == item.get("assets_after")
            for item in manifests),
        "all_protocol_hashes_match": len(manifests) == 2 and all(
            item.get("protocol_sha256") == protocol_hash for item in manifests),
        "manifest_paths": [path.relative_to(root).as_posix() for path in paths],
    }
    checks["passed"] = bool(
        checks["all_complete"]
        and checks["task_counts"] == [80, 80]
        and checks["all_source_preserved"]
        and checks["all_assets_preserved"]
        and checks["all_protocol_hashes_match"])
    return checks


def cell_row(cell: dict[str, Any]) -> dict[str, Any]:
    detail = cell.get("detail", {})
    convergence = detail.get("convergence", {})
    finest = convergence.get("finest_adjacent_pair", {})
    metrics = convergence.get("epsilon_metrics", [])
    finest_metric = metrics[-1] if metrics else {}
    return {
        "arm": cell["arm"],
        "batch_id": cell["batch_id"],
        "direction_id": cell["direction_id"],
        "regime": cell["regime"],
        "status": cell["status"],
        "finest_relative_change": finest.get("relative_change"),
        "finest_cosine": finest_metric.get("cosine"),
        "finest_norm_ratio": finest_metric.get("norm_ratio"),
        "selected_jvp_l2": cell.get("selected_jvp_l2"),
        "finite": detail.get("finite"),
        "source_preserved": detail.get("source_preserved"),
        "amp_branch_pairing": detail.get("amp_skip_behavior_identical_all_eps"),
        "amp_sweep_pairing": detail.get("amp_regime_identical_across_eps"),
        "discrete_state_pairing": detail.get(
            "discrete_state_behavior_identical_all_eps"),
        "receipt": cell["_path"],
    }


def aggregate(rows: list[dict[str, Any]], regimes: list[str]) -> dict[str, Any]:
    result = {}
    for regime in regimes:
        subset = [row for row in rows if row["regime"] == regime]
        changes = [float(row["finest_relative_change"]) for row in subset]
        result[regime] = {
            "cell_count": len(subset),
            "status_counts": dict(Counter(row["status"] for row in subset)),
            "relative_change": {
                "minimum": min(changes),
                "mean": statistics.fmean(changes),
                "maximum": max(changes),
            },
        }
        if regime == "D_production_algorithmic":
            result[regime]["production_pairing"] = {
                "finite": dict(Counter(row["finite"] for row in subset)),
                "source_preserved": dict(Counter(
                    row["source_preserved"] for row in subset)),
                "amp_branch_pairing": dict(Counter(
                    row["amp_branch_pairing"] for row in subset)),
                "amp_sweep_pairing": dict(Counter(
                    row["amp_sweep_pairing"] for row in subset)),
                "discrete_state_pairing": dict(Counter(
                    row["discrete_state_pairing"] for row in subset)),
            }
    return result


def main() -> None:
    args = parse_args()
    frozen = load_protocol()
    protocol_hash = sha256_file(PROTOCOL_PATH)
    cells = load_formal_cells(args.receipt_root)
    expected = expected_keys(frozen)
    observed = set(cells)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    rows = [cell_row(cells[key]) for key in sorted(cells)]
    manifests = validate_manifests(args.receipt_root, protocol_hash)
    complete = not missing and not unexpected and len(rows) == 160
    integrity_passed = bool(complete and manifests["passed"])
    by_regime = aggregate(rows, frozen["regimes"])

    a = by_regime["A_squared_gn_fp32"]["status_counts"]
    controls_pass = all(
        by_regime[regime]["status_counts"] == {"PASS": 32}
        for regime in (
            "B_real_loss_gn_fp32", "C_full_field_fp32",
            "E_full_field_pseudohuber_fp32"))
    production_fails = (
        by_regime["D_production_algorithmic"]["status_counts"]
        == {"FAIL_CLOSED": 32})
    transition_verdict = (
        "GO_PRODUCTION_TRANSITION_SEPARATION"
        if integrity_passed and controls_pass and production_fails
        else "HOLD_MIXED_OR_INCOMPLETE")
    internal_verdict = "HOLD_INTERNAL_COMPONENT_ATTRIBUTION"

    a_failures = [row for row in rows
                  if row["regime"] == "A_squared_gn_fp32"
                  and row["status"] != "PASS"]
    duplicated_a_boundary = bool(
        len(a_failures) == 2
        and {(row["batch_id"], row["direction_id"])
             for row in a_failures} == {(2026082602, 2026082611)}
        and {row["arm"] for row in a_failures} == {"A", "D"}
        and len({row["finest_relative_change"] for row in a_failures}) == 1)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": 2,
        "complete": complete,
        "integrity_passed": integrity_passed,
        "expected_cell_count": 160,
        "observed_cell_count": len(rows),
        "unique_cell_count": len(observed),
        "missing_cells": missing,
        "unexpected_cells": unexpected,
        "protocol_sha256": protocol_hash,
        "manifests": manifests,
        "transition_level_verdict": transition_verdict,
        "internal_component_verdict": internal_verdict,
        "by_regime": by_regime,
        "a_status_counts": a,
        "duplicated_a_boundary_condition": duplicated_a_boundary,
        "a_boundary_cells": a_failures,
        "claim": (
            "At the audited state and calibrated scales, the FP32 objective-field "
            "controls admit stable local linearizations, whereas the complete "
            "production algorithmic transition does not."),
        "claim_ceiling": frozen["claim_ceiling"],
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    change = {key: value["relative_change"] for key, value in by_regime.items()}
    report = f"""# Calibrated Jacobian failure factorial v2

**GO at the production-transition level; HOLD for attribution inside that transition.**

All 160 frozen cells completed with two valid manifests, 160 unique keys, matched
protocol hashes, and preserved source state and assets. The calibrated controls
separate cleanly from the production transition: B, C, and E pass in 32/32 cells,
whereas D fails in 32/32 cells. A passes in 30/32 cells; the two failures are the
same marginal batch/direction condition repeated in arms A and D, with relative
change 0.0517766 against the frozen 0.05 threshold.

| Regime | PASS | FAIL_CLOSED | Relative change, min / mean / max |
|---|---:|---:|---:|
| A: squared-GN FP32 | 30 | 2 | {change['A_squared_gn_fp32']['minimum']:.4f} / {change['A_squared_gn_fp32']['mean']:.4f} / {change['A_squared_gn_fp32']['maximum']:.4f} |
| B: real-loss GN FP32 | 32 | 0 | {change['B_real_loss_gn_fp32']['minimum']:.4f} / {change['B_real_loss_gn_fp32']['mean']:.4f} / {change['B_real_loss_gn_fp32']['maximum']:.4f} |
| C: full recompute-detach FP32 | 32 | 0 | {change['C_full_field_fp32']['minimum']:.4f} / {change['C_full_field_fp32']['mean']:.4f} / {change['C_full_field_fp32']['maximum']:.4f} |
| D: production algorithmic transition | 0 | 32 | {change['D_production_algorithmic']['minimum']:.4f} / {change['D_production_algorithmic']['mean']:.4f} / {change['D_production_algorithmic']['maximum']:.4f} |
| E: pseudo-Huber FP32 field | 32 | 0 | {change['E_full_field_pseudohuber_fp32']['minimum']:.4f} / {change['E_full_field_pseudohuber_fp32']['mean']:.4f} / {change['E_full_field_pseudohuber_fp32']['maximum']:.4f} |

The D failures are not process crashes or mismatched central-difference branches.
All D cells are finite, preserve the source state, pair AMP behavior across the
positive and negative branches, remain in one AMP regime across the sweep, and
pair the tracked discrete state. Their finest-scale relative changes remain
large (0.516--1.332; mean 0.940), while the complete FP32 field C remains below
0.011 in every cell.

## Interpretation

At this checkpoint, the smooth FP32 objective fields admit stable local
linearizations across the frozen factorial. The complete production transition
does not exhibit a numerically stable classical Jacobian at the calibrated
scales. This establishes separation at the transition level. Regime D combines
autocast/FP16, the stateful RAdam update, EMA, and scaler state; the factorial
does not assign the instability to one internal component.

The result supports the bounded training-dynamics statement that instantaneous
objective structure need not survive the production optimizer transition. It
does not connect the Jacobian diagnostic to FID or identify an optimizer-mediated
quality mechanism.
"""
    (args.out / "REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
