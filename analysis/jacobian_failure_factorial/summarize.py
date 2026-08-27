"""Summarize the complete frozen factorial into CSV, JSON, and Markdown."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol.json"
REQUIRED_COLUMNS = (
    "arm", "batch_id", "direction_id", "regime", "epsilon", "jvp_norm",
    "relative_error", "cosine", "finite", "amp_scale_plus",
    "amp_scale_minus", "step_executed_plus", "step_executed_minus",
    "same_amp_regime", "pass_convergence_gate",
)
EXTRA_COLUMNS = (
    "norm_ratio", "coarse_epsilon", "cell_status", "source_preserved",
    "amp_scale_before_plus", "amp_scale_before_minus",
    "overflow_detected_plus", "overflow_detected_minus",
    "same_discrete_state", "error_type", "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE)
    return parser.parse_args()


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def expected_keys(protocol: dict[str, Any]) -> set[tuple[str, int, int, str]]:
    return {
        (arm, int(batch), int(direction), regime)
        for arm in protocol["arms"]
        for batch in protocol["audit_minibatch_ids"]
        for direction in protocol["projection_direction_seeds"]
        for regime in protocol["regimes"]
    }


def load_cells(root: Path) -> dict[tuple[str, int, int, str], dict[str, Any]]:
    cells = {}
    for path in sorted(root.rglob("arm*_batch*_dir*_*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        key = (value["arm"], int(value["batch_id"]),
               int(value["direction_id"]), value["regime"])
        if key in cells:
            raise RuntimeError(f"duplicate cell {key}: {path}")
        value["_path"] = str(path.resolve())
        cells[key] = value
    return cells


def rows_for_cell(cell: dict[str, Any]) -> list[dict[str, Any]]:
    detail = cell.get("detail", {})
    convergence = detail.get("convergence", {})
    metrics = convergence.get("epsilon_metrics", [])
    branches = {float(item["epsilon"]): item
                for item in detail.get("branches", [])}
    if not metrics:
        metrics = [{"epsilon": epsilon, "jvp_norm": None, "finite": False,
                    "relative_error": None, "cosine": None,
                    "norm_ratio": None, "coarse_epsilon": None}
                   for epsilon in cell["epsilon_grid"]]
    result = []
    for metric in metrics:
        epsilon = float(metric["epsilon"])
        branch = branches.get(epsilon, {})
        plus = branch.get("plus", {})
        minus = branch.get("minus", {})
        result.append({
            "arm": cell["arm"],
            "batch_id": cell["batch_id"],
            "direction_id": cell["direction_id"],
            "regime": cell["regime"],
            "epsilon": epsilon,
            "jvp_norm": metric.get("jvp_norm"),
            "relative_error": metric.get("relative_error"),
            "cosine": metric.get("cosine"),
            "finite": bool(metric.get("finite", False) and detail.get("finite", False)),
            "amp_scale_plus": plus.get("grad_scale_after"),
            "amp_scale_minus": minus.get("grad_scale_after"),
            "step_executed_plus": branch.get("step_executed_plus"),
            "step_executed_minus": branch.get("step_executed_minus"),
            "same_amp_regime": branch.get("same_amp_regime"),
            "pass_convergence_gate": bool(convergence.get("passed", False)),
            "norm_ratio": metric.get("norm_ratio"),
            "coarse_epsilon": metric.get("coarse_epsilon"),
            "cell_status": cell["status"],
            "source_preserved": detail.get("source_preserved"),
            "amp_scale_before_plus": plus.get("grad_scale_before"),
            "amp_scale_before_minus": minus.get("grad_scale_before"),
            "overflow_detected_plus": branch.get("overflow_detected_plus"),
            "overflow_detected_minus": branch.get("overflow_detected_minus"),
            "same_discrete_state": branch.get("discrete_state_identical"),
            "error_type": cell.get("error_type"),
            "error": cell.get("error"),
        })
    return result


def regime_state(cells: list[dict[str, Any]]) -> str:
    passed = sum(cell["status"] == "PASS" for cell in cells)
    if passed == len(cells):
        return "PASS"
    if passed == 0:
        return "FAIL"
    return "MIXED"


def decide(cells: dict[tuple[str, int, int, str], dict[str, Any]],
           protocol: dict[str, Any]) -> tuple[str, str]:
    by_regime = defaultdict(list)
    for cell in cells.values():
        by_regime[cell["regime"]].append(cell)
    states = {regime: regime_state(by_regime[regime])
              for regime in protocol["regimes"]}
    if any(value == "MIXED" for value in states.values()):
        return "HOLD", "mixed outcomes across frozen arms, batches, or directions"
    a, b, c, d, e = (states[name] for name in protocol["regimes"])
    if a != "PASS":
        return "NO-GO", "squared-GN correctness baseline is not stable"
    if c == "FAIL" and e == "PASS":
        return "GO", "c=0 loss geometry materially contributes at the audited state"
    if b == "FAIL":
        if e == "PASS":
            return "GO", "loss-geometry source isolated by fixed smoothing diagnostic"
        return "HOLD", "instability enters with residual geometry but is not isolated"
    if b == "PASS" and c == "FAIL":
        return "GO", "network-curvature/full-field source at the audited state"
    if c == "PASS" and d == "FAIL":
        d_cells = by_regime["D_production_algorithmic"]
        mismatch = []
        for cell in d_cells:
            mismatch.append(not all(
                branch.get("same_amp_regime", False)
                for branch in cell.get("detail", {}).get("branches", [])))
        if mismatch and all(mismatch):
            return "GO", "production discrete-transition source"
        return "HOLD", "internal-FP16/stateful layer localized but source not separated"
    if all(value == "PASS" for value in (a, b, c, d, e)):
        return "GO", "earlier failure localizes to removed state-direction coordinates"
    return "HOLD", "frozen result pattern does not isolate one source"


def main() -> None:
    args = parse_args()
    protocol = load_protocol()
    cells = load_cells(args.receipt_root)
    expected = expected_keys(protocol)
    missing = sorted(expected - set(cells))
    unexpected = sorted(set(cells) - expected)
    complete = not missing and not unexpected
    rows = [row for key in sorted(cells) for row in rows_for_cell(cells[key])]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS + EXTRA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    if complete:
        verdict, conclusion = decide(cells, protocol)
    else:
        verdict, conclusion = "NO-GO", "formal factorial is incomplete"
    by_regime = {}
    for regime in protocol["regimes"]:
        subset = [cell for cell in cells.values() if cell["regime"] == regime]
        by_regime[regime] = {
            "expected_cells": 32,
            "observed_cells": len(subset),
            "status_counts": dict(Counter(cell["status"] for cell in subset)),
            "state": regime_state(subset) if subset else "MISSING",
        }
    summary = {
        "schema_version": 1,
        "complete": complete,
        "expected_cell_count": len(expected),
        "observed_cell_count": len(cells),
        "row_count": len(rows),
        "missing_cells": missing,
        "unexpected_cells": unexpected,
        "verdict": verdict,
        "bounded_conclusion": conclusion,
        "by_regime": by_regime,
        "claim_ceiling": (
            "One state, two frozen batches, four paired directions, and four "
            "diagnostic arms; no FID, training-quality, population, or global-clock claim."),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Jacobian failure factorial report", "",
        f"**{verdict}: {conclusion}.**", "",
        ("The frozen matrix is complete." if complete else
         f"The matrix is incomplete: {len(missing)} cells are missing."), "",
        "## Regime-level result", "",
        "| Regime | Pass | Fail | State |", "|---|---:|---:|---|",
    ]
    for regime, item in by_regime.items():
        counts = item["status_counts"]
        lines.append(
            f"| {regime} | {counts.get('PASS', 0)} | "
            f"{counts.get('FAIL_CLOSED', 0)} | {item['state']} |")
    lines.extend([
        "", "## Interpretation boundary", "",
        summary["claim_ceiling"], "",
        ("Cell-level values, including failures and AMP pairing metadata, are "
         "retained in `results.csv`; batches, directions, epsilons, regimes, and "
         "arms are paired repeated measurements rather than independent replicates."),
        "",
    ])
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
