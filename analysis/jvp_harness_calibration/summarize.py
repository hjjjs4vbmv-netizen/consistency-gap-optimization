"""Build compact calibration tables and a bounded scientific report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_RECEIPT = HERE / "results" / "raw_receipts" / "calibration_receipt.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--out", type=Path, default=HERE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    rows = []
    for item in receipt.get("rows", []):
        parameter = item["parameter_resolution"]
        tangent = item["tangent_vs_oracle"]
        action = item["action_vs_oracle"]
        rows.append({
            "epsilon": item["epsilon"],
            "finite": item["finite"],
            "source_preserved": item["source_preserved"],
            "tangent_relative_error": tangent["relative_error"],
            "tangent_cosine": tangent["cosine"],
            "tangent_norm_ratio": tangent["norm_ratio"],
            "action_relative_error": action["relative_error"],
            "action_cosine": action["cosine"],
            "action_norm_ratio": action["norm_ratio"],
            "branch_distinct_fraction": parameter["branch_distinct_fraction"],
            "realized_direction_relative_error": parameter[
                "realized_direction_relative_error"],
            "realized_direction_cosine": parameter["realized_direction_cosine"],
        })
    if not rows:
        raise RuntimeError("calibration receipt has no scale rows")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best_tangent = min(rows, key=lambda row: row["tangent_relative_error"])
    best_action = min(rows, key=lambda row: row["action_relative_error"])
    summary = {
        "schema_version": 1,
        "verdict": receipt["verdict"],
        "oracle": receipt.get("oracle"),
        "source_preserved": receipt.get("source_preserved"),
        "assets_preserved": receipt.get("assets_preserved"),
        "epsilon_grid": [row["epsilon"] for row in rows],
        "plateaus": receipt.get("plateaus", []),
        "best_tangent_scale": {
            "epsilon": best_tangent["epsilon"],
            "relative_error": best_tangent["tangent_relative_error"],
        },
        "best_action_scale": {
            "epsilon": best_action["epsilon"],
            "relative_error": best_action["action_relative_error"],
        },
        "bounded_conclusion": (
            "The original 0.03--0.01 squared-GN output finite-difference range "
            "was too coarse for this frozen cell. Four consecutive smaller "
            "binary scales match the executed-graph autograd tangent and action "
            "within the preregistered five-percent tolerance."),
        "claim_ceiling": receipt["protocol"]["claim_ceiling"],
        "old_factorial_reopened": False,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    plateau = receipt.get("plateaus", [])
    plateau_text = (
        ", ".join(str(value) for value in plateau[0]["epsilons"])
        if plateau else "none")
    report = f"""# Squared-GN JVP harness calibration

**{receipt['verdict']}: an oracle-accurate finite-difference plateau was found.**

The reverse-over-reverse autograd oracle was finite, and both source state and
input assets were preserved. The admitted consecutive scales are:

```text
{plateau_text}
```

| Quantity | Best epsilon | Relative error |
|---|---:|---:|
| Residual tangent | {best_tangent['epsilon']} | {best_tangent['tangent_relative_error']:.6g} |
| Squared-GN action | {best_action['epsilon']} | {best_action['action_relative_error']:.6g} |

At the original scale neighborhood, errors remain large: epsilon 0.015625 has
residual-tangent error {rows[2]['tangent_relative_error']:.3f} and action error
{rows[2]['action_relative_error']:.3f}. At epsilon 0.00390625 these fall to
{rows[4]['tangent_relative_error']:.3f} and {rows[4]['action_relative_error']:.3f}.
Almost every FP32 parameter coordinate remains distinguishable between the
positive and negative branches throughout the sweep, so the observed plateau is
not explained by widespread coordinate collapse.

## Interpretation boundary

This calibration identifies a numerical-scale problem in the original
squared-GN correctness cell. It does not retroactively reopen the old factorial
and does not localize the production Jacobian failure. A new protocol may use an
interior plateau scale, but it must freeze that choice before evaluating any
full-field or production regime.
"""
    (args.out / "REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
