#!/usr/bin/env python3
"""Create a paper-ready budget-curve PDF from a validated paired CSV.

The input uses the same long-form schema as collect_multibudget_results.py.
Only a complete two-method paired matrix is accepted. The command writes:
  - budget_curves.pdf, .png, .svg: mean curves with sample-SD whiskers;
  - budget_curve_summary.csv: figure-ready means and sample SDs;
  - paired_summary.csv: descriptive paired deltas at each budget.

Example:
  python scripts/summarize_budget_curve.py --input-csv new_results.csv \
      --outdir paper/figures --baseline-method fixed --candidate-method global110
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # Supports both `python scripts/...py` and package-level tests.
    from .collect_multibudget_results import (
        aggregate_rows, paired_rows, plot_budget_curves, read_rows, validate, write_csv,
    )
except ImportError:
    from collect_multibudget_results import (
        aggregate_rows, paired_rows, plot_budget_curves, read_rows, validate, write_csv,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--baseline-method", default="fixed")
    parser.add_argument("--candidate-method", default="global110")
    args = parser.parse_args(argv)

    rows = read_rows(args.input_csv.resolve())
    matrix = validate(rows, args.baseline_method, args.candidate_method)
    paired = paired_rows(matrix)
    curves, summary = aggregate_rows(rows, paired, matrix)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "budget_curve_summary.csv", curves)
    write_csv(outdir / "paired_summary.csv", summary)
    plot_budget_curves(curves, matrix, outdir)
    print("Validated {} metric rows; wrote {}".format(len(rows), outdir / "budget_curves.pdf"))


if __name__ == "__main__":
    main()
