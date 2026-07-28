#!/usr/bin/env python3
"""Fail-closed validation for one completed gap-factorial training arm."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"[verify_gap_factorial_arm] ERROR: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(row: dict[str, str], field: str, row_number: int) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        fail(f"row {row_number}: invalid {field}: {exc}")
    if not math.isfinite(value):
        fail(f"row {row_number}: non-finite {field}: {value}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-kimg", type=float, required=True)
    parser.add_argument("--expected-schedule", required=True)
    parser.add_argument("--require-controller-active", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    required = [
        "experiment_meta.env",
        "training_options.json",
        "train_summary.csv",
        "runner.log",
        "network-snapshot-latest.pkl",
        "training-state-latest.pt",
        "checkpoint.sha256",
    ]
    for name in required:
        path = run_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            fail(f"missing or empty required output: {path}")

    numbered = [
        *run_dir.glob("network-snapshot-[0-9]*.pkl"),
        *run_dir.glob("training-state-[0-9]*.pt"),
    ]
    if numbered:
        fail(f"unexpected redundant numbered checkpoints: {numbered}")

    meta = (run_dir / "experiment_meta.env").read_text(encoding="utf-8")
    if "exit_code=0\n" not in meta:
        fail("experiment metadata does not record exit_code=0")
    if "Exiting..." not in (run_dir / "runner.log").read_text(
        encoding="utf-8", errors="replace"
    ):
        fail("runner log does not contain the training completion marker")

    with (run_dir / "train_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        rows = list(reader)
    if not rows:
        fail("train_summary.csv contains no data rows")

    geometry_fields = {
        "gap_over_sigmoid_gap_mean",
        "lower_gap_clip_rate",
        "upper_gap_clip_rate",
    }
    present_geometry_fields = geometry_fields & fieldnames
    if present_geometry_fields and present_geometry_fields != geometry_fields:
        fail(
            "partial realized-gap diagnostic schema: "
            f"{sorted(present_geometry_fields)}"
        )
    geometry_recorded = present_geometry_fields == geometry_fields

    schedules = {row.get("schedule") for row in rows}
    if schedules != {args.expected_schedule}:
        fail(f"schedule mismatch: expected {args.expected_schedule}, got {schedules}")

    skipped_steps = 0
    for index, row in enumerate(rows, start=2):
        finite_float(row, "loss", index)
        if finite_float(row, "grad_scale", index) <= 0:
            fail(f"row {index}: grad_scale must be positive")
        finite_float(row, "r_over_t_mean", index)
        finite_float(row, "gap_mean", index)
        if geometry_recorded:
            if finite_float(row, "gap_over_sigmoid_gap_mean", index) < 0:
                fail(
                    f"row {index}: gap_over_sigmoid_gap_mean must be "
                    "non-negative"
                )
            for field in ("lower_gap_clip_rate", "upper_gap_clip_rate"):
                value = finite_float(row, field, index)
                if not 0 <= value <= 1:
                    fail(f"row {index}: {field} must be in [0, 1]")
        try:
            skipped = int(row["step_skipped"])
        except (KeyError, TypeError, ValueError) as exc:
            fail(f"row {index}: invalid step_skipped: {exc}")
        if skipped not in (0, 1):
            fail(f"row {index}: step_skipped must be 0 or 1")
        skipped_steps += skipped

    final_kimg = finite_float(rows[-1], "processed_kimg", len(rows) + 1)
    if not args.expected_kimg <= final_kimg < args.expected_kimg + 0.128:
        fail(
            f"final processed_kimg {final_kimg} is outside "
            f"[{args.expected_kimg}, {args.expected_kimg + 0.128})"
        )
    # The frozen GradScaler starts at 65536 and normally needs roughly nine
    # overflow-driven reductions before reaching its stable scale.  Prior
    # paired baselines show 9--10 total skips, occasionally late in training.
    if skipped_steps > 16:
        fail(f"unexpected AMP instability: {skipped_steps} skipped steps")

    if args.require_controller_active:
        active = rows[-1].get("adaptive_active")
        signal_updates = int(rows[-1].get("signal_updates", "0"))
        correction = finite_float(rows[-1], "correction", len(rows) + 1)
        if active not in {"1", "True", "true"}:
            fail("local controller is not active in the final row")
        if signal_updates <= 64:
            fail(f"local controller has only {signal_updates} signal updates")
        if correction == 0:
            fail("local controller final correction is zero")

    checkpoint = run_dir / "network-snapshot-latest.pkl"
    recorded = (run_dir / "checkpoint.sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    actual = sha256(checkpoint)
    if recorded != actual:
        fail(f"checkpoint hash mismatch: recorded {recorded}, actual {actual}")

    report = {
        "status": "passed",
        "run_dir": str(run_dir),
        "expected_schedule": args.expected_schedule,
        "rows": len(rows),
        "final_processed_kimg": final_kimg,
        "skipped_steps": skipped_steps,
        "controller_required": args.require_controller_active,
        "final_controller_active": rows[-1].get("adaptive_active"),
        "final_signal_updates": int(rows[-1].get("signal_updates", "0")),
        "final_correction": finite_float(
            rows[-1], "correction", len(rows) + 1
        ),
        "realized_gap_diagnostics": {
            "status": (
                "recorded"
                if geometry_recorded
                else "not_recorded_pre_instrumentation"
            ),
            "fields": sorted(geometry_fields),
        },
        "checkpoint_sha256": actual,
    }
    (run_dir / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
