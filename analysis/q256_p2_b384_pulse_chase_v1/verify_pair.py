#!/usr/bin/env python3
"""Verify same-source identity, matched randomness, and common-A chase."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training import pulse_chase, reproducibility


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-full-tape", action="store_true")
    args = parser.parse_args()
    seed_dir = args.seed_dir.resolve(strict=True)
    completions = {}
    manifests = {}
    for branch in pulse_chase.BRANCHES:
        root = seed_dir / branch
        completions[branch] = json.loads(
            (root / "trajectory_completion_receipt.json").read_text()
        )
        manifests[branch] = pulse_chase.load_run_manifest(
            root / "formal_run_manifest.json"
        )
    early = completions["Early-switch"]
    late = completions["Late-switch"]
    failures = []
    for key in ("source_state_sha256", "source_internal_state_sha256"):
        if early[key] != late[key]:
            failures.append(f"source identity mismatch: {key}")
    if manifests["Early-switch"]["source_state"] != manifests["Late-switch"]["source_state"]:
        failures.append("branch manifests do not name the identical source record")
    early512 = early["endpoints"][0]["training_state"]["internal_state_sha256"]
    late512 = late["endpoints"][0]["training_state"]["internal_state_sha256"]
    for key in (
        "rank_rng", "sampler", "data_cursor", "attempted_iteration", "cur_nimg"
    ):
        if early512[key] != late512[key]:
            failures.append(f"512 matched-state field differs: {key}")
    early_pulse = rows(seed_dir / "Early-switch" / "p2_pulse_training_telemetry_v1.csv")
    late_pulse = rows(seed_dir / "Late-switch" / "p2_pulse_training_telemetry_v1.csv")
    tape_fields = ["attempted_iteration", "batch_sha256", "t_sha256"]
    if args.require_full_tape:
        tape_fields += [
            "cuda_rng_before_loss_sha256", "cuda_rng_after_loss_sha256",
        ]
    if len(early_pulse) != len(late_pulse):
        failures.append("pulse tape row counts differ")
    else:
        for index, (a, b) in enumerate(zip(early_pulse, late_pulse), start=1):
            for key in tape_fields:
                if a.get(key) != b.get(key):
                    failures.append(f"pulse tape mismatch row {index}: {key}")
                    break
            if failures:
                break
    for branch in pulse_chase.BRANCHES:
        chase_rows = rows(seed_dir / branch / "p2_chase_training_telemetry_v1.csv")
        if any(row["arm"] != "A" for row in chase_rows):
            failures.append(f"{branch} chase is not uniformly objective A")
    seed = manifests["Early-switch"]["seed"]
    expected_order = (
        ["Early-switch", "Late-switch"]
        if seed % 2 else ["Late-switch", "Early-switch"]
    )
    order_receipt = json.loads((seed_dir / "branch_order_receipt.json").read_text())
    if order_receipt.get("order") != expected_order:
        failures.append("branch order is not counterbalanced")
    payload = {
        "schema": "ect.q256.p2-pair-integrity/v1",
        "status": "PASS" if not failures else "FAIL_CLOSED",
        "seed": seed,
        "source_identity": "MATCH" if not any(
            "source" in item for item in failures
        ) else "MISMATCH",
        "endpoint_512_rng_sampler_counters": "MATCH" if not any(
            "512" in item for item in failures
        ) else "MISMATCH",
        "matched_randomness_fields": tape_fields,
        "common_a_chase": not any("chase" in item for item in failures),
        "counterbalanced_order": expected_order,
        "failures": failures,
    }
    reproducibility.atomic_json_dump(payload, args.output, overwrite=False)
    print(json.dumps({"status": payload["status"], "seed": seed}))
    return 0 if not failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
