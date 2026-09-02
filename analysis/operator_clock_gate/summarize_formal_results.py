#!/usr/bin/env python3
"""Validate completeness and summarize a formal operator-clock result tree."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.operator_clock_gate import cli_common
from analysis.operator_clock_gate.core import write_json


def _distribution(values):
    values = [float(value) for value in values]
    return {
        "count": len(values), "min": min(values),
        "median": statistics.median(values), "max": max(values),
    }


def _rows(root: Path, kind: str):
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(root.joinpath(kind).glob("shard*/*_dir*.json"))]


def _coverage(rows):
    expected = {
        (arm, batch, direction)
        for arm in "ABCD"
        for batch in cli_common.protocol()["audit_minibatch_ids"]
        for direction in range(8)
    }
    keys = [(row["arm"], row["audit_minibatch_id"],
             row["projection_direction_index"]) for row in rows]
    counts = Counter(keys)
    return {
        "expected": len(expected), "observed": len(rows),
        "unique": len(counts),
        "missing": [list(item) for item in sorted(expected - set(counts))],
        "extra": [list(item) for item in sorted(set(counts) - expected)],
        "duplicates": [list(item) for item, count in counts.items() if count != 1],
    }


def build(root: Path) -> dict:
    field = _rows(root, "field")
    algorithmic = _rows(root, "algorithmic")
    field_coverage = _coverage(field)
    algorithmic_coverage = _coverage(algorithmic)
    tensor_files = sorted(root.glob("field/shard*/*.pt")) + sorted(
        root.glob("algorithmic/shard*/*.pt"))
    missing_tensor_pairs = []
    for path in sorted(root.glob("field/shard*/*_dir*.json")) + sorted(
            root.glob("algorithmic/shard*/*_dir*.json")):
        if not path.with_suffix(".pt").is_file():
            missing_tensor_pairs.append(str(path.with_suffix(".pt")))
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("field/shard*/field_jvp_manifest.json"))
        + sorted(root.glob("algorithmic/shard*/algorithmic_jvp_manifest.json"))
    ]
    matched_path = root / "matched" / "matched_micro_rollout.json"
    matched = json.loads(matched_path.read_text(encoding="utf-8"))
    completeness = (
        field_coverage["observed"] == field_coverage["unique"] == 128
        and not any(field_coverage[key] for key in ("missing", "extra", "duplicates"))
        and algorithmic_coverage["observed"] == algorithmic_coverage["unique"] == 128
        and not any(algorithmic_coverage[key]
                    for key in ("missing", "extra", "duplicates"))
        and not missing_tensor_pairs and len(tensor_files) == 256
        and len(manifests) == 4
        and all(item["source_files_preserved"] for item in manifests)
        and matched["status"] == "PASS" and matched["source_preserved"]
        and all(len(matched["branches"][arm]["steps"]) == 64 for arm in "ABCD")
        and all(set(matched["branches"][arm]["horizons"]) == {"1", "4", "16", "64"}
                for arm in "ABCD")
    )
    field_changes = [
        row["recompute_detach_field"]["convergence"]
        ["finest_adjacent_pair"]["relative_change"] for row in field]
    algorithmic_changes = [
        row["convergence"]["finest_adjacent_pair"]["relative_change"]
        for row in algorithmic]
    return {
        "schema_version": 1,
        "status": ("PASS_EXECUTION_COMPLETE_WITH_FAIL_CLOSED_JACOBIANS"
                   if completeness else "INVALID_INCOMPLETE"),
        "protocol": cli_common.protocol(),
        "coverage": {"field": field_coverage,
                     "algorithmic": algorithmic_coverage},
        "raw_tensors": {
            "file_count": len(tensor_files),
            "total_bytes": sum(path.stat().st_size for path in tensor_files),
            "missing_pairs": missing_tensor_pairs,
        },
        "manifests": {
            "count": len(manifests),
            "source_files_preserved": all(
                item["source_files_preserved"] for item in manifests),
            "statuses": dict(Counter(item["status"] for item in manifests)),
        },
        "field": {
            "status_counts": dict(Counter(row["status"] for row in field)),
            "source_preserved_count": sum(
                row["recompute_detach_field"]["source_preserved"] for row in field),
            "convergence_pass_count": sum(
                row["recompute_detach_field"]["convergence"]["passed"]
                for row in field),
            "finest_adjacent_relative_change": _distribution(field_changes),
            "by_arm": {
                arm: _distribution([
                    row["recompute_detach_field"]["convergence"]
                    ["finest_adjacent_pair"]["relative_change"]
                    for row in field if row["arm"] == arm]) for arm in "ABCD"
            },
        },
        "algorithmic": {
            "status_counts": dict(Counter(row["status"] for row in algorithmic)),
            "source_preserved_count": sum(row["source_preserved"]
                                           for row in algorithmic),
            "convergence_pass_count": sum(row["convergence"]["passed"]
                                           for row in algorithmic),
            "amp_pair_pass_count": sum(
                row["amp_skip_behavior_identical_all_eps"] for row in algorithmic),
            "amp_regime_pass_count": sum(
                row["amp_regime_identical_across_eps"] for row in algorithmic),
            "discrete_pair_pass_count": sum(
                row["discrete_state_behavior_identical_all_eps"]
                for row in algorithmic),
            "finest_adjacent_relative_change": _distribution(algorithmic_changes),
        },
        "matched": {
            "status": matched["status"],
            "source_preserved": matched["source_preserved"],
            "amp_skip_all_arms_identical": matched["amp_skip_all_arms_identical"],
            "steps_by_arm": {arm: len(matched["branches"][arm]["steps"])
                             for arm in "ABCD"},
            "amp_skips_by_arm": {
                arm: sum(step["step_skipped"]
                         for step in matched["branches"][arm]["steps"])
                for arm in "ABCD"
            },
            "horizons": matched["horizons"],
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build(args.results)
    write_json(args.out, payload)
    return 0 if payload["status"].startswith("PASS_") else 3


if __name__ == "__main__":
    raise SystemExit(main())
