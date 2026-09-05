#!/usr/bin/env python3
"""Summarize M1 endpoint metrics at the paired training-seed grain."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import build_m1_evaluation_slots as slot_contract


BRANCHES = ("K_A", "K_B", "R_A", "R_B")
BLOCKS = ("B0", "B1", "B2")
PASS = "SEALED_PASS"
SCIENTIFIC_MISSING = {"NOT_RUN_NO_ENDPOINT", "SCIENTIFIC_READOUT_INVALID"}
TECHNICAL = {"PLANNED", "RUNNING", "INCOMPLETE_TECHNICAL", "MISSING_RESULT"}
INVALID = {"INVALID_IMPLEMENTATION"}
KNOWN_STATUSES = {PASS} | SCIENTIFIC_MISSING | TECHNICAL | INVALID
T95 = {
    1: 12.7062047364, 2: 4.3026527297, 3: 3.1824463053,
    4: 2.7764451052, 5: 2.5705818366, 6: 2.4469118511,
    7: 2.3646242510, 8: 2.3060041350, 9: 2.2621571629,
    10: 2.2281388520, 11: 2.2009851601, 12: 2.1788128297,
    13: 2.1603686565, 14: 2.1447866879, 15: 2.1314495456,
}


class SummaryError(ValueError):
    pass


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _manifest_index(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    rows = list(rows)
    try:
        slot_contract.validate_slots(rows)
    except slot_contract.SlotError as exc:
        raise SummaryError(str(exc)) from exc
    index = {}
    for row in rows:
        slot_id = row.get("slot_id")
        if not isinstance(slot_id, str) or not slot_id or slot_id in index:
            raise SummaryError(f"invalid or duplicate manifest slot_id: {slot_id!r}")
        index[slot_id] = row
    return index


def _result_index(
    rows: Iterable[Mapping[str, Any]], expected: set[str]
) -> dict[str, Mapping[str, Any]]:
    index = {}
    for row in rows:
        slot_id = row.get("slot_id")
        if slot_id not in expected:
            raise SummaryError(f"unexpected result slot_id: {slot_id!r}")
        if slot_id in index:
            raise SummaryError(f"duplicate result slot_id: {slot_id}")
        status = row.get("fid_status", row.get("status"))
        if status not in KNOWN_STATUSES:
            raise SummaryError(f"unknown FID result status for {slot_id}: {status!r}")
        kid_status = row.get("kid_status")
        if kid_status is not None and kid_status not in KNOWN_STATUSES:
            raise SummaryError(f"unknown KID result status for {slot_id}: {kid_status!r}")
        index[slot_id] = row
    return index


def _seed_map(manifest: Mapping[str, Mapping[str, Any]]) -> dict[int, dict[tuple[str, str], str]]:
    seeds: dict[int, dict[tuple[str, str], str]] = {}
    for slot_id, row in manifest.items():
        try:
            seed = int(row["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SummaryError(f"invalid seed in manifest slot {slot_id}") from exc
        branch, readout, block = row.get("branch"), row.get("readout"), row.get("block")
        if branch in BRANCHES and readout == "E_512" and block in BLOCKS:
            key = (branch, block)
            if key in seeds.setdefault(seed, {}):
                raise SummaryError(f"duplicate E_512 cell for seed {seed}: {key}")
            seeds[seed][key] = slot_id
    if len(seeds) != 16 or any(len(cells) != 12 for cells in seeds.values()):
        raise SummaryError("manifest must contain 12 E_512 slots for each of 16 seeds")
    return seeds


def _status(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return "MISSING_RESULT"
    status = row.get("fid_status")
    return str(row.get("status") if status is None else status)


def _fid(row: Mapping[str, Any], slot_id: str) -> float:
    try:
        value = float(row["fid50k_full"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SummaryError(f"SEALED_PASS slot lacks numeric FID: {slot_id}") from exc
    if not math.isfinite(value) or value <= 0:
        raise SummaryError(f"SEALED_PASS slot has invalid FID: {slot_id}")
    return value


def _kid_status(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return "MISSING_RESULT"
    if row.get("kid_status") is not None:
        return str(row["kid_status"])
    return PASS if row.get("kid50k_full") not in (None, "") else "MISSING_RESULT"


def _kid(row: Mapping[str, Any], slot_id: str) -> float:
    try:
        value = float(row["kid50k_full"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SummaryError(f"SEALED_PASS slot lacks numeric KID: {slot_id}") from exc
    if not math.isfinite(value):
        raise SummaryError(f"SEALED_PASS slot has invalid KID: {slot_id}")
    return value


def _aggregate_status(statuses: Sequence[str]) -> str:
    if any(status in INVALID for status in statuses):
        return "INVALID_IMPLEMENTATION"
    if any(status in TECHNICAL for status in statuses):
        return "INCOMPLETE_TECHNICAL"
    if any(status in SCIENTIFIC_MISSING for status in statuses):
        return "SCIENTIFIC_MISSING"
    return "COMPLETE"


def _paired_block_details(
    cells: Mapping[tuple[str, str], str],
    results: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    details = []
    for block in BLOCKS:
        a_slot, b_slot = cells[("R_A", block)], cells[("R_B", block)]
        a_row, b_row = results.get(a_slot), results.get(b_slot)
        a_status, b_status = _status(a_row), _status(b_row)
        status = _aggregate_status((a_status, b_status))
        detail: dict[str, Any] = {
            "block": block,
            "status": status,
            "R_A_status": a_status,
            "R_B_status": b_status,
        }
        if a_status == PASS:
            detail["R_A_fid50k_full"] = _fid(a_row, a_slot)
        if b_status == PASS:
            detail["R_B_fid50k_full"] = _fid(b_row, b_slot)
        if status == "COMPLETE":
            a_fid = detail["R_A_fid50k_full"]
            b_fid = detail["R_B_fid50k_full"]
            log_difference = math.log(b_fid) - math.log(a_fid)
            direction = "B_BETTER" if log_difference < 0 else "A_BETTER" if log_difference > 0 else "EQUAL"
            detail.update(
                R_A_fid50k_full=a_fid,
                R_B_fid50k_full=b_fid,
                log_fid_difference=log_difference,
                direction=direction,
            )
        details.append(detail)
    return details


def build_description_rows(
    manifest: Mapping[str, Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for slot_id, slot in sorted(manifest.items(), key=lambda item: int(item[1]["slot_index"])):
        result = results.get(slot_id)
        fid_status, kid_status = _status(result), _kid_status(result)
        row = {
            key: slot[key]
            for key in ("slot_index", "slot_id", "roster_slot", "seed", "branch", "readout", "block")
        }
        row.update(
            fid_status=fid_status,
            fid50k_full=_fid(result, slot_id) if fid_status == PASS else "",
            kid_status=kid_status,
            kid50k_full=_kid(result, slot_id) if kid_status == PASS else "",
        )
        output.append(row)
    return output


def _interval(values: Sequence[float]) -> dict[str, Any]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"mean": mean, "sample_sd": None, "ci95": None}
    sd = statistics.stdev(values)
    if sd == 0:
        return {"mean": mean, "sample_sd": 0.0, "ci95": None}
    radius = T95[len(values) - 1] * sd / math.sqrt(len(values))
    return {"mean": mean, "sample_sd": sd, "ci95": [mean - radius, mean + radius]}


def _seed_contrast(
    cells: Mapping[tuple[str, str], str],
    results: Mapping[str, Mapping[str, Any]],
    branches: Sequence[str],
) -> tuple[str, float | None]:
    slot_ids = [cells[(branch, block)] for branch in branches for block in BLOCKS]
    statuses = [_status(results.get(slot_id)) for slot_id in slot_ids]
    if any(status in INVALID for status in statuses):
        return "INVALID_IMPLEMENTATION", None
    if any(status in TECHNICAL for status in statuses):
        return "INCOMPLETE_TECHNICAL", None
    if any(status in SCIENTIFIC_MISSING for status in statuses):
        return "SCIENTIFIC_MISSING", None
    values = {
        (branch, block): _fid(results[cells[(branch, block)]], cells[(branch, block)])
        for branch in branches for block in BLOCKS
    }
    if tuple(branches) == ("R_A", "R_B"):
        contrast = statistics.fmean(
            math.log(values[("R_B", block)]) - math.log(values[("R_A", block)])
            for block in BLOCKS
        )
    else:
        contrast = statistics.fmean(
            (math.log(values[("R_B", block)]) - math.log(values[("R_A", block)]))
            - (math.log(values[("K_B", block)]) - math.log(values[("K_A", block)]))
            for block in BLOCKS
        )
    return "COMPLETE", contrast


def summarize(
    manifest_rows: Iterable[Mapping[str, Any]],
    result_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    manifest = _manifest_index(manifest_rows)
    results = _result_index(result_rows, set(manifest))
    by_seed = _seed_map(manifest)
    status_counts = Counter(_status(results.get(slot_id)) for slot_id in manifest)
    kid_status_counts = Counter(_kid_status(results.get(slot_id)) for slot_id in manifest)

    primary_rows, secondary_rows = [], []
    arm_status_rows = []
    for seed in sorted(by_seed):
        primary_status, d_value = _seed_contrast(
            by_seed[seed], results, ("R_A", "R_B")
        )
        secondary_status, i_value = _seed_contrast(
            by_seed[seed], results, BRANCHES
        )
        primary_rows.append(
            {
                "seed": seed,
                "status": primary_status,
                "d": d_value,
                "blocks": _paired_block_details(by_seed[seed], results),
            }
        )
        secondary_rows.append({"seed": seed, "status": secondary_status, "i": i_value})
        arms = {}
        for branch in BRANCHES:
            block_statuses = {
                block: _status(results.get(by_seed[seed][(branch, block)]))
                for block in BLOCKS
            }
            arms[branch] = {
                "status": _aggregate_status(tuple(block_statuses.values())),
                "blocks": block_statuses,
            }
        arm_status_rows.append({"seed": seed, "arms": arms})

    primary = _finalize_primary(primary_rows)
    secondary = _finalize_secondary(secondary_rows)
    block_directions = Counter(
        block["direction"]
        for row in primary_rows
        for block in row["blocks"]
        if block["status"] == "COMPLETE"
    )
    block_directions["UNAVAILABLE"] = 48 - sum(block_directions.values())
    primary["block_direction_counts"] = {
        key: block_directions.get(key, 0)
        for key in ("B_BETTER", "A_BETTER", "EQUAL", "UNAVAILABLE")
    }
    descriptions = build_description_rows(manifest, results)
    if any(
        status_counts.get(status) or kid_status_counts.get(status)
        for status in INVALID
    ):
        matrix_status = "INVALID_IMPLEMENTATION"
    elif status_counts.get("MISSING_RESULT"):
        matrix_status = "INCOMPLETE_SLOT_LEDGER"
    elif any(
        status_counts.get(status) for status in TECHNICAL - {"MISSING_RESULT"}
    ):
        matrix_status = "INCOMPLETE_TECHNICAL"
    elif any(kid_status_counts.get(status) for status in TECHNICAL):
        matrix_status = "RESOLVED_WITH_KID_MISSINGNESS"
    else:
        matrix_status = "RESOLVED"
    return {
        "schema": "ect.m1.endpoint-summary/v1",
        "matrix_status": matrix_status,
        "planned_slots": len(manifest),
        "reported_slots": len(results),
        "slot_status_counts": dict(sorted(status_counts.items())),
        "fid_slot_status_counts": dict(sorted(status_counts.items())),
        "kid_slot_status_counts": dict(sorted(kid_status_counts.items())),
        "full_seed_arm_status": arm_status_rows,
        "description_table": {
            "rows": len(descriptions),
            "b0_three_readout_rows": sum(row["block"] == "B0" for row in descriptions),
            "planned_kid_rows": len(descriptions),
        },
        "primary": primary,
        "secondary": secondary,
    }


def _finalize_primary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row["status"] == "COMPLETE"]
    output = {
        "n_R": len(complete),
        "n_R_denominator": 16,
        "S_R": [row["seed"] for row in complete],
        "per_seed": rows,
    }
    statuses = {row["status"] for row in rows}
    if "INVALID_IMPLEMENTATION" in statuses:
        return output | {"status": "INVALID_IMPLEMENTATION_PRIMARY"}
    if "INCOMPLETE_TECHNICAL" in statuses:
        return output | {"status": "INCOMPLETE_TECHNICAL"}
    if len(complete) < 2:
        return output | {"status": "INSUFFICIENT_COMPLETE_R_PAIRS"}
    summary = _interval([row["d"] for row in complete])
    if summary["ci95"] is None:
        return output | summary | {
            "status": "DEGENERATE_ZERO_SD_DESCRIPTIVE_ONLY",
            "geometric_fid_ratio": math.exp(summary["mean"]),
            "geometric_fid_ratio_ci95": None,
            "improvement_percent": 100 * (1 - math.exp(summary["mean"])),
        }
    lower, upper = summary["ci95"]
    verdict = (
        "B_ADVANTAGE_SUPPORTED_CONDITIONAL" if upper < 0 else
        "B_DISADVANTAGE_SUPPORTED_CONDITIONAL" if lower > 0 else "INCONCLUSIVE"
    )
    ratio_ci = [math.exp(lower), math.exp(upper)]
    return output | summary | {
        "status": verdict,
        "geometric_fid_ratio": math.exp(summary["mean"]),
        "geometric_fid_ratio_ci95": ratio_ci,
        "improvement_percent": 100 * (1 - math.exp(summary["mean"])),
    }


def _finalize_secondary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row["status"] == "COMPLETE"]
    output = {
        "n_4": len(complete),
        "n_4_denominator": 16,
        "S_4": [row["seed"] for row in complete],
        "per_seed": rows,
    }
    statuses = {row["status"] for row in rows}
    if "INVALID_IMPLEMENTATION" in statuses:
        return output | {"status": "INVALID_IMPLEMENTATION_SECONDARY"}
    if "INCOMPLETE_TECHNICAL" in statuses:
        return output | {"status": "INCOMPLETE_TECHNICAL_SECONDARY"}
    if len(complete) < 2:
        return output | {"status": "INSUFFICIENT_COMPLETE_FOUR_ARM_PAIRS"}
    summary = _interval([row["i"] for row in complete])
    status = "ESTIMATE_ONLY" if summary["ci95"] is not None else "DEGENERATE_ZERO_SD"
    return output | summary | {"status": status}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--description-csv", type=Path, required=True)
    args = parser.parse_args()
    manifest_rows = read_csv(args.manifest_csv)
    result_rows = read_csv(args.results_csv)
    payload = summarize(manifest_rows, result_rows)
    payload["description_table"]["path"] = str(args.description_csv.resolve())
    manifest = _manifest_index(manifest_rows)
    results = _result_index(result_rows, set(manifest))
    descriptions = build_description_rows(manifest, results)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    args.description_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.description_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(descriptions[0]))
        writer.writeheader()
        writer.writerows(descriptions)
    print(f"M1_SUMMARY {payload['primary']['status']} n_R={payload['primary']['n_R']}")


if __name__ == "__main__":
    main()
