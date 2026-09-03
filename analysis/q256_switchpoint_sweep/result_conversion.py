from __future__ import annotations

import csv
import math
from pathlib import Path


SEEDS = tuple(range(81, 93))
SWITCH_POINTS = (128, 256, 384, 512)
ENDPOINTS = {128: 640, 256: 768, 384: 896, 512: 1024}
CSV_FIELDS = (
    "seed", "s_kimg", "ba_kimg", "ctrl_kimg", "ba_job_id", "ctrl_job_id",
    "ba_fid50k", "ctrl_fid50k", "G", "ba_valid", "ctrl_valid", "seed_complete",
    "root_cause",
)


def _boolean(value: str) -> bool:
    if value.lower() not in {"true", "false"}:
        raise ValueError(f"expected true/false, got {value!r}")
    return value.lower() == "true"


def _fid(row: dict) -> float | None:
    if row["status"] == "EXHAUSTED_FAILURE":
        return None
    if row["status"] != "PASS":
        raise ValueError(f"unexpected decoded status {row['status']!r}")
    value = float(row["fid50k_full"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("PASS decoded cell must have a finite positive FID")
    return value


def _expected_cells() -> dict[tuple[int, str, int], str]:
    expected = {}
    for seed in SEEDS:
        for kimg in (640, 768, 896, 1024):
            expected[(seed, "CTRL", kimg)] = "primary"
        for switch in SWITCH_POINTS:
            expected[(seed, f"BA{switch}", ENDPOINTS[switch])] = "primary"
        for switch in (128, 256, 384):
            expected[(seed, f"BA{switch}", 1024)] = "secondary"
    return expected


def convert_decoded(decoded: dict) -> tuple[list[dict[str, object]], dict[int, dict[int, float]]]:
    if decoded.get("status") != "PASS" or not isinstance(decoded.get("results"), list):
        raise ValueError("decoded_results.json is not a completed decode")
    expected = _expected_cells()
    cells = {}
    for row in decoded["results"]:
        key = (int(row["seed"]), str(row["trajectory"]), int(row["kimg"]))
        if key in cells:
            raise ValueError(f"duplicate decoded cell {key}")
        cells[key] = row
    if set(cells) != set(expected):
        raise ValueError("decoded results do not contain the exact 132-cell formal matrix")
    if any(cells[key]["role"] != role for key, role in expected.items()):
        raise ValueError("decoded cell role does not match the frozen matrix")

    rows: list[dict[str, object]] = []
    h_values = {switch: {} for switch in SWITCH_POINTS}
    for seed in SEEDS:
        primary = []
        for switch in SWITCH_POINTS:
            endpoint = ENDPOINTS[switch]
            primary.extend((cells[(seed, f"BA{switch}", endpoint)], cells[(seed, "CTRL", endpoint)]))
        failed = [row for row in primary if _fid(row) is None]
        causes = sorted({str(row.get("root_cause") or "UNCLASSIFIED_FAILURE") for row in failed})
        root_cause = " + ".join(causes)
        complete = not failed
        for switch in SWITCH_POINTS:
            endpoint = ENDPOINTS[switch]
            ba = cells[(seed, f"BA{switch}", endpoint)]
            ctrl = cells[(seed, "CTRL", endpoint)]
            ba_fid, ctrl_fid = _fid(ba), _fid(ctrl)
            rows.append({
                "seed": seed, "s_kimg": switch, "ba_kimg": endpoint, "ctrl_kimg": endpoint,
                "ba_job_id": ba["opaque_id"], "ctrl_job_id": ctrl["opaque_id"],
                "ba_fid50k": ba_fid, "ctrl_fid50k": ctrl_fid,
                "G": None if ba_fid is None or ctrl_fid is None else math.log(ba_fid) - math.log(ctrl_fid),
                "ba_valid": ba_fid is not None, "ctrl_valid": ctrl_fid is not None,
                "seed_complete": complete, "root_cause": root_cause,
            })
            common_ba = cells[(seed, f"BA{switch}", 1024)]
            common_ctrl = cells[(seed, "CTRL", 1024)]
            common_ba_fid, common_ctrl_fid = _fid(common_ba), _fid(common_ctrl)
            if common_ba_fid is not None and common_ctrl_fid is not None:
                h_values[switch][seed] = math.log(common_ba_fid) - math.log(common_ctrl_fid)
    return rows, h_values


def write_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({key: str(row[key]).lower() if isinstance(row[key], bool) else row[key] for key in CSV_FIELDS} for row in rows)


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            seed, switch = int(raw["seed"]), int(raw["s_kimg"])
            if switch not in ENDPOINTS:
                raise ValueError(f"seed {seed}: invalid switch point {switch}")
            endpoint = ENDPOINTS[switch]
            if int(raw["ba_kimg"]) != endpoint or int(raw["ctrl_kimg"]) != endpoint:
                raise ValueError(f"seed {seed}, s={switch}: fixed-chase endpoint must be {endpoint}")
            ba_valid, ctrl_valid = _boolean(raw["ba_valid"]), _boolean(raw["ctrl_valid"])
            ba_fid = float(raw["ba_fid50k"]) if raw["ba_fid50k"] else None
            ctrl_fid = float(raw["ctrl_fid50k"]) if raw["ctrl_fid50k"] else None
            if ba_valid and (ba_fid is None or not math.isfinite(ba_fid) or ba_fid <= 0):
                raise ValueError(f"seed {seed}, s={switch}: invalid BA FID")
            if ctrl_valid and (ctrl_fid is None or not math.isfinite(ctrl_fid) or ctrl_fid <= 0):
                raise ValueError(f"seed {seed}, s={switch}: invalid CTRL FID")
            rows.append({**raw, "seed": seed, "s_kimg": switch, "ba_valid": ba_valid,
                         "ctrl_valid": ctrl_valid, "ba_fid50k": ba_fid, "ctrl_fid50k": ctrl_fid})
    expected = [(seed, switch) for seed in SEEDS for switch in SWITCH_POINTS]
    if sorted((row["seed"], row["s_kimg"]) for row in rows) != expected:
        raise ValueError("input must contain exactly one row for every seed 81..92 and switch point")
    return sorted(rows, key=lambda row: (row["seed"], row["s_kimg"]))
