#!/usr/bin/env python3
"""Verify the realized common-random-number stream for one M1 seed."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


BRANCHES = ("K_A", "K_B", "R_A", "R_B")
PROTOCOL_ID = "m1_r1_history_persistence_q256"
RUN_MANIFEST_SCHEMA = "ect.q256.schedule-switch-run-manifest/v1"
CRN_FIELDS = (
    "attempted_iteration",
    "seed",
    "batch_sha256",
    "t_sha256",
    "base_r_sha256",
    "eps_sha256",
    "dropout_rng_sha256",
    "online_input_sha256",
    "target_input_sha256",
)
WINDOWS = {
    "gate16": (4001, 4016),
    "gate32": (4001, 4032),
    "formal": (4001, 8000),
}


def load_series(
    path: Path, branch: str, seed: int, mode: str
) -> list[tuple[str, ...]]:
    if mode not in WINDOWS:
        raise RuntimeError(f"invalid M1 CRN mode: {mode}")
    first, last = WINDOWS[mode]
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != last - first + 1:
        raise RuntimeError(
            f"{branch} {mode} telemetry must contain {last - first + 1} attempts"
        )
    if [int(row["attempted_iteration"]) for row in rows] != list(
        range(first, last + 1)
    ):
        raise RuntimeError(f"{branch} attempt coverage mismatch")
    if any(
        row.get("branch") != branch
        or row.get("continuation_arm") != "A"
        or int(row.get("seed", -1)) != seed
        for row in rows
    ):
        raise RuntimeError(f"{branch} seed, identity, or continuation mismatch")
    try:
        return [tuple(row[field] for field in CRN_FIELDS) for row in rows]
    except KeyError as exc:
        raise RuntimeError(f"{branch} lacks M1 CRN field: {exc.args[0]}") from exc


def verify(paths: dict[str, Path], seed: int, mode: str) -> str:
    if set(paths) != set(BRANCHES):
        raise RuntimeError("CRN verification requires exactly four M1 branches")
    series = [
        load_series(paths[branch], branch, seed, mode) for branch in BRANCHES
    ]
    if any(value != series[0] for value in series[1:]):
        raise RuntimeError("M1 batch/t/base_r/noise/dropout CRN mismatch")
    payload = json.dumps(series[0], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_pair(
    paths: dict[str, Path], seed: int, mode: str, branches: tuple[str, str]
) -> str:
    if len(set(branches)) != 2 or set(paths) != set(branches):
        raise RuntimeError("CRN pair verification requires the named branch pair")
    series = [load_series(paths[branch], branch, seed, mode) for branch in branches]
    if series[0] != series[1]:
        raise RuntimeError("M1 batch/t/base_r/noise/dropout CRN mismatch")
    payload = json.dumps(series[0], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_manifest_bindings(paths: dict[str, Path]) -> tuple[int, dict[str, str]]:
    if set(paths) != set(BRANCHES):
        raise RuntimeError("manifest binding requires exactly four M1 branches")
    seeds = set()
    hashes = {}
    for branch in BRANCHES:
        raw = paths[branch].read_bytes()
        manifest = json.loads(raw)
        if (
            manifest.get("schema") != RUN_MANIFEST_SCHEMA
            or manifest.get("experiment_protocol") != PROTOCOL_ID
            or manifest.get("branch") != branch
            or manifest.get("continuation_arm") != "A"
        ):
            raise RuntimeError(f"{branch} manifest identity mismatch")
        seed = manifest.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise RuntimeError(f"{branch} manifest seed is invalid")
        if seed not in range(50, 80):
            raise RuntimeError(f"{branch} manifest seed is outside M1 cohort")
        seeds.add(seed)
        hashes[branch] = hashlib.sha256(raw).hexdigest()
    if len(seeds) != 1:
        raise RuntimeError("M1 manifests do not bind one common seed")
    return seeds.pop(), hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--telemetry", action="append", required=True,
        help="BRANCH=/absolute/path/to/schedule_switch_training_telemetry_v1.csv",
    )
    parser.add_argument(
        "--manifest", action="append", required=True,
        help="BRANCH=/absolute/path/to/formal_run_manifest.json",
    )
    parser.add_argument("--mode", choices=tuple(WINDOWS), required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    def bindings(values, label):
        result = {}
        for value in values:
            branch, separator, raw_path = value.partition("=")
            if not separator or branch in result:
                raise RuntimeError(f"invalid {label} binding: {value}")
            result[branch] = Path(raw_path).resolve(strict=True)
        return result

    paths = bindings(args.telemetry, "telemetry")
    manifests = bindings(args.manifest, "manifest")
    seed, manifest_hashes = load_manifest_bindings(manifests)
    series_sha256 = verify(paths, seed, args.mode)
    first, last = WINDOWS[args.mode]
    receipt = {
        "schema": "ect.m1.crn-gate/v1",
        "status": "PASS",
        "seed": seed,
        "mode": args.mode,
        "branches": list(BRANCHES),
        "fields": list(CRN_FIELDS),
        "attempts": [first, last],
        "series_sha256": series_sha256,
        "manifest_sha256_by_branch": manifest_hashes,
    }
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from training import reproducibility
    reproducibility.atomic_json_dump(
        receipt, args.receipt.resolve(), overwrite=False
    )
    print(f"M1_CRN_PASS seed={seed} sha256={series_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
