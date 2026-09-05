#!/usr/bin/env python3
"""Expand the frozen M1 roster into the 320 endpoint evaluation slots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence


PROTOCOL_ID = "m1_r1_history_persistence_q256"
EVALUATOR_COMMIT = "d6aba02fb88e9db0993623895eb2228ed717d810"
METRIC_SEED = 20_260_730
BRANCHES = ("K_A", "K_B", "R_A", "R_B")
BLOCKS = {
    "B0": (0, 49_999),
    "B1": (50_000, 99_999),
    "B2": (100_000, 149_999),
}
READOUT_BLOCKS = {
    "ONLINE": ("B0",),
    "E_KEEP": ("B0",),
    "E_512": tuple(BLOCKS),
}
FIELDS = (
    "slot_index",
    "slot_id",
    "roster_slot",
    "seed",
    "branch",
    "readout",
    "block",
    "sample_seed_start",
    "sample_seed_end",
    "sample_count",
    "budget_kimg",
    "nfe",
    "precision",
    "metrics",
    "metric_seed",
    "evaluator_commit",
    "training_manifest_sha256",
    "implementation_commit",
    "frozen_source_state_sha256",
    "status",
)


class SlotError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_identity(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "ect.m1.training-run-manifest/v1"
        or payload.get("experiment_protocol") != PROTOCOL_ID
    ):
        raise SlotError("training manifest schema/protocol mismatch")
    implementation = payload.get("implementation_commit")
    if not isinstance(implementation, str) or re.fullmatch(r"[0-9a-f]{40}", implementation) is None:
        raise SlotError("training manifest implementation commit is invalid")
    runtime_receipt = payload.get("runtime_receipt")
    if not isinstance(runtime_receipt, dict):
        raise SlotError("training manifest runtime receipt identity is missing")
    runtime_path = Path(str(runtime_receipt.get("path", "")))
    if (
        not runtime_path.is_absolute() or runtime_path.is_symlink()
        or not runtime_path.is_file()
        or sha256_file(runtime_path) != runtime_receipt.get("sha256")
    ):
        raise SlotError("training manifest runtime receipt identity is invalid")
    output_root = Path(str(payload.get("output_root", "")))
    if not output_root.is_absolute():
        raise SlotError("training manifest output root must be absolute")
    roster = payload.get("roster")
    if not isinstance(roster, list) or len(roster) != 16:
        raise SlotError("training manifest must contain 16 roster rows")
    normalized = []
    sources = {}
    for index, row in enumerate(roster, start=1):
        if (
            not isinstance(row, dict)
            or row.get("roster_slot") != f"S{index:02d}"
            or not isinstance(row.get("seed"), int)
            or set(row.get("sources", {})) != {"A", "B"}
        ):
            raise SlotError("training manifest roster identity is invalid")
        normalized.append({"roster_slot": row["roster_slot"], "seed": row["seed"]})
        for arm in ("A", "B"):
            digest = row["sources"][arm].get("source_state_sha256")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise SlotError("training manifest source SHA256 is invalid")
            sources[(row["seed"], arm)] = digest
    seeds = [row["seed"] for row in normalized]
    if seeds != sorted(seeds) or len(set(seeds)) != 16 or any(seed not in range(50, 80) for seed in seeds):
        raise SlotError("training manifest roster seeds are invalid")
    return {
        "training_manifest_path": str(path),
        "training_manifest_sha256": sha256_file(path),
        "implementation_commit": implementation,
        "training_runtime_receipt_path": str(runtime_path),
        "training_runtime_receipt_sha256": runtime_receipt["sha256"],
        "output_root": str(output_root),
        "roster": normalized,
        "sources": sources,
    }


def normalize_roster(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        raise SlotError("roster input must be a frozen source inventory object")
    if value.get("schema") != "ect.m1.source-inventory/v1" or value.get("status") != "PASS":
        raise SlotError("source inventory schema/status mismatch")
    rows = value.get("candidates")
    if not isinstance(rows, list) or len(rows) != 30:
        raise SlotError("source inventory must contain seeds 50..79")
    if [row.get("seed") for row in rows if isinstance(row, dict)] != list(range(50, 80)):
        raise SlotError("source inventory candidates must be strictly ordered seeds 50..79")

    selected = []
    for row in rows:
        seed = row["seed"]
        checked, qualified = row.get("checked"), row.get("qualified")
        reason = row.get("reason")
        if not isinstance(checked, bool) or not isinstance(qualified, bool):
            raise SlotError(f"candidate {seed} checked/qualified must be boolean")
        if not isinstance(reason, str) or not reason.strip():
            raise SlotError(f"candidate {seed} requires a non-empty reason")
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", reason) is None:
            raise SlotError(f"candidate {seed} reason must be a machine-readable code")
        if qualified and not checked:
            raise SlotError(f"candidate {seed} cannot qualify without being checked")
        if len(selected) == 16:
            if checked:
                raise SlotError("candidates after the 16th qualified seed must remain unchecked")
            continue
        if not checked:
            raise SlotError("inventory scan cannot skip an unchecked seed before roster completion")
        if qualified:
            if reason != "QUALIFIED":
                raise SlotError(f"qualified candidate {seed} must use reason QUALIFIED")
            _validate_source_identity(row, seed)
        else:
            if reason == "QUALIFIED":
                raise SlotError(f"unqualified candidate {seed} needs a failure reason")
            _validate_incomplete_observations(row, seed)
        if qualified:
            selected.append({"roster_slot": f"S{len(selected) + 1:02d}", "seed": seed})
    if len(selected) != 16:
        raise SlotError(f"inventory contains only {len(selected)} checked qualified seeds")
    return selected


def _validate_source_identity(row: dict[str, Any], seed: int) -> None:
    sources = row.get("sources")
    if not isinstance(sources, dict) or set(sources) != {"A", "B"}:
        raise SlotError(f"checked candidate {seed} requires A/B source identities")
    for arm in ("A", "B"):
        source = sources[arm]
        if not isinstance(source, dict):
            raise SlotError(f"candidate {seed}/{arm} source identity must be an object")
        path = source.get("source_state_path")
        receipt = source.get("provenance_receipt_path")
        receipt_digest = source.get("provenance_receipt_sha256")
        size = source.get("source_state_bytes")
        digest = source.get("source_state_sha256")
        internal_digest = source.get("internal_state_sha256")
        if not isinstance(path, str) or not path or not isinstance(receipt, str) or not receipt:
            raise SlotError(f"candidate {seed}/{arm} lacks source paths")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise SlotError(f"candidate {seed}/{arm} has invalid source bytes")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise SlotError(f"candidate {seed}/{arm} has invalid source SHA256")
        if (
            not isinstance(receipt_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", receipt_digest) is None
        ):
            raise SlotError(f"candidate {seed}/{arm} has invalid provenance receipt SHA256")
        if not _valid_internal_hashes(internal_digest):
            raise SlotError(f"candidate {seed}/{arm} has invalid internal state SHA256")


def _valid_internal_hashes(value: Any) -> bool:
    if isinstance(value, str):
        return re.fullmatch(r"[0-9a-f]{64}", value) is not None
    if isinstance(value, list):
        return bool(value) and all(_valid_internal_hashes(item) for item in value)
    if isinstance(value, dict):
        return bool(value) and all(
            isinstance(key, str) and key and _valid_internal_hashes(item)
            for key, item in value.items()
        )
    return False


def _validate_incomplete_observations(row: dict[str, Any], seed: int) -> None:
    sources = row.get("sources")
    if not isinstance(sources, dict) or set(sources) != {"A", "B"}:
        raise SlotError(f"checked unqualified candidate {seed} requires A/B observations")
    fields = (
        "source_state_path", "source_state_bytes", "source_state_sha256",
        "provenance_receipt_path", "provenance_receipt_sha256",
    )
    for arm in ("A", "B"):
        source = sources[arm]
        if not isinstance(source, dict):
            raise SlotError(f"candidate {seed}/{arm} observations must be an object")
        for kind in ("expected", "actual"):
            observation = source.get(kind)
            if not isinstance(observation, dict) or set(observation) != set(fields):
                raise SlotError(
                    f"candidate {seed}/{arm} requires fixed expected/actual observations"
                )
            for field in ("source_state_path", "provenance_receipt_path"):
                value = observation[field]
                if value is not None and (not isinstance(value, str) or not value):
                    raise SlotError(f"candidate {seed}/{arm} has invalid {kind} {field}")
            size = observation["source_state_bytes"]
            if size is not None and (
                isinstance(size, bool) or not isinstance(size, int) or size < 0
            ):
                raise SlotError(
                    f"candidate {seed}/{arm} has invalid {kind} source_state_bytes"
                )
            digest = observation["source_state_sha256"]
            if digest is not None and (
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise SlotError(
                    f"candidate {seed}/{arm} has invalid {kind} source_state_sha256"
                )
            receipt_digest = observation["provenance_receipt_sha256"]
            if receipt_digest is not None and (
                not isinstance(receipt_digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", receipt_digest) is None
            ):
                raise SlotError(
                    f"candidate {seed}/{arm} has invalid {kind} provenance receipt SHA256"
                )


def validate_slots(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(rows)
    if len(rows) != 320:
        raise SlotError(f"M1 manifest must contain 320 slots, got {len(rows)}")
    roster_by_slot: dict[str, int] = {}
    for row in rows:
        roster_slot = str(row.get("roster_slot", ""))
        try:
            seed = int(row.get("seed", ""))
        except (TypeError, ValueError) as exc:
            raise SlotError("manifest contains a non-integer seed") from exc
        if roster_slot in roster_by_slot and roster_by_slot[roster_slot] != seed:
            raise SlotError(f"roster slot maps to multiple seeds: {roster_slot}")
        roster_by_slot[roster_slot] = seed
    expected_roster_slots = {f"S{index:02d}" for index in range(1, 17)}
    if set(roster_by_slot) != expected_roster_slots:
        raise SlotError("manifest roster slots must be exactly S01..S16")
    roster = [
        {"roster_slot": roster_slot, "seed": roster_by_slot[roster_slot]}
        for roster_slot in sorted(roster_by_slot)
    ]
    seeds = [row["seed"] for row in roster]
    if seeds != sorted(seeds) or any(seed not in range(50, 80) for seed in seeds):
        raise SlotError("manifest roster seeds must be strictly increasing within 50..79")
    training_hashes = {str(row.get("training_manifest_sha256", "")) for row in rows}
    implementations = {str(row.get("implementation_commit", "")) for row in rows}
    if (
        len(training_hashes) != 1
        or re.fullmatch(r"[0-9a-f]{64}", next(iter(training_hashes))) is None
        or len(implementations) != 1
        or re.fullmatch(r"[0-9a-f]{40}", next(iter(implementations))) is None
    ):
        raise SlotError("manifest training provenance is invalid or inconsistent")
    source_hashes = {}
    for row in rows:
        key = (int(row["seed"]), "B" if str(row["branch"]).endswith("_B") else "A")
        digest = str(row.get("frozen_source_state_sha256", ""))
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise SlotError("manifest frozen source SHA256 is invalid")
        if key in source_hashes and source_hashes[key] != digest:
            raise SlotError("manifest source SHA256 varies within a seed/arm")
        source_hashes[key] = digest
    identity = {
        "training_manifest_sha256": next(iter(training_hashes)),
        "implementation_commit": next(iter(implementations)),
        "sources": source_hashes,
    }
    expected_rows = build_slots(roster, identity)
    expected = [{field: str(row[field]) for field in FIELDS} for row in expected_rows]
    actual = [{field: str(row.get(field, "")) for field in FIELDS} for row in rows]
    if actual != expected:
        raise SlotError("manifest is not the canonical M1 320-slot expansion")
    return rows


def build_slots(
    roster: Sequence[dict[str, Any]], training_identity: dict[str, Any]
) -> list[dict[str, Any]]:
    slots = []
    for roster_row in roster:
        for branch in BRANCHES:
            for readout, blocks in READOUT_BLOCKS.items():
                for block in blocks:
                    start, end = BLOCKS[block]
                    slot_id = (
                        f"{roster_row['roster_slot']}-{branch}-{readout}-{block}"
                    )
                    slots.append(
                        {
                            "slot_index": len(slots),
                            "slot_id": slot_id,
                            "roster_slot": roster_row["roster_slot"],
                            "seed": roster_row["seed"],
                            "branch": branch,
                            "readout": readout,
                            "block": block,
                            "sample_seed_start": start,
                            "sample_seed_end": end,
                            "sample_count": 50_000,
                            "budget_kimg": 1024,
                            "nfe": 1,
                            "precision": "fp32",
                            "metrics": "kid50k_full,fid50k_full",
                            "metric_seed": METRIC_SEED,
                            "evaluator_commit": EVALUATOR_COMMIT,
                            "training_manifest_sha256": training_identity[
                                "training_manifest_sha256"
                            ],
                            "implementation_commit": training_identity[
                                "implementation_commit"
                            ],
                            "frozen_source_state_sha256": training_identity["sources"][
                                (
                                    roster_row["seed"],
                                    "B" if branch.endswith("_B") else "A",
                                )
                            ],
                            "status": "PLANNED",
                        }
                    )
    if len(slots) != 320 or len({row["slot_id"] for row in slots}) != 320:
        raise SlotError("M1 slot expansion did not produce 320 unique jobs")
    return slots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    identity = load_training_identity(args.training_manifest)
    slots = build_slots(identity["roster"], identity)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(slots)
    print(f"M1_EVALUATION_SLOTS_PASS jobs={len(slots)} protocol={PROTOCOL_ID}")


if __name__ == "__main__":
    main()
