#!/usr/bin/env python3
"""Compare exact computational state while ignoring serialization/timing bytes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training import pulse_chase, reproducibility


def tape(path: Path) -> list[tuple[str, str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            (row["attempted_iteration"], row["batch_sha256"], row["t_sha256"])
            for row in csv.DictReader(handle)
        ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uninterrupted", type=Path, required=True)
    parser.add_argument("--segmented", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left_path = args.uninterrupted.resolve(strict=True)
    right_path = args.segmented.resolve(strict=True)
    left = torch.load(left_path, map_location="cpu", weights_only=False)
    right = torch.load(right_path, map_location="cpu", weights_only=False)
    left_hashes = pulse_chase.internal_state_hashes(left)
    right_hashes = pulse_chase.internal_state_hashes(right)
    differing = sorted(key for key in left_hashes if left_hashes[key] != right_hashes[key])
    left_tape = tape(left_path.parent / "factorial_training_telemetry_v1.csv")
    right_tape = tape(right_path.parent / "factorial_training_telemetry_v1.csv")
    if left_tape != right_tape:
        differing.append("batch_t_tape")
    payload = {
        "schema": "ect.q256.p2-segmented-resume-parity/v1",
        "status": "COMPUTATIONAL_STATE_MATCH" if not differing else "FAIL_CLOSED",
        "uninterrupted": {"path": str(left_path),
                          "file_sha256": pulse_chase.sha256_file(left_path),
                          "internal": left_hashes},
        "segmented": {"path": str(right_path),
                      "file_sha256": pulse_chase.sha256_file(right_path),
                      "internal": right_hashes},
        "batch_t_tape_match": left_tape == right_tape,
        "differing_fields": differing,
    }
    reproducibility.atomic_json_dump(payload, args.output, overwrite=False)
    print(json.dumps({"status": payload["status"], "differing_fields": differing}))
    return 0 if not differing else 3


if __name__ == "__main__":
    raise SystemExit(main())
