#!/usr/bin/env python3
"""Verify one immutable B@384 continuation branch and write its receipt."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training import reproducibility


ARMS = {"A": (1.0, 1.0), "B": (1.1, 1.1), "C": (1.1, 1.0), "D": (1.0, 1.1)}
EXOGENOUS = (
    "batch_sha256", "t_sha256", "base_r_sha256", "input_noise_sha256",
    "dropout_rng_sha256", "augmentation_rng_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nonfinite(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int((~torch.isfinite(value)).sum()) if value.is_floating_point() else 0
    if isinstance(value, dict):
        return sum(nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(nonfinite(item) for item in value)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(3, 4, 5), required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--end-kimg", type=int, choices=(448, 512), required=True)
    args = parser.parse_args()
    failures: list[str] = []
    source_after = sha256_file(args.source_state)
    if source_after != args.source_sha256:
        failures.append("source state SHA256 changed")
    final_path = args.run_dir / f"training-state-kimg{args.end_kimg:06d}.pt"
    if not final_path.is_file():
        failures.append("final full state missing")
        state = None
    else:
        state = torch.load(final_path, map_location="cpu", weights_only=False)
    expected_attempt = args.end_kimg * 1000 // 128
    if state is not None:
        target, denominator = ARMS[args.arm]
        if int(state.get("cur_nimg", -1)) != args.end_kimg * 1000:
            failures.append("final cur_nimg mismatch")
        if int(state.get("attempted_iteration", -1)) != expected_attempt:
            failures.append("final attempted_iteration mismatch")
        factorial = state.get("factorial", {})
        if factorial.get("arm") != args.arm:
            failures.append("continuation arm mismatch")
        if float(factorial.get("target_gap_scale", -1)) != target:
            failures.append("target scale mismatch")
        if float(factorial.get("denominator_gap_scale", -1)) != denominator:
            failures.append("denominator scale mismatch")
        fork = state.get("same_state_fork", {})
        expected_fork = {
            "schema": "ect.q256.same-state-fork/v1", "origin_arm": "B",
            "source_kimg": 384, "protocol_sha256": args.protocol_sha256,
            "continuation_arm": args.arm, "branch_label": f"B384_to_{args.arm}",
            "source_attempted_iteration": 3000, "source_cur_nimg": 384000,
        }
        if fork != expected_fork:
            failures.append("origin-B fork metadata mismatch")
        rank_states = state.get("rank_states", [])
        if len(rank_states) != 1 or int(rank_states[0].get("sampler_state", {}).get("consumed_samples", -1)) != args.end_kimg * 1000:
            failures.append("sampler endpoint mismatch")
        if nonfinite({
            "net": state["net"].state_dict(), "ema": state["ema"].state_dict(),
            "optimizer": state["optimizer_state"], "scaler": state["gradscaler_state"],
        }):
            failures.append("non-finite final tensors")
    telemetry_path = args.run_dir / "matched_training_telemetry_v1.csv"
    rows = []
    if telemetry_path.is_file():
        with telemetry_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    expected_rows = expected_attempt - 3000
    if len(rows) != expected_rows:
        failures.append(f"telemetry row count {len(rows)} != {expected_rows}")
    elif rows:
        if int(rows[0]["attempted_iteration"]) != 3001 or int(rows[-1]["attempted_iteration"]) != expected_attempt:
            failures.append("telemetry attempt endpoints mismatch")
        if any(not row.get(field) for row in rows for field in EXOGENOUS):
            failures.append("missing production exogenous receipt")
    formal_state = args.run_dir / "training-state-kimg000448.pt"
    formal_snapshot = args.run_dir / "network-snapshot-kimg000448.pkl"
    if not formal_state.is_file() or not formal_snapshot.is_file():
        failures.append("448-kimg formal state or EMA snapshot missing")
    else:
        state448 = torch.load(formal_state, map_location="cpu", weights_only=False)
        with formal_snapshot.open("rb") as handle:
            snapshot = pickle.load(handle)
        if reproducibility.module_state_sha256(state448["ema"]) != reproducibility.module_state_sha256(snapshot["ema"]):
            failures.append("448-kimg EMA snapshot does not match full state")
    artifacts = sorted(path for path in args.run_dir.iterdir() if path.is_file())
    manifest = {path.name: sha256_file(path) for path in artifacts if path.name not in {"branch_receipt.json", "SHA256SUMS.txt"}}
    payload = {
        "schema": "ect.q256.b384-same-state-branch-receipt/v1",
        "status": "PASS" if not failures else "FAIL_CLOSED",
        "seed": args.seed, "origin_arm": "B", "continuation_arm": args.arm,
        "branch_label": f"B384_to_{args.arm}", "source_state": str(args.source_state),
        "source_sha256_before_and_after": args.source_sha256,
        "protocol_sha256": args.protocol_sha256, "end_kimg": args.end_kimg,
        "final_attempted_iteration": expected_attempt,
        "final_state_sha256": manifest.get(final_path.name),
        "formal_448_state_sha256": manifest.get(formal_state.name),
        "formal_448_snapshot_sha256": manifest.get(formal_snapshot.name),
        "exogenous_fields": list(EXOGENOUS), "failures": failures,
    }
    (args.run_dir / "branch_receipt.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["branch_receipt.json"] = sha256_file(args.run_dir / "branch_receipt.json")
    (args.run_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(manifest.items())),
        encoding="utf-8",
    )
    return 0 if not failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
