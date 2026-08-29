#!/usr/bin/env python3
"""Compare B384_to_B@512 against the PR #79 canonical computational state."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training import reproducibility


EXPECTED_FILE_SHA = {
    3: "207e31ab74b5759d5cbad507b5df8b4a523eca44cc6bca38413b7a8e02a572a7",
    4: "ecc97d6d2e3894fa693bda9bfb5f2aad9229b0c6cb54110475ff941a456b3967",
    5: "d4efc5a3111864bf4a0c617e16db0b6f998e1c13a91a6198f3406de6d5bc0971",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(state: dict) -> dict:
    steps = []
    for item in state["optimizer_state"]["state"].values():
        value = item["step"]
        steps.append(int(value.item()) if isinstance(value, torch.Tensor) else int(value))
    return {
        "online_model": reproducibility.module_state_sha256(state["net"]),
        "ema_model": reproducibility.module_state_sha256(state["ema"]),
        "optimizer": reproducibility.state_sha256(state["optimizer_state"]),
        "gradscaler": reproducibility.state_sha256(state["gradscaler_state"]),
        "radam_steps": steps,
        "rank_rng": reproducibility.state_sha256(state["rank_states"][0]["rng_state"]),
        "sampler": reproducibility.state_sha256(state["rank_states"][0]["sampler_state"]),
        "attempted_iteration": int(state["attempted_iteration"]),
        "successful_optimizer_steps": int(state["successful_optimizer_steps"]),
        "cur_nimg": int(state["cur_nimg"]),
        "sampler_consumed_samples": int(state["rank_states"][0]["sampler_state"]["consumed_samples"]),
    }


def write_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()
    with args.inventory.open(newline="", encoding="utf-8") as handle:
        rows = {
            int(row["seed"]): row for row in csv.DictReader(handle)
            if row["arm"] == "B" and row["budget_kimg"] == "512"
        }
    results = []
    for seed in (3, 4, 5):
        actual_path = args.output_root / "runs" / f"seed{seed}" / "B384_to_B" / "training-state-kimg000512.pt"
        canonical_path = args.canonical_root / f"seed{seed}" / "armB" / "training-state-kimg000512.pt"
        failures = []
        canonical_file_sha = sha256_file(canonical_path)
        if canonical_file_sha != EXPECTED_FILE_SHA[seed] or canonical_file_sha != rows[seed]["replay_state_sha256"]:
            failures.append("canonical whole-file SHA mismatch")
        canonical = torch.load(canonical_path, map_location="cpu", weights_only=False)
        actual = torch.load(actual_path, map_location="cpu", weights_only=False)
        expected = fingerprint(canonical)
        observed = fingerprint(actual)
        differing = sorted(key for key in expected if expected[key] != observed[key])
        failures.extend(f"computational field mismatch: {key}" for key in differing)
        for key, column in (
            ("online_model", "online_model_canonical_sha256"),
            ("ema_model", "ema_model_canonical_sha256"),
            ("optimizer", "optimizer_canonical_sha256"),
        ):
            if observed[key] != rows[seed][column]:
                failures.append(f"inventory internal SHA mismatch: {key}")
        results.append({
            "seed": seed, "status": "COMPUTATIONAL_STATE_MATCH" if not failures else "FAIL_CLOSED",
            "actual_path": str(actual_path), "actual_file_sha256": sha256_file(actual_path),
            "canonical_path": str(canonical_path), "canonical_file_sha256": canonical_file_sha,
            "observed": observed, "expected": expected, "differing_fields": differing,
            "failures": failures,
        })
    passed = all(item["status"] == "COMPUTATIONAL_STATE_MATCH" for item in results)
    payload = {
        "schema": "ect.q256.b384-same-state-b-noop-parity/v1",
        "status": "PASS" if passed else "FAIL_CLOSED",
        "required": "3/3 COMPUTATIONAL_STATE_MATCH",
        "results": results,
    }
    write_exclusive(args.out_json, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = ["# B@384 → B@512 no-op parity", "", f"- Status: **{payload['status']}**", "",
             "| seed | result | differing fields |", "|---:|---|---|"]
    for item in results:
        lines.append(f"| {item['seed']} | {item['status']} | {', '.join(item['differing_fields']) or 'none'} |")
    lines += ["", "A/C/D formal launch is authorized only when this report is PASS.", ""]
    write_exclusive(args.out_md, "\n".join(lines))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
