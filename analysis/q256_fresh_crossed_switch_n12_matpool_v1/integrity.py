#!/usr/bin/env python3
"""Fail-closed integrity audit of all fresh prefixes and crossed suffixes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.q256_fresh_crossed_switch_n12_matpool_v1 import experiment  # noqa: E402


def checked(path: Path, schema: str | None = None) -> dict:
    value = experiment.load_json(path.resolve(strict=True))
    if value.get("status") != "PASS":
        raise RuntimeError(f"non-PASS receipt: {path}")
    if schema is not None and value.get("schema") != schema:
        raise RuntimeError(f"receipt schema mismatch: {path}")
    return value


def verify_artifact(record: dict) -> None:
    path = Path(record["path"])
    if path.is_symlink() or not path.is_file() or path.stat().st_size != record["bytes"]:
        raise RuntimeError(f"artifact identity failure: {path}")
    if experiment.sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"artifact SHA256 failure: {path}")


def telemetry_integrity(path: Path, first_attempt: int, last_attempt: int) -> dict:
    with path.open("rt", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    attempts = [int(float(row["attempted_iteration"])) for row in rows]
    if attempts != list(range(first_attempt, last_attempt + 1)):
        raise RuntimeError(f"telemetry attempt sequence mismatch: {path}")
    nonfinite_fields = ("loss_nonfinite_count", "update_nonfinite_count",
                        "model_nonfinite_count", "ema_nonfinite_count",
                        "factor_nonfinite_count", "nonpositive_denominator_count")
    totals = {field: sum(int(float(row[field])) for row in rows) for field in nonfinite_fields}
    if any(totals.values()):
        raise RuntimeError(f"scientific nonfinite instability in {path}: {totals}")
    return {"attempts": len(rows),
            "final_successful_optimizer_steps": int(float(rows[-1]["successful_optimizer_steps"])),
            "amp_skips": sum(str(row["step_skipped"]).strip().lower() in {"1", "true"}
                             for row in rows),
            "nonfinite_totals": totals}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    protocol = experiment.load_json(protocol_path)
    experiment.validate_protocol(protocol, protocol_path)
    root = Path(protocol["paths"]["formal_output_root"])
    protocol_sha = experiment.sha256_file(protocol_path)
    checked(root / "training_matrix_completion_receipt.json")
    prefixes = suffixes = b384 = source512 = suffix_final_states = 0
    matched_prefix = matched_suffix = source_identity = 0
    records = []
    for seed in experiment.SEEDS:
        seed_root = root / "training" / f"seed{seed}"
        checked(seed_root / "seed_completion_receipt.json")
        checked(seed_root / "prefix_matched_randomness_receipt.json")
        checked(seed_root / "suffix_matched_randomness_receipt.json")
        matched_prefix += 1; matched_suffix += 1
        sources = {}
        initial_receipts = {}
        seed_telemetry = {}
        for arm in experiment.ARMS:
            prefix = seed_root / f"prefix_{arm}"
            checked(prefix / "preparation_receipt.json")
            checked(prefix / "trajectory_manifest.json")
            checked(prefix / "matpool_gpu_receipt.json")
            checked(prefix / "compute_completion_receipt.json")
            checked(prefix / "compute_time_receipt.json")
            checked(prefix / "gpu_exclusivity_before.json")
            checked(prefix / "gpu_exclusivity_after.json")
            source = checked(prefix / "source_state_receipt.json")
            seed_telemetry[f"prefix_{arm}"] = telemetry_integrity(
                prefix / "factorial_training_telemetry_v1.csv", 1, 4000)
            initial_receipts[arm] = experiment.load_json(prefix / "initial_state_receipt_v1.json")
            if source.get("protocol_sha256") != protocol_sha or source.get("source_kimg") != 512:
                raise RuntimeError("source-state protocol/counter mismatch")
            verify_artifact(source["training_state"])
            receipt512 = checked(prefix / "kimg0512" / "milestone_receipt.json")
            verify_artifact(receipt512["training_state"]); verify_artifact(receipt512["network_snapshot"])
            if receipt512.get("attempted_iteration") != 4000 or receipt512.get("cur_nimg") != 512000:
                raise RuntimeError("512-kimg prefix counter mismatch")
            sources[arm] = source
            prefixes += 1; source512 += 1
            if arm == "B":
                receipt384 = checked(prefix / "kimg0384" / "milestone_receipt.json")
                verify_artifact(receipt384["training_state"]); verify_artifact(receipt384["network_snapshot"])
                if receipt384.get("attempted_iteration") != 3000 or receipt384.get("cur_nimg") != 384000:
                    raise RuntimeError("384-kimg B-prefix counter mismatch")
                b384 += 1
        if (initial_receipts["A"].get("common_initial_state_sha256")
                != initial_receipts["B"].get("common_initial_state_sha256")):
            raise RuntimeError(f"A/B common initialization mismatch: seed{seed}")
        source_rng = [sources[arm]["training_state"]["internal_state_sha256"]["rank_rng"]
                      for arm in experiment.ARMS]
        source_sampler = [sources[arm]["training_state"]["internal_state_sha256"]["rank_sampler"]
                          for arm in experiment.ARMS]
        if source_rng[0] != source_rng[1] or source_sampler[0] != source_sampler[1]:
            raise RuntimeError(f"prefix dropout-RNG/sampler pairing failure: seed{seed}")
        suffix_rng = []
        suffix_sampler = []
        for cell, (origin, _) in experiment.CELLS.items():
            run = seed_root / cell
            manifest = experiment.load_json(run / "formal_run_manifest.json")
            source_path = Path(manifest["source_state"]["path"])
            if (manifest.get("protocol_sha256") != protocol_sha
                    or manifest["source_state"]["sha256"] != sources[origin]["training_state"]["sha256"]
                    or source_path.resolve() != Path(sources[origin]["training_state"]["path"]).resolve()):
                raise RuntimeError(f"same-origin source identity failure: seed{seed}/{cell}")
            source_identity += 1
            suffix_source = checked(run / "source_state_receipt.json")
            seed_telemetry[cell] = telemetry_integrity(
                run / "schedule_switch_training_telemetry_v1.csv", 4001, 8000)
            if suffix_source.get("source_state", {}).get("sha256") != sources[origin]["training_state"]["sha256"]:
                raise RuntimeError(f"suffix source receipt mismatch: seed{seed}/{cell}")
            checked(run / "preparation_receipt.json")
            checked(run / "trajectory_manifest.json")
            checked(run / "matpool_gpu_receipt.json")
            checked(run / "compute_completion_receipt.json")
            checked(run / "compute_time_receipt.json")
            checked(run / "gpu_exclusivity_before.json")
            checked(run / "gpu_exclusivity_after.json")
            checked(run / "trajectory_completion_receipt.json")
            for budget, attempt in ((640, 5000), (768, 6000), (896, 7000), (1024, 8000)):
                receipt = checked(run / f"kimg{budget:04d}" / "milestone_receipt.json")
                verify_artifact(receipt["training_state"]); verify_artifact(receipt["network_snapshot"])
                if receipt.get("attempted_iteration") != attempt or receipt.get("cur_nimg") != budget * 1000:
                    raise RuntimeError(f"suffix milestone counter mismatch: seed{seed}/{cell}/{budget}")
                runtime = receipt.get("runtime", {})
                if (runtime.get("deterministic_algorithms") is not True
                        or runtime.get("tf32_cudnn") is not False
                        or runtime.get("tf32_matmul") is not False
                        or runtime.get("world_size") != 1):
                    raise RuntimeError("suffix exporter runtime policy mismatch")
                if budget == 1024:
                    suffix_final_states += 1
                    suffix_rng.append(receipt["training_state"]["internal_state_sha256"]["rank_rng"])
                    suffix_sampler.append(receipt["training_state"]["internal_state_sha256"]["rank_sampler"])
            suffixes += 1
        if len({json.dumps(value, sort_keys=True) for value in suffix_rng}) != 1:
            raise RuntimeError(f"suffix dropout-RNG pairing failure: seed{seed}")
        if len({json.dumps(value, sort_keys=True) for value in suffix_sampler}) != 1:
            raise RuntimeError(f"suffix sampler pairing failure: seed{seed}")
        records.append({"seed": seed, "prefixes": 2, "suffixes": 4,
                        "prefix_order": experiment.assignment(seed)["prefix_order"],
                        "suffix_order": experiment.assignment(seed)["suffix_order"],
                        "telemetry": seed_telemetry})
    counts = {
        "prefixes": prefixes, "suffixes": suffixes, "B_at_384_full_states": b384,
        "A_or_B_at_512_source_states": source512, "suffix_at_1024_full_states": suffix_final_states,
        "same_origin_source_references": source_identity,
        "prefix_matched_randomness_receipts": matched_prefix,
        "suffix_matched_randomness_receipts": matched_suffix,
    }
    expected = {"prefixes": 24, "suffixes": 48, "B_at_384_full_states": 12,
                "A_or_B_at_512_source_states": 24, "suffix_at_1024_full_states": 48,
                "same_origin_source_references": 48, "prefix_matched_randomness_receipts": 12,
                "suffix_matched_randomness_receipts": 12}
    if counts != expected:
        raise RuntimeError(f"training integrity counts mismatch: {counts}")
    payload = {
        "schema": "ect.q256.fresh-crossed-switch-training-integrity/v1", "status": "PASS",
        "protocol_sha256": protocol_sha, "counts": counts, "expected": expected,
        "same_origin_source_identity": "PASS", "matched_randomness": "PASS",
        "scientific_nonfinite_instability": "ABSENT",
        "per_seed": records,
    }
    experiment.atomic_json(args.output, payload)
    print(json.dumps({"status": "PASS", "counts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
