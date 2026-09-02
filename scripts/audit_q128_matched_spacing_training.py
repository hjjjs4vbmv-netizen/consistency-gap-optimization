#!/usr/bin/env python3
"""Audit all 15 q128 matched-spacing trajectories from the consolidated card."""

import argparse
import csv
import datetime
import hashlib
import json
import os
import re
from pathlib import Path


ARMS = ["A", "Bsame", "Bmatch", "Cmatch", "Dmatch"]
BUDGETS = [256, 384, 512, 640, 768, 896, 1024]
RESUME_RE = re.compile(
    r"resum(?:e|ing)|restart|loading training state|restor(?:e|ing)", re.I
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--verify-file-hashes", action="store_true")
    return parser.parse_args()


def iso_mtime(path):
    value = datetime.datetime.fromtimestamp(path.stat().st_mtime, datetime.timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def run_layout():
    layouts = []
    seed3_sources = {
        "A": Path("/root/q128_matched_spacing_v1"),
        "Bsame": Path("/root/q128_data_card/sources/seed3-Bsame"),
        "Bmatch": Path("/root/q128_data_card/sources/seed3-Bmatch"),
        "Cmatch": Path("/root/q128_data_card/sources/seed3-Cmatch"),
        "Dmatch": Path("/root/q128_data_card/sources/seed3-Dmatch"),
    }
    for arm, source in seed3_sources.items():
        layouts.append((3, arm, source, source / "runs/seed3" / ("arm" + arm), 0))
    source7 = Path("/root/q128_data_card/sources/multigpu7-takeover")
    for arm in ARMS:
        launch = load_json(source7 / "runs/seed4" / ("arm" + arm) / "launch_record.json")
        layouts.append(
            (4, arm, source7, source7 / "runs/seed4" / ("arm" + arm), int(launch["gpu_id"]))
        )
    source5 = Path("/root/q128_data_card/sources/multigpu5")
    for arm in ("A", "Bsame"):
        launch = load_json(source7 / "runs/seed5" / ("arm" + arm) / "launch_record.json")
        layouts.append(
            (5, arm, source7, source7 / "runs/seed5" / ("arm" + arm), int(launch["gpu_id"]))
        )
    for arm in ("Bmatch", "Cmatch", "Dmatch"):
        launch = load_json(source5 / "runs/seed5" / ("arm" + arm) / "launch_record.json")
        layouts.append(
            (5, arm, source5, source5 / "runs/seed5" / ("arm" + arm), int(launch["gpu_id"]))
        )
    return sorted(layouts)


def normalized_pairing_config(receipt):
    value = json.loads(json.dumps(receipt["trajectory_config"]))
    loss = value["loss_kwargs"]
    for key in ("factorial_protocol", "target_gap_scale", "denominator_gap_scale"):
        loss.pop(key, None)
    return value


def write_csv(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.outdir.exists():
        raise RuntimeError("refusing to overwrite {}".format(args.outdir))
    args.outdir.mkdir(parents=True)

    integrity_rows = []
    artifact_rows = []
    hardware_rows = []
    initial_by_seed = {3: [], 4: [], 5: []}
    stage_values = set()

    for seed, arm, source, run_dir, gpu_id in run_layout():
        launch = load_json(run_dir / "launch_record.json")
        initial = load_json(run_dir / "initial_state_receipt_v1.json")
        preflight = load_json(source / "preflight/node_preflight.json")
        summary_path = run_dir / "train_summary.csv"
        with summary_path.open(newline="") as handle:
            summary = list(csv.DictReader(handle))
        attempts = [int(row["attempted_iteration"]) for row in summary]
        accepted = [int(row["successful_optimizer_steps"]) for row in summary]
        stages = [int(row["stage"]) for row in summary]
        stage_values.update(stages)
        amp_skips = sum(int(row["step_skipped"]) for row in summary)
        continuous = attempts == list(range(1, max(attempts) + 1))
        log_text = (run_dir / "log.txt").read_text(errors="replace")
        resume_markers = RESUME_RE.findall(log_text)
        receipts = []
        for budget in BUDGETS:
            receipt_path = run_dir / "network-snapshot-kimg{:06d}.receipt.json".format(budget)
            receipt = load_json(receipt_path)
            state_path = run_dir / "training-state-kimg{:06d}.pt".format(budget)
            snapshot_path = run_dir / "network-snapshot-kimg{:06d}.pkl".format(budget)
            state_verified = "not_rehashed"
            snapshot_verified = "not_rehashed"
            if args.verify_file_hashes:
                state_verified = sha256_file(state_path) == receipt["source_state_sha256"]
                snapshot_verified = sha256_file(snapshot_path) == receipt["snapshot_sha256"]
                if not state_verified or not snapshot_verified:
                    raise RuntimeError("artifact hash mismatch: {} {}".format(seed, arm))
            artifact_rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "budget_kimg": budget,
                    "state_sha256": receipt["source_state_sha256"],
                    "state_hash_verified": state_verified,
                    "snapshot_sha256": receipt["snapshot_sha256"],
                    "snapshot_hash_verified": snapshot_verified,
                    "ema_canonical_sha256": receipt["ema_canonical_sha256"],
                    "rng_unchanged_during_export": receipt["rng_unchanged"],
                    "receipt_status": receipt["status"],
                }
            )
            receipts.append(receipt)

        initial_by_seed[seed].append(
            {
                "arm": arm,
                "common_initial_state_sha256": initial["common_initial_state_sha256"],
                "model_sha256": initial["hashes"]["model"],
                "ema_sha256": initial["hashes"]["ema"],
                "optimizer_sha256": initial["hashes"]["optimizer"],
                "gradscaler_sha256": initial["hashes"]["gradscaler"],
                "rng_sha256": initial["hashes"]["rank_rng"],
                "sampler_sha256": initial["hashes"]["rank_sampler"],
                "normalized_pairing_config_sha256": sha256_json(
                    normalized_pairing_config(initial)
                ),
            }
        )
        completion = (
            len(receipts) == 7
            and all(receipt["status"] == "PASS" for receipt in receipts)
            and max(attempts) == 8000
            and float(summary[-1]["processed_kimg"]) == 1024.0
            and "Exiting..." in log_text
        )
        fresh_cell = (
            launch["smoke"] is False
            and initial["attempted_iteration"] == 0
            and initial["factorial"]["protocol"] == "q128_matched_spacing_v1"
            and initial["factorial"]["arm"] == arm
        )
        integrity_rows.append(
            {
                "seed": seed,
                "arm": arm,
                "source_hostname": preflight["hostname"],
                "gpu_id": gpu_id,
                "gpu_model": preflight["gpu"].splitlines()[gpu_id],
                "gpu_uuid": "not_recorded_in_preflight_v1",
                "launch_utc_from_preserved_mtime": iso_mtime(run_dir / "launch_record.json"),
                "completion_utc_from_log_mtime": iso_mtime(run_dir / "log.txt"),
                "attempted_iterations": max(attempts),
                "accepted_optimizer_steps": max(accepted),
                "amp_skips": amp_skips,
                "processed_kimg": summary[-1]["processed_kimg"],
                "max_stage": max(stages),
                "summary_rows": len(summary),
                "attempt_sequence_continuous": continuous,
                "resume_markers": len(resume_markers),
                "restart_resume_status": (
                    "no_resume_marker_continuous_1_to_8000"
                    if not resume_markers and continuous
                    else "review_required"
                ),
                "immutable_checkpoint_receipts": len(receipts),
                "fresh_five_arm_cell": fresh_cell,
                "completion_status": "PASS" if completion else "FAIL",
            }
        )
        hardware_rows.append(
            {
                "seed": seed,
                "arm": arm,
                "hostname": preflight["hostname"],
                "gpu_id": gpu_id,
                "gpu_model": preflight["gpu"].splitlines()[gpu_id],
                "gpu_uuid": "not_recorded_in_preflight_v1",
                "runtime_sif_sha256": preflight["runtime_sif_sha256"],
                "dataset_sha256": preflight["dataset_sha256"],
                "transfer_sha256": preflight["transfer_sha256"],
                "config_sha256": launch["config_sha256"],
            }
        )

    pairing = {}
    for seed, values in initial_by_seed.items():
        keys = [
            "common_initial_state_sha256", "model_sha256", "ema_sha256",
            "optimizer_sha256", "gradscaler_sha256", "rng_sha256",
            "sampler_sha256", "normalized_pairing_config_sha256",
        ]
        checks = {key: len({json.dumps(value[key], sort_keys=True) for value in values}) == 1 for key in keys}
        pairing[str(seed)] = {
            "arms": sorted(value["arm"] for value in values),
            "all_five_arms_present": sorted(value["arm"] for value in values) == sorted(ARMS),
            "within_seed_equalities": checks,
            "status": "PASS" if all(checks.values()) else "FAIL",
        }

    summary = {
        "schema": "ect.q128-matched-spacing-training-integrity/v1",
        "status": "PASS" if all(row["completion_status"] == "PASS" for row in integrity_rows) else "FAIL",
        "trajectories_completed": sum(row["completion_status"] == "PASS" for row in integrity_rows),
        "trajectories_expected": 15,
        "checkpoint_receipts": len(artifact_rows),
        "checkpoints_expected": 105,
        "attempted_iterations_total": sum(row["attempted_iterations"] for row in integrity_rows),
        "accepted_optimizer_steps_total": sum(row["accepted_optimizer_steps"] for row in integrity_rows),
        "amp_skips_total": sum(row["amp_skips"] for row in integrity_rows),
        "stages_observed": sorted(stage_values),
        "stage0_only": stage_values == {0},
        "fresh_a_cells": sum(row["arm"] == "A" and row["fresh_five_arm_cell"] for row in integrity_rows),
        "fresh_bsame_cells": sum(row["arm"] == "Bsame" and row["fresh_five_arm_cell"] for row in integrity_rows),
        "hardware_models": sorted({row["gpu_model"] for row in integrity_rows}),
        "gpu_uuid_status": "not_recorded_in_preflight_v1",
        "gpu_uuid_limitation": (
            "The immutable preflight schema recorded model and memory but not UUID. "
            "Released nodes cannot be queried retrospectively; UUIDs are not fabricated."
        ),
        "runtime_sif_sha256_values": sorted({row["runtime_sif_sha256"] for row in hardware_rows}),
        "dataset_sha256_values": sorted({row["dataset_sha256"] for row in hardware_rows}),
        "transfer_sha256_values": sorted({row["transfer_sha256"] for row in hardware_rows}),
        "pairing": pairing,
        "artifact_file_hashes_recomputed": bool(args.verify_file_hashes),
    }
    if summary["trajectories_completed"] != 15 or summary["checkpoint_receipts"] != 105:
        summary["status"] = "FAIL"
    if not summary["stage0_only"] or any(value["status"] != "PASS" for value in pairing.values()):
        summary["status"] = "FAIL"

    write_csv(args.outdir / "training_integrity.csv", list(integrity_rows[0]), integrity_rows)
    write_csv(args.outdir / "training_artifact_hashes.csv", list(artifact_rows[0]), artifact_rows)
    write_csv(args.outdir / "hardware_assignment.csv", list(hardware_rows[0]), hardware_rows)
    (args.outdir / "training_integrity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    checksums = []
    for path in sorted(args.outdir.iterdir()):
        if path.name != "SHA256SUMS.txt":
            checksums.append("{}  {}".format(sha256_file(path), path.name))
    (args.outdir / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
