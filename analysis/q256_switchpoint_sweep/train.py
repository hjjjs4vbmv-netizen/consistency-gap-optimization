#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.q256_fresh_crossed_switch_n12_matpool_v1 import experiment
from training import reproducibility, schedule_switch


SEED_GROUPS = ((81, 85, 89), (82, 86, 90), (83, 87, 91), (84, 88, 92))
REQUIRED_MILESTONES = {
    "CTRL": (640, 768, 896, 1024),
    "BA128": (640, 1024),
    "BA256": (768, 1024),
    "BA384": (896, 1024),
    "BA512": (1024,),
}


def training_jobs() -> list[dict]:
    jobs = []
    for group, seeds in enumerate(SEED_GROUPS):
        for seed in seeds:
            jobs += [
                {"phase": "prefix", "gpu": 2 * group, "seed": seed, "name": "prefix_A", "arm": "A"},
                {"phase": "prefix", "gpu": 2 * group + 1, "seed": seed, "name": "prefix_B", "arm": "B"},
            ]
            for name, switch, gpu in (
                ("CTRL", 512, group), ("BA512", 512, group),
                ("BA384", 384, group), ("BA128", 128, group + 4),
                ("BA256", 256, group + 4),
            ):
                jobs.append({"phase": "continuation", "gpu": gpu, "seed": seed,
                             "name": name, "arm": "A", "switch_kimg": switch})
    return sorted(jobs, key=lambda row: (row["phase"], row["gpu"], row["seed"], row["name"]))


def normalize_command(command: list[str], job: dict) -> list[str]:
    replacements = {"--tick=": "--tick=128", "--snap=": "--snap=0", "--dump=": "--dump=0"}
    result = []
    for item in command:
        if any(item.startswith(prefix) for prefix in replacements):
            prefix = next(prefix for prefix in replacements if item.startswith(prefix))
            result.append(replacements[prefix])
        elif item.startswith("--immutable-checkpoint-kimg="):
            if job["phase"] == "prefix":
                result.append("--immutable-checkpoint-kimg=" + ("512" if job["arm"] == "A" else "128,256,384,512"))
            else:
                result.append(
                    "--immutable-checkpoint-kimg="
                    + ",".join(str(value) for value in REQUIRED_MILESTONES[job["name"]])
                )
        elif item.startswith("--planned-pause-protocol="):
            result.append(f"--planned-pause-protocol={schedule_switch.SWITCHPOINT_SWEEP_PROTOCOL}")
        else:
            result.append(item)
    return result


def bound_protocol(protocol: dict) -> dict:
    value = json.loads(json.dumps(protocol))
    runtime = Path(protocol["paths"]["runtime"]) / "runtime-manifest.json"
    value["assets"]["runtime_manifest"] = {"path": str(runtime), "sha256": experiment.sha256_file(runtime)}
    value["gpus"] = experiment.query_gpus()
    if [gpu["index"] for gpu in value["gpus"]] != list(range(8)):
        raise RuntimeError("TASK 2 requires exactly GPU indices 0..7")
    value["paths"]["formal_output_root"] = protocol["root"]
    return value


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError(f"refuse existing published artifact: {destination}")
    os.link(source, destination)


def _state_record(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": experiment.sha256_file(path),
            "internal_state_sha256": schedule_switch.internal_state_hashes(state)}


def publish_prefix(run_dir: Path, seed: int, arm: str, protocol_sha: str) -> None:
    points = (512,) if arm == "A" else (128, 256, 384, 512)
    for kimg in points:
        raw = run_dir / f"training-state-kimg{kimg:06d}.pt"
        published = run_dir / f"kimg{kimg:04d}" / "training-state.pt"
        _link(raw, published)
        receipt = {"status": "PASS", "seed": seed, "arm": arm, "kimg": kimg,
                   "training_state": _state_record(published), "protocol_sha256": protocol_sha}
        experiment.atomic_json(published.parent / "milestone_receipt.json", receipt)
    latest = run_dir / "training-state-latest.pt"
    if latest.exists():
        latest.unlink()
    for snapshot in run_dir.glob("network-snapshot-*.pkl"):
        snapshot.unlink()


def copy_history(source_dir: Path, destination: Path, switch_kimg: int) -> None:
    last_attempt = switch_kimg * 1000 // 128
    for source_name, target_name in (
        ("train_summary.csv", "train_summary.csv"),
        ("factorial_training_telemetry_v1.csv", "source_factorial_training_telemetry_v1.csv"),
    ):
        with (source_dir / source_name).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle); rows = [row for row in reader if int(float(row["attempted_iteration"])) <= last_attempt]
            fields = reader.fieldnames
        with (destination / target_name).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    experiment.copy_exclusive(source_dir / "initial_state_receipt_v1.json", destination / "initial_state_receipt_v1.json")
    experiment.copy_exclusive(source_dir / "training_options.json", destination / "source_training_options.json")


def make_suffix_manifest(protocol: dict, protocol_path: Path, job: dict, run_dir: Path) -> Path:
    prefix = Path(protocol["paths"]["training"]) / f"seed{job['seed']:03d}" / ("prefix_A" if job["name"] == "CTRL" else "prefix_B")
    source = prefix / f"kimg{job['switch_kimg']:04d}" / "training-state.pt"
    source_receipt = source.parent / "milestone_receipt.json"
    copy_history(prefix, run_dir, job["switch_kimg"])
    manifest = {
        "schema": schedule_switch.RUN_MANIFEST_SCHEMA,
        "experiment_protocol": schedule_switch.SWITCHPOINT_SWEEP_PROTOCOL,
        "run_kind": "formal", "branch": "CTRL" if job["name"] == "CTRL" else "BA",
        "seed": job["seed"], "origin_arm": "A" if job["name"] == "CTRL" else "B",
        "continuation_arm": "A", "switch_kimg": job["switch_kimg"], "final_kimg": 1024,
        "protocol_sha256": experiment.sha256_file(protocol_path),
        "implementation_commit": subprocess.check_output(
            ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_checkpoint_manifest_sha256": experiment.sha256_file(source_receipt),
        "source_state": _state_record(source), "source_history_prefix": {"last_attempt": job["switch_kimg"] * 1000 // 128},
        "immutable_output_root": str(run_dir.resolve()),
    }
    path = run_dir / "formal_run_manifest.json"
    experiment.atomic_json(path, manifest)
    schedule_switch.load_run_manifest(path)
    return path


def publish_continuation(run_dir: Path, job: dict, protocol_sha: str) -> None:
    for kimg in REQUIRED_MILESTONES[job["name"]]:
        source = run_dir / f"training-state-kimg{kimg:06d}.pt"
        state = torch.load(source, map_location="cpu", weights_only=False)
        ema = copy.deepcopy(state["ema"]).eval().requires_grad_(False)
        ema_sha = reproducibility.module_state_sha256(ema)
        published = run_dir / f"kimg{kimg:04d}" / "network-snapshot.pkl"
        reproducibility.atomic_pickle_dump(
            {
                "ema": ema,
                "loss_fn": None,
                "augment_pipe": None,
                "dataset_kwargs": dict(state["trajectory_config"]["dataset_kwargs"]),
            },
            published,
        )
        receipt = {"status": "PASS", "seed": job["seed"], "trajectory": job["name"],
                   "kimg": kimg, "snapshot": {"path": str(published.resolve()),
                   "bytes": published.stat().st_size, "sha256": experiment.sha256_file(published),
                   "ema_internal_sha256": ema_sha}, "protocol_sha256": protocol_sha}
        if kimg == 1024:
            state_record = _state_record(source)
            if state_record["internal_state_sha256"]["ema"] != ema_sha:
                raise RuntimeError("terminal state/snapshot EMA mismatch")
            full = published.parent / "training-state.pt"; _link(source, full)
            receipt["training_state"] = _state_record(full)
        experiment.atomic_json(published.parent / "milestone_receipt.json", receipt)
        if kimg != 1024:
            source.unlink()
    latest_snapshot = run_dir / "network-snapshot-latest.pkl"
    if latest_snapshot.exists():
        latest_snapshot.unlink()
    latest_state = run_dir / "training-state-latest.pt"
    if latest_state.exists():
        latest_state.unlink()


def execute(protocol_path: Path, job: dict) -> None:
    protocol = json.loads(protocol_path.read_text()); bound = bound_protocol(protocol)
    protocol_sha = experiment.sha256_file(protocol_path)
    run_dir = Path(protocol["paths"]["training"]) / f"seed{job['seed']:03d}" / job["name"]
    if job["phase"] == "prefix":
        base = experiment.training_command(bound, run_dir, job["seed"], job["arm"], job["gpu"], prefix=True)
        experiment.run_cell(bound, run_dir, job["gpu"], normalize_command(base, job), f"seed{job['seed']}:{job['name']}")
        publish_prefix(run_dir, job["seed"], job["arm"], protocol_sha)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = make_suffix_manifest(protocol, protocol_path, job, run_dir)
        source = Path(json.loads(manifest.read_text())["source_state"]["path"])
        base = experiment.training_command(bound, run_dir, job["seed"], "A", job["gpu"], prefix=False, manifest=manifest, source=source)
        experiment.run_cell(bound, run_dir, job["gpu"], normalize_command(base, job), f"seed{job['seed']}:{job['name']}")
        publish_continuation(run_dir, job, protocol_sha)


def run_job(protocol_path: Path, job: dict) -> dict:
    protocol = json.loads(protocol_path.read_text())
    training_root = Path(protocol["paths"]["training"])
    run_dir = training_root / f"seed{job['seed']:03d}" / job["name"]
    errors = []
    for attempt in (1, 2):
        try:
            execute(protocol_path, job)
            return {"status": "PASS", "attempts": attempt, "errors": errors}
        except Exception as error:
            errors.append(repr(error))
            if attempt == 1 and run_dir.exists():
                shutil.move(run_dir, run_dir.with_name(f"{run_dir.name}-attempt1"))
    return {"status": "EXHAUSTED_FAILURE", "attempts": 2, "errors": errors,
            "root_cause": "TRAJECTORY_TRAINING_FAILURE"}


def worker(protocol_path: Path, phase: str, gpu: int) -> None:
    protocol = json.loads(protocol_path.read_text())
    terminal = Path(protocol["paths"]["training"]) / "terminal"
    terminal.mkdir(parents=True, exist_ok=True)
    for job in training_jobs():
        if job["phase"] != phase or job["gpu"] != gpu:
            continue
        receipt = None
        if phase == "continuation":
            prefix = "prefix_A" if job["name"] == "CTRL" else "prefix_B"
            upstream = terminal / f"prefix-seed{job['seed']:03d}-{prefix}.json"
            upstream_receipt = json.loads(upstream.read_text())
            if upstream_receipt["status"] != "PASS":
                receipt = {"status": "EXHAUSTED_FAILURE", "attempts": 0, "errors": [],
                           "root_cause": "A_TRUNK_FAILURE" if prefix == "prefix_A" else "B_TRUNK_FAILURE",
                           "upstream_terminal_sha256": experiment.sha256_file(upstream)}
        if receipt is None:
            receipt = run_job(protocol_path, job)
            if receipt["status"] != "PASS" and phase == "prefix":
                receipt["root_cause"] = "A_TRUNK_FAILURE" if job["arm"] == "A" else "B_TRUNK_FAILURE"
        receipt.update(job={key: job[key] for key in sorted(job)})
        experiment.atomic_json(
            terminal / f"{phase}-seed{job['seed']:03d}-{job['name']}.json",
            receipt,
        )


def matrix_summary(receipts: list[dict]) -> dict:
    complete = [seed for seed in range(81, 93)
                if sum(row["status"] == "PASS" and row["job"]["seed"] == seed for row in receipts) == 7]
    failures = sum(row["status"] != "PASS" for row in receipts)
    return {"status": "PASS" if len(complete) == 12 else "COMPLETE_WITH_FAILURES",
            "jobs": len(receipts), "expected_jobs": 84, "failures": failures,
            "complete_seeds": complete, "n_complete_seeds": len(complete)}


def launch(protocol_path: Path) -> None:
    protocol = json.loads(protocol_path.read_text())
    preflight_path = Path(protocol["paths"]["evidence"]) / "preflight.json"
    preflight = json.loads(preflight_path.read_text())
    head = subprocess.check_output(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], text=True).strip()
    if (preflight.get("status") != "PASS"
            or preflight.get("protocol_sha256") != experiment.sha256_file(protocol_path)
            or preflight.get("implementation_commit") != head):
        raise RuntimeError("training requires a current protocol/commit-bound preflight PASS")
    log_root = Path(protocol["paths"]["training"]) / "launcher_logs"
    log_root.mkdir(parents=True, exist_ok=False)
    runtime = json.loads((Path(protocol["paths"]["runtime"]) / "runtime-manifest.json").read_text())
    python = str(Path(runtime["environment_prefix"]) / "bin" / "python")
    for phase in ("prefix", "continuation"):
        processes = []
        for gpu in range(8):
            log = (log_root / f"{phase}-gpu{gpu}.log").open("xb")
            command = [python, str(Path(__file__).resolve()), "worker", "--protocol", str(protocol_path),
                       "--phase", phase, "--gpu", str(gpu)]
            processes.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
        for process, log in processes:
            code = process.wait(); log.close()
            if code:
                raise RuntimeError(f"{phase} worker exited {code}")
    terminal = Path(protocol["paths"]["training"]) / "terminal"
    receipts = [json.loads(path.read_text()) for path in terminal.glob("*.json")]
    summary = matrix_summary(receipts)
    summary.update(protocol_sha256=experiment.sha256_file(protocol_path), implementation_commit=head)
    experiment.atomic_json(Path(protocol["paths"]["training"]) / "training_matrix_receipt.json", summary)
    if len(receipts) != 84:
        raise RuntimeError("training launcher did not produce 84 terminal receipts")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("plan", "launch", "worker"))
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--phase", choices=("prefix", "continuation")); parser.add_argument("--gpu", type=int, choices=range(8))
    args = parser.parse_args(); jobs = training_jobs()
    if args.command == "plan":
        print(json.dumps(jobs, indent=2)); return 0
    if args.command == "launch":
        launch(args.protocol.resolve(strict=True)); return 0
    if args.phase is None or args.gpu is None:
        parser.error("worker requires --phase and --gpu")
    worker(args.protocol.resolve(strict=True), args.phase, args.gpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
