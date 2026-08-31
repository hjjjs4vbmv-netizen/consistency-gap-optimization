#!/usr/bin/env python3
"""Runtime-specific uninterrupted-vs-resume exact parity gate."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.q256_fresh_crossed_switch_n12_matpool_v1 import experiment  # noqa: E402
from training import reproducibility, schedule_switch  # noqa: E402

ENGINEERING_SEED = 20260831


def selected_gpu_apps(gpus: list[dict], gpu_indices: list[int] | tuple[int, ...],
                      apps: list[dict]) -> list[dict]:
    """Return compute applications attached to the explicitly selected GPUs."""
    if len(gpu_indices) != 4 or len(set(gpu_indices)) != 4:
        raise RuntimeError("engineering parity requires four unique GPU indices")
    if any(index < 0 or index >= len(gpus) for index in gpu_indices):
        raise RuntimeError("engineering parity GPU index is unavailable")
    selected_uuids = {gpus[index]["uuid"] for index in gpu_indices}
    return [row for row in apps if row["gpu_uuid"] in selected_uuids]


def continuous_command(protocol: dict, run_dir: Path, arm: str, gpu: int) -> list[str]:
    command = experiment.training_command(
        protocol, run_dir, ENGINEERING_SEED, arm, gpu, prefix=True, final_kimg=640
    )
    command = [
        item for item in command
        if item != "--stop-after-attempts=4000"
        and not item.startswith("--planned-pause-protocol=")
        and not item.startswith("--immutable-checkpoint-kimg=")
    ]
    command.append("--immutable-checkpoint-kimg=512,640")
    return command


def engineering_switch_manifest(
    protocol: dict, engineering_protocol_sha: str, arm: str,
    source_receipt: dict, output: Path,
) -> Path:
    branch = "A_to_A" if arm == "A" else "B_to_B"
    source_path = output.parent / "prefix" / "source_state_receipt.json"
    manifest = {
        "schema": schedule_switch.RUN_MANIFEST_SCHEMA,
        "experiment_protocol": schedule_switch.FRESH_N12_ENGINEERING_PROTOCOL,
        "run_kind": "parity", "branch": branch, "seed": ENGINEERING_SEED,
        "origin_arm": arm, "continuation_arm": arm,
        "switch_kimg": 512, "final_kimg": 640,
        "protocol_sha256": engineering_protocol_sha,
        "implementation_commit": protocol["implementation_commit"],
        "source_checkpoint_manifest_sha256": experiment.sha256_file(source_path),
        "source_state": source_receipt["training_state"],
        "immutable_output_root": str(output),
    }
    output.mkdir(exist_ok=False)
    source_history = experiment.prepare_resume_history(output.parent / "prefix", output)
    manifest["source_history_prefix"] = source_history
    path = output / "formal_run_manifest.json"
    experiment.atomic_json(path, manifest)
    schedule_switch.load_run_manifest(path)
    return path


def worker(args: argparse.Namespace) -> None:
    engineering = experiment.load_json(args.engineering_protocol)
    protocol = engineering["execution_protocol"]
    protocol["protocol_sha256_runtime"] = engineering["engineering_protocol_sha256"]
    root = Path(engineering["engineering_root"])
    arm = args.arm
    gpu = args.gpu_index
    if args.mode == "continuous":
        run_dir = root / f"continuous_{arm}"
        command = continuous_command(protocol, run_dir, arm, gpu)
        experiment.run_cell(protocol, run_dir, gpu, command, f"parity:continuous_{arm}")
        return
    mode_root = root / f"segmented_{arm}"
    mode_root.mkdir(exist_ok=False)
    prefix_dir = mode_root / "prefix"
    prefix_command = experiment.training_command(
        protocol, prefix_dir, ENGINEERING_SEED, arm, gpu, prefix=True
    )
    experiment.run_cell(protocol, prefix_dir, gpu, prefix_command, f"parity:segmented_{arm}:prefix")
    source = experiment.export_prefix(
        prefix_dir, ENGINEERING_SEED, arm, engineering["engineering_protocol_sha256"]
    )
    suffix_dir = mode_root / "suffix"
    manifest = engineering_switch_manifest(
        protocol, engineering["engineering_protocol_sha256"], arm, source, suffix_dir
    )
    source_state = Path(source["training_state"]["path"])
    suffix_command = experiment.training_command(
        protocol, suffix_dir, ENGINEERING_SEED, arm, gpu, prefix=False,
        manifest=manifest, source=source_state, final_kimg=640,
    )
    experiment.run_cell(protocol, suffix_dir, gpu, suffix_command, f"parity:segmented_{arm}:suffix")


def computational_state(state: dict) -> dict:
    value = {
        key: copy.deepcopy(item)
        for key, item in state.items()
        if key not in {"net", "ema", "schedule_switch", "elapsed_sec"}
    }
    value["net_sha256"] = reproducibility.module_state_sha256(state["net"])
    value["ema_sha256"] = reproducibility.module_state_sha256(state["ema"])
    return value


def launch(args: argparse.Namespace) -> None:
    runtime = experiment.load_json(args.runtime_manifest.resolve(strict=True))
    experiment.validate_runtime(runtime)
    gpus = experiment.query_gpus()
    if len(gpus) != 6:
        raise RuntimeError("engineering parity requires six visible A100 GPUs")
    if selected_gpu_apps(gpus, args.gpu_indices, experiment.compute_apps()):
        raise RuntimeError("engineering parity requires the four selected GPUs to be idle")
    root = args.output_root.absolute()
    root.mkdir(parents=True, exist_ok=False)
    protocol = {
        "implementation_commit": args.implementation_commit,
        "gpus": gpus,
        "assets": {
            "dataset": {"path": str(args.dataset.resolve(strict=True)), "sha256": experiment.DATASET_SHA256},
            "transfer": {"path": str(args.transfer.resolve(strict=True)), "sha256": experiment.TRANSFER_SHA256},
            "runtime_manifest": {"path": str(args.runtime_manifest.resolve(strict=True)),
                                 "sha256": experiment.sha256_file(args.runtime_manifest)},
        },
    }
    if experiment.sha256_file(args.dataset) != experiment.DATASET_SHA256:
        raise RuntimeError("engineering dataset SHA mismatch")
    if experiment.sha256_file(args.transfer) != experiment.TRANSFER_SHA256:
        raise RuntimeError("engineering transfer SHA mismatch")
    frozen = {
        "schema": "ect.q256.fresh-crossed-switch-engineering-protocol/v1",
        "seed": ENGINEERING_SEED, "arms": ["A", "B"],
        "checks": ["A_continuous_vs_segmented_640", "B_continuous_vs_segmented_640"],
        "execution_protocol": protocol, "engineering_root": str(root),
    }
    digest_source = copy.deepcopy(frozen)
    digest = experiment.canonical_sha256(digest_source)
    frozen["engineering_protocol_sha256"] = digest
    engineering_path = root / "engineering_protocol.json"
    experiment.atomic_json(engineering_path, frozen)
    python = str(Path(runtime["environment_prefix"]) / "bin" / "python")
    plans = tuple(
        (arm, mode, gpu) for (arm, mode), gpu in zip(
            (("A", "continuous"), ("B", "continuous"),
             ("A", "segmented"), ("B", "segmented")),
            args.gpu_indices,
        )
    )
    processes = []
    for arm, mode, gpu in plans:
        log = (root / f"{mode}_{arm}.launcher.log").open("xb")
        command = [python, str(Path(__file__).resolve()), "worker",
                   "--engineering-protocol", str(engineering_path),
                   "--arm", arm, "--mode", mode, "--gpu-index", str(gpu)]
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        processes.append((arm, mode, gpu, process, log))
    failures = []
    for arm, mode, gpu, process, log in processes:
        returncode = process.wait()
        log.close()
        if returncode:
            failures.append({"arm": arm, "mode": mode, "gpu": gpu, "exit_code": returncode})
    if failures:
        experiment.atomic_json(root / "parity_report.json", {
            "schema": "ect.q256.fresh-crossed-switch-runtime-parity/v1",
            "status": "FAIL", "failures": failures, "automatic_retry_count": 0,
        })
        raise RuntimeError(f"engineering parity process failure: {failures}")
    comparisons = []
    for arm in ("A", "B"):
        uninterrupted_path = root / f"continuous_{arm}" / "training-state-kimg000640.pt"
        resumed_path = root / f"segmented_{arm}" / "suffix" / "training-state-kimg000640.pt"
        uninterrupted = torch.load(uninterrupted_path, map_location="cpu", weights_only=False)
        resumed = torch.load(resumed_path, map_location="cpu", weights_only=False)
        first = computational_state(uninterrupted)
        second = computational_state(resumed)
        hashes_first = schedule_switch.internal_state_hashes(uninterrupted)
        hashes_second = schedule_switch.internal_state_hashes(resumed)
        detail = {
            "arm": arm,
            "parameters": hashes_first["net"] == hashes_second["net"],
            "ema": hashes_first["ema"] == hashes_second["ema"],
            "optimizer": hashes_first["optimizer"] == hashes_second["optimizer"],
            "gradscaler": hashes_first["gradscaler"] == hashes_second["gradscaler"],
            "rng": hashes_first["rank_rng"] == hashes_second["rank_rng"],
            "sampler": hashes_first["rank_sampler"] == hashes_second["rank_sampler"],
            "counters": (
                uninterrupted["attempted_iteration"] == resumed["attempted_iteration"] == 5000
                and uninterrupted["cur_nimg"] == resumed["cur_nimg"] == 640000
                and uninterrupted["successful_optimizer_steps"] == resumed["successful_optimizer_steps"]
            ),
            "complete_computational_state_sha256_uninterrupted": reproducibility.state_sha256(first),
            "complete_computational_state_sha256_resumed": reproducibility.state_sha256(second),
        }
        detail["complete_state"] = (
            detail["complete_computational_state_sha256_uninterrupted"]
            == detail["complete_computational_state_sha256_resumed"]
        )
        detail["status"] = "PASS" if all(
            detail[key] for key in ("parameters", "ema", "optimizer", "gradscaler", "rng", "sampler", "counters", "complete_state")
        ) else "FAIL"
        comparisons.append(detail)
    report = {
        "schema": "ect.q256.fresh-crossed-switch-runtime-parity/v1",
        "status": "PASS" if all(item["status"] == "PASS" for item in comparisons) else "FAIL",
        "runtime_manifest_sha256": experiment.sha256_file(args.runtime_manifest),
        "implementation_commit": args.implementation_commit,
        "engineering_seed": ENGINEERING_SEED, "comparisons": comparisons,
        "automatic_retry_count": 0,
    }
    experiment.atomic_json(root / "parity_report.json", report)
    if report["status"] != "PASS":
        raise RuntimeError("engineering exact parity mismatch")
    print(json.dumps(report, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--runtime-manifest", type=Path, required=True)
    launch_parser.add_argument("--dataset", type=Path, required=True)
    launch_parser.add_argument("--transfer", type=Path, required=True)
    launch_parser.add_argument("--implementation-commit", required=True)
    launch_parser.add_argument("--output-root", type=Path, required=True)
    launch_parser.add_argument("--gpu-indices", type=int, choices=range(6), nargs=4,
                               default=(0, 1, 2, 3))
    launch_parser.set_defaults(func=launch)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--engineering-protocol", type=Path, required=True)
    worker_parser.add_argument("--arm", choices=("A", "B"), required=True)
    worker_parser.add_argument("--mode", choices=("continuous", "segmented"), required=True)
    worker_parser.add_argument("--gpu-index", type=int, required=True)
    worker_parser.set_defaults(func=worker)
    return root


def main() -> int:
    args = parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
