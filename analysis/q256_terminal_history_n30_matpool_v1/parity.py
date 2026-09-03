#!/usr/bin/env python3
"""Runtime-specific 512-kimg pause/resume parity gate for the n30 launcher."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.q256_terminal_history_n30_matpool_v1 import run_node  # noqa: E402
from training import reproducibility, schedule_switch  # noqa: E402


ENGINEERING_SEED = 20260831
ENGINEERING_PROTOCOL = schedule_switch.FRESH_N12_ENGINEERING_PROTOCOL


def replace_option(command: list[str], prefix: str, value: str) -> list[str]:
    matches = [index for index, item in enumerate(command) if item.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {prefix} option")
    result = list(command)
    result[matches[0]] = value
    return result


def prefix_command(protocol: dict, run_dir: Path, arm: str, gpu: int) -> list[str]:
    command = run_node.training_command(
        protocol, run_dir, 50, arm, gpu, prefix=True
    )
    command = replace_option(command, "--seed=", f"--seed={ENGINEERING_SEED}")
    command = replace_option(
        command,
        "--planned-pause-protocol=",
        f"--planned-pause-protocol={ENGINEERING_PROTOCOL}",
    )
    return command


def continuous_command(protocol: dict, run_dir: Path, arm: str, gpu: int) -> list[str]:
    command = prefix_command(protocol, run_dir, arm, gpu)
    command = [
        item for item in command
        if item != "--stop-after-attempts=4000"
        and not item.startswith("--planned-pause-protocol=")
        and not item.startswith("--immutable-checkpoint-kimg=")
    ]
    command = replace_option(command, "--duration=", "--duration=0.640")
    command.append("--immutable-checkpoint-kimg=512,640")
    return command


def engineering_manifest(
    protocol: dict,
    protocol_sha: str,
    arm: str,
    source_receipt: dict,
    prefix_dir: Path,
    output: Path,
) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    branch = "A_to_A" if arm == "A" else "B_to_B"
    source_receipt_path = prefix_dir / "source_state_receipt.json"
    manifest = {
        "schema": schedule_switch.RUN_MANIFEST_SCHEMA,
        "experiment_protocol": ENGINEERING_PROTOCOL,
        "run_kind": "parity",
        "branch": branch,
        "seed": ENGINEERING_SEED,
        "origin_arm": arm,
        "continuation_arm": arm,
        "switch_kimg": 512,
        "final_kimg": 640,
        "protocol_sha256": protocol_sha,
        "implementation_commit": protocol["implementation_commit"],
        "source_checkpoint_manifest_sha256": run_node.sha256_file(source_receipt_path),
        "source_state": source_receipt["training_state"],
        "source_history_prefix": run_node.prepare_resume_history(prefix_dir, output),
        "immutable_output_root": str(output),
    }
    path = output / "formal_run_manifest.json"
    run_node.atomic_json(path, manifest)
    schedule_switch.load_run_manifest(path)
    return path


def suffix_command(
    protocol: dict,
    run_dir: Path,
    arm: str,
    gpu: int,
    manifest: Path,
    source: Path,
) -> list[str]:
    command = run_node.training_command(
        protocol, run_dir, 50, arm, gpu, prefix=False,
        manifest=manifest, source=source,
    )
    command = replace_option(command, "--seed=", f"--seed={ENGINEERING_SEED}")
    command = replace_option(command, "--duration=", "--duration=0.640")
    command = replace_option(
        command,
        "--immutable-checkpoint-kimg=",
        "--immutable-checkpoint-kimg=640",
    )
    return command


def worker(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve(strict=True)
    protocol = run_node.validate_protocol(protocol_path)
    gate = run_node.load_json(args.gate.resolve(strict=True))
    protocol_sha = gate["engineering_protocol_sha256"]
    protocol["protocol_sha256_runtime"] = protocol_sha
    root = Path(gate["output_root"])
    if args.mode == "continuous":
        run_dir = root / f"continuous_{args.arm}"
        receipt = run_node.run_cell(
            protocol, run_dir, args.gpu_index,
            continuous_command(protocol, run_dir, args.arm, args.gpu_index),
            f"parity:continuous_{args.arm}",
        )
        if receipt["status"] != "PASS":
            raise RuntimeError(f"continuous parity cell failed: {args.arm}")
        return
    segmented = root / f"segmented_{args.arm}"
    segmented.mkdir(exist_ok=False)
    prefix_dir = segmented / "prefix"
    receipt = run_node.run_cell(
        protocol, prefix_dir, args.gpu_index,
        prefix_command(protocol, prefix_dir, args.arm, args.gpu_index),
        f"parity:segmented_{args.arm}:prefix",
    )
    if receipt["status"] != "PASS":
        raise RuntimeError(f"segmented prefix failed: {args.arm}")
    source = run_node.export_prefix(
        prefix_dir, ENGINEERING_SEED, args.arm, protocol_sha
    )
    suffix_dir = segmented / "suffix"
    manifest = engineering_manifest(
        protocol, protocol_sha, args.arm, source, prefix_dir, suffix_dir
    )
    receipt = run_node.run_cell(
        protocol, suffix_dir, args.gpu_index,
        suffix_command(
            protocol, suffix_dir, args.arm, args.gpu_index,
            manifest, Path(source["training_state"]["path"]),
        ),
        f"parity:segmented_{args.arm}:suffix",
    )
    if receipt["status"] != "PASS":
        raise RuntimeError(f"segmented suffix failed: {args.arm}")


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
    protocol_path = args.protocol.resolve(strict=True)
    protocol = run_node.validate_protocol(protocol_path)
    gpus = run_node.query_gpus()
    if len(gpus) < 4 or any("A100" not in row["name"] for row in gpus[:4]):
        raise RuntimeError("parity requires four visible A100 GPUs")
    if run_node.compute_apps():
        raise RuntimeError("parity requires all visible GPUs to be idle")
    root = args.output_root.absolute()
    root.mkdir(parents=True, exist_ok=False)
    frozen = {
        "schema": "ect.q256.terminal-history-runtime-parity-protocol/v1",
        "status": "FROZEN",
        "output_root": str(root),
        "engineering_seed": ENGINEERING_SEED,
        "arms": ["A", "B"],
        "checks": ["A_continuous_vs_segmented_640", "B_continuous_vs_segmented_640"],
        "formal_protocol_sha256": run_node.sha256_file(protocol_path),
        "implementation_commit": protocol["implementation_commit"],
        "runtime": protocol["runtime"],
        "dataset": protocol["assets"]["dataset"],
        "transfer": protocol["assets"]["transfer"],
    }
    frozen["engineering_protocol_sha256"] = run_node.canonical_sha256(frozen)
    gate_path = root / "engineering_protocol.json"
    run_node.atomic_json(gate_path, frozen)
    plans = (
        ("A", "continuous", 0),
        ("B", "continuous", 1),
        ("A", "segmented", 2),
        ("B", "segmented", 3),
    )
    processes = []
    for arm, mode, gpu in plans:
        log = (root / f"{mode}_{arm}.worker.log").open("xb")
        command = [
            protocol["runtime"]["python"], str(Path(__file__).resolve()), "worker",
            "--protocol", str(protocol_path), "--gate", str(gate_path),
            "--arm", arm, "--mode", mode, "--gpu-index", str(gpu),
        ]
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append((arm, mode, gpu, process, log))
    failures = []
    for arm, mode, gpu, process, log in processes:
        returncode = process.wait()
        log.close()
        if returncode:
            failures.append({"arm": arm, "mode": mode, "gpu": gpu,
                             "exit_code": returncode})
    if failures:
        report = {
            "schema": "ect.q256.terminal-history-runtime-parity/v1",
            "status": "FAIL",
            "failures": failures,
            "automatic_retry_count": 0,
        }
        run_node.atomic_json(root / "parity_report.json", report)
        raise RuntimeError(f"parity workers failed: {failures}")
    comparisons = []
    for arm in ("A", "B"):
        continuous_path = root / f"continuous_{arm}" / "training-state-kimg000640.pt"
        segmented_path = root / f"segmented_{arm}" / "suffix" / "training-state-kimg000640.pt"
        continuous = torch.load(continuous_path, map_location="cpu", weights_only=False)
        segmented = torch.load(segmented_path, map_location="cpu", weights_only=False)
        first_hashes = schedule_switch.internal_state_hashes(continuous)
        second_hashes = schedule_switch.internal_state_hashes(segmented)
        first_complete = reproducibility.state_sha256(computational_state(continuous))
        second_complete = reproducibility.state_sha256(computational_state(segmented))
        detail = {
            "arm": arm,
            "parameters": first_hashes["net"] == second_hashes["net"],
            "ema": first_hashes["ema"] == second_hashes["ema"],
            "optimizer": first_hashes["optimizer"] == second_hashes["optimizer"],
            "gradscaler": first_hashes["gradscaler"] == second_hashes["gradscaler"],
            "rng": first_hashes["rank_rng"] == second_hashes["rank_rng"],
            "sampler": first_hashes["rank_sampler"] == second_hashes["rank_sampler"],
            "counters": continuous["attempted_iteration"] == segmented["attempted_iteration"] == 5000,
            "complete_state": first_complete == second_complete,
            "complete_state_sha256_continuous": first_complete,
            "complete_state_sha256_segmented": second_complete,
        }
        detail["status"] = "PASS" if all(
            detail[key] for key in (
                "parameters", "ema", "optimizer", "gradscaler",
                "rng", "sampler", "counters", "complete_state",
            )
        ) else "FAIL"
        comparisons.append(detail)
    report = {
        "schema": "ect.q256.terminal-history-runtime-parity/v1",
        "status": "PASS" if all(item["status"] == "PASS" for item in comparisons) else "FAIL",
        "formal_protocol_sha256": run_node.sha256_file(protocol_path),
        "engineering_protocol_sha256": frozen["engineering_protocol_sha256"],
        "implementation_commit": protocol["implementation_commit"],
        "runtime": run_node.runtime_fingerprint(Path(protocol["runtime"]["python"])),
        "comparisons": comparisons,
        "automatic_retry_count": 0,
    }
    run_node.atomic_json(root / "parity_report.json", report)
    if report["status"] != "PASS":
        raise RuntimeError("exact resume parity mismatch")
    print(json.dumps(report, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--protocol", type=Path, required=True)
    launch_parser.add_argument("--output-root", type=Path, required=True)
    launch_parser.set_defaults(func=launch)
    work = sub.add_parser("worker")
    work.add_argument("--protocol", type=Path, required=True)
    work.add_argument("--gate", type=Path, required=True)
    work.add_argument("--arm", choices=("A", "B"), required=True)
    work.add_argument("--mode", choices=("continuous", "segmented"), required=True)
    work.add_argument("--gpu-index", type=int, required=True)
    work.set_defaults(func=worker)
    return root


def main() -> int:
    args = parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
