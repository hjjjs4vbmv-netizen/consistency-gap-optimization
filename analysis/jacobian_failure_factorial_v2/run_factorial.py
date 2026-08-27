"""Run the calibrated v2 correctness gate or formal Jacobian factorial."""
from __future__ import annotations

import argparse
import json
import traceback
from itertools import product
from pathlib import Path
from typing import Any

import torch

from analysis.jacobian_failure_factorial import core
from analysis.operator_clock_gate import cli_common
from analysis.operator_clock_gate import core as gate


HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol.json"
DEFAULT_OUT = HERE / "results" / "raw_receipts"


def protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    cli_common.add_common_args(parser)
    cli_common.add_shard_args(parser)
    parser.set_defaults(out=DEFAULT_OUT)
    parser.add_argument("--mode", choices=("correctness", "formal"), required=True)
    parser.add_argument("--correctness-receipt", type=Path)
    return parser.parse_args()


def apply_hashes(args: argparse.Namespace, frozen: dict[str, Any]) -> None:
    mapping = {
        "expected_training_state_sha256": "training_state_sha256",
        "expected_checkpoint_sha256": "checkpoint_sha256",
        "expected_batch_file_sha256": "batch_file_sha256",
    }
    for argument, key in mapping.items():
        expected = frozen["parent_audit"][key]
        supplied = getattr(args, argument)
        if supplied is not None and supplied != expected:
            raise RuntimeError(f"{argument} conflicts with frozen v2 hash")
        setattr(args, argument, expected)


def source_assets(args: argparse.Namespace) -> dict[str, Any]:
    assets = cli_common.source_assets(args)
    assets["v2_protocol"] = {
        "path": str(PROTOCOL_PATH.resolve()),
        "sha256": cli_common.sha256_file(PROTOCOL_PATH),
    }
    assets["v2_implementation"] = {
        path.name: cli_common.sha256_file(path)
        for path in (
            Path(core.__file__).resolve(), Path(__file__).resolve())
    }
    return assets


def load_correctness(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        raise RuntimeError("v2 formal mode requires --correctness-receipt")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("formal_admissible"):
        raise RuntimeError("v2 correctness gate did not admit the formal factorial")
    if value.get("protocol_sha256") != cli_common.sha256_file(PROTOCOL_PATH):
        raise RuntimeError("correctness receipt belongs to another protocol")
    return value


def tasks(frozen: dict[str, Any], mode: str) -> list[tuple[str, int, int, str]]:
    if mode == "correctness":
        cell = frozen["correctness_gate"]["cell"]
        return [(
            cell["arm"], int(cell["audit_minibatch_id"]),
            int(cell["projection_direction_seed"]), "A_squared_gn_fp32")]
    return list(product(
        frozen["arms"], frozen["audit_minibatch_ids"],
        frozen["projection_direction_seeds"], frozen["regimes"]))


def cell_name(arm: str, batch_id: int, direction_seed: int,
              regime: str) -> str:
    return (f"arm{arm}_batch{batch_id}_dir{direction_seed}_"
            f"{regime}.json")


def main() -> None:
    args = parse_args()
    frozen = protocol()
    apply_hashes(args, frozen)
    if args.mode == "correctness" and (
            args.shard_index != 0 or args.num_shards != 1):
        raise RuntimeError("v2 correctness is exactly one unsharded cell")
    correctness = load_correctness(
        args.correctness_receipt) if args.mode == "formal" else None
    determinism = cli_common.configure_determinism()
    before_assets = source_assets(args)
    state = cli_common.load_algorithmic_state(args)
    source_before = state.sha256()
    batches = {item.audit_id: item
               for item in cli_common.load_frozen_batches(args, state.loss_fn)}
    selected_tasks = cli_common.select_shard(
        tasks(frozen, args.mode), shard_index=args.shard_index,
        num_shards=args.num_shards)
    if not selected_tasks:
        raise RuntimeError("v2 shard has no frozen tasks")
    receipt_dir = args.out / args.mode / f"shard{args.shard_index}"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    parameter_values = gate.parameter_vector(state.net)
    receipts = []
    statuses = []
    for arm, batch_id, direction_seed, regime in selected_tasks:
        direction = gate.state_relative_direction_like(
            parameter_values, int(direction_seed))
        destination = receipt_dir / cell_name(
            arm, int(batch_id), int(direction_seed), regime)
        cell: dict[str, Any] = {
            "schema_version": 2,
            "arm": arm,
            "batch_id": int(batch_id),
            "direction_id": int(direction_seed),
            "regime": regime,
            "epsilon_grid": frozen["epsilon_grid"],
            "direction_sha256": gate.tensor_map_sha256(direction),
            "direction_l2": gate.vector_l2(direction),
        }
        try:
            selected, detail = core.run_regime(
                regime, state, batches[int(batch_id)], direction, arm=arm,
                epsilons=frozen["epsilon_grid"],
                tolerance=frozen["convergence_relative_tolerance"],
                pseudo_huber_c=frozen["pseudo_huber_c"])
            cell.update({
                "status": detail["status"],
                "selected_jvp_sha256": gate.tensor_map_sha256(selected),
                "selected_jvp_l2": gate.vector_l2(selected),
                "detail": detail,
            })
            del selected
        except Exception as exc:
            cell.update({
                "status": "FAIL_CLOSED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
        gate.write_json(destination, cell)
        receipts.append(str(destination.resolve()))
        statuses.append(cell["status"])
        del direction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    after_assets = source_assets(args)
    source_after = state.sha256()
    source_preserved = (
        source_before == source_after and before_assets == after_assets)
    manifest = {
        "schema_version": 2,
        "mode": args.mode,
        "status": "COMPLETE" if source_preserved else "FAIL_CLOSED",
        "scientific_cell_status_counts": {
            value: statuses.count(value) for value in sorted(set(statuses))},
        "determinism": determinism,
        "protocol": frozen,
        "protocol_sha256": cli_common.sha256_file(PROTOCOL_PATH),
        "correctness_receipt": correctness,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "task_count": len(selected_tasks),
        "receipts": receipts,
        "source_state_sha256_before": source_before,
        "source_state_sha256_after": source_after,
        "source_preserved": source_preserved,
        "assets_before": before_assets,
        "assets_after": after_assets,
    }
    manifest_path = receipt_dir / f"{args.mode}_manifest.json"
    gate.write_json(manifest_path, manifest)
    if args.mode == "correctness":
        only = json.loads(Path(receipts[0]).read_text(encoding="utf-8"))
        detail = only.get("detail", {})
        admissible = bool(
            source_preserved and only["status"] == "PASS"
            and detail.get("finite") and detail.get("source_preserved")
            and detail.get("convergence", {}).get("passed"))
        gate.write_json(receipt_dir / "correctness_gate.json", {
            "schema_version": 2,
            "status": "PASS" if admissible else "NO_GO",
            "formal_admissible": admissible,
            "protocol_sha256": manifest["protocol_sha256"],
            "cell_receipt": receipts[0],
            "manifest": str(manifest_path.resolve()),
            "failure_action": (
                None if admissible else
                "NO_GO_DO_NOT_RUN_V2_FORMAL_FACTORIAL"),
        })


if __name__ == "__main__":
    main()

