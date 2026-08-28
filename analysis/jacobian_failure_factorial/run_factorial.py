"""Run the preregistered Jacobian-failure correctness gate or factorial.

The correctness mode must pass before formal mode is admitted.  Formal mode
retains scientific failures as receipts and never drops a frozen cell.
"""
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


def apply_frozen_hash_defaults(args: argparse.Namespace,
                               frozen: dict[str, Any]) -> None:
    parent = frozen["parent_audit"]
    defaults = {
        "expected_training_state_sha256": parent["training_state_sha256"],
        "expected_checkpoint_sha256": parent["checkpoint_sha256"],
        "expected_batch_file_sha256": parent["batch_file_sha256"],
    }
    for name, expected in defaults.items():
        supplied = getattr(args, name)
        if supplied is not None and supplied != expected:
            raise RuntimeError(
                f"{name}={supplied} conflicts with frozen hash {expected}")
        setattr(args, name, expected)


def source_assets(args: argparse.Namespace) -> dict[str, Any]:
    assets = cli_common.source_assets(args)
    assets["factorial_protocol"] = {
        "path": str(PROTOCOL_PATH.resolve()),
        "sha256": cli_common.sha256_file(PROTOCOL_PATH),
    }
    assets["factorial_implementation"] = {
        path.name: cli_common.sha256_file(path)
        for path in (HERE / "core.py", Path(__file__).resolve())
    }
    return assets


def cell_name(arm: str, batch_id: int, direction_seed: int,
              regime: str) -> str:
    return (f"arm{arm}_batch{batch_id}_dir{direction_seed}_"
            f"{regime}.json")


def load_correctness_receipt(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        raise RuntimeError("formal mode requires --correctness-receipt")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("formal_admissible"):
        raise RuntimeError("correctness gate did not admit the formal factorial")
    if value.get("protocol_sha256") != cli_common.sha256_file(PROTOCOL_PATH):
        raise RuntimeError("correctness receipt was produced under another protocol")
    return value


def frozen_tasks(frozen: dict[str, Any], mode: str) -> list[tuple[str, int, int, str]]:
    gate_cell = frozen["correctness_gate"]["cell"]
    if mode == "correctness":
        return [(
            gate_cell["arm"], gate_cell["audit_minibatch_id"],
            gate_cell["projection_direction_seed"], "A_squared_gn_fp32",
        )]
    return list(product(
        frozen["arms"], frozen["audit_minibatch_ids"],
        frozen["projection_direction_seeds"], frozen["regimes"],
    ))


def main() -> None:
    args = parse_args()
    frozen = protocol()
    apply_frozen_hash_defaults(args, frozen)
    if args.mode == "correctness" and (
            args.shard_index != 0 or args.num_shards != 1):
        raise RuntimeError("correctness mode is exactly one unsharded frozen cell")
    determinism = cli_common.configure_determinism()
    if args.mode == "formal":
        correctness = load_correctness_receipt(args.correctness_receipt)
    else:
        correctness = None

    before_assets = source_assets(args)
    state = cli_common.load_algorithmic_state(args)
    source_state_before = state.sha256()
    all_batches = cli_common.load_frozen_batches(args, state.loss_fn)
    batches = {batch.audit_id: batch for batch in all_batches}
    expected_batches = set(frozen["audit_minibatch_ids"])
    if not expected_batches.issubset(batches):
        raise RuntimeError("frozen audit minibatches are missing")

    tasks = frozen_tasks(frozen, args.mode)
    tasks = cli_common.select_shard(
        tasks, shard_index=args.shard_index, num_shards=args.num_shards)
    if not tasks:
        raise RuntimeError("this shard has no frozen factorial tasks")
    receipt_dir = args.out / args.mode / f"shard{args.shard_index}"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    parameter_values = gate.parameter_vector(state.net)
    receipts = []
    statuses = []

    for arm, batch_id, direction_seed, regime in tasks:
        direction = gate.state_relative_direction_like(
            parameter_values, int(direction_seed))
        destination = receipt_dir / cell_name(
            arm, int(batch_id), int(direction_seed), regime)
        cell: dict[str, Any] = {
            "schema_version": 1,
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
                pseudo_huber_c=frozen["pseudo_huber_c"],
            )
            cell.update({
                "status": detail["status"],
                "selected_jvp_sha256": gate.tensor_map_sha256(selected),
                "selected_jvp_l2": gate.vector_l2(selected),
                "detail": detail,
            })
            del selected
        except Exception as exc:  # scientific/numerical failures are retained.
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

    source_state_after = state.sha256()
    after_assets = source_assets(args)
    source_preserved = (source_state_before == source_state_after
                        and before_assets == after_assets)
    manifest = {
        "schema_version": 1,
        "mode": args.mode,
        "status": ("COMPLETE" if source_preserved and len(receipts) == len(tasks)
                   else "FAIL_CLOSED"),
        "scientific_cell_status_counts": {
            value: statuses.count(value) for value in sorted(set(statuses))
        },
        "determinism": determinism,
        "protocol": frozen,
        "protocol_sha256": cli_common.sha256_file(PROTOCOL_PATH),
        "correctness_receipt": correctness,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "task_count": len(tasks),
        "receipts": receipts,
        "source_state_sha256_before": source_state_before,
        "source_state_sha256_after": source_state_after,
        "source_preserved": source_preserved,
        "assets_before": before_assets,
        "assets_after": after_assets,
    }
    manifest_path = receipt_dir / f"{args.mode}_manifest.json"
    gate.write_json(manifest_path, manifest)

    if args.mode == "correctness":
        only = json.loads(Path(receipts[0]).read_text(encoding="utf-8"))
        detail = only.get("detail", {})
        formal_admissible = bool(
            manifest["status"] == "COMPLETE"
            and only["status"] == "PASS"
            and detail.get("finite")
            and detail.get("source_preserved")
            and detail.get("convergence", {}).get("passed"))
        gate.write_json(receipt_dir / "correctness_gate.json", {
            "schema_version": 1,
            "status": "PASS" if formal_admissible else "NO_GO",
            "formal_admissible": formal_admissible,
            "protocol_sha256": manifest["protocol_sha256"],
            "cell_receipt": receipts[0],
            "manifest": str(manifest_path.resolve()),
            "failure_action": (None if formal_admissible else
                               "NO_GO_DO_NOT_RUN_FORMAL_FACTORIAL"),
        })


if __name__ == "__main__":
    main()
