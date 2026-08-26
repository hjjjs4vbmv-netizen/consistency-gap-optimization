#!/usr/bin/env python3
"""Run squared-baseline and true recompute-and-detach ECT field JVPs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from analysis.operator_clock_gate import cli_common
from analysis.operator_clock_gate.core import (
    ARM_SPECS,
    field_jvp,
    state_relative_direction_like,
    squared_gn_operator_jvp,
    tensor_map_sha256,
    write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    cli_common.add_common_args(parser)
    parser.add_argument("--epsilons", type=cli_common.parse_epsilons,
                        default=tuple(cli_common.protocol()["finite_difference_epsilons"]))
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--convergence-tolerance", type=float, default=0.05)
    return parser.parse_args(argv)


def run(args) -> int:
    assets = cli_common.source_assets(args)
    cli_common.write_run_manifest(
        args.out, "field_jvp", assets, [], "RUNNING")
    state = cli_common.load_algorithmic_state(args)
    batches = cli_common.load_frozen_batches(args, state.loss_fn)
    learning_rate = (float(args.learning_rate) if args.learning_rate is not None
                     else float(state.optimizer.param_groups[0]["lr"]))
    seeds = cli_common.protocol()["projection_direction_seeds"]
    base = {name: value.detach().double().cpu()
            for name, value in state.net.named_parameters()}
    receipt_paths = []
    all_pass = True
    for arm in ARM_SPECS:
        for batch_index, batch in enumerate(batches):
            for direction_index, seed in enumerate(seeds):
                direction = state_relative_direction_like(base, seed)
                square, square_receipt = squared_gn_operator_jvp(
                    state.net, state.loss_fn, [batch], direction, arm=arm,
                    learning_rate=learning_rate)
                field, field_receipt = field_jvp(
                    state.net, state.loss_fn, [batch], direction, arm=arm,
                    epsilons=args.epsilons, learning_rate=learning_rate,
                    convergence_tolerance=args.convergence_tolerance)
                payload = {
                    "schema_version": 1, "arm": arm,
                    "audit_minibatch_id": batch.audit_id,
                    "projection_direction_seed": seed,
                    "projection_direction_index": direction_index,
                    "audit_minibatch_index": batch_index,
                    "squared_baseline": square_receipt,
                    "recompute_detach_field": field_receipt,
                    "vector_hashes": {
                        "direction": tensor_map_sha256(direction),
                        "squared_operator_jvp": tensor_map_sha256(square),
                        "field_operator_jvp": tensor_map_sha256(field),
                    },
                }
                payload["status"] = (
                    "PASS" if square_receipt["status"] == "PASS"
                    and field_receipt["status"] == "PASS" else "FAIL_CLOSED")
                all_pass &= payload["status"] == "PASS"
                name = f"field_arm{arm}_batch{batch.audit_id}_dir{direction_index}.json"
                write_json(args.out / name, payload)
                torch.save({"squared_operator_jvp": square,
                            "field_operator_jvp": field}, args.out / name.replace(".json", ".pt"))
                receipt_paths.append(name)
    status = "PASS" if all_pass else "FAIL_CLOSED"
    assets_after = cli_common.source_assets(args)
    preserved = assets_after == assets
    cli_common.write_run_manifest(
        args.out, "field_jvp", assets, receipt_paths, status,
        assets_after=assets_after)
    return 0 if all_pass and preserved else 3


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
