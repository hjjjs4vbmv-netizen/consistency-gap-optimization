#!/usr/bin/env python3
"""Run full-state RAdam/AMP/EMA algorithmic transition JVPs."""
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
    algorithmic_jvp,
    safe_algorithmic_direction_like,
    tensor_map_sha256,
    write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    cli_common.add_common_args(parser)
    cli_common.add_shard_args(parser)
    parser.add_argument("--epsilons", type=cli_common.parse_epsilons,
                        default=tuple(cli_common.protocol()[
                            "algorithmic_finite_difference_epsilons"]))
    parser.add_argument("--convergence-tolerance", type=float,
                        default=cli_common.protocol()[
                            "convergence_relative_tolerance"])
    return parser.parse_args(argv)


def run(args) -> int:
    assets = cli_common.source_assets(args)
    cli_common.write_run_manifest(
        args.out, "algorithmic_jvp", assets, [], "RUNNING")
    state = cli_common.load_algorithmic_state(args)
    batches = cli_common.load_frozen_batches(args, state.loss_fn)
    seeds = cli_common.protocol()["projection_direction_seeds"]
    base = state.continuous_vector()
    receipt_paths = []
    all_pass = True
    tasks = [
        (arm, batch_index, direction_index, seed)
        for arm in ARM_SPECS
        for batch_index, _batch in enumerate(batches)
        for direction_index, seed in enumerate(seeds)
    ]
    tasks = cli_common.select_shard(
        tasks, shard_index=args.shard_index, num_shards=args.num_shards)
    for arm, batch_index, direction_index, seed in tasks:
        batch = batches[batch_index]
        direction = safe_algorithmic_direction_like(
            base, seed, max_epsilon=max(args.epsilons))
        result, receipt = algorithmic_jvp(
            state, batch, direction, arm=arm, epsilons=args.epsilons,
            convergence_tolerance=args.convergence_tolerance)
        receipt.update({
            "audit_minibatch_index": batch_index,
            "audit_minibatch_id": batch.audit_id,
            "projection_direction_index": direction_index,
            "projection_direction_seed": seed,
            "direction_sha256": tensor_map_sha256(direction),
            "algorithmic_jvp_sha256": tensor_map_sha256(result),
        })
        all_pass &= receipt["status"] == "PASS"
        name = f"algorithmic_arm{arm}_batch{batch.audit_id}_dir{direction_index}.json"
        write_json(args.out / name, receipt)
        torch.save(result, args.out / name.replace(".json", ".pt"))
        receipt_paths.append(name)
    status = "PASS" if all_pass else "FAIL_CLOSED"
    assets_after = cli_common.source_assets(args)
    preserved = assets_after == assets
    cli_common.write_run_manifest(
        args.out, "algorithmic_jvp", assets, receipt_paths, status,
        assets_after=assets_after)
    return 0 if all_pass and preserved else 3


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
