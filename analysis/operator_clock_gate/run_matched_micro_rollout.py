#!/usr/bin/env python3
"""Run matched A/B/C/D counterfactual continuations for 1/4/16/64 steps."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from analysis.operator_clock_gate import cli_common
from analysis.operator_clock_gate.core import matched_micro_rollout, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    cli_common.add_common_args(parser)
    parser.add_argument("--fixed-latent-file", type=Path, default=None,
                        help="Optional trusted torch tensor; otherwise first frozen noise is used")
    return parser.parse_args(argv)


def run(args) -> int:
    assets = cli_common.source_assets(args)
    cli_common.write_run_manifest(
        args.out, "matched_micro_rollout", assets, [], "RUNNING")
    state = cli_common.load_algorithmic_state(args)
    batches = cli_common.load_frozen_batches(args, state.loss_fn)
    frozen = cli_common.protocol()
    latent = None
    if args.fixed_latent_file is not None:
        latent = torch.load(args.fixed_latent_file, map_location=args.device,
                            weights_only=False)
        if isinstance(latent, dict):
            latent = latent.get("latents")
        if not isinstance(latent, torch.Tensor):
            raise RuntimeError("fixed latent file must contain a tensor or {'latents': tensor}")
        assets["fixed_latent"] = {
            "path": str(args.fixed_latent_file.resolve()),
            "sha256": cli_common.sha256_file(args.fixed_latent_file),
        }
    receipt = matched_micro_rollout(
        state, batches, horizons=frozen["horizons"],
        projection_seeds=frozen["projection_direction_seeds"],
        fixed_latent=latent)
    receipt["assets"] = assets
    output = args.out / "matched_micro_rollout.json"
    write_json(output, receipt)
    assets_after = cli_common.source_assets(args)
    if args.fixed_latent_file is not None:
        assets_after["fixed_latent"] = {
            "path": str(args.fixed_latent_file.resolve()),
            "sha256": cli_common.sha256_file(args.fixed_latent_file),
        }
    preserved = assets_after == assets
    cli_common.write_run_manifest(
        args.out, "matched_micro_rollout", assets, [output.name], receipt["status"],
        assets_after=assets_after)
    return 0 if receipt["status"] == "PASS" and preserved else 3


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
