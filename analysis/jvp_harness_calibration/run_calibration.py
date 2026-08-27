"""Run the separately frozen squared-GN harness calibration."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch

from analysis.jvp_harness_calibration import core
from analysis.operator_clock_gate import cli_common
from analysis.operator_clock_gate import core as gate


HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol.json"
DEFAULT_OUT = HERE / "results" / "raw_receipts"


def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    cli_common.add_common_args(parser)
    parser.set_defaults(out=DEFAULT_OUT)
    return parser.parse_args()


def apply_hashes(args: argparse.Namespace, frozen: dict) -> None:
    mapping = {
        "expected_training_state_sha256": "training_state_sha256",
        "expected_checkpoint_sha256": "checkpoint_sha256",
        "expected_batch_file_sha256": "batch_file_sha256",
    }
    for argument, key in mapping.items():
        expected = frozen["parent"][key]
        supplied = getattr(args, argument)
        if supplied is not None and supplied != expected:
            raise RuntimeError(f"{argument} conflicts with frozen hash")
        setattr(args, argument, expected)


def source_assets(args: argparse.Namespace) -> dict:
    assets = cli_common.source_assets(args)
    assets["calibration_protocol"] = {
        "path": str(PROTOCOL_PATH.resolve()),
        "sha256": cli_common.sha256_file(PROTOCOL_PATH),
    }
    assets["calibration_implementation"] = {
        path.name: cli_common.sha256_file(path)
        for path in (HERE / "core.py", Path(__file__).resolve())
    }
    return assets


def main() -> None:
    args = parse_args()
    frozen = protocol()
    apply_hashes(args, frozen)
    determinism = cli_common.configure_determinism()
    assets_before = source_assets(args)
    state = cli_common.load_algorithmic_state(args)
    source_before = state.sha256()
    batches = {item.audit_id: item
               for item in cli_common.load_frozen_batches(args, state.loss_fn)}
    cell = frozen["cell"]
    batch = batches[int(cell["audit_minibatch_id"])]
    direction = gate.state_relative_direction_like(
        gate.parameter_vector(state.net),
        int(cell["projection_direction_seed"]))
    receipt = {
        "schema_version": 1,
        "protocol": frozen,
        "protocol_sha256": cli_common.sha256_file(PROTOCOL_PATH),
        "determinism": determinism,
        "direction_sha256": gate.tensor_map_sha256(direction),
        "direction_l2": gate.vector_l2(direction),
        "rows": [],
    }
    try:
        oracle_tangent, oracle_action, oracle_detail = core.exact_oracle(
            state.net, state.loss_fn, [batch], direction, arm=cell["arm"])
        receipt["oracle"] = oracle_detail
        oracle_ok = bool(
            oracle_detail["finite"] and oracle_detail["source_preserved"])
        if oracle_ok:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for epsilon in frozen["epsilon_grid"]:
                tangent, action, detail = core.finite_difference_estimate(
                    state.net, state.loss_fn, [batch], direction,
                    arm=cell["arm"], epsilon=float(epsilon))
                detail["tangent_vs_oracle"] = core.comparison(
                    tangent, oracle_tangent)
                detail["action_vs_oracle"] = core.comparison(
                    [action[name] for name in sorted(action)],
                    [oracle_action[name] for name in sorted(oracle_action)])
                receipt["rows"].append(detail)
                del tangent, action
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        decision = core.classify(
            receipt["rows"],
            tolerance=float(frozen["primary_relative_error_tolerance"]),
            minimum_consecutive=int(
                frozen["plateau_minimum_consecutive_scales"]),
            oracle_ok=oracle_ok)
        receipt.update(decision)
    except Exception as exc:
        receipt.update({
            "verdict": "ORACLE_UNAVAILABLE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
    receipt["source_preserved"] = source_before == state.sha256()
    receipt["assets_before"] = assets_before
    receipt["assets_after"] = source_assets(args)
    receipt["assets_preserved"] = receipt["assets_before"] == receipt["assets_after"]
    if not receipt["source_preserved"] or not receipt["assets_preserved"]:
        receipt["verdict"] = "ORACLE_UNAVAILABLE"
    args.out.mkdir(parents=True, exist_ok=True)
    gate.write_json(args.out / "calibration_receipt.json", receipt)


if __name__ == "__main__":
    main()

