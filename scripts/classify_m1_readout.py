#!/usr/bin/env python3
"""Classify one fixed M1 terminal readout as finite or scientifically invalid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import build_m1_evaluation_slots as slots
from scripts import validate_m1_evaluation_job as validation
from training import m1, reproducibility, schedule_switch


def classify_fixed_input(model: torch.nn.Module, device: torch.device) -> dict:
    """Run the frozen, deterministic readout probe and return its observations."""
    resolution = getattr(model, "img_resolution", None)
    channels = getattr(model, "img_channels", None)
    label_dim = getattr(model, "label_dim", None)
    if (
        isinstance(resolution, bool) or not isinstance(resolution, int)
        or resolution <= 0
        or isinstance(channels, bool) or not isinstance(channels, int)
        or channels <= 0
        or isinstance(label_dim, bool) or not isinstance(label_dim, int)
        or label_dim < 0
    ):
        raise RuntimeError("readout lacks a valid image/label input contract")

    model = model.to(device).eval().requires_grad_(False)
    nonfinite_state = [
        name for name, tensor in model.state_dict().items()
        if torch.is_tensor(tensor) and not torch.isfinite(tensor).all().item()
    ]
    x = torch.zeros(
        [1, channels, resolution, resolution], dtype=torch.float32, device=device
    )
    sigma = torch.ones([1], dtype=torch.float32, device=device)
    labels = (
        None if label_dim == 0
        else torch.zeros([1, label_dim], dtype=torch.float32, device=device)
    )
    output = None
    forward_error = None
    try:
        with torch.no_grad():
            output = model(x, sigma, labels, force_fp32=True)
        if not torch.is_tensor(output):
            raise RuntimeError("fixed-input readout forward did not return a tensor")
    except Exception as exc:
        if not nonfinite_state:
            raise
        forward_error = {"type": type(exc).__name__, "message": str(exc)}
    output_nonfinite = (
        None if output is None else int((~torch.isfinite(output)).sum().item())
    )
    if nonfinite_state and output_nonfinite:
        classification = "NONFINITE_READOUT_STATE_AND_FIXED_OUTPUT"
    elif nonfinite_state:
        classification = "NONFINITE_READOUT_STATE"
    elif output_nonfinite:
        classification = "NONFINITE_FIXED_INPUT_OUTPUT"
    else:
        classification = "FINITE_READOUT"
    return {
        "classification": classification,
        "fixed_input": output is not None,
        "fixed_input_executed": output is not None,
        "fixed_input_spec": {
            "x": {
                "shape": [1, channels, resolution, resolution],
                "dtype": "float32", "fill_value": 0.0,
            },
            "sigma": {"shape": [1], "dtype": "float32", "fill_value": 1.0},
            "class_labels": (
                None if labels is None else {
                    "shape": [1, label_dim], "dtype": "float32", "fill_value": 0.0,
                }
            ),
            "force_fp32": True,
            "model_mode": "eval",
            "autograd": False,
            "device": str(device),
        },
        "nonfinite_state_tensor_paths": nonfinite_state,
        "fixed_input_forward_error": forward_error,
        "output_shape": None if output is None else list(output.shape),
        "output_dtype": (
            None if output is None else str(output.dtype).removeprefix("torch.")
        ),
        "output_nonfinite_count": output_nonfinite,
        "invalid_fields": (
            [f"state_dict:{name}" for name in nonfinite_state]
            + (["fixed_input_output"] if output_nonfinite else [])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--branch-manifest", type=Path, required=True)
    parser.add_argument("--terminal-state", type=Path, required=True)
    parser.add_argument("--readout", choices=m1.READOUTS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal readout classification requires an available CUDA device")

    training = slots.load_training_identity(args.training_manifest)
    implementation_checkout = validation.verify_implementation_checkout(
        training["implementation_commit"]
    )
    branch_path = args.branch_manifest.resolve(strict=True)
    manifest = schedule_switch.load_run_manifest(branch_path)
    if (
        manifest.get("experiment_protocol") != m1.PROTOCOL_ID
        or manifest.get("training_manifest_sha256")
        != training["training_manifest_sha256"]
        or manifest.get("implementation_commit") != training["implementation_commit"]
    ):
        raise RuntimeError("branch manifest does not bind the frozen M1 training manifest")
    state_path = args.terminal_state.resolve(strict=True)
    state = torch.load(state_path, map_location=torch.device("cpu"), weights_only=False)
    schedule_switch.verify_switched_state(state, manifest)
    if int(state.get("attempted_iteration", -1)) != 8_000 or int(state.get("cur_nimg", -1)) != 1_024_000:
        raise RuntimeError("readout classification requires the fixed M1 terminal state")
    state_sha256 = schedule_switch.sha256_file(str(state_path))
    try:
        training_slot = validation.validate_canonical_training_milestone(
            training, manifest["seed"], manifest["branch"], state_path,
            state_sha256,
        )
    except validation.ValidationError as exc:
        raise RuntimeError(str(exc)) from exc
    snapshot = m1.evaluator_snapshot(state, args.readout)
    source_readout_sha256 = reproducibility.module_state_sha256(snapshot["ema"])
    observation = classify_fixed_input(snapshot["ema"], device)
    invalid = observation["classification"] != "FINITE_READOUT"
    status = "SCIENTIFIC_READOUT_INVALID" if invalid else "READOUT_VALID"
    payload = {
        "schema": validation.CLASSIFIER_SCHEMA,
        "status": status,
        "protocol_id": slots.PROTOCOL_ID,
        **observation,
        "training_manifest_sha256": training["training_manifest_sha256"],
        "implementation_commit": training["implementation_commit"],
        "implementation_checkout": implementation_checkout,
        "seed": manifest["seed"], "branch": manifest["branch"],
        "readout": args.readout,
        "frozen_source_state_sha256": manifest["source_state"]["sha256"],
        "source_attempted_iteration": int(state["attempted_iteration"]),
        "source_cur_nimg": int(state["cur_nimg"]),
        "source_readout_sha256": source_readout_sha256,
        "terminal_state_path": str(state_path),
        "terminal_state_sha256": state_sha256,
        "branch_manifest_path": str(branch_path),
        "branch_manifest_sha256": schedule_switch.sha256_file(str(branch_path)),
        **training_slot,
    }
    reproducibility.atomic_json_dump(payload, args.output.resolve(), overwrite=False)
    print(json.dumps({
        "status": status,
        "nonfinite_state_tensor_count": len(
            observation["nonfinite_state_tensor_paths"]
        ),
        "output_nonfinite_count": observation["output_nonfinite_count"],
    }))
    return 2 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
