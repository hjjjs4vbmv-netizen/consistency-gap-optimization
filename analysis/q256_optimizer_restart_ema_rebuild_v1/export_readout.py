#!/usr/bin/env python3
"""Export one M1 terminal readout in the evaluator snapshot format."""

import argparse
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import build_m1_evaluation_slots as evaluation_slots
from scripts import validate_m1_evaluation_job as evaluation_validation
from training import m1, reproducibility, schedule_switch


REPO_ROOT = Path(__file__).resolve().parents[2]


def verify_implementation_checkout(manifest: dict) -> dict:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        text=True,
    )
    if head != manifest.get("implementation_commit") or dirty:
        raise RuntimeError("M1 exporter requires the clean frozen implementation checkout")
    return {"head": head, "clean": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--readout", choices=m1.READOUTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--classifier-receipt", type=Path)
    parser.add_argument("--gate-state", action="store_true")
    args = parser.parse_args()

    manifest = schedule_switch.load_run_manifest(args.manifest)
    if not m1.is_m1_manifest(manifest):
        raise RuntimeError("manifest is not M1")
    implementation_checkout = verify_implementation_checkout(manifest)
    state = torch.load(
        args.state.resolve(strict=True),
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    schedule_switch.verify_switched_state(state, manifest)
    if args.gate_state:
        m1.validate_resumed_state(state, manifest)
        if (
            int(state.get("attempted_iteration", -1)) != 4032
            or int(state.get("cur_nimg", -1)) != 516096
        ):
            raise RuntimeError("M1 G4 export requires the attempt-4032 gate state")
    else:
        m1.validate_terminal_state(state, manifest)
    source_readout_sha256 = reproducibility.module_state_sha256(
        m1.readout_module(state, args.readout)
    )
    state_path = args.state.resolve()
    manifest_path = args.manifest.resolve()
    state_sha256 = schedule_switch.sha256_file(str(state_path))
    manifest_sha256 = schedule_switch.sha256_file(str(manifest_path))
    classifier = None
    training_slot = None
    if not args.gate_state:
        if args.classifier_receipt is None or args.training_manifest is None:
            raise RuntimeError("formal M1 export requires a READOUT_VALID classifier receipt")
        try:
            training = evaluation_slots.load_training_identity(args.training_manifest)
            if (
                training["training_manifest_sha256"]
                != manifest["training_manifest_sha256"]
                or training["implementation_commit"]
                != manifest["implementation_commit"]
            ):
                raise evaluation_validation.ValidationError(
                    "branch and training manifests do not match"
                )
            training_slot = evaluation_validation.validate_canonical_training_milestone(
                training, manifest["seed"], manifest["branch"], state_path,
                state_sha256,
            )
            classifier = evaluation_validation.load_valid_readout_classifier_receipt(
                args.classifier_receipt,
                {
                    "training_manifest_sha256": manifest["training_manifest_sha256"],
                    "implementation_commit": manifest["implementation_commit"],
                    "implementation_checkout": implementation_checkout,
                    "seed": manifest["seed"],
                    "branch": manifest["branch"],
                    "readout": args.readout,
                    "frozen_source_state_sha256": manifest["source_state"]["sha256"],
                    "terminal_state_path": str(state_path),
                    "terminal_state_sha256": state_sha256,
                    "branch_manifest_path": str(manifest_path),
                    "branch_manifest_sha256": manifest_sha256,
                    "source_readout_sha256": source_readout_sha256,
                    **training_slot,
                },
            )
        except evaluation_validation.ValidationError as exc:
            raise RuntimeError(str(exc)) from exc
    snapshot = m1.evaluator_snapshot(state, args.readout)
    snapshot_readout_sha256 = reproducibility.module_state_sha256(snapshot["ema"])
    if snapshot["ema"].training or snapshot_readout_sha256 != source_readout_sha256:
        raise RuntimeError("M1 exported readout changed weights or is not eval-mode")
    output = args.output.resolve()
    reproducibility.atomic_pickle_dump(snapshot, output, overwrite=False)
    receipt = {
            "schema": "ect.m1.readout-export/v1",
            "status": "PASS",
            "protocol_id": m1.PROTOCOL_ID,
            "seed": manifest["seed"],
            "branch": manifest["branch"],
            "readout": args.readout,
            "source_state_path": str(state_path),
            "terminal_state_sha256": state_sha256,
            "source_attempted_iteration": int(state["attempted_iteration"]),
            "source_cur_nimg": int(state["cur_nimg"]),
            "quality_eligible": not args.gate_state,
            "gate_state": args.gate_state,
            "branch_manifest_path": str(manifest_path),
            "branch_manifest_sha256": manifest_sha256,
            "training_manifest_sha256": manifest["training_manifest_sha256"],
            "implementation_commit": manifest["implementation_commit"],
            "implementation_checkout": implementation_checkout,
            "frozen_source_state_sha256": manifest["source_state"]["sha256"],
            "snapshot_path": str(output),
            "snapshot_sha256": schedule_switch.sha256_file(str(output)),
            "source_readout_sha256": source_readout_sha256,
            "snapshot_readout_sha256": snapshot_readout_sha256,
        }
    if classifier is not None:
        receipt.update(
            classifier_receipt_path=classifier["receipt_path"],
            classifier_receipt_sha256=classifier["receipt_sha256"],
            **training_slot,
        )
    reproducibility.atomic_json_dump(
        receipt, args.receipt.resolve(), overwrite=False
    )
    print(
        f"exported {manifest['seed']} {manifest['branch']} "
        f"{args.readout} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
