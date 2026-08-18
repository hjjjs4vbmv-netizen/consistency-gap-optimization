"""Formal q256 observed/exact-scalar x real/reset RAdam factorial audit.

This runner performs no training, sampling, FID, or KID computation.  Each
frozen audit batch uses one observed G_1.00/G_1.10 backward pair.  The exact
scalar treatment is then constructed from the saved G_1.00 without another
forward pass, and all optimizer steps occur only on disposable deep clones.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch

import radam_stateful_update_audit as audit_lib
import radam_update_gauge as gauge


DEFAULT_AUDIT_SEEDS = (
    2026081101, 2026081102, 2026081103, 2026081104,
    2026081105, 2026081106, 2026081107, 2026081108,
)
CELL_LABELS = {
    "observed_real": "A",
    "observed_reset": "B",
    "exact_scalar_real": "C",
    "exact_scalar_reset": "D",
}
OPERATIONS_EXCLUDED = [
    "training", "sample_generation", "FID", "KID",
    "continuation_training", "moment_only_variants",
]


def _parse_audit_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("audit seeds must be comma-separated integers") from exc
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("audit seeds must be unique non-negative integers")
    return seeds


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-seed", type=int, required=True, choices=(3, 4, 5))
    parser.add_argument("--training-state", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-training-state-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--code-commit", default=None,
                        help="explicit 40-hex commit for exported worktrees without .git")
    parser.add_argument("--reference-gap-scale", type=float, default=1.0)
    parser.add_argument("--probe-gap-scale", type=float, default=1.1)
    parser.add_argument("--audit-seeds", type=_parse_audit_seeds,
                        default=DEFAULT_AUDIT_SEEDS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batch-gpu", type=int, default=16)
    parser.add_argument("--support-atol", type=float, default=0.0)
    parser.add_argument("--a-star-denominator-atol", type=float, default=1e-30)
    parser.add_argument("--exact-scalar-r-grad-tolerance", type=float, default=1e-12)
    parser.add_argument("--exact-scale-identity-tolerance", type=float, default=1e-3)
    parser.add_argument("--order-numeric-tolerance", type=float, default=1e-12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--betas", default="0.9,0.999")
    parser.add_argument("--eps", dest="eps_opt", type=float, default=1e-8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--initial-scale", type=float, default=65536.0)
    parser.add_argument("--state-kimg", type=float, default=256.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="run only the first frozen audit batch")
    args = parser.parse_args(argv)
    try:
        args.betas = tuple(float(item) for item in args.betas.split(","))
    except ValueError as exc:
        parser.error("--betas must be beta1,beta2")
    if len(args.betas) != 2:
        parser.error("--betas must contain exactly two values")
    scales = (args.reference_gap_scale, args.probe_gap_scale)
    tolerances = (
        args.support_atol, args.a_star_denominator_atol,
        args.exact_scalar_r_grad_tolerance, args.exact_scale_identity_tolerance,
        args.order_numeric_tolerance,
    )
    if (not all(math.isfinite(value) and value > 0 for value in scales)
            or scales[0] == scales[1]):
        parser.error("reference/probe gap scales must be distinct, finite, and > 0")
    if any(not math.isfinite(value) or value < 0 for value in tolerances):
        parser.error("all tolerances must be finite and >= 0")
    if args.batch_size < 1 or args.batch_gpu < 1 or args.batch_size % args.batch_gpu:
        parser.error("--batch-size must be divisible by positive --batch-gpu")
    if args.initial_scale <= 0 or not math.isfinite(args.initial_scale):
        parser.error("--initial-scale must be finite and > 0")
    for label in ("expected_training_state_sha256", "expected_checkpoint_sha256",
                  "expected_data_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", getattr(args, label)):
            parser.error(f"--{label.replace('_', '-')} must be a lowercase SHA256")
    if args.code_commit is not None and not re.fullmatch(r"[0-9a-f]{40}", args.code_commit):
        parser.error("--code-commit must be a 40-hex Git commit")
    return args


def _strict_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(audit_lib._json_safe(payload), handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")


def _asset_record(path: Path, expected_sha256: str) -> dict:
    record = {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "expected_sha256": expected_sha256,
    }
    record["actual_sha256"] = gauge.sha256_file(path) if path.is_file() else None
    record["sha256_matches"] = record["actual_sha256"] == expected_sha256
    return record


def dry_run_receipt(args) -> dict:
    assets = {
        "training_state": _asset_record(
            args.training_state, args.expected_training_state_sha256),
        "checkpoint": _asset_record(args.checkpoint, args.expected_checkpoint_sha256),
        "data": _asset_record(args.data, args.expected_data_sha256),
    }
    return {
        "status": ("PASS" if all(item["exists"] and item["sha256_matches"]
                                  for item in assets.values()) else "FAIL_CLOSED"),
        "mode": "dry_run",
        "training_seed": args.training_seed,
        "reference_gap_scale": args.reference_gap_scale,
        "probe_gap_scale": args.probe_gap_scale,
        "audit_seeds": list(args.audit_seeds),
        "cells": [{"cell": label, "gradient_mode": key.rsplit("_", 1)[0],
                   "state_mode": key.rsplit("_", 1)[1]}
                  for key, label in CELL_LABELS.items()],
        "operations_excluded": OPERATIONS_EXCLUDED,
        "assets": assets,
    }


def _next_batch(dataset, *, batch_size: int, seed: int, device: torch.device):
    from torch.utils.data import DataLoader
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=0, generator=generator)
    images, labels = next(iter(loader))
    return images.to(device).to(torch.float32) / 127.5 - 1, labels.to(device)


def _provenance(args, loss, state_meta, *, training_state_sha256: str,
                checkpoint_sha256: str, dataset_sha256: str,
                dataset_hash_algorithm: str) -> dict:
    return {
        "code_commit": args.code_commit or audit_lib._source_commit(),
        "runner_sha256": gauge.sha256_file(Path(__file__)),
        "audit_library_sha256": gauge.sha256_file(Path(audit_lib.__file__)),
        "preregistration_sha256": gauge.sha256_file(
            REPO_ROOT / "analysis" / "q256_gradient_state_factorial_preregistration.json"),
        "training_seed": args.training_seed,
        "source_state": str(args.training_state.resolve()),
        "source_state_sha256": training_state_sha256,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "data": str(args.data.resolve()),
        "dataset_sha256": dataset_sha256,
        "dataset_hash_algorithm": dataset_hash_algorithm,
        "state_kimg": args.state_kimg,
        "reference_gap_scale": args.reference_gap_scale,
        "probe_gap_scale": args.probe_gap_scale,
        "batch_size": args.batch_size,
        "batch_gpu": args.batch_gpu,
        "support_atol": args.support_atol,
        "a_star_denominator_atol": args.a_star_denominator_atol,
        "exact_scalar_r_grad_tolerance": args.exact_scalar_r_grad_tolerance,
        "exact_scale_identity_tolerance": args.exact_scale_identity_tolerance,
        "order_numeric_tolerance": args.order_numeric_tolerance,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "schedule": loss.schedule.name,
        "q": float(loss.q),
        "k": float(loss.k),
        "b": float(loss.b),
        "training_state_meta": state_meta,
        "operations_excluded": OPERATIONS_EXCLUDED,
    }


def _cell_receipt(batch_receipt: dict, key: str, *, training_seed: int,
                  audit_seed: int, provenance: dict) -> dict:
    cell = copy.deepcopy(batch_receipt["cells"][key])
    return {
        "schema_version": 1,
        "training_seed": training_seed,
        "audit_batch_id": audit_seed,
        "audit_seed": audit_seed,
        "cell": CELL_LABELS[key],
        "gradient_mode": cell["gradient_mode"],
        "state_mode": cell["state_mode"],
        "reference_gap_scale": batch_receipt["reference_gap_scale"],
        "probe_gap_scale": batch_receipt["probe_gap_scale"],
        "a_star": batch_receipt["a_star"],
        "source_state_hash": cell["source_state_hash"],
        "result_hash": cell["result_hash"],
        "finite_gate": cell["finite_gate"],
        "branch_skipped_flag": cell["branch_skipped_flag"],
        "whole_model": cell["whole_model"],
        "branches": cell["branches"],
        "batch_correctness_gate": batch_receipt["correctness_gate"],
        "source_state_non_committing": batch_receipt["source_state_non_committing"],
        "control_control_identity": batch_receipt["control_control_identity"],
        "order_invariance_and_rerun": batch_receipt["order_invariance_and_rerun"][key],
        "gradient_contract": batch_receipt["gradient_contract"],
        "randomness_contract": batch_receipt["randomness_contract"],
        "provenance": provenance,
    }


def run(args) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    dry = dry_run_receipt(args)
    _strict_dump(args.out / "dry_run.json", dry)
    if dry["status"] != "PASS":
        print(json.dumps(dry, indent=2))
        return 2
    if args.dry_run:
        print(json.dumps(dry, indent=2))
        return 0
    device = torch.device(args.device)
    if args.amp and (device.type != "cuda" or not torch.cuda.is_available()):
        raise SystemExit("AMP formal audit requires an available CUDA device")
    loss = audit_lib.load_loss_from_checkpoint(args.checkpoint)
    if float(loss.q) != 256.0:
        raise SystemExit(f"formal audit requires q=256, got q={loss.q}")
    net, optimizer, scaler_state, loss_fn_state, state_meta = audit_lib.load_training_state(
        args.training_state, device, lr=args.lr, betas=args.betas, eps_opt=args.eps_opt)
    if args.amp and scaler_state is None:
        raise SystemExit("formal AMP audit requires gradscaler_state")
    if loss_fn_state is not None and hasattr(loss, "load_schedule_state_dict"):
        if not loss.load_schedule_state_dict(copy.deepcopy(loss_fn_state)):
            raise SystemExit("loss_fn_state is incompatible with checkpoint loss")
    if state_meta["cur_nimg"] is None or int(state_meta["cur_nimg"]) != 256000:
        raise SystemExit(f"formal audit requires cur_nimg=256000, got {state_meta['cur_nimg']}")
    from training.dataset import ImageFolderDataset
    dataset = ImageFolderDataset(path=str(args.data), use_labels=False, xflip=False,
                                 cache=True, resolution=net.img_resolution)
    training_state_sha256 = gauge.sha256_file(args.training_state)
    checkpoint_sha256 = gauge.sha256_file(args.checkpoint)
    dataset_sha256, dataset_hash_algorithm = gauge.dataset_sha256(args.data)
    provenance = _provenance(
        args, loss, state_meta, training_state_sha256=training_state_sha256,
        checkpoint_sha256=checkpoint_sha256, dataset_sha256=dataset_sha256,
        dataset_hash_algorithm=dataset_hash_algorithm)
    if provenance["code_commit"] is None:
        raise SystemExit("code commit is unavailable; pass --code-commit for exported worktrees")
    audit_seeds = args.audit_seeds[:1] if args.smoke else args.audit_seeds
    manifest = {
        "schema_version": 1,
        "status": "RUNNING",
        "training_seed": args.training_seed,
        "audit_seeds": list(audit_seeds),
        "frozen_formal_audit_seeds": list(DEFAULT_AUDIT_SEEDS),
        "smoke": args.smoke,
        "provenance": provenance,
        "receipts": [],
    }
    _strict_dump(args.out / "provenance_manifest.json", manifest)
    source_optimizer_before = gauge.state_sha256(optimizer.state_dict())
    source_parameter_before = gauge.module_state_hashes(net)
    source_gradient_before = audit_lib._source_gradient_buffers_hash(net)
    all_valid = True
    for audit_seed in audit_seeds:
        images, labels = _next_batch(dataset, batch_size=args.batch_size,
                                     seed=audit_seed, device=device)
        batch_receipt, layers_by_cell = audit_lib.run_gradient_state_factorial(
            net, optimizer, loss, images, labels,
            reference_gap_scale=args.reference_gap_scale,
            probe_gap_scale=args.probe_gap_scale, amp=args.amp,
            initial_scale=args.initial_scale, scaler_state=scaler_state,
            random_seed=audit_seed, microbatch_size=args.batch_gpu,
            support_atol=args.support_atol,
            a_star_denominator_atol=args.a_star_denominator_atol,
            exact_scalar_r_grad_tolerance=args.exact_scalar_r_grad_tolerance,
            exact_scale_identity_tolerance=args.exact_scale_identity_tolerance,
            order_numeric_tolerance=args.order_numeric_tolerance,
            verify_reverse_order=True)
        batch_receipt.update({
            "training_seed": args.training_seed,
            "audit_batch_id": audit_seed,
            "audit_seed": audit_seed,
            "provenance": provenance,
        })
        receipt_dir = (args.out / "receipts" / f"seed{args.training_seed}"
                       / f"audit{audit_seed}")
        batch_path = receipt_dir / "batch_receipt.json"
        _strict_dump(batch_path, batch_receipt)
        all_valid &= bool(batch_receipt["correctness_gate"]["valid"])
        for key in CELL_LABELS:
            cell_receipt = _cell_receipt(
                batch_receipt, key, training_seed=args.training_seed,
                audit_seed=audit_seed, provenance=provenance)
            receipt_path = receipt_dir / f"{key}.json"
            layer_path = receipt_dir / f"{key}_layerwise.csv"
            _strict_dump(receipt_path, cell_receipt)
            layer_path.parent.mkdir(parents=True, exist_ok=True)
            with layer_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=audit_lib.GENERIC_LAYERWISE_FIELDS,
                                        lineterminator="\n")
                writer.writeheader()
                writer.writerows(audit_lib._json_safe(layers_by_cell[key]))
            manifest["receipts"].append({
                "training_seed": args.training_seed,
                "audit_batch_id": audit_seed,
                "cell": CELL_LABELS[key],
                "gradient_mode": cell_receipt["gradient_mode"],
                "state_mode": cell_receipt["state_mode"],
                "receipt": str(receipt_path.relative_to(args.out)),
                "layerwise": str(layer_path.relative_to(args.out)),
                "result_hash": cell_receipt["result_hash"],
            })
        manifest.setdefault("batch_receipts", []).append(str(batch_path.relative_to(args.out)))
        _strict_dump(args.out / "provenance_manifest.json", manifest)
        print(json.dumps({
            "training_seed": args.training_seed,
            "audit_batch_id": audit_seed,
            "a_star": batch_receipt["a_star"],
            "valid": batch_receipt["correctness_gate"]["valid"],
            "R_opt": {CELL_LABELS[key]: batch_receipt["cells"][key]["whole_model"]["R_opt"]
                      for key in CELL_LABELS},
        }, sort_keys=True), flush=True)
        if not batch_receipt["correctness_gate"]["valid"]:
            break
    preservation = {
        "optimizer_hash_before": source_optimizer_before,
        "optimizer_hash_after": gauge.state_sha256(optimizer.state_dict()),
        "parameter_hash_before": source_parameter_before,
        "parameter_hash_after": gauge.module_state_hashes(net),
        "gradient_buffers_hash_before": source_gradient_before,
        "gradient_buffers_hash_after": audit_lib._source_gradient_buffers_hash(net),
        "source_state_file_hash_before": training_state_sha256,
        "source_state_file_hash_after": gauge.sha256_file(args.training_state),
    }
    preservation["preserved"] = (
        preservation["optimizer_hash_before"] == preservation["optimizer_hash_after"]
        and preservation["parameter_hash_before"] == preservation["parameter_hash_after"]
        and preservation["gradient_buffers_hash_before"] == preservation["gradient_buffers_hash_after"]
        and preservation["source_state_file_hash_before"] == preservation["source_state_file_hash_after"]
    )
    manifest["source_preservation_across_all_audits"] = preservation
    manifest["status"] = "PASS" if all_valid and preservation["preserved"] else "INVALID"
    _strict_dump(args.out / "provenance_manifest.json", manifest)
    return 0 if manifest["status"] == "PASS" else 3


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
