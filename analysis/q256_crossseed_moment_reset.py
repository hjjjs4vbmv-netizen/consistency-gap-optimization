"""Formal q256 real-state vs moment-reset RAdam manipulation audit.

This runner never trains, samples, or computes FID/KID.  For each frozen audit
RNG seed it computes one reference/probe unscaled gradient pair, then reuses
that exact pair in disposable real-state and reset-moments optimizer branches.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "analysis") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "analysis"))

import torch

import radam_stateful_update_audit as audit_lib
import radam_update_gauge as gauge


DEFAULT_AUDIT_SEEDS = (
    2026081101, 2026081102, 2026081103, 2026081104,
    2026081105, 2026081106, 2026081107, 2026081108,
)


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
    parser.add_argument("--reference-gap-scale", type=float, default=1.0)
    parser.add_argument("--probe-gap-scale", type=float, default=1.1)
    parser.add_argument("--audit-seeds", type=_parse_audit_seeds,
                        default=DEFAULT_AUDIT_SEEDS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batch-gpu", type=int, default=16)
    parser.add_argument("--support-atol", type=float, default=0.0)
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
                        help="run only the first frozen audit seed")
    args = parser.parse_args(argv)
    try:
        args.betas = tuple(float(item) for item in args.betas.split(","))
    except ValueError as exc:
        parser.error("--betas must be beta1,beta2")
    if len(args.betas) != 2:
        parser.error("--betas must contain exactly two values")
    if (not math.isfinite(args.reference_gap_scale)
            or not math.isfinite(args.probe_gap_scale)
            or args.reference_gap_scale <= 0 or args.probe_gap_scale <= 0
            or args.reference_gap_scale == args.probe_gap_scale):
        parser.error("reference/probe gap scales must be distinct, finite, and > 0")
    if (args.batch_size < 1 or args.batch_gpu < 1
            or args.batch_size % args.batch_gpu or args.initial_scale <= 0
            or not math.isfinite(args.support_atol) or args.support_atol < 0):
        parser.error("batch sizes, initial scale, and support atol must be valid")
    return args


def _strict_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(audit_lib._json_safe(payload), handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")


def _asset_record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def dry_run_receipt(args) -> dict:
    assets = {
        "training_state": _asset_record(args.training_state),
        "checkpoint": _asset_record(args.checkpoint),
        "data": _asset_record(args.data),
    }
    return {
        "status": "PASS" if all(item["exists"] for item in assets.values()) else "FAIL_CLOSED",
        "mode": "dry_run",
        "training_seed": args.training_seed,
        "reference_gap_scale": args.reference_gap_scale,
        "probe_gap_scale": args.probe_gap_scale,
        "audit_seeds": list(args.audit_seeds),
        "conditions": ["real", "reset_moments"],
        "operations_excluded": ["training", "sample_generation", "FID", "KID"],
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
        "code_commit": audit_lib._source_commit(),
        "runner_sha256": gauge.sha256_file(Path(__file__)),
        "audit_library_sha256": gauge.sha256_file(Path(audit_lib.__file__)),
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
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "schedule": loss.schedule.name,
        "q": float(loss.q),
        "k": float(loss.k),
        "b": float(loss.b),
        "training_state_meta": state_meta,
        "operations_excluded": ["training", "sample_generation", "FID", "KID"],
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
    audit_seeds = args.audit_seeds[:1] if args.smoke else args.audit_seeds
    manifest = {
        "status": "RUNNING",
        "training_seed": args.training_seed,
        "audit_seeds": list(audit_seeds),
        "frozen_formal_audit_seeds": list(DEFAULT_AUDIT_SEEDS),
        "smoke": args.smoke,
        "provenance": provenance,
        "receipts": [],
    }
    _strict_dump(args.out / "provenance_manifest.json", manifest)
    source_optimizer_hash_before_all = gauge.state_sha256(optimizer.state_dict())
    source_parameter_hash_before_all = gauge.module_state_hashes(net)
    for audit_seed in audit_seeds:
        images, labels = _next_batch(dataset, batch_size=args.batch_size,
                                     seed=audit_seed, device=device)
        audits, layers_by_condition = audit_lib.run_moment_reset_manipulation(
            net, optimizer, loss, images, labels,
            reference_gap_scale=args.reference_gap_scale,
            probe_gap_scale=args.probe_gap_scale, amp=args.amp,
            initial_scale=args.initial_scale, scaler_state=scaler_state,
            random_seed=audit_seed, microbatch_size=args.batch_gpu,
            support_atol=args.support_atol)
        real = audits["real"]["whole_model"]
        reset = audits["reset_moments"]["whole_model"]
        if real["R_opt"] == 0:
            raise RuntimeError("suppression is undefined because R_opt_real is zero")
        suppression = 1.0 - reset["R_opt"] / real["R_opt"]
        r_grad_identical = real["R_grad"] == reset["R_grad"]
        comparison = {
            "R_opt_real": real["R_opt"],
            "R_opt_reset": reset["R_opt"],
            "suppression": suppression,
            "R_grad_real": real["R_grad"],
            "R_grad_reset": reset["R_grad"],
            "R_grad_identical": r_grad_identical,
            "R_grad_frozen_tolerance": 0.0,
        }
        for condition in ("real", "reset_moments"):
            receipt = audits[condition]
            receipt["training_seed"] = args.training_seed
            receipt["audit_seed"] = audit_seed
            receipt["comparison"] = comparison
            receipt["provenance"] = provenance
            receipt_dir = args.out / "receipts" / f"seed{args.training_seed}" / f"audit{audit_seed}"
            receipt_path = receipt_dir / f"{condition}.json"
            layer_path = receipt_dir / f"{condition}_layerwise.csv"
            _strict_dump(receipt_path, receipt)
            layer_path.parent.mkdir(parents=True, exist_ok=True)
            with layer_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=audit_lib.LAYERWISE_FIELDS)
                writer.writeheader()
                writer.writerows(audit_lib._json_safe(layers_by_condition[condition]))
            manifest["receipts"].append({
                "training_seed": args.training_seed,
                "audit_seed": audit_seed,
                "condition": condition,
                "receipt": str(receipt_path.relative_to(args.out)),
                "layerwise": str(layer_path.relative_to(args.out)),
            })
        _strict_dump(args.out / "provenance_manifest.json", manifest)
        print(json.dumps({"training_seed": args.training_seed, "audit_seed": audit_seed,
                          **comparison}, sort_keys=True), flush=True)
    source_optimizer_hash_after_all = gauge.state_sha256(optimizer.state_dict())
    source_parameter_hash_after_all = gauge.module_state_hashes(net)
    source_state_file_hash_after_all = gauge.sha256_file(args.training_state)
    manifest["status"] = "PASS"
    manifest["source_preservation_across_all_audits"] = {
        "optimizer_hash_before": source_optimizer_hash_before_all,
        "optimizer_hash_after": source_optimizer_hash_after_all,
        "parameter_hash_before": source_parameter_hash_before_all,
        "parameter_hash_after": source_parameter_hash_after_all,
        "source_state_file_hash_before": training_state_sha256,
        "source_state_file_hash_after": source_state_file_hash_after_all,
        "preserved": (source_optimizer_hash_before_all == source_optimizer_hash_after_all
                      and source_parameter_hash_before_all == source_parameter_hash_after_all
                      and training_state_sha256 == source_state_file_hash_after_all),
    }
    if not manifest["source_preservation_across_all_audits"]["preserved"]:
        raise RuntimeError("source state changed across audit batches")
    _strict_dump(args.out / "provenance_manifest.json", manifest)
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
