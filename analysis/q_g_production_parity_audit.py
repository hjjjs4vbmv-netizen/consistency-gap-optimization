#!/usr/bin/env python3
"""Production parity audit for the stage-0 q--g reparameterization.

The audit compares

    reference: q=256, g=1.10
    candidate: q=256/1.10, g=1

through the production ``global_sigmoid`` schedule.  It first evaluates a
dense deterministic t-grid and then, at one frozen online network state and
shared minibatch/RNG realization, compares realized pairs, target inputs,
detached target outputs, explicit weights, per-sample losses, and one-sided
parameter gradients.  It never constructs or steps an optimizer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.gap_gradient_hook import (  # noqa: E402
    module_state_hashes,
    set_dropout_rng_state,
    sha256_file,
    tensor_collection_sha256,
)
from analysis.q256_target_component_audit import (  # noqa: E402
    CANONICAL_CIFAR10_SHA256,
    collect_gradients,
    configure_deterministic_runtime,
    load_common_state,
    publish_json_artifact,
    validate_asset_receipt,
)
from training.loss import ECMLoss  # noqa: E402
from training.schedules import get_schedule  # noqa: E402


SCHEMA = "ect.q-g-production-parity-audit/v1"
REFERENCE_Q = 256.0
REFERENCE_G = 1.10
CANDIDATE_Q = REFERENCE_Q / REFERENCE_G
CANDIDATE_G = 1.0
K = 8.0
B = 1.0
STAGE = 0
DEFAULT_GRID_SIZE = 8192
DEFAULT_BATCHES = 2
DEFAULT_BATCH_SIZE = 16
DEFAULT_AUDIT_SEED = 20260825
DEFAULT_FIELD_TOLERANCE = 1e-6
DEFAULT_COORDINATE_EPS_MULTIPLIER = 32.0


def _relative_l2(first: torch.Tensor, second: torch.Tensor) -> float:
    a = first.detach().double().reshape(-1)
    b = second.detach().double().reshape(-1)
    denominator = max(float(a.norm()), float(b.norm()), 1e-30)
    return float((a - b).norm()) / denominator


def _relative_named_l2(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> float:
    if first.keys() != second.keys():
        raise ValueError("gradient parameter keys differ")
    numerator_sq = 0.0
    first_sq = 0.0
    second_sq = 0.0
    for name in first:
        a = first[name].detach().double()
        b = second[name].detach().double()
        numerator_sq += float((a - b).square().sum())
        first_sq += float(a.square().sum())
        second_sq += float(b.square().sum())
    denominator = max(math.sqrt(first_sq), math.sqrt(second_sq), 1e-30)
    return math.sqrt(numerator_sq) / denominator


def _normalized_max_abs(first: torch.Tensor, second: torch.Tensor) -> float:
    a = first.detach().double()
    b = second.detach().double()
    scale = torch.maximum(torch.maximum(a.abs(), b.abs()), torch.ones_like(a))
    return float(((a - b).abs() / scale).max())


def _normalized_max_abs_with_scale(
    first: torch.Tensor, second: torch.Tensor, scale_tensor: torch.Tensor
) -> float:
    a = first.detach().double()
    b = second.detach().double()
    scale = torch.maximum(
        scale_tensor.detach().double().abs(), torch.ones_like(a)
    )
    return float(((a - b).abs() / scale).max())


def _finite_pair(first: torch.Tensor, second: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(first).all()) or not bool(torch.isfinite(second).all()):
        raise FloatingPointError(f"non-finite values in {label}")


def dense_grid_parity(
    *,
    grid_size: int,
    dtype: torch.dtype,
    eps_multiplier: float,
    field_tolerance: float = DEFAULT_FIELD_TOLERANCE,
) -> dict[str, Any]:
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    reference = get_schedule(
        "global_sigmoid", q=REFERENCE_Q, k=K, b=B,
        global_gap_scale=REFERENCE_G,
    )
    candidate = get_schedule(
        "global_sigmoid", q=CANDIDATE_Q, k=K, b=B,
        global_gap_scale=CANDIDATE_G,
    )
    # The range covers far more than the mass of exp(N(-1.1, 2^2)) while
    # retaining very small positive values that stress subtractive rounding.
    t = torch.logspace(-6, 4, grid_size, dtype=dtype).reshape(-1, 1, 1, 1)
    r_reference = reference.compute_r(t=t, stage=STAGE)
    r_candidate = candidate.compute_r(t=t, stage=STAGE)
    delta_reference = t - r_reference
    delta_candidate = t - r_candidate
    for label, first, second in (
        ("r", r_reference, r_candidate),
        ("delta", delta_reference, delta_candidate),
    ):
        _finite_pair(first, second, label)
    if not bool((delta_reference > 0).all()) or not bool((delta_candidate > 0).all()):
        raise FloatingPointError("dense-grid gaps must be strictly positive")
    coordinate_tolerance = eps_multiplier * torch.finfo(dtype).eps
    r_error = _normalized_max_abs(r_reference, r_candidate)
    delta_error = _normalized_max_abs_with_scale(
        delta_reference, delta_candidate, t
    )
    weight_reference = delta_reference.reciprocal()
    weight_candidate = delta_candidate.reciprocal()
    return {
        "dtype": str(dtype).replace("torch.", ""),
        "grid_size": grid_size,
        "t_min": float(t.min()),
        "t_max": float(t.max()),
        "reference_zero_or_clipped_count": int((r_reference <= 0).sum()),
        "candidate_zero_or_clipped_count": int((r_candidate <= 0).sum()),
        "r_normalized_max_abs": r_error,
        "delta_normalized_max_abs": delta_error,
        "delta_relative_l2": _relative_l2(delta_reference, delta_candidate),
        "weight_relative_l2": _relative_l2(weight_reference, weight_candidate),
        "field_tolerance": field_tolerance,
        "weight_gate_passed": (
            _relative_l2(weight_reference, weight_candidate) <= field_tolerance
        ),
        "coordinate_tolerance": coordinate_tolerance,
        "coordinate_gate_passed": (
            r_error <= coordinate_tolerance and delta_error <= coordinate_tolerance
        ),
    }


def construct_paths(loss_template: ECMLoss) -> tuple[ECMLoss, ECMLoss]:
    common = {
        "P_mean": float(loss_template.P_mean),
        "P_std": float(loss_template.P_std),
        "sigma_data": float(loss_template.sigma_data),
        "c": float(loss_template.c),
        "k": float(loss_template.k),
        "b": float(loss_template.b),
        "cut": 4.0,
        "adj": "global_sigmoid",
        "factorial_protocol": "none",
    }
    reference = ECMLoss(q=REFERENCE_Q, global_gap_scale=REFERENCE_G, **common)
    candidate = ECMLoss(q=CANDIDATE_Q, global_gap_scale=CANDIDATE_G, **common)
    reference.update_schedule(STAGE)
    candidate.update_schedule(STAGE)
    return reference, candidate


def fixed_path_loss(
    net: torch.nn.Module,
    loss: ECMLoss,
    images: torch.Tensor,
    labels: torch.Tensor,
    t: torch.Tensor,
    eps: torch.Tensor,
    dropout_rng_state: torch.Tensor,
    *,
    force_fp32: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    r = loss.schedule.compute_r(t=t, stage=loss.stage)
    delta = t - r
    if not bool(torch.isfinite(r).all()) or not bool((delta > 0).all()):
        raise FloatingPointError("invalid realized pair")
    target_input = images + eps * r
    set_dropout_rng_state(dropout_rng_state, images.device)
    d_t = net(
        images + eps * t,
        t,
        labels,
        augment_labels=None,
        force_fp32=force_fp32,
    )
    set_dropout_rng_state(dropout_rng_state, images.device)
    with torch.no_grad():
        d_r = net(
            target_input,
            r,
            labels,
            augment_labels=None,
            force_fp32=force_fp32,
        )
    d_r = torch.nan_to_num(d_r)
    mask = r > 0
    d_r = mask * d_r + (~mask) * images
    squared = (d_t - d_r).square().reshape(images.shape[0], -1).sum(dim=1)
    zero_residual_count = int((squared == 0).sum().detach().cpu())
    if float(loss.c) > 0:
        numerator = torch.sqrt(squared + float(loss.c) ** 2) - float(loss.c)
    else:
        if zero_residual_count:
            raise FloatingPointError("c=0 path has an exact zero residual")
        numerator = torch.sqrt(squared)
    weight = delta.flatten().reciprocal()
    return numerator * weight, {
        "r": r,
        "delta": delta,
        "target_input": target_input,
        "target_output": d_r,
        "weight": weight,
        "zero_residual_count": zero_residual_count,
    }


def run_path(
    net: torch.nn.Module,
    loss: ECMLoss,
    images: torch.Tensor,
    labels: torch.Tensor,
    t: torch.Tensor,
    eps: torch.Tensor,
    dropout_rng_state: torch.Tensor,
    *,
    force_fp32: bool,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor | int]]:
    net.zero_grad(set_to_none=True)
    per_sample, tensors = fixed_path_loss(
        net,
        loss,
        images,
        labels,
        t,
        eps,
        dropout_rng_state,
        force_fp32=force_fp32,
    )
    per_sample.mean().backward()
    gradients = collect_gradients(net)
    return gradients, per_sample.detach().cpu(), tensors


def minibatch_parity(
    net: torch.nn.Module,
    loss_template: ECMLoss,
    data_iter,
    *,
    batches: int,
    device: torch.device,
    audit_seed: int,
    force_fp32: bool,
    field_tolerance: float,
    coordinate_eps_multiplier: float,
) -> dict[str, Any]:
    reference, candidate = construct_paths(loss_template)
    generator = torch.Generator(device=device).manual_seed(audit_seed)
    rows = []
    for batch_index in range(batches):
        images, labels = next(data_iter)
        images = images.to(device).to(torch.float32) / 127.5 - 1
        labels = labels.to(device)
        t = torch.randn(
            (images.shape[0], 1, 1, 1), generator=generator, device=device
        )
        t = (t * float(loss_template.P_std) + float(loss_template.P_mean)).exp()
        eps = torch.randn(
            images.shape, generator=generator, device=device, dtype=images.dtype
        )
        dropout_generator = torch.Generator(device=device).manual_seed(
            audit_seed + 100_000 + batch_index
        )
        dropout_state = dropout_generator.get_state()
        grad_ref, loss_ref, tensors_ref = run_path(
            net, reference, images, labels, t, eps, dropout_state,
            force_fp32=force_fp32,
        )
        grad_cand, loss_cand, tensors_cand = run_path(
            net, candidate, images, labels, t, eps, dropout_state,
            force_fp32=force_fp32,
        )
        coordinate_tolerance = (
            coordinate_eps_multiplier * torch.finfo(t.dtype).eps
        )
        r_error = _normalized_max_abs(tensors_ref["r"], tensors_cand["r"])
        delta_error = _normalized_max_abs_with_scale(
            tensors_ref["delta"], tensors_cand["delta"], t
        )
        target_input_error = _normalized_max_abs(
            tensors_ref["target_input"], tensors_cand["target_input"]
        )
        row = {
            "batch_index": batch_index,
            "sample_count": int(images.shape[0]),
            "force_fp32": force_fp32,
            "images_sha256": tensor_collection_sha256((("images", images),)),
            "t_sha256": tensor_collection_sha256((("t", t),)),
            "eps_sha256": tensor_collection_sha256((("eps", eps),)),
            "r_reference_sha256": tensor_collection_sha256(
                (("r", tensors_ref["r"]),)
            ),
            "r_candidate_sha256": tensor_collection_sha256(
                (("r", tensors_cand["r"]),)
            ),
            "r_normalized_max_abs": r_error,
            "delta_normalized_max_abs": delta_error,
            "target_input_normalized_max_abs": target_input_error,
            "weight_relative_l2": _relative_l2(
                tensors_ref["weight"], tensors_cand["weight"]
            ),
            "target_output_relative_l2": _relative_l2(
                tensors_ref["target_output"], tensors_cand["target_output"]
            ),
            "loss_relative_l2": _relative_l2(loss_ref, loss_cand),
            "gradient_relative_l2": _relative_named_l2(grad_ref, grad_cand),
            "coordinate_tolerance": coordinate_tolerance,
        }
        row["coordinate_gate_passed"] = max(
            r_error, delta_error, target_input_error
        ) <= coordinate_tolerance
        row["field_gate_passed"] = max(
            row["weight_relative_l2"],
            row["target_output_relative_l2"],
            row["loss_relative_l2"],
            row["gradient_relative_l2"],
        ) <= field_tolerance
        rows.append(row)
        del grad_ref, grad_cand
    return {
        "precision_path": "fp32_reference" if force_fp32 else "native_network_precision",
        "batch_count": batches,
        "field_tolerance": field_tolerance,
        "coordinate_eps_multiplier": coordinate_eps_multiplier,
        "max_r_normalized_max_abs": max(row["r_normalized_max_abs"] for row in rows),
        "max_delta_normalized_max_abs": max(
            row["delta_normalized_max_abs"] for row in rows
        ),
        "max_target_input_normalized_max_abs": max(
            row["target_input_normalized_max_abs"] for row in rows
        ),
        "max_weight_relative_l2": max(row["weight_relative_l2"] for row in rows),
        "max_target_output_relative_l2": max(
            row["target_output_relative_l2"] for row in rows
        ),
        "max_loss_relative_l2": max(row["loss_relative_l2"] for row in rows),
        "max_gradient_relative_l2": max(
            row["gradient_relative_l2"] for row in rows
        ),
        "coordinate_gate_passed": all(row["coordinate_gate_passed"] for row in rows),
        "field_gate_passed": all(row["field_gate_passed"] for row in rows),
        "batches": rows,
    }


def implementation_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "loss": REPO_ROOT / "training" / "loss.py",
        "schedules": REPO_ROOT / "training" / "schedules.py",
        "networks": REPO_ROOT / "training" / "networks.py",
        "common_state_loader": REPO_ROOT / "analysis" / "q256_target_component_audit.py",
        "fixed_randomness_helper": REPO_ROOT / "analysis" / "gap_gradient_hook.py",
        "theory_note": REPO_ROOT / "docs" / "Q_VS_G_NOTE.md",
        "protocol": REPO_ROOT / "analysis" / "Q_G_PRODUCTION_PARITY_AUDIT_PROTOCOL.md",
    }
    return {label: sha256_file(path) for label, path in paths.items()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-state", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--expected-data-sha256", default=CANONICAL_CIFAR10_SHA256
    )
    parser.add_argument("--state-arm", default="A")
    parser.add_argument("--state-kimg", type=int, default=512)
    parser.add_argument("--training-seed", type=int, default=3)
    parser.add_argument("--batches", type=int, default=DEFAULT_BATCHES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--audit-seed", type=int, default=DEFAULT_AUDIT_SEED)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument(
        "--field-tolerance", type=float, default=DEFAULT_FIELD_TOLERANCE
    )
    parser.add_argument(
        "--coordinate-eps-multiplier",
        type=float,
        default=DEFAULT_COORDINATE_EPS_MULTIPLIER,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.batches < 1 or args.batch_size < 1 or args.grid_size < 2:
        parser.error("batches, batch-size, and grid-size must be positive")
    if args.field_tolerance <= 0 or args.coordinate_eps_multiplier <= 0:
        parser.error("tolerances must be positive")
    if args.state_arm != "A":
        parser.error("the parity audit is frozen to an arm-A common state")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    runtime = configure_deterministic_runtime(
        device, run_kind="smoke", preflight_only=False
    )
    receipt, asset_hashes = validate_asset_receipt(args)
    net, loss_template, state = load_common_state(
        args.training_state,
        args.checkpoint,
        device,
        expected_arm=args.state_arm,
        expected_kimg=args.state_kimg,
        expected_training_seed=args.training_seed,
    )
    from torch.utils.data import DataLoader
    from training.dataset import ImageFolderDataset

    loader_generator = torch.Generator(device="cpu").manual_seed(args.audit_seed)
    dataset = ImageFolderDataset(
        path=str(args.data), use_labels=False, xflip=False, cache=False,
        resolution=net.img_resolution,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=loader_generator,
    )
    hashes_before = module_state_hashes(net)
    grid = [
        dense_grid_parity(
            grid_size=args.grid_size,
            dtype=dtype,
            eps_multiplier=args.coordinate_eps_multiplier,
            field_tolerance=args.field_tolerance,
        )
        for dtype in (torch.float32, torch.float64)
    ]
    native = minibatch_parity(
        net,
        loss_template,
        iter(loader),
        batches=args.batches,
        device=device,
        audit_seed=args.audit_seed,
        force_fp32=False,
        field_tolerance=args.field_tolerance,
        coordinate_eps_multiplier=args.coordinate_eps_multiplier,
    )
    # Reset the loader so both precision paths see the same examples.
    loader_generator = torch.Generator(device="cpu").manual_seed(args.audit_seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=loader_generator,
    )
    fp32 = minibatch_parity(
        net,
        loss_template,
        iter(loader),
        batches=args.batches,
        device=device,
        audit_seed=args.audit_seed,
        force_fp32=True,
        field_tolerance=args.field_tolerance,
        coordinate_eps_multiplier=args.coordinate_eps_multiplier,
    )
    net.zero_grad(set_to_none=True)
    hashes_after = module_state_hashes(net)
    state_preserved = hashes_before == hashes_after
    gate = (
        state_preserved
        and all(
            row["coordinate_gate_passed"] and row["weight_gate_passed"]
            for row in grid
        )
        and native["coordinate_gate_passed"]
        and native["field_gate_passed"]
        and fp32["coordinate_gate_passed"]
        and fp32["field_gate_passed"]
    )
    payload = {
        "schema": SCHEMA,
        "status": "PASS_PRODUCTION_PARITY" if gate else "FAIL_PRODUCTION_PARITY",
        "reference": {"q": REFERENCE_Q, "g": REFERENCE_G},
        "candidate": {"q": CANDIDATE_Q, "g": CANDIDATE_G},
        "analytic_relation": "candidate_q = reference_q / reference_g at stage 0",
        "scope": "stage-0 unclipped production global_sigmoid pair construction",
        "dense_grid": grid,
        "minibatch_native": native,
        "minibatch_fp32_reference": fp32,
        "runtime": runtime,
        "implementation_sha256": implementation_hashes(),
        "asset_sha256": asset_hashes,
        "checkpoint_receipt_sha256": sha256_file(args.checkpoint_receipt),
        "receipt_status": receipt.get("status"),
        "state": {
            "training_seed": args.training_seed,
            "state_arm": args.state_arm,
            "state_kimg": args.state_kimg,
            "trajectory_config_sha256": state["trajectory_config_sha256"],
            "trajectory_dynamics_sha256": state["trajectory_dynamics_sha256"],
        },
        "optimizer_constructed_or_stepped": False,
        "network_state_preserved": state_preserved,
        "gate_passed": gate,
        "claim_boundary": (
            "Passing establishes numerical parity for the audited grid, state, batches, "
            "and precision paths. It does not prove bitwise identity for every future "
            "training update or license attribution of cross-run differences to q."
        ),
    }
    publish_json_artifact(args.out, "q_g_production_parity.json", payload)
    print(json.dumps({
        "status": payload["status"],
        "native_gradient_relative_l2": native["max_gradient_relative_l2"],
        "fp32_gradient_relative_l2": fp32["max_gradient_relative_l2"],
        "out": str(args.out),
    }, indent=2, sort_keys=True))
    return 0 if gate else 3


if __name__ == "__main__":
    raise SystemExit(main())
