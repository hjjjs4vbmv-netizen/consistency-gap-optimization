"""Frozen-state target-component audit for the q=256 ECT factorial.

The audit evaluates virtual A/B/C/D cells at one *common online training
state* and one shared stochastic realization.  It never constructs or steps
an optimizer.  For the unclipped stage-0 protocol, with s=1/1.10, it measures

    tau_tar = G_B - s G_A,
    R_tar   = ||tau_tar|| / ||G_A||,
    cos(tau_tar, G_A),
    a_star  = <G_B, G_A> / ||G_A||^2,

and gates the exact denominator identities G_D=sG_A and G_B=sG_C.  The
implementation is deliberately separate from finite-training outcome analysis:
it characterizes realized common-state gradient geometry, not mediation.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform as python_platform
import pickle
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.gap_gradient_hook import (  # noqa: E402
    layer_name,
    module_state_hashes,
    set_dropout_rng_state,
    sha256_file,
    tensor_collection_sha256,
)
from training.loss import (  # noqa: E402
    TARGET_WEIGHT_FACTORIAL_ARMS,
    compute_target_weight_times,
)


ARM_FACTORS = {
    arm: factors for factors, arm in TARGET_WEIGHT_FACTORIAL_ARMS.items()
}
REPLAY_RECEIPT_SCHEMA = "ect.q256.replay-ema-export/v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CANONICAL_CIFAR10_SHA256 = (
    "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372"
)
PRIMARY_TRAINING_SEEDS = frozenset({3, 4, 5})
PRIMARY_STATE_BUDGETS_KIMG = frozenset({256, 512, 768, 1024})
PRIMARY_STATE_ARM = "A"
PRIMARY_BATCHES = 8
PRIMARY_BATCH_SIZE = 16
PRIMARY_AUDIT_SEED = 20260823
MAX_IDENTITY_RELATIVE_TOLERANCE = 1e-4
TRAJECTORY_HORIZON_FIELD = "total_kimg"
PRIMARY_LOSS_CONTRACT = {
    "P_mean": -1.1,
    "P_std": 2.0,
    "sigma_data": 0.5,
    "k": 8.0,
    "b": 1.0,
}


def configure_deterministic_runtime(
    device: torch.device,
    *,
    run_kind: str,
    preflight_only: bool,
) -> dict[str, Any]:
    """Freeze the FP32 reference-field runtime before any model execution."""
    if run_kind not in {"primary", "smoke"}:
        raise ValueError(f"unsupported run kind: {run_kind!r}")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if device.type == "cuda" and not preflight_only:
        if run_kind == "primary" and cublas_config != ":4096:8":
            raise SystemExit(
                "a full primary CUDA audit requires "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8 before process launch; "
                ":16:8 is admitted only for smoke runs"
            )
        if run_kind == "smoke" and cublas_config not in (":4096:8", ":16:8"):
            raise SystemExit(
                "a CUDA smoke audit requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                "or :16:8 before process launch"
            )
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": getattr(
            torch.backends.cuda.matmul, "allow_tf32", None
        ),
        "cudnn_allow_tf32": getattr(torch.backends.cudnn, "allow_tf32", None),
        "float32_matmul_precision": (
            torch.get_float32_matmul_precision()
            if hasattr(torch, "get_float32_matmul_precision")
            else None
        ),
        "cublas_workspace_config": cublas_config,
        "cudnn_version": torch.backends.cudnn.version(),
        "python_version": python_platform.python_version(),
        "platform": python_platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _named_norm_sq(values: Iterable[torch.Tensor]) -> float:
    return sum(float(value.detach().double().square().sum()) for value in values)


def _named_dot(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> float:
    if first.keys() != second.keys():
        raise ValueError("named tensor maps have different parameter keys")
    return sum(
        float((first[name].detach().double() * second[name].detach().double()).sum())
        for name in first
    )


def _scaled_residual(
    observed: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    scale: float,
) -> dict[str, float]:
    observed_sq = _named_norm_sq(observed.values())
    reference_sq = _named_norm_sq(reference.values())
    if observed.keys() != reference.keys():
        raise ValueError("named tensor maps have different parameter keys")
    # Form the residual directly.  The equivalent norm/dot expansion can lose
    # every significant bit when an identity is satisfied to FP32 precision.
    residual_sq = sum(
        float(
            (
                observed[name].detach().double()
                - scale * reference[name].detach().double()
            )
            .square()
            .sum()
        )
        for name in observed
    )
    denominator = max(math.sqrt(observed_sq), abs(scale) * math.sqrt(reference_sq), 1e-30)
    return {
        "relative_l2": math.sqrt(residual_sq) / denominator,
        "absolute_l2": math.sqrt(residual_sq),
        "observed_l2": math.sqrt(observed_sq),
        "expected_l2": abs(scale) * math.sqrt(reference_sq),
    }


def target_geometry(
    gradient_a: dict[str, torch.Tensor],
    gradient_b: dict[str, torch.Tensor],
    scale: float,
) -> dict[str, float]:
    """Geometry of tau=G_B-scale*G_A, accumulated in float64."""
    if gradient_a.keys() != gradient_b.keys():
        raise ValueError("named tensor maps have different parameter keys")
    norm_a_sq = _named_norm_sq(gradient_a.values())
    norm_b_sq = _named_norm_sq(gradient_b.values())
    if norm_a_sq <= 0:
        raise RuntimeError("G_A is zero; target-component geometry is undefined")
    dot_ba = _named_dot(gradient_b, gradient_a)
    tau_sq = 0.0
    tau_dot_a = 0.0
    for name in gradient_a:
        a = gradient_a[name].detach().double()
        tau = gradient_b[name].detach().double() - scale * a
        tau_sq += float(tau.square().sum())
        tau_dot_a += float((tau * a).sum())
    tau_l2 = math.sqrt(tau_sq)
    norm_a = math.sqrt(norm_a_sq)
    norm_b = math.sqrt(norm_b_sq)
    best_residual_sq = 0.0
    a_star = dot_ba / norm_a_sq
    for name in gradient_a:
        a = gradient_a[name].detach().double()
        residual = gradient_b[name].detach().double() - a_star * a
        best_residual_sq += float(residual.square().sum())
    return {
        "g_a_l2": norm_a,
        "g_b_l2": norm_b,
        "tau_tar_l2": tau_l2,
        "r_tar": tau_l2 / norm_a,
        "r_tar_over_g_b": tau_l2 / norm_b if norm_b > 0 else math.nan,
        "r_tar_over_s_g_a": tau_l2 / (abs(scale) * norm_a),
        "cos_tau_g_a": (
            tau_dot_a / (tau_l2 * norm_a) if tau_l2 > 0 else math.nan
        ),
        "cos_g_b_g_a": dot_ba / (norm_b * norm_a) if norm_b > 0 else math.nan,
        "a_star": a_star,
        "r_best_over_g_b": (
            math.sqrt(best_residual_sq) / norm_b if norm_b > 0 else math.nan
        ),
        "s_explicit": scale,
    }


def layerwise_target_geometry(
    gradient_a: dict[str, torch.Tensor],
    gradient_b: dict[str, torch.Tensor],
    scale: float,
) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for name in gradient_a:
        groups[layer_name(name)].append(name)
    rows = []
    for layer, names in sorted(groups.items()):
        a = {name: gradient_a[name] for name in names}
        b = {name: gradient_b[name] for name in names}
        norm_a_sq = _named_norm_sq(a.values())
        if norm_a_sq > 0:
            row = target_geometry(a, b, scale)
            row["reference_gradient_nonzero"] = True
        else:
            norm_b_sq = _named_norm_sq(b.values())
            row = {
                "g_a_l2": 0.0,
                "g_b_l2": math.sqrt(norm_b_sq),
                "tau_tar_l2": math.sqrt(norm_b_sq),
                "r_tar": math.nan,
                "r_tar_over_g_b": 1.0 if norm_b_sq > 0 else math.nan,
                "r_tar_over_s_g_a": math.nan,
                "cos_tau_g_a": math.nan,
                "cos_g_b_g_a": math.nan,
                "a_star": math.nan,
                "r_best_over_g_b": 1.0 if norm_b_sq > 0 else math.nan,
                "s_explicit": scale,
                "reference_gradient_nonzero": False,
            }
        row["layer"] = layer
        row["fit_scope"] = "layer_specific_scalar"
        row["parameter_tensor_count"] = len(names)
        row["parameter_coordinate_count"] = sum(gradient_a[name].numel() for name in names)
        rows.append(row)
    return rows


def _linear_quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def summarize_layer_geometry(
    layers: list[dict[str, Any]], aggregate: dict[str, Any]
) -> dict[str, Any]:
    """Summarize layer-local fits and reconcile their vector energies."""
    values = sorted(
        float(row["r_tar"])
        for row in layers
        if row["reference_gradient_nonzero"] and math.isfinite(float(row["r_tar"]))
    )
    energy_reconstruction: dict[str, dict[str, float]] = {}
    for key in ("g_a_l2", "g_b_l2", "tau_tar_l2"):
        reconstructed = math.sqrt(sum(float(row[key]) ** 2 for row in layers))
        whole_model = float(aggregate[key])
        relative_error = abs(reconstructed - whole_model) / max(
            abs(whole_model), 1e-30
        )
        energy_reconstruction[key] = {
            "whole_model": whole_model,
            "from_layer_energies": reconstructed,
            "relative_error": relative_error,
        }
    energy_gate_passed = all(
        item["relative_error"] <= 1e-12
        for item in energy_reconstruction.values()
    )
    return {
        "fit_scope": (
            "each layer has its own best-fit scalar; these are not the "
            "whole-model a_star"
        ),
        "layer_count": len(layers),
        "nonzero_reference_layer_count": len(values),
        "zero_reference_layer_count": len(layers) - len(values),
        "r_tar_unweighted_across_nonzero_reference_layers": {
            "median": _linear_quantile(values, 0.5),
            "q90": _linear_quantile(values, 0.9),
            "q95": _linear_quantile(values, 0.95),
            "max": max(values) if values else math.nan,
        },
        "energy_reconstruction": energy_reconstruction,
        "energy_reconstruction_gate_tolerance": 1e-12,
        "energy_reconstruction_gate_passed": energy_gate_passed,
    }


def collect_gradients(
    net: torch.nn.Module, *, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    gradients: dict[str, torch.Tensor] = {}
    saw_nonzero = False
    for name, parameter in net.named_parameters():
        if parameter.grad is None:
            value = torch.zeros(parameter.shape, dtype=dtype)
        else:
            source = parameter.grad.detach()
            if not bool(torch.isfinite(source).all()):
                raise FloatingPointError(f"non-finite gradient in {name}")
            value = source.to(device="cpu", dtype=dtype).clone()
        gradients[name] = value
        saw_nonzero |= bool(value.count_nonzero())
    if not saw_nonzero:
        raise RuntimeError("all parameter gradients are zero")
    return gradients


def add_in_place(
    accumulator: dict[str, torch.Tensor] | None,
    values: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if accumulator is None:
        return {name: value.double().clone() for name, value in values.items()}
    if accumulator.keys() != values.keys():
        raise ValueError("gradient accumulator keys changed")
    for name, value in values.items():
        accumulator[name].add_(value.double())
    return accumulator


def divide_in_place(values: dict[str, torch.Tensor], denominator: float) -> None:
    for value in values.values():
        value.div_(denominator)


def realized_scale_summary(
    delta_baseline: torch.Tensor,
    delta_enlarged: torch.Tensor,
    *,
    expected_scale: float,
    base_r: torch.Tensor | None = None,
    enlarged_r: torch.Tensor | None = None,
    ratio_tolerance: float = 1e-4,
) -> dict[str, float | int | bool]:
    ratios = (delta_baseline / delta_enlarged).detach().double().flatten()
    if not bool(torch.isfinite(ratios).all()) or not bool((ratios > 0).all()):
        raise FloatingPointError("realized denominator ratios must be finite and positive")
    max_error = float((ratios - expected_scale).abs().max())
    if not math.isfinite(ratio_tolerance) or ratio_tolerance < 0:
        raise ValueError("ratio_tolerance must be finite and non-negative")
    support_gate = True
    clipped_count = 0
    if (base_r is None) != (enlarged_r is None):
        raise ValueError("base_r and enlarged_r must be supplied together")
    if base_r is not None and enlarged_r is not None:
        clipped = (base_r <= 0) | (enlarged_r <= 0)
        clipped_count = int(clipped.sum().detach().cpu())
        support_gate = clipped_count == 0
    ratio_gate = max_error <= ratio_tolerance
    return {
        "count": ratios.numel(),
        "min": float(ratios.min()),
        "max": float(ratios.max()),
        "mean": float(ratios.mean()),
        "std": float(ratios.std(unbiased=False)),
        "expected": expected_scale,
        "max_abs_error_to_expected": max_error,
        "unclipped_support_gate_passed": support_gate,
        "clipped_or_zero_support_count": clipped_count,
        "ratio_numeric_gate_passed": ratio_gate,
        "constant_scale_gate_passed": support_gate and ratio_gate,
        "constant_scale_tolerance": ratio_tolerance,
    }


def factorial_loss_with_fixed_randomness(
    net: torch.nn.Module,
    loss_template,
    images: torch.Tensor,
    labels: torch.Tensor,
    t: torch.Tensor,
    eps: torch.Tensor,
    dropout_rng_state: torch.Tensor,
    *,
    target_gap_scale: float,
    denominator_gap_scale: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Evaluate one virtual factorial cell with supplied stochastic inputs."""
    if getattr(loss_template.schedule, "name", None) != "sigmoid":
        raise ValueError("target-component audit requires the native sigmoid base schedule")
    base_r = loss_template.schedule.compute_r(t=t, stage=loss_template.stage)
    r_target, r_denominator, delta_target, delta_denominator = (
        compute_target_weight_times(
            t,
            base_r,
            target_gap_scale=target_gap_scale,
            denominator_gap_scale=denominator_gap_scale,
        )
    )
    device = images.device
    set_dropout_rng_state(dropout_rng_state, device)
    d_t = net(
        images + eps * t,
        t,
        labels,
        augment_labels=None,
        force_fp32=True,
    )
    if bool((r_target > 0).any()):
        set_dropout_rng_state(dropout_rng_state, device)
        with torch.no_grad():
            d_r = net(
                images + eps * r_target,
                r_target,
                labels,
                augment_labels=None,
                force_fp32=True,
            )
        d_r = torch.nan_to_num(d_r)
        mask = r_target > 0
        d_r = mask * d_r + (~mask) * images
    else:
        d_r = images

    squared = (d_t - d_r).square().reshape(images.shape[0], -1).sum(dim=1)
    residual_zero_count = int((squared == 0).sum().detach().cpu())
    if float(loss_template.c) > 0:
        numerator = torch.sqrt(squared + float(loss_template.c) ** 2) - float(loss_template.c)
    else:
        numerator = torch.sqrt(squared)
    per_sample = numerator / delta_denominator.flatten()
    return per_sample, {
        "base_r": base_r,
        "r_target": r_target,
        "r_denominator": r_denominator,
        "delta_target": delta_target,
        "delta_denominator": delta_denominator,
        "residual_zero_count": residual_zero_count,
    }


def run_cell(
    net: torch.nn.Module,
    loss_template,
    images: torch.Tensor,
    labels: torch.Tensor,
    t: torch.Tensor,
    eps: torch.Tensor,
    dropout_rng_state: torch.Tensor,
    arm: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, Any]]:
    target_scale, denominator_scale = ARM_FACTORS[arm]
    net.zero_grad(set_to_none=True)
    per_sample, times = factorial_loss_with_fixed_randomness(
        net,
        loss_template,
        images,
        labels,
        t,
        eps,
        dropout_rng_state,
        target_gap_scale=target_scale,
        denominator_gap_scale=denominator_scale,
    )
    if float(loss_template.c) == 0.0 and times["residual_zero_count"]:
        raise FloatingPointError(
            f"arm {arm} has {times['residual_zero_count']} exact zero pair residuals; "
            "the production c=0 norm derivative is undefined there"
        )
    per_sample.mean().backward()
    return collect_gradients(net), per_sample.detach().double().cpu(), times


def _loss_identity_error(observed: torch.Tensor, expected: torch.Tensor) -> float:
    if not bool(torch.isfinite(observed).all()) or not bool(
        torch.isfinite(expected).all()
    ):
        raise FloatingPointError("non-finite per-sample loss in identity gate")
    difference = (observed - expected).norm()
    denominator = max(float(observed.norm()), float(expected.norm()), 1e-30)
    return float(difference) / denominator


def _run_probe_impl(
    net: torch.nn.Module,
    loss_template,
    data_iter,
    *,
    batches: int,
    device: torch.device,
    random_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if batches < 1:
        raise ValueError("batches must be positive")
    expected_scale = 1.0 / 1.1
    generator = torch.Generator(device=device).manual_seed(random_seed)
    sum_a: dict[str, torch.Tensor] | None = None
    sum_b: dict[str, torch.Tensor] | None = None
    batch_rows: list[dict[str, Any]] = []

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
        dropout_seed = random_seed + 100_000 + batch_index
        dropout_generator = torch.Generator(device=device).manual_seed(dropout_seed)
        dropout_state = dropout_generator.get_state()

        grad_a, loss_a, times_a = run_cell(
            net, loss_template, images, labels, t, eps, dropout_state, "A"
        )
        grad_d, loss_d, times_d = run_cell(
            net, loss_template, images, labels, t, eps, dropout_state, "D"
        )
        identity_da = _scaled_residual(grad_d, grad_a, expected_scale)
        del grad_d

        grad_c, loss_c, times_c = run_cell(
            net, loss_template, images, labels, t, eps, dropout_state, "C"
        )
        grad_b, loss_b, times_b = run_cell(
            net, loss_template, images, labels, t, eps, dropout_state, "B"
        )
        identity_bc = _scaled_residual(grad_b, grad_c, expected_scale)
        del grad_c
        time_contract = {
            "base_r_all_cells_equal": all(
                torch.equal(times_a["base_r"], times["base_r"])
                for times in (times_d, times_c, times_b)
            ),
            "target_r_a_equals_d": torch.equal(
                times_a["r_target"], times_d["r_target"]
            ),
            "target_r_c_equals_b": torch.equal(
                times_c["r_target"], times_b["r_target"]
            ),
            "denominator_r_a_equals_c": torch.equal(
                times_a["r_denominator"], times_c["r_denominator"]
            ),
            "denominator_r_d_equals_b": torch.equal(
                times_d["r_denominator"], times_b["r_denominator"]
            ),
        }
        if not all(time_contract.values()):
            raise RuntimeError(f"factorial time contract failed: {time_contract}")
        scale_summary = realized_scale_summary(
            times_a["delta_denominator"],
            times_b["delta_denominator"],
            expected_scale=expected_scale,
            base_r=times_a["base_r"],
            enlarged_r=times_b["r_denominator"],
        )
        if not scale_summary["constant_scale_gate_passed"]:
            raise RuntimeError(
                "clipping or a denominator-ratio contract violation was detected; "
                "the constant-s minibatch identity is not licensed"
            )
        geometry = target_geometry(grad_a, grad_b, expected_scale)
        batch_rows.append({
            "batch_index": batch_index,
            "sample_count": int(images.shape[0]),
            "images_sha256": tensor_collection_sha256((("images", images),)),
            "labels_sha256": tensor_collection_sha256((("labels", labels),)),
            "t_sha256": tensor_collection_sha256((("t", t),)),
            "eps_sha256": tensor_collection_sha256((("eps", eps),)),
            "dropout_rng_sha256": tensor_collection_sha256(
                (("dropout_rng", dropout_state),)
            ),
            "baseline_target_r_sha256": tensor_collection_sha256(
                (("r_target", times_a["r_target"]),)
            ),
            "enlarged_target_r_sha256": tensor_collection_sha256(
                (("r_target", times_b["r_target"]),)
            ),
            "baseline_denominator_r_sha256": tensor_collection_sha256(
                (("r_denominator", times_a["r_denominator"]),)
            ),
            "enlarged_denominator_r_sha256": tensor_collection_sha256(
                (("r_denominator", times_b["r_denominator"]),)
            ),
            **time_contract,
            **geometry,
            "identity_d_equals_s_a_relative_l2": identity_da["relative_l2"],
            "identity_b_equals_s_c_relative_l2": identity_bc["relative_l2"],
            "loss_identity_d_equals_s_a_relative_l2": _loss_identity_error(
                loss_d, expected_scale * loss_a
            ),
            "loss_identity_b_equals_s_c_relative_l2": _loss_identity_error(
                loss_b, expected_scale * loss_c
            ),
            "base_r_zero_count": int((times_a["base_r"] == 0).sum().cpu()),
            "enlarged_r_zero_count": int((times_b["r_target"] == 0).sum().cpu()),
            "baseline_pair_residual_zero_count": times_a["residual_zero_count"],
            "enlarged_pair_residual_zero_count": times_b["residual_zero_count"],
            "s_realized_min": scale_summary["min"],
            "s_realized_max": scale_summary["max"],
            "s_realized_mean": scale_summary["mean"],
            "s_max_abs_error": scale_summary["max_abs_error_to_expected"],
            "s_unclipped_support_gate_passed": scale_summary[
                "unclipped_support_gate_passed"
            ],
            "s_ratio_numeric_gate_passed": scale_summary[
                "ratio_numeric_gate_passed"
            ],
        })
        sum_a = add_in_place(sum_a, grad_a)
        sum_b = add_in_place(sum_b, grad_b)
        del grad_a, grad_b

    assert sum_a is not None and sum_b is not None
    divide_in_place(sum_a, float(batches))
    divide_in_place(sum_b, float(batches))
    aggregate = {
        "aggregation": "gradient_of_equal-weight_mean_over_fixed_batches",
        "fit_scope": "whole_model_common_scalar",
        "batch_count": batches,
        **target_geometry(sum_a, sum_b, expected_scale),
        "max_identity_d_equals_s_a_relative_l2": max(
            row["identity_d_equals_s_a_relative_l2"] for row in batch_rows
        ),
        "max_identity_b_equals_s_c_relative_l2": max(
            row["identity_b_equals_s_c_relative_l2"] for row in batch_rows
        ),
        "max_loss_identity_d_equals_s_a_relative_l2": max(
            row["loss_identity_d_equals_s_a_relative_l2"] for row in batch_rows
        ),
        "max_loss_identity_b_equals_s_c_relative_l2": max(
            row["loss_identity_b_equals_s_c_relative_l2"] for row in batch_rows
        ),
        "total_base_r_zero_count": sum(row["base_r_zero_count"] for row in batch_rows),
        "total_enlarged_r_zero_count": sum(
            row["enlarged_r_zero_count"] for row in batch_rows
        ),
        "total_baseline_pair_residual_zero_count": sum(
            row["baseline_pair_residual_zero_count"] for row in batch_rows
        ),
        "total_enlarged_pair_residual_zero_count": sum(
            row["enlarged_pair_residual_zero_count"] for row in batch_rows
        ),
    }
    layers = layerwise_target_geometry(sum_a, sum_b, expected_scale)
    return aggregate, layers, batch_rows


def run_probe(
    net: torch.nn.Module,
    loss_template,
    data_iter,
    *,
    batches: int,
    device: torch.device,
    random_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the probe while restoring process RNG and gradient-buffer state."""
    if any(parameter.grad is not None for parameter in net.parameters()):
        raise RuntimeError("audit requires empty source gradient buffers")
    cpu_rng_before = torch.get_rng_state()
    cuda_rng_before = (
        torch.cuda.get_rng_state(device=device) if device.type == "cuda" else None
    )
    try:
        return _run_probe_impl(
            net,
            loss_template,
            data_iter,
            batches=batches,
            device=device,
            random_seed=random_seed,
        )
    finally:
        net.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng_before)
        if cuda_rng_before is not None:
            torch.cuda.set_rng_state(cuda_rng_before, device=device)


def load_common_state(
    training_state_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    *,
    expected_arm: str,
    expected_kimg: int,
    expected_training_seed: int,
):
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if checkpoint.get("augment_pipe") is not None:
        raise SystemExit("augmentation-enabled checkpoints are unsupported")
    checkpoint_loss = checkpoint.get("loss_fn")
    checkpoint_ema = checkpoint.get("ema")
    if checkpoint_ema is None:
        raise SystemExit("network snapshot does not contain ema")
    checkpoint_ema_hashes = module_state_hashes(checkpoint_ema)
    del checkpoint_ema
    # Replay snapshots intentionally omit loss_fn.  In that case the strict
    # training state's trajectory_config is the authoritative constructor.
    del checkpoint

    state = torch.load(training_state_path, map_location="cpu", weights_only=False)
    if "net" not in state:
        raise SystemExit("training state does not contain online net")
    trajectory_config = state.get("trajectory_config")
    if not isinstance(trajectory_config, dict):
        raise SystemExit(
            "strict training state does not contain trajectory_config"
        )
    from training import reproducibility

    if state.get("reproducibility_schema") != reproducibility.TRAINING_STATE_SCHEMA:
        raise SystemExit("unexpected or missing strict training-state schema")
    if trajectory_config.get("schema") != reproducibility.TRAJECTORY_CONFIG_SCHEMA:
        raise SystemExit("unexpected or missing trajectory-config schema")
    if trajectory_config.get("seed") != expected_training_seed:
        raise SystemExit(
            f"trajectory seed={trajectory_config.get('seed')!r}, expected "
            f"{expected_training_seed}"
        )
    if "augment_kwargs" not in trajectory_config:
        raise SystemExit("trajectory config does not record augment_kwargs")
    if trajectory_config["augment_kwargs"] is not None:
        raise SystemExit("augmentation-enabled trajectories are unsupported")
    trajectory_loss_kwargs = trajectory_config.get("loss_kwargs")
    if not isinstance(trajectory_loss_kwargs, dict):
        raise SystemExit("trajectory config does not contain loss_kwargs")
    dataset_kwargs = trajectory_config.get("dataset_kwargs")
    if not isinstance(dataset_kwargs, dict):
        raise SystemExit("trajectory config does not contain dataset_kwargs")
    dataset_contract = {
        "use_labels": False,
        "xflip": False,
        "resolution": 32,
    }
    for key, expected in dataset_contract.items():
        if dataset_kwargs.get(key) != expected:
            raise SystemExit(
                f"trajectory dataset {key}={dataset_kwargs.get(key)!r}, "
                f"expected {expected!r}"
            )
    import dnnlib

    loss = dnnlib.util.construct_class_by_name(
        **copy.deepcopy(trajectory_loss_kwargs)
    )
    loss_source = "training_state.trajectory_config.loss_kwargs"
    if getattr(loss.schedule, "name", None) != "sigmoid":
        raise SystemExit("factorial audit requires a sigmoid-base checkpoint")
    if float(loss.q) != 256.0 or float(loss.c) != 0.0:
        raise SystemExit("factorial audit requires q=256 and c=0")
    observed_loss_contract = {
        key: float(getattr(loss, key)) for key in PRIMARY_LOSS_CONTRACT
    }
    if observed_loss_contract != PRIMARY_LOSS_CONTRACT:
        raise SystemExit(
            "factorial audit loss contract mismatch: "
            f"observed {observed_loss_contract}, expected {PRIMARY_LOSS_CONTRACT}"
        )
    embedded_arm = loss.factorial.get("arm") if loss.factorial.get("enabled") else None
    if embedded_arm != expected_arm:
        raise SystemExit(
            f"checkpoint factorial arm {embedded_arm!r} does not match {expected_arm!r}"
        )

    if checkpoint_loss is not None:
        checkpoint_contract = (
            getattr(checkpoint_loss.schedule, "name", None),
            float(checkpoint_loss.P_mean),
            float(checkpoint_loss.P_std),
            float(checkpoint_loss.sigma_data),
            float(checkpoint_loss.q),
            float(checkpoint_loss.k),
            float(checkpoint_loss.b),
            float(checkpoint_loss.c),
            int(checkpoint_loss.stage),
            checkpoint_loss.factorial,
        )
        state_contract = (
            getattr(loss.schedule, "name", None),
            float(loss.P_mean),
            float(loss.P_std),
            float(loss.sigma_data),
            float(loss.q),
            float(loss.k),
            float(loss.b),
            float(loss.c),
            int(loss.stage),
            loss.factorial,
        )
        if checkpoint_contract != state_contract:
            raise SystemExit(
                "network-snapshot loss contract does not match training-state config"
            )

    saved_trajectory_hash = state.get("trajectory_config_sha256")
    if not isinstance(saved_trajectory_hash, str):
        raise SystemExit("strict training state has no trajectory_config_sha256")
    actual_trajectory_hash = reproducibility.state_sha256(trajectory_config)
    if actual_trajectory_hash != saved_trajectory_hash:
        raise SystemExit("training-state trajectory_config hash mismatch")
    trajectory_total_kimg = trajectory_config.get(TRAJECTORY_HORIZON_FIELD)
    if (
        isinstance(trajectory_total_kimg, bool)
        or not isinstance(trajectory_total_kimg, int)
        or trajectory_total_kimg < expected_kimg
    ):
        raise SystemExit(
            "trajectory total_kimg must be an integer horizon at least as large "
            f"as the audited state: {trajectory_total_kimg!r} < {expected_kimg}"
        )
    # Source and continuation configs may differ only in their declared
    # terminal horizon. Bind both the raw config and a continuation-invariant
    # dynamics config instead of treating total_kimg as an update-rule change.
    trajectory_dynamics_config = copy.deepcopy(trajectory_config)
    del trajectory_dynamics_config[TRAJECTORY_HORIZON_FIELD]
    trajectory_dynamics_sha256 = reproducibility.state_sha256(
        trajectory_dynamics_config
    )

    state_factorial = state.get("factorial")
    if state_factorial is None:
        raise SystemExit("strict training state has no factorial metadata")
    if state_factorial != loss.factorial:
        raise SystemExit("training-state factorial metadata does not match loss config")
    state_ema = state.get("ema")
    if state_ema is None:
        raise SystemExit("strict training state does not contain ema")
    state_ema_hashes = module_state_hashes(state_ema)
    if state_ema_hashes != checkpoint_ema_hashes:
        raise SystemExit("snapshot EMA does not match training-state EMA")

    cur_nimg = state.get("cur_nimg")
    if cur_nimg is None or int(cur_nimg) != expected_kimg * 1000:
        raise SystemExit(
            f"training-state cur_nimg={cur_nimg!r}, expected {expected_kimg * 1000}"
        )
    loss_state = state.get("loss_fn_state")
    if loss_state is None or not hasattr(loss, "load_schedule_state_dict"):
        raise SystemExit("strict training state has no loadable loss_fn_state")
    if not loss.load_schedule_state_dict(loss_state):
        raise SystemExit("training-state loss_fn_state is incompatible with config")
    if (
        checkpoint_loss is not None
        and checkpoint_loss.schedule_state_dict() != loss.schedule_state_dict()
    ):
        raise SystemExit(
            "network-snapshot loss schedule state does not match training state"
        )
    if int(loss.stage) != 0:
        raise SystemExit(f"target-component audit is frozen to stage 0, got {loss.stage}")
    net = state["net"].to(device).train().requires_grad_(True)
    meta = {
        "cur_nimg": int(cur_nimg),
        "cur_tick": state.get("cur_tick"),
        "successful_optimizer_steps": state.get("successful_optimizer_steps"),
        "attempted_iteration": state.get("attempted_iteration"),
        "loss_stage": int(loss.stage),
        "loss_q": float(loss.q),
        "loss_P_mean": float(loss.P_mean),
        "loss_P_std": float(loss.P_std),
        "loss_sigma_data": float(loss.sigma_data),
        "loss_k": float(loss.k),
        "loss_b": float(loss.b),
        "loss_c": float(loss.c),
        "state_arm": expected_arm,
        "state_kimg": expected_kimg,
        "loss_source": loss_source,
        "trajectory_config_sha256": actual_trajectory_hash,
        "trajectory_dynamics_sha256": trajectory_dynamics_sha256,
        "trajectory_total_kimg": trajectory_total_kimg,
        "snapshot_ema_hashes": checkpoint_ema_hashes,
        "state_ema_hashes": state_ema_hashes,
        "training_enable_amp": trajectory_config.get("enable_amp"),
        "training_batch_gpu": trajectory_config.get("batch_gpu"),
        "training_network_use_fp16": (
            trajectory_config.get("network_kwargs", {}).get("use_fp16")
            if isinstance(trajectory_config.get("network_kwargs"), dict)
            else None
        ),
        "trajectory_dataset_kwargs": dataset_kwargs,
    }
    return net, loss, meta


def validate_asset_receipt(args) -> tuple[dict[str, Any], dict[str, str]]:
    with args.checkpoint_receipt.open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    expected_fields = {
        "schema": REPLAY_RECEIPT_SCHEMA,
        "status": "PASS",
        "seed": args.training_seed,
        "arm": args.state_arm,
        "budget_kimg": args.state_kimg,
    }
    for key, expected in expected_fields.items():
        if receipt.get(key) != expected:
            raise SystemExit(
                f"checkpoint receipt {key}={receipt.get(key)!r}, expected {expected!r}"
            )
    expected_hashes = {
        "training_state": receipt.get("source_state_sha256"),
        "checkpoint": receipt.get("snapshot_sha256"),
        "data": args.expected_data_sha256,
    }
    for label, expected in expected_hashes.items():
        if not isinstance(expected, str) or not SHA256_PATTERN.fullmatch(expected):
            raise SystemExit(f"invalid expected SHA256 for {label}")
    paths = {
        "training_state": args.training_state,
        "checkpoint": args.checkpoint,
        "data": args.data,
    }
    actual_hashes = {label: sha256_file(path) for label, path in paths.items()}
    mismatches = {
        label: {"expected": expected_hashes[label], "actual": actual_hashes[label]}
        for label in paths
        if expected_hashes[label] != actual_hashes[label]
    }
    if mismatches:
        raise SystemExit(f"asset SHA256 mismatch: {mismatches}")
    return receipt, actual_hashes


def implementation_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "protocol": REPO_ROOT
        / "analysis"
        / "Q256_TARGET_COMPONENT_AUDIT_PROTOCOL.md",
        "protocol_amendment_001": REPO_ROOT
        / "analysis"
        / "Q256_TARGET_COMPONENT_AUDIT_AMENDMENT_001.md",
        "fixed_randomness_helper": REPO_ROOT / "analysis" / "gap_gradient_hook.py",
        "loss": REPO_ROOT / "training" / "loss.py",
        "schedules": REPO_ROOT / "training" / "schedules.py",
        "reproducibility": REPO_ROOT / "training" / "reproducibility.py",
        "dataset": REPO_ROOT / "training" / "dataset.py",
        "networks": REPO_ROOT / "training" / "networks.py",
        "dnnlib_init": REPO_ROOT / "dnnlib" / "__init__.py",
        "dnnlib_util": REPO_ROOT / "dnnlib" / "util.py",
        "torch_utils_persistence": (
            REPO_ROOT / "torch_utils" / "persistence.py"
        ),
        "torch_utils_distributed": (
            REPO_ROOT / "torch_utils" / "distributed.py"
        ),
        "matrix_validator": (
            REPO_ROOT / "analysis" / "validate_q256_target_component_matrix.py"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing audit implementation dependency: {missing}")
    return {label: sha256_file(path) for label, path in paths.items()}


def _validate_run_contract(args, parser: argparse.ArgumentParser) -> None:
    """Prevent smoke or sensitivity runs from masquerading as the primary."""
    if args.identity_relative_tolerance > MAX_IDENTITY_RELATIVE_TOLERANCE:
        parser.error(
            "identity-relative-tolerance may not exceed the frozen 1e-4 gate"
        )
    if args.run_kind != "primary":
        return
    if not args.preflight_only and torch.device(args.device).type != "cuda":
        parser.error(
            "a primary gradient measurement requires a CUDA device; CPU is "
            "admitted only for preflight or smoke"
        )
    expected = {
        "state_arm": PRIMARY_STATE_ARM,
        "batches": PRIMARY_BATCHES,
        "batch_size": PRIMARY_BATCH_SIZE,
        "audit_seed": PRIMARY_AUDIT_SEED,
        "identity_relative_tolerance": MAX_IDENTITY_RELATIVE_TOLERANCE,
        "expected_data_sha256": CANONICAL_CIFAR10_SHA256,
    }
    mismatches = {
        key: {"observed": getattr(args, key), "expected": value}
        for key, value in expected.items()
        if getattr(args, key) != value
    }
    if args.training_seed not in PRIMARY_TRAINING_SEEDS:
        mismatches["training_seed"] = {
            "observed": args.training_seed,
            "expected": sorted(PRIMARY_TRAINING_SEEDS),
        }
    if args.state_kimg not in PRIMARY_STATE_BUDGETS_KIMG:
        mismatches["state_kimg"] = {
            "observed": args.state_kimg,
            "expected": sorted(PRIMARY_STATE_BUDGETS_KIMG),
        }
    if mismatches:
        parser.error(f"primary run contract mismatch: {mismatches}")


def measurement_labels(run_kind: str, audit_gate_passed: bool) -> tuple[str, str]:
    """Return machine-distinct schema/status pairs for primary and smoke runs."""
    if run_kind == "primary":
        return (
            "ect.q256.target-component-audit-primary/v2",
            (
                "PASS_PRIMARY_COMMON_STATE_GRADIENT_AUDIT"
                if audit_gate_passed
                else "INVALID_PRIMARY_AUDIT_GATE"
            ),
        )
    if run_kind == "smoke":
        return (
            "ect.q256.target-component-audit-smoke/v1",
            (
                "PASS_SMOKE_NOT_PRIMARY"
                if audit_gate_passed
                else "INVALID_SMOKE_AUDIT_GATE"
            ),
        )
    raise ValueError(f"unsupported run kind: {run_kind!r}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-kind", choices=("smoke", "primary"), required=True)
    parser.add_argument("--training-state", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--state-arm", choices=sorted(ARM_FACTORS), required=True)
    parser.add_argument("--state-kimg", type=int, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-seed", type=int, default=20260823)
    parser.add_argument("--identity-relative-tolerance", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate hashes/state contracts without loading data or backpropagating",
    )
    args = parser.parse_args(argv)
    if args.state_kimg <= 0 or args.batches <= 0 or args.batch_size <= 0:
        parser.error("state-kimg, batches, and batch-size must be positive")
    if (
        not math.isfinite(args.identity_relative_tolerance)
        or args.identity_relative_tolerance < 0
    ):
        parser.error("identity-relative-tolerance must be finite and non-negative")
    if not SHA256_PATTERN.fullmatch(args.expected_data_sha256):
        parser.error("expected-data-sha256 must be a lowercase 64-hex digest")
    _validate_run_contract(args, parser)
    return args


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def publish_measurement_artifacts(
    out: Path,
    layers: list[dict[str, Any]],
    batch_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, str]:
    """Publish a complete, hash-bound artifact directory by one rename."""
    if out.exists():
        raise FileExistsError(f"output path already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.staging-", dir=out.parent))
    try:
        layer_path = staging / "target_component_layers.csv"
        batch_path = staging / "target_component_batches.csv"
        _write_csv(layer_path, layers)
        _write_csv(batch_path, batch_rows)
        artifact_hashes = {
            layer_path.name: sha256_file(layer_path),
            batch_path.name: sha256_file(batch_path),
        }
        manifest["artifact_sha256"] = artifact_hashes
        manifest_path = staging / "target_component_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(manifest), handle, indent=2, sort_keys=True)
            handle.write("\n")
        staging.replace(out)
        return artifact_hashes
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def publish_json_artifact(out: Path, filename: str, payload: dict[str, Any]) -> None:
    """Atomically publish a single-manifest artifact directory."""
    if out.exists():
        raise FileExistsError(f"output path already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.staging-", dir=out.parent))
    try:
        with (staging / filename).open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        staging.replace(out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"output path already exists: {args.out}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but unavailable")
    runtime_contract = configure_deterministic_runtime(
        device,
        run_kind=args.run_kind,
        preflight_only=args.preflight_only,
    )
    if (
        args.run_kind == "primary"
        and not args.preflight_only
        and runtime_contract["cublas_workspace_config"] != ":4096:8"
    ):
        raise SystemExit(
            "primary matrix requires CUBLAS_WORKSPACE_CONFIG=:4096:8; "
            "the :16:8 variant is admitted only for smoke"
        )
    code_hashes = implementation_hashes()
    receipt, asset_hashes = validate_asset_receipt(args)
    net, loss, state_meta = load_common_state(
        args.training_state,
        args.checkpoint,
        device,
        expected_arm=args.state_arm,
        expected_kimg=args.state_kimg,
        expected_training_seed=args.training_seed,
    )
    if args.preflight_only:
        preflight = {
            "schema": "ect.q256.target-component-audit-preflight/v4",
            "status": f"PASS_{args.run_kind.upper()}_CONTRACT_NO_GRADIENT_MEASUREMENT",
            "declared_run_kind": args.run_kind,
            "training_state": str(args.training_state.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_receipt": str(args.checkpoint_receipt.resolve()),
            "data": str(args.data.resolve()),
            "asset_sha256": asset_hashes,
            "checkpoint_receipt_payload": receipt,
            "checkpoint_receipt_sha256": sha256_file(args.checkpoint_receipt),
            "implementation_sha256": code_hashes,
            "runtime_contract": runtime_contract,
            "state": state_meta,
            "model_parameter_buffer_hashes": module_state_hashes(net),
            "gradient_measurement_performed": False,
            "optimizer_constructed_or_stepped": False,
        }
        publish_json_artifact(args.out, "preflight_manifest.json", preflight)
        print(json.dumps(_json_safe(preflight), indent=2, sort_keys=True))
        return 0
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
    cuda_memory = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        cuda_memory = {
            "allocated_before_bytes": torch.cuda.memory_allocated(device),
            "reserved_before_bytes": torch.cuda.memory_reserved(device),
        }
    aggregate, layers, batch_rows = run_probe(
        net,
        loss,
        iter(loader),
        batches=args.batches,
        device=device,
        random_seed=args.audit_seed,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        assert cuda_memory is not None
        cuda_memory.update(
            {
                "allocated_after_bytes": torch.cuda.memory_allocated(device),
                "reserved_after_bytes": torch.cuda.memory_reserved(device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
        )
    hashes_after = module_state_hashes(net)
    state_preserved = hashes_before == hashes_after
    if not state_preserved:
        raise SystemExit("network parameter/buffer state changed during gradient-only audit")

    identity_errors = {
        key: aggregate[key]
        for key in (
            "max_identity_d_equals_s_a_relative_l2",
            "max_identity_b_equals_s_c_relative_l2",
            "max_loss_identity_d_equals_s_a_relative_l2",
            "max_loss_identity_b_equals_s_c_relative_l2",
        )
    }
    identity_gate_passed = all(
        math.isfinite(value) and value <= args.identity_relative_tolerance
        for value in identity_errors.values()
    )
    layerwise_summary = summarize_layer_geometry(layers, aggregate)
    audit_gate_passed = (
        identity_gate_passed
        and layerwise_summary["energy_reconstruction_gate_passed"]
    )
    schema, status = measurement_labels(args.run_kind, audit_gate_passed)
    manifest = {
        "schema": schema,
        "estimand": "fp32_reference_one_sided_stop_gradient_objective_gradient",
        "status": status,
        "run_kind": args.run_kind,
        "training_state": str(args.training_state.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_receipt_path": str(args.checkpoint_receipt.resolve()),
        "data": str(args.data.resolve()),
        "training_state_sha256": asset_hashes["training_state"],
        "checkpoint_sha256": asset_hashes["checkpoint"],
        "checkpoint_receipt_sha256": sha256_file(args.checkpoint_receipt),
        "checkpoint_receipt_payload": receipt,
        "dataset_sha256": asset_hashes["data"],
        "implementation_sha256": code_hashes,
        "training_seed": args.training_seed,
        "audit_seed": args.audit_seed,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "device": str(device),
        "force_fp32": True,
        "amp_used": False,
        "optimizer_constructed_or_stepped": False,
        "runtime_contract": runtime_contract,
        "cuda_memory": cuda_memory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "network_state_preserved": state_preserved,
        "network_hashes_before": hashes_before,
        "network_hashes_after": hashes_after,
        "identity_relative_tolerance": args.identity_relative_tolerance,
        "identity_errors": identity_errors,
        "identity_gate_passed": identity_gate_passed,
        "audit_gate_passed": audit_gate_passed,
        "arm_factors": ARM_FACTORS,
        "aggregate": aggregate,
        "layerwise_summary": layerwise_summary,
        "state": state_meta,
    }
    artifact_hashes = publish_measurement_artifacts(
        args.out, layers, batch_rows, manifest
    )
    print(json.dumps(_json_safe(aggregate), indent=2, sort_keys=True))
    print(json.dumps({"artifact_sha256": artifact_hashes}, sort_keys=True))
    print(f"wrote {args.out}")
    return 0 if audit_gate_passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
