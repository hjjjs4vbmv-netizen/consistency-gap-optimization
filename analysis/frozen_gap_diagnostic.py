"""Frozen diagnostics for paired gap interventions.

The canonical scaling-law assumptions live in ``theory/gap_scaling_theorem``.
This module only records empirical schedule facts: realized gaps, clipping,
the theorem-valid subset supplied to a scalar-law evaluator, and frozen-state
gradient contrasts.  It deliberately keeps whole-batch projections separate
from per-sample scalar quantities.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import NamedTuple

import torch


class RealizedGapPair(NamedTuple):
    """Realized schedule quantities for one paired diagnostic batch."""

    baseline_gap: torch.Tensor
    probe_gap: torch.Tensor
    ratio: torch.Tensor
    valid: torch.Tensor
    upper_clipped: torch.Tensor
    theorem_valid: torch.Tensor
    upper_clip_rate: float


def scaling_exponent(nu: float, p: float, alpha: float) -> float:
    """Return κ = ν(p - 1) - α after validating finite inputs."""
    values = {"nu": nu, "p": p, "alpha": alpha}
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise ValueError("nu, p, and alpha must be finite")
    if nu <= 0:
        raise ValueError("nu must be positive")
    return float(nu) * (float(p) - 1.0) - float(alpha)


def global_sigmoid_upper_clip_free(
    *, q: float, k: float, b: float, stage: int, requested_gain: float,
) -> bool:
    """Certify no upper-gap clipping for the nonnegative sigmoid regime.

    For ``k >= 0``, ``b >= 0``, and the canonical ``t >= 0`` support,
    sigmoid(-b*t) <= 1/2, so
    ``requested_gain * (1 + k / 2) / q**(stage + 1) <= 1`` is sufficient for
    the global-gap schedule to remain unclipped for every sample.
    """
    if stage < 0:
        raise ValueError("stage must be nonnegative")
    values = {"q": q, "k": k, "b": b, "requested_gain": requested_gain}
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise ValueError("q, k, b, and requested_gain must be finite")
    if q <= 1 or k < 0 or b < 0 or requested_gain <= 0:
        raise ValueError(
            "require q > 1, k >= 0, b >= 0, and requested_gain > 0"
        )
    return requested_gain * (1.0 + k / 2.0) / q ** (stage + 1) <= 1.0


def _as_float_tensor(value: torch.Tensor | float, *, like: torch.Tensor | None = None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if like is not None:
        tensor = tensor.to(device=like.device, dtype=like.dtype)
    elif not tensor.is_floating_point():
        tensor = tensor.to(torch.get_default_dtype())
    return tensor


def realized_gap_pair(
    t: torch.Tensor | float,
    r_baseline: torch.Tensor | float,
    r_probe: torch.Tensor | float,
    *,
    requested_gain: float,
) -> RealizedGapPair:
    """Compute realized ``Δ_g / Δ_1`` and the upper-gap clipping mask.

    ``r_baseline`` and ``r_probe`` must be the schedule outputs actually
    supplied to the loss.  For a global-gap intervention, the unclipped
    identity is ``Δ_g = requested_gain * Δ_1``.  The schedule enforces
    ``Δ_g <= t``.  The returned ratio remains useful for an empirical receipt
    even when clipping occurs, but those clipped samples are marked outside
    the local theorem's domain through ``theorem_valid``.
    """
    if not math.isfinite(float(requested_gain)) or requested_gain <= 0:
        raise ValueError("requested_gain must be finite and positive")
    t_tensor = _as_float_tensor(t)
    r1 = _as_float_tensor(r_baseline, like=t_tensor)
    rg = _as_float_tensor(r_probe, like=t_tensor)
    t_tensor, r1, rg = torch.broadcast_tensors(t_tensor, r1, rg)

    finite = torch.isfinite(t_tensor) & torch.isfinite(r1) & torch.isfinite(rg)
    base = (t_tensor - r1).clamp_min(0)
    probe = (t_tensor - rg).clamp_min(0)
    valid = finite & (base > 0)
    ratio = torch.full_like(base, float("nan"))
    ratio[valid] = probe[valid] / base[valid]

    intended = base * float(requested_gain)
    tolerance = 16 * torch.finfo(t_tensor.dtype).eps * torch.maximum(
        torch.maximum(intended.abs(), probe.abs()), t_tensor.abs()
    )
    upper_clipped = valid & (probe < intended - tolerance)
    theorem_valid = valid & ~upper_clipped
    clip_rate = (
        float(upper_clipped[valid].to(torch.float64).mean().cpu())
        if bool(valid.any())
        else float("nan")
    )
    return RealizedGapPair(
        base, probe, ratio, valid, upper_clipped, theorem_valid, clip_rate,
    )


def a_pred_realized(
    t: torch.Tensor | float,
    r_baseline: torch.Tensor | float,
    r_probe: torch.Tensor | float,
    *,
    requested_gain: float,
    nu: float,
    p: float,
    alpha: float,
) -> tuple[torch.Tensor, RealizedGapPair]:
    """Return predictions only on the unclipped theorem-valid subset.

    The realized ratio is still available in the returned ``RealizedGapPair``
    for every finite positive-baseline-gap sample.  A clipped/boundary sample
    receives ``NaN`` here rather than an extrapolated theorem prediction.
    """
    pair = realized_gap_pair(
        t, r_baseline, r_probe, requested_gain=requested_gain,
    )
    prediction = torch.full_like(pair.ratio, float("nan"))
    prediction[pair.theorem_valid] = pair.ratio[pair.theorem_valid].pow(
        scaling_exponent(nu, p, alpha)
    )
    return prediction, pair


def batch_optimal_scalar(reference: torch.Tensor, probe: torch.Tensor) -> torch.Tensor:
    """Return the whole-batch least-squares scalar ``<Gg,G1>/||G1||²``."""
    reference, probe = torch.broadcast_tensors(reference, probe)
    denominator = torch.sum(reference * reference)
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
        raise ValueError("reference batch gradient must have finite nonzero norm")
    return torch.sum(probe * reference) / denominator


def batch_decomposition(
    reference_per_sample: torch.Tensor,
    scalars: torch.Tensor,
    local_residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(mean_scalar, heterogeneity, local)`` in Proposition 2.

    If ``g'_i = a_i g_i + r_i``, then
    ``G' = mean_scalar * G + heterogeneity + local`` for batch means.  The
    identity does not rank batch and sample residual magnitudes.
    """
    if reference_per_sample.ndim < 2:
        raise ValueError("reference_per_sample must have a sample dimension and parameter dimensions")
    if local_residual.shape != reference_per_sample.shape:
        raise ValueError("local_residual must match reference_per_sample")
    if scalars.ndim != 1 or scalars.shape[0] != reference_per_sample.shape[0]:
        raise ValueError("scalars must contain exactly one value per sample")
    scalar_shape = (scalars.shape[0],) + (1,) * (reference_per_sample.ndim - 1)
    mean_scalar = scalars.mean()
    heterogeneity = ((scalars.reshape(scalar_shape) - mean_scalar) * reference_per_sample).mean(dim=0)
    local = local_residual.mean(dim=0)
    return mean_scalar, heterogeneity, local


def four_arm_interaction(gradients: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Compute ``G_B - G_C - G_D + G_A`` after a four-arm run completes."""
    if set(gradients) != {"A", "B", "C", "D"}:
        raise ValueError("gradients must contain exactly A, B, C, and D")
    reference = gradients["A"]
    if not isinstance(reference, torch.Tensor) or not reference.is_floating_point():
        raise ValueError("four-arm gradients must be floating-point tensors")
    for cell, gradient in gradients.items():
        if (not isinstance(gradient, torch.Tensor) or gradient.shape != reference.shape
                or gradient.dtype != reference.dtype or gradient.device != reference.device):
            raise ValueError(f"gradient {cell} does not match gradient A")
    return gradients["B"] - gradients["C"] - gradients["D"] + gradients["A"]
