"""Autograd-oracle calibration of the squared-GN finite-difference harness."""
from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import torch

from analysis.jacobian_failure_factorial import core as factorial
from analysis.operator_clock_gate import core as gate


TensorMap = Mapping[str, torch.Tensor]


def _flat_l2(tensors: Sequence[torch.Tensor]) -> float:
    return math.sqrt(sum(float(value.detach().double().square().sum())
                         for value in tensors))


def _flat_relative_error(left: Sequence[torch.Tensor],
                         right: Sequence[torch.Tensor]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("tensor sequences must be nonempty and aligned")
    delta = [a.detach().double() - b.detach().double()
             for a, b in zip(left, right)]
    return _flat_l2(delta) / max(
        _flat_l2(right), torch.finfo(torch.float64).tiny)


def _flat_cosine(left: Sequence[torch.Tensor],
                 right: Sequence[torch.Tensor]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("tensor sequences must be nonempty and aligned")
    dot = sum(float(a.detach().double().mul(b.detach().double()).sum())
              for a, b in zip(left, right))
    denominator = _flat_l2(left) * _flat_l2(right)
    if denominator <= torch.finfo(torch.float64).tiny:
        return float("nan")
    return max(-1.0, min(1.0, dot / denominator))


def comparison(left: Sequence[torch.Tensor],
               oracle: Sequence[torch.Tensor]) -> dict[str, float | bool]:
    left_norm = _flat_l2(left)
    oracle_norm = _flat_l2(oracle)
    finite = all(bool(torch.isfinite(value).all()) for value in left)
    return {
        "finite": finite,
        "relative_error": _flat_relative_error(left, oracle),
        "cosine": _flat_cosine(left, oracle),
        "norm": left_norm,
        "oracle_norm": oracle_norm,
        "norm_ratio": left_norm / max(
            oracle_norm, torch.finfo(torch.float64).tiny),
    }


def _directional_residual_autograd(
    net: torch.nn.Module,
    loss_template: Any,
    batch: gate.AuditBatch,
    direction: TensorMap,
    *,
    arm: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return exact executed-graph ``(J_i-J_j)u`` and ``J_i^T`` action."""
    branch = copy.deepcopy(net).train().requires_grad_(True)
    params = tuple(branch.parameters())
    names = tuple(name for name, _ in branch.named_parameters())
    if set(names) != set(direction):
        raise ValueError("direction does not match network parameters")
    online, target, _, _ = factorial._pair_tensors(
        branch, loss_template, batch, arm, force_fp32=True, target_grad=True)
    residual = online - target
    cotangent = torch.zeros_like(residual, requires_grad=True)
    parameter_vjp = torch.autograd.grad(
        residual, params, grad_outputs=cotangent, create_graph=True,
        retain_graph=True, allow_unused=True)
    directional_scalar = None
    for name, value in zip(names, parameter_vjp):
        if value is None:
            continue
        term = (value * direction[name].to(value)).sum()
        directional_scalar = term if directional_scalar is None else (
            directional_scalar + term)
    if directional_scalar is None:
        raise RuntimeError("residual has no parameter-dependent coordinates")
    tangent = torch.autograd.grad(
        directional_scalar, cotangent, retain_graph=True)[0]
    action_scalar = (online * tangent.detach()).sum()
    action_grads = torch.autograd.grad(
        action_scalar, params, allow_unused=True)
    action = {
        name: (torch.zeros_like(parameter) if grad is None else grad)
        .detach().double().cpu()
        for name, parameter, grad in zip(names, params, action_grads)
    }
    tangent_cpu = tangent.detach().double().cpu()
    del branch, online, target, residual, cotangent, parameter_vjp
    return tangent_cpu, action


def exact_oracle(
    source_net: torch.nn.Module,
    loss_template: Any,
    batches: Sequence[gate.AuditBatch | gate.AuditBatchGroup],
    direction: TensorMap,
    *,
    arm: str,
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    source_hash = gate.state_sha256(source_net.state_dict())
    source_rng = gate.rng_sha256()
    tangents: list[torch.Tensor] = []
    action = {name: torch.zeros_like(value)
              for name, value in gate.parameter_vector(source_net).items()}
    micros = gate._flatten_batches(batches)
    total_samples = sum(batch.images.shape[0] for batch in micros)
    with gate.preserved_rng():
        for batch in micros:
            tangent, micro_action = _directional_residual_autograd(
                source_net, loss_template, batch, direction, arm=arm)
            tangents.append(tangent)
            for name in action:
                action[name].add_(micro_action[name] / total_samples)
            del micro_action
    source_preserved = (
        source_hash == gate.state_sha256(source_net.state_dict())
        and source_rng == gate.rng_sha256())
    finite = (all(bool(torch.isfinite(value).all()) for value in tangents)
              and all(bool(torch.isfinite(value).all()) for value in action.values()))
    return tangents, action, {
        "method": "reverse_over_reverse_autograd_executed_graph",
        "microbatch_count": len(micros),
        "tangent_norm": _flat_l2(tangents),
        "action_norm": gate.vector_l2(action),
        "finite": finite,
        "source_preserved": source_preserved,
    }


def _parameter_resolution(
    base: torch.nn.Module,
    plus: torch.nn.Module,
    minus: torch.nn.Module,
    direction: TensorMap,
    epsilon: float,
) -> dict[str, float | int]:
    base_params = dict(base.named_parameters())
    plus_params = dict(plus.named_parameters())
    minus_params = dict(minus.named_parameters())
    total = plus_changed = minus_changed = branch_distinct = 0
    intended_sq = realized_sq = delta_sq = dot = 0.0
    for name in sorted(base_params):
        center = base_params[name].detach()
        positive = plus_params[name].detach()
        negative = minus_params[name].detach()
        intended = direction[name].to(center).double()
        realized = (positive.double() - negative.double()) / (2 * float(epsilon))
        total += center.numel()
        plus_changed += int((positive != center).sum())
        minus_changed += int((negative != center).sum())
        branch_distinct += int((positive != negative).sum())
        intended_sq += float(intended.square().sum())
        realized_sq += float(realized.square().sum())
        delta_sq += float((realized - intended).square().sum())
        dot += float((realized * intended).sum())
    intended_norm = math.sqrt(intended_sq)
    realized_norm = math.sqrt(realized_sq)
    return {
        "coordinate_count": total,
        "plus_changed_fraction": plus_changed / total,
        "minus_changed_fraction": minus_changed / total,
        "branch_distinct_fraction": branch_distinct / total,
        "realized_direction_relative_error": math.sqrt(delta_sq) / max(
            intended_norm, torch.finfo(torch.float64).tiny),
        "realized_direction_cosine": dot / max(
            intended_norm * realized_norm, torch.finfo(torch.float64).tiny),
        "realized_direction_norm_ratio": realized_norm / max(
            intended_norm, torch.finfo(torch.float64).tiny),
    }


def finite_difference_estimate(
    source_net: torch.nn.Module,
    loss_template: Any,
    batches: Sequence[gate.AuditBatch | gate.AuditBatchGroup],
    direction: TensorMap,
    *,
    arm: str,
    epsilon: float,
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    source_hash = gate.state_sha256(source_net.state_dict())
    source_rng = gate.rng_sha256()
    tangents: list[torch.Tensor] = []
    action = {name: torch.zeros_like(value)
              for name, value in gate.parameter_vector(source_net).items()}
    micros = gate._flatten_batches(batches)
    total_samples = sum(batch.images.shape[0] for batch in micros)
    resolution_rows = []
    with gate.preserved_rng():
        for batch in micros:
            plus = copy.deepcopy(source_net).train().requires_grad_(False)
            minus = copy.deepcopy(source_net).train().requires_grad_(False)
            gate._perturb_parameters(plus, direction, float(epsilon))
            gate._perturb_parameters(minus, direction, -float(epsilon))
            resolution_rows.append(_parameter_resolution(
                source_net, plus, minus, direction, float(epsilon)))
            with torch.no_grad():
                online_plus, target_plus, _, _ = factorial._pair_tensors(
                    plus, loss_template, batch, arm, force_fp32=True,
                    target_grad=False)
                online_minus, target_minus, _, _ = factorial._pair_tensors(
                    minus, loss_template, batch, arm, force_fp32=True,
                    target_grad=False)
                tangent = ((online_plus - target_plus)
                           - (online_minus - target_minus)) / (2 * float(epsilon))
            tangents.append(tangent.detach().double().cpu())
            del plus, minus, online_plus, target_plus, online_minus, target_minus

            live = copy.deepcopy(source_net).train().requires_grad_(True)
            online, _, _, _ = factorial._pair_tensors(
                live, loss_template, batch, arm, force_fp32=True,
                target_grad=False)
            scalar = (online * tangent.detach()).sum() / total_samples
            grads = torch.autograd.grad(
                scalar, tuple(live.parameters()), allow_unused=True)
            for (name, parameter), grad in zip(live.named_parameters(), grads):
                if grad is not None:
                    action[name].add_(grad.detach().double().cpu())
            del live, online, tangent, grads
    source_preserved = (
        source_hash == gate.state_sha256(source_net.state_dict())
        and source_rng == gate.rng_sha256())
    finite = (all(bool(torch.isfinite(value).all()) for value in tangents)
              and all(bool(torch.isfinite(value).all()) for value in action.values()))
    # A frozen batch group repeats the same parameter perturbation across its
    # microbatches, so these rows must agree exactly.
    resolution = resolution_rows[0]
    if any(row != resolution for row in resolution_rows[1:]):
        raise RuntimeError("parameter-resolution diagnostics vary across microbatches")
    return tangents, action, {
        "epsilon": float(epsilon),
        "finite": finite,
        "source_preserved": source_preserved,
        "parameter_resolution": resolution,
    }


def classify(rows: Sequence[Mapping[str, Any]], *, tolerance: float,
             minimum_consecutive: int, oracle_ok: bool) -> dict[str, Any]:
    if not oracle_ok:
        return {"verdict": "ORACLE_UNAVAILABLE", "plateaus": []}
    admissible = [
        bool(row.get("finite") and row.get("source_preserved")
             and row["tangent_vs_oracle"]["relative_error"] <= tolerance
             and row["action_vs_oracle"]["relative_error"] <= tolerance)
        for row in rows
    ]
    plateaus = []
    start = None
    for index, passed in enumerate(admissible + [False]):
        if passed and start is None:
            start = index
        elif not passed and start is not None:
            if index - start >= minimum_consecutive:
                plateaus.append({
                    "start_index": start,
                    "end_index": index - 1,
                    "epsilons": [float(rows[item]["epsilon"])
                                 for item in range(start, index)],
                })
            start = None
    return {
        "verdict": "PASS_CALIBRATED" if plateaus else "NO_PLATEAU",
        "plateaus": plateaus,
        "admissible_by_scale": admissible,
    }

