"""Paired finite-difference regimes for the Jacobian failure factorial.

All five regimes use one state-relative *parameter* direction.  Regime D maps
that direction into the full algorithmic state with zeros in optimizer, EMA,
buffer, and scaler input coordinates, then evaluates the complete transition.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import torch

from analysis.operator_clock_gate import core as gate


REGIMES = (
    "A_squared_gn_fp32",
    "B_real_loss_gn_fp32",
    "C_full_field_fp32",
    "D_production_algorithmic",
    "E_full_field_pseudohuber_fp32",
)


def optimizer_step_executed(
    before: Mapping[str, Any], after: Mapping[str, Any],
) -> bool:
    """Return whether every tracked optimizer parameter advanced one step."""
    before_steps = before["optimizer_steps"]
    after_steps = after["optimizer_steps"]
    return (len(before_steps) == len(after_steps)
            and bool(before_steps)
            and all(new == old + 1
                    for old, new in zip(before_steps, after_steps)))


def amp_regime_signature(
    detail: Mapping[str, Any], *, step_executed: bool,
) -> tuple[Any, ...]:
    """Discrete/numerical regime that must pair across central-FD branches."""
    return (
        bool(detail["amp_enabled"]),
        float(detail["grad_scale_before"]),
        float(detail["grad_scale_after"]),
        bool(detail["step_skipped"]),
        bool(step_executed),
    )


def vector_cosine(left: Mapping[str, torch.Tensor],
                  right: Mapping[str, torch.Tensor]) -> float:
    if set(left) != set(right) or not left:
        raise ValueError("vector maps must have identical nonempty keys")
    dot = sum(float(left[key].detach().double().mul(
        right[key].detach().double()).sum()) for key in left)
    denominator = gate.vector_l2(left) * gate.vector_l2(right)
    if denominator <= torch.finfo(torch.float64).tiny:
        return float("nan")
    return max(-1.0, min(1.0, dot / denominator))


def convergence_with_geometry(
    estimates: Mapping[float, Mapping[str, torch.Tensor]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Apply the PR #87 gate and add cosine/norm diagnostics."""
    report = gate.fd_convergence(estimates, tolerance=tolerance)
    epsilons = report["epsilons"]
    by_epsilon: list[dict[str, Any]] = []
    for index, epsilon in enumerate(epsilons):
        current = estimates[epsilon]
        row = {
            "epsilon": epsilon,
            "jvp_norm": gate.vector_l2(current),
            "finite": all(bool(torch.isfinite(value).all())
                          for value in current.values()),
            "relative_error": None,
            "cosine": None,
            "norm_ratio": None,
            "coarse_epsilon": None,
        }
        if index:
            coarse_epsilon = epsilons[index - 1]
            coarse = estimates[coarse_epsilon]
            row.update({
                "relative_error": gate.relative_difference(current, coarse),
                "cosine": vector_cosine(current, coarse),
                "norm_ratio": gate.vector_l2(current) / max(
                    gate.vector_l2(coarse), torch.finfo(torch.float64).tiny),
                "coarse_epsilon": coarse_epsilon,
            })
        by_epsilon.append(row)
    report["epsilon_metrics"] = by_epsilon
    return report


def parameter_direction_in_full_state(
    continuous_state: Mapping[str, torch.Tensor],
    parameter_direction: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Embed one parameter direction while holding all other inputs fixed."""
    result = {key: torch.zeros_like(value) for key, value in continuous_state.items()}
    expected = {key.removeprefix("theta.") for key in continuous_state
                if key.startswith("theta.")}
    if set(parameter_direction) != expected:
        raise ValueError("parameter direction does not match theta coordinates")
    for name, value in parameter_direction.items():
        result[f"theta.{name}"] = value.detach().double().cpu().clone()
    return result


def _forward(net: torch.nn.Module, x: torch.Tensor, sigma: torch.Tensor,
             labels: torch.Tensor, *, force_fp32: bool) -> torch.Tensor:
    # Production ECT exposes both keywords.  The fallbacks are only for the
    # deliberately tiny CPU test networks; unrelated TypeErrors still surface.
    attempts = ([{"augment_labels": None, "force_fp32": True},
                 {"force_fp32": True}, {"augment_labels": None}, {}]
                if force_fp32 else [{"augment_labels": None}, {}])
    last_error: TypeError | None = None
    for kwargs in attempts:
        try:
            return net(x, sigma, labels, **kwargs)
        except TypeError as exc:
            message = str(exc)
            if not any(key in message for key in ("force_fp32", "augment_labels")):
                raise
            last_error = exc
    assert last_error is not None
    raise last_error


def _pair_tensors(
    net: torch.nn.Module,
    loss_template: Any,
    batch: gate.AuditBatch,
    arm: str,
    *,
    force_fp32: bool,
    target_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    spec = gate.ARM_SPECS[arm]
    r_target = gate._schedule(loss_template, spec["target_scale"]).compute_r(
        batch.t, stage=int(loss_template.stage))
    r_denominator = gate._schedule(
        loss_template, spec["denominator_scale"]).compute_r(
            batch.t, stage=int(loss_template.stage))
    denominator = batch.t - r_denominator
    if not bool(torch.isfinite(denominator).all()) or not bool((denominator > 0).all()):
        raise RuntimeError("ECT denominator must be positive and finite")
    device = batch.images.device
    gate.set_device_rng_state(batch.dropout_rng_state, device)
    online = _forward(
        net, batch.images + batch.noise * batch.t, batch.t, batch.labels,
        force_fp32=force_fp32)
    target_input = batch.images + batch.noise * r_target
    if target_grad:
        gate.set_device_rng_state(batch.dropout_rng_state, device)
        target = _forward(
            net, target_input, r_target, batch.labels, force_fp32=force_fp32)
    else:
        with torch.no_grad():
            gate.set_device_rng_state(batch.dropout_rng_state, device)
            target = _forward(
                net, target_input, r_target, batch.labels, force_fp32=force_fp32)
    target = torch.nan_to_num(target)
    positive = r_target > 0
    target = positive * target + (~positive) * batch.images
    return online, target, r_target, denominator


def _output_residual_at_perturbation(
    source_net: torch.nn.Module,
    loss_template: Any,
    batch: gate.AuditBatch,
    arm: str,
    direction: Mapping[str, torch.Tensor],
    scale: float,
) -> torch.Tensor:
    branch = copy.deepcopy(source_net).train().requires_grad_(False)
    gate._perturb_parameters(branch, direction, scale)
    with torch.no_grad():
        online, target, _, _ = _pair_tensors(
            branch, loss_template, batch, arm, force_fp32=True,
            target_grad=False)
        residual = (online - target).detach()
    del branch
    return residual


def _residual_hessian_action(residual: torch.Tensor,
                             tangent: torch.Tensor, *, c: float) -> torch.Tensor:
    flat_r = residual.reshape(residual.shape[0], -1)
    flat_v = tangent.reshape(tangent.shape[0], -1)
    squared = flat_r.square().sum(dim=1, keepdim=True)
    if c == 0.0 and bool((squared == 0).any()):
        raise RuntimeError("c=0 residual norm has an undefined Hessian at zero")
    scale = torch.sqrt(squared + float(c) ** 2)
    dot = (flat_r * flat_v).sum(dim=1, keepdim=True)
    action = flat_v / scale - flat_r * dot / scale.pow(3)
    return action.reshape_as(residual)


def gauss_newton_action(
    source_net: torch.nn.Module,
    loss_template: Any,
    batches: Sequence[gate.AuditBatch | gate.AuditBatchGroup],
    direction: Mapping[str, torch.Tensor],
    *,
    arm: str,
    output_fd_epsilon: float,
    residual_geometry: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Estimate squared or real-loss GN directional action in forced FP32."""
    source_hash = gate.state_sha256(source_net.state_dict())
    source_rng = gate.rng_sha256()
    gate._validate_vector_pair(gate.parameter_vector(source_net), direction)
    accumulated = {name: torch.zeros_like(value)
                   for name, value in gate.parameter_vector(source_net).items()}
    residual_min = math.inf
    with gate.preserved_rng():
        live = copy.deepcopy(source_net).train().requires_grad_(True)
        micros = gate._flatten_batches(batches)
        total_samples = sum(batch.images.shape[0] for batch in micros)
        for batch in micros:
            plus = _output_residual_at_perturbation(
                source_net, loss_template, batch, arm, direction,
                float(output_fd_epsilon))
            minus = _output_residual_at_perturbation(
                source_net, loss_template, batch, arm, direction,
                -float(output_fd_epsilon))
            jdiff_u = (plus - minus) / (2 * float(output_fd_epsilon))
            online, target, _, _ = _pair_tensors(
                live, loss_template, batch, arm, force_fp32=True,
                target_grad=False)
            residual = online - target
            norms = residual.detach().double().reshape(
                residual.shape[0], -1).norm(dim=1)
            residual_min = min(residual_min, float(norms.min().cpu()))
            transformed = (_residual_hessian_action(
                residual.detach(), jdiff_u.detach(), c=0.0)
                if residual_geometry else jdiff_u.detach())
            # Regimes A/B are the frozen unweighted diagnostic fields
            # J_i^T (J_i-J_j)u and J_i^T H_rho(r)(J_i-J_j)u.  Explicit ECT
            # denominator weighting re-enters only in the full fields C-E.
            scalar = (online * transformed).sum() / total_samples
            grads = torch.autograd.grad(
                scalar, tuple(live.parameters()), allow_unused=True)
            for (name, parameter), grad in zip(live.named_parameters(), grads):
                if grad is not None:
                    accumulated[name].add_(grad.detach().double().cpu())
            del plus, minus, jdiff_u, online, target, residual, transformed
        del live
    after_hash = gate.state_sha256(source_net.state_dict())
    after_rng = gate.rng_sha256()
    finite = all(bool(torch.isfinite(value).all()) for value in accumulated.values())
    return accumulated, {
        "predictor": ("real_loss_gauss_newton_action" if residual_geometry
                      else "squared_loss_gauss_newton_action"),
        "output_fd_epsilon": float(output_fd_epsilon),
        "force_fp32": True,
        "residual_min_l2": residual_min,
        "finite": finite,
        "source_preserved": source_hash == after_hash and source_rng == after_rng,
    }


def gauss_newton_convergence(
    source_net: torch.nn.Module,
    loss_template: Any,
    batches: Sequence[gate.AuditBatch | gate.AuditBatchGroup],
    direction: Mapping[str, torch.Tensor],
    *,
    arm: str,
    epsilons: Sequence[float],
    tolerance: float,
    residual_geometry: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    estimates: dict[float, dict[str, torch.Tensor]] = {}
    branches = []
    for epsilon in epsilons:
        estimate, detail = gauss_newton_action(
            source_net, loss_template, batches, direction, arm=arm,
            output_fd_epsilon=float(epsilon),
            residual_geometry=residual_geometry)
        estimates[float(epsilon)] = estimate
        branches.append(detail)
    convergence = convergence_with_geometry(estimates, tolerance=tolerance)
    selected_epsilon = min(float(value) for value in epsilons)
    selected = estimates[selected_epsilon]
    source_preserved = all(item["source_preserved"] for item in branches)
    finite = all(item["finite"] for item in branches)
    receipt = {
        "predictor": branches[0]["predictor"],
        "arm": arm,
        "selected_epsilon": selected_epsilon,
        "convergence": convergence,
        "branches": branches,
        "source_preserved": source_preserved,
        "finite": finite,
    }
    receipt["status"] = (
        "PASS" if source_preserved and finite and convergence["passed"]
        else "FAIL_CLOSED")
    return selected, receipt


def _gradient_field_fp32(
    source_net: torch.nn.Module,
    loss_template: Any,
    batches: Sequence[gate.AuditBatch | gate.AuditBatchGroup],
    *,
    arm: str,
    c_override: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    source_net.zero_grad(set_to_none=True)
    micros = gate._flatten_batches(batches)
    total_samples = sum(batch.images.shape[0] for batch in micros)
    target_hashes = []
    target_requires_grad = []
    residual_min = math.inf
    for batch in micros:
        online, target, _, denominator = _pair_tensors(
            source_net, loss_template, batch, arm, force_fp32=True,
            target_grad=False)
        residual = online - target
        raw_sq = residual.square().reshape(batch.images.shape[0], -1).sum(dim=1)
        residual_min = min(residual_min, float(raw_sq.detach().sqrt().min().cpu()))
        c = float(c_override)
        numerator = torch.sqrt(raw_sq + c * c) - c if c > 0 else torch.sqrt(raw_sq)
        loss = numerator / denominator.flatten()
        (loss.sum() / total_samples).backward()
        target_hashes.append(gate.state_sha256(target))
        target_requires_grad.append(bool(target.requires_grad))
    field = {
        name: (torch.zeros_like(parameter) if parameter.grad is None
               else parameter.grad).detach().double().cpu().clone()
        for name, parameter in source_net.named_parameters()
    }
    finite = bool(field) and all(bool(torch.isfinite(value).all())
                                 for value in field.values())
    return field, {
        "target_recompute_count": len(micros),
        "all_targets_detached": not any(target_requires_grad),
        "target_hashes": target_hashes,
        "force_fp32": True,
        "c_override": float(c_override),
        "residual_min_l2": residual_min,
        "finite": finite,
    }


def full_field_fp32_jvp(
    source_net: torch.nn.Module,
    loss_template: Any,
    batches: Sequence[gate.AuditBatch | gate.AuditBatchGroup],
    direction: Mapping[str, torch.Tensor],
    *,
    arm: str,
    epsilons: Sequence[float],
    tolerance: float,
    c_override: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    source_hash = gate.state_sha256(source_net.state_dict())
    source_rng = gate.rng_sha256()
    estimates: dict[float, dict[str, torch.Tensor]] = {}
    branches = []
    with gate.preserved_rng():
        for epsilon in epsilons:
            epsilon = float(epsilon)
            plus = copy.deepcopy(source_net).train().requires_grad_(True)
            gate._perturb_parameters(plus, direction, epsilon)
            plus_field, plus_detail = _gradient_field_fp32(
                plus, copy.deepcopy(loss_template), batches, arm=arm,
                c_override=c_override)
            del plus
            minus = copy.deepcopy(source_net).train().requires_grad_(True)
            gate._perturb_parameters(minus, direction, -epsilon)
            minus_field, minus_detail = _gradient_field_fp32(
                minus, copy.deepcopy(loss_template), batches, arm=arm,
                c_override=c_override)
            del minus
            estimates[epsilon] = gate._difference(
                plus_field, minus_field, 2 * epsilon)
            branches.append({
                "epsilon": epsilon,
                "plus": plus_detail,
                "minus": minus_detail,
                "paired_target_recomputation": (
                    plus_detail["target_recompute_count"]
                    == minus_detail["target_recompute_count"]),
            })
    convergence = convergence_with_geometry(estimates, tolerance=tolerance)
    selected_epsilon = min(float(value) for value in epsilons)
    selected = estimates[selected_epsilon]
    after_hash = gate.state_sha256(source_net.state_dict())
    after_rng = gate.rng_sha256()
    finite = all(item[side]["finite"] for item in branches
                 for side in ("plus", "minus"))
    receipt = {
        "predictor": "full_recompute_detach_field_fp32",
        "arm": arm,
        "selected_epsilon": selected_epsilon,
        "c_override": float(c_override),
        "force_fp32": True,
        "convergence": convergence,
        "branches": branches,
        "finite": finite,
        "source_preserved": source_hash == after_hash and source_rng == after_rng,
        "definition_guard": (
            "Every branch reruns online and target forwards, detaches the fresh "
            "target inside the branch, and forces the production network FP32 path."),
    }
    receipt["status"] = (
        "PASS" if receipt["source_preserved"] and finite
        and convergence["passed"] else "FAIL_CLOSED")
    return selected, receipt


def parameter_partial_algorithmic_jvp(
    source: gate.AlgorithmicState,
    batch: gate.AuditBatch | gate.AuditBatchGroup,
    parameter_direction: Mapping[str, torch.Tensor],
    *,
    arm: str,
    epsilons: Sequence[float],
    tolerance: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    source_hash = source.sha256()
    source_rng = gate.rng_sha256()
    full_direction = parameter_direction_in_full_state(
        source.continuous_vector(), parameter_direction)
    estimates: dict[float, dict[str, torch.Tensor]] = {}
    branches = []
    amp_pairing = True
    discrete_pairing = True
    before_discrete = gate._optimizer_discrete_signature(source)

    with gate.preserved_rng():
        for epsilon in epsilons:
            epsilon = float(epsilon)
            plus = source.clone()
            plus.add_direction_(full_direction, epsilon)
            with gate.preserved_rng(batch.audit_id):
                plus_after, plus_detail = gate.transition_step(
                    plus, batch, arm=arm, clone_input=False)
            plus_vector = plus_after.continuous_vector()
            plus_discrete = gate._optimizer_discrete_signature(plus_after)
            plus_executed = optimizer_step_executed(
                before_discrete, plus_discrete)
            del plus, plus_after
            minus = source.clone()
            minus.add_direction_(full_direction, -epsilon)
            with gate.preserved_rng(batch.audit_id):
                minus_after, minus_detail = gate.transition_step(
                    minus, batch, arm=arm, clone_input=False)
            minus_vector = minus_after.continuous_vector()
            minus_discrete = gate._optimizer_discrete_signature(minus_after)
            minus_executed = optimizer_step_executed(
                before_discrete, minus_discrete)
            del minus, minus_after
            estimates[epsilon] = gate._difference(
                plus_vector, minus_vector, 2 * epsilon)
            plus_amp = amp_regime_signature(
                plus_detail, step_executed=plus_executed)
            minus_amp = amp_regime_signature(
                minus_detail, step_executed=minus_executed)
            amp_equal = plus_amp == minus_amp
            discrete_equal = plus_discrete == minus_discrete
            amp_pairing &= amp_equal
            discrete_pairing &= discrete_equal
            branches.append({
                "epsilon": epsilon,
                "plus": plus_detail,
                "minus": minus_detail,
                "step_executed_plus": plus_executed,
                "step_executed_minus": minus_executed,
                "overflow_detected_plus": bool(plus_detail["step_skipped"]),
                "overflow_detected_minus": bool(minus_detail["step_skipped"]),
                "amp_regime_plus": plus_amp,
                "amp_regime_minus": minus_amp,
                "same_amp_regime": amp_equal,
                "discrete_state_plus": plus_discrete,
                "discrete_state_minus": minus_discrete,
                "discrete_state_identical": discrete_equal,
            })
    convergence = convergence_with_geometry(estimates, tolerance=tolerance)
    sweep_regimes = [(tuple(item["amp_regime_plus"]),
                      tuple(item["amp_regime_minus"])) for item in branches]
    amp_regime_identical = len(set(sweep_regimes)) == 1
    selected_epsilon = min(float(value) for value in epsilons)
    selected = estimates[selected_epsilon]
    after_hash = source.sha256()
    after_rng = gate.rng_sha256()
    finite = all(item["finite"] for item in convergence["epsilon_metrics"])
    receipt = {
        "predictor": "parameter_partial_production_state_transition",
        "arm": arm,
        "input_direction_scope": "theta_only",
        "selected_epsilon": selected_epsilon,
        "convergence": convergence,
        "branches": branches,
        "amp_skip_behavior_identical_all_eps": amp_pairing,
        "amp_regime_identical_across_eps": amp_regime_identical,
        "discrete_state_behavior_identical_all_eps": discrete_pairing,
        "finite": finite,
        "source_preserved": source_hash == after_hash and source_rng == after_rng,
        "no_in_place_source_pollution": source_hash == after_hash,
    }
    receipt["status"] = (
        "PASS" if receipt["source_preserved"] and finite
        and convergence["passed"] and amp_pairing and amp_regime_identical
        and discrete_pairing else "FAIL_CLOSED")
    return selected, receipt


def run_regime(
    regime: str,
    state: gate.AlgorithmicState,
    batch: gate.AuditBatch | gate.AuditBatchGroup,
    parameter_direction: Mapping[str, torch.Tensor],
    *,
    arm: str,
    epsilons: Sequence[float],
    tolerance: float,
    pseudo_huber_c: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if regime == "A_squared_gn_fp32":
        return gauss_newton_convergence(
            state.net, state.loss_fn, [batch], parameter_direction, arm=arm,
            epsilons=epsilons, tolerance=tolerance,
            residual_geometry=False)
    if regime == "B_real_loss_gn_fp32":
        return gauss_newton_convergence(
            state.net, state.loss_fn, [batch], parameter_direction, arm=arm,
            epsilons=epsilons, tolerance=tolerance,
            residual_geometry=True)
    if regime == "C_full_field_fp32":
        return full_field_fp32_jvp(
            state.net, state.loss_fn, [batch], parameter_direction, arm=arm,
            epsilons=epsilons, tolerance=tolerance, c_override=0.0)
    if regime == "D_production_algorithmic":
        return parameter_partial_algorithmic_jvp(
            state, batch, parameter_direction, arm=arm,
            epsilons=epsilons, tolerance=tolerance)
    if regime == "E_full_field_pseudohuber_fp32":
        return full_field_fp32_jvp(
            state.net, state.loss_fn, [batch], parameter_direction, arm=arm,
            epsilons=epsilons, tolerance=tolerance,
            c_override=float(pseudo_huber_c))
    raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")
