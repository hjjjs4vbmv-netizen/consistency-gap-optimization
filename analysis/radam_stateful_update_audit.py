"""Stateful RAdam update audit from a real non-zero optimizer state.

Fresh-state ``c_0^*`` (``analysis/radam_update_gauge.py``) is only an
implementation sanity probe.  The core check starts from a restored training
state

    z_K = (θ_K, m_K, v_K, n_K, GradScaler_K)

forks two disposable branches that share one minibatch / ``t`` / noise /
dropout RNG, and differs only by ``global_gap_scale`` ``g ∈ {1.0, 1.3}``.

Each branch reports gradient scalars ``a_K^*``, ``R_grad(K)`` and optimizer
scalars ``s_K^*``, ``c_K^*``, ``R_opt(K)``, plus layer summaries ``h_{K,i}``,
``H_K``, and off-support energy.  The critical comparison is
``R_opt(K) - R_grad(K)`` (not the identity ``H_K = R_opt``), together with
whether idealized moment-predicted ``h_{K,i}`` matches actual-update
``h_{K,i}``.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "analysis") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "analysis"))

import torch

import radam_update_gauge as gauge
from training.schedules import get_schedule

LAYERWISE_FIELDS = (
    "layer",
    "h_K_i_actual",
    "h_K_i_predicted",
    "h_K_i_abs_diff",
    "update_1_l2",
    "update_1p3_l2",
    "update_cosine",
    "c_K_star_layer",
    "predicted_1_l2",
    "predicted_1p3_l2",
    "s_K_star_layer",
    "off_support_energy_actual",
    "off_support_energy_predicted",
)


def _norm_sq(values: Iterable[torch.Tensor]) -> float:
    return sum(float(value.square().sum()) for value in values)


def _dot(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    return sum(float((left[name] * right[name]).sum()) for name in left)


def _scale_and_residual(reference: dict[str, torch.Tensor],
                        probe: dict[str, torch.Tensor],
                        *,
                        fit_probe_to_reference: bool) -> tuple[float, float, float]:
    """Least-squares scalar fit and relative residual between two named tensors.

    If ``fit_probe_to_reference`` is True (optimizer / ``c`` convention):
        c * probe ≈ reference,  R = ||c*probe - reference|| / ||reference||
    Else (gradient / ``a`` convention):
        a * reference ≈ probe,  R = ||probe - a*reference|| / ||probe||
    """
    n_ref_sq = _norm_sq(reference.values())
    n_probe_sq = _norm_sq(probe.values())
    if n_ref_sq <= 0 or n_probe_sq <= 0:
        raise RuntimeError("zero vector; scalar fit is undefined")
    dot = _dot(probe, reference)
    if fit_probe_to_reference:
        if abs(dot) == 0:
            raise RuntimeError("orthogonal vectors; scalar fit is undefined")
        scale = dot / n_probe_sq
        residual_sq = max(_norm_sq(scale * probe[name] - reference[name] for name in reference), 0.0)
        residual = math.sqrt(residual_sq) / math.sqrt(n_ref_sq)
    else:
        scale = dot / n_ref_sq
        residual_sq = max(_norm_sq(probe[name] - scale * reference[name] for name in reference), 0.0)
        residual = math.sqrt(residual_sq) / math.sqrt(n_probe_sq)
    cosine = dot / math.sqrt(n_ref_sq * n_probe_sq)
    return scale, residual, cosine


def off_support_energy(reference: dict[str, torch.Tensor],
                       probe: dict[str, torch.Tensor],
                       scale: float) -> float:
    """Relative energy of ``reference`` orthogonal to ``span(probe)``.

    With the LS coefficient ``scale`` for ``scale * probe ≈ reference``, this is
    ``||reference - scale*probe||² / ||reference||²`` (= ``R_opt²`` when scale
    is ``c_K^*``).
    """
    n_ref_sq = _norm_sq(reference.values())
    if n_ref_sq <= 0:
        return math.nan
    residual_sq = max(_norm_sq(reference[name] - scale * probe[name] for name in reference), 0.0)
    return residual_sq / n_ref_sq


def layerwise_h(reference: dict[str, torch.Tensor],
                probe: dict[str, torch.Tensor],
                model_scale: float) -> tuple[float, list[dict[str, float]]]:
    """Per-layer residuals with the whole-model scale, plus energy-weighted H.

    ``H`` is the energy-weighted RMS of layer residuals and equals the
    whole-model residual by construction (the identity ``H_K = R_opt``),
    because every layer's residual energy is included even when a layer's
    relative ``h_{K,i}`` is undefined (zero reference energy).
    """
    by_layer: dict[str, list[str]] = defaultdict(list)
    for name in reference:
        by_layer[gauge.layer_name(name)].append(name)
    layers = []
    weighted_res_sq = 0.0
    total_ref_sq = 0.0
    for layer, names in sorted(by_layer.items()):
        ref_sq = _norm_sq(reference[name] for name in names)
        probe_sq = _norm_sq(probe[name] for name in names)
        res_sq = max(_norm_sq(model_scale * probe[name] - reference[name] for name in names), 0.0)
        weighted_res_sq += res_sq
        total_ref_sq += ref_sq
        if ref_sq == 0:
            # Relative residual undefined; energy still feeds H_K.
            h_i = 0.0 if res_sq == 0 else math.nan
            off = 0.0 if res_sq == 0 else math.nan
        else:
            h_i = math.sqrt(res_sq) / math.sqrt(ref_sq)
            off = res_sq / ref_sq
        layers.append({
            "layer": layer,
            "h_K_i": h_i,
            "off_support_energy": off,
            "ref_l2": math.sqrt(ref_sq),
            "probe_l2": math.sqrt(probe_sq),
        })
    H = math.sqrt(weighted_res_sq / total_ref_sq) if total_ref_sq > 0 else math.nan
    return H, layers


def _grad_by_name(net: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a full-parameter gradient map (zeros where ``grad is None``)."""
    grads = {}
    saw_nonzero = False
    for name, parameter in net.named_parameters():
        if parameter.grad is None:
            grads[name] = torch.zeros(parameter.shape, dtype=torch.float64)
            continue
        value = parameter.grad.detach().double().cpu().clone()
        grads[name] = value
        saw_nonzero |= float(value.abs().sum()) > 0
    if not saw_nonzero:
        raise RuntimeError("no gradients after backward")
    return grads


def idealized_radam_update(net: torch.nn.Module, optimizer: torch.optim.RAdam) -> dict[str, torch.Tensor]:
    """Predict ``Δθ`` from current grads + optimizer moments without stepping.

    Mirrors ``torch.optim.radam._single_tensor_radam`` for the repository's
    default settings (``weight_decay=0``, no maximize / complex / capturable).
    Parameters with ``grad is None`` contribute a zero delta so the predicted
    map always covers the full parameter set (matching actual ``Δθ`` keys).
    """
    predicted: dict[str, torch.Tensor] = {
        name: torch.zeros(parameter.shape, dtype=torch.float64)
        for name, parameter in net.named_parameters()
    }
    name_by_param = {parameter: name for name, parameter in net.named_parameters()}
    saw_grad = False
    for group in optimizer.param_groups:
        beta1, beta2 = group["betas"]
        lr = group["lr"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        if group.get("maximize", False):
            raise RuntimeError("idealized RAdam predictor does not support maximize=True")
        if group.get("differentiable", False):
            raise RuntimeError("idealized RAdam predictor does not support differentiable=True")
        if weight_decay != 0:
            raise RuntimeError("idealized RAdam predictor only supports weight_decay=0")
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            saw_grad = True
            name = name_by_param[parameter]
            state = optimizer.state[parameter]
            # Stay in parameter dtype so the analytical map matches torch.optim.
            grad = parameter.grad.detach()
            exp_avg = state["exp_avg"].detach().clone()
            exp_avg_sq = state["exp_avg_sq"].detach().clone()
            step = int(state["step"].item()) + 1
            exp_avg = exp_avg.lerp(grad, 1 - beta1)
            exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(grad, grad, value=1 - beta2)
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            bias_corrected_exp_avg = exp_avg / bias_correction1
            rho_inf = 2 / (1 - beta2) - 1
            rho_t = rho_inf - 2 * step * (beta2 ** step) / bias_correction2
            if rho_t > 5.0:
                rect = math.sqrt(
                    (rho_t - 4) * (rho_t - 2) * rho_inf
                    / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                )
                adaptive_lr = (bias_correction2 ** 0.5) / (exp_avg_sq.sqrt().add(eps))
                delta = -(bias_corrected_exp_avg * lr * adaptive_lr * rect)
            else:
                delta = -(bias_corrected_exp_avg * lr)
            predicted[name] = delta.detach().double().cpu()
    if not saw_grad:
        raise RuntimeError("idealized RAdam predictor produced an empty update")
    return predicted


def _optimizer_step_count(optimizer: torch.optim.Optimizer) -> int:
    if not optimizer.state:
        return 0
    steps = [int(state["step"].item()) for state in optimizer.state.values() if "step" in state]
    return max(steps) if steps else 0


def _scalar_step(value: Any) -> float:
    """Return a finite scalar optimizer step, rejecting malformed state."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("step is not scalar")
        value = value.item()
    step = float(value)
    if not math.isfinite(step):
        raise ValueError("step is not finite")
    return step


def stateful_radam_state_summary(net: torch.nn.Module,
                                 optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Validate that every model parameter belongs to one coherent live state.

    RAdam normally advances every participating parameter in lockstep.  The
    audit therefore rejects partially initialized states or divergent counters
    instead of silently labelling their maximum as the single requested
    ``n_K``.  A zero entry in an individual moment tensor is valid; the two
    moment *collections* must each contain non-zero finite energy.
    """
    errors: list[str] = []
    steps: list[float] = []
    m_energy = 0.0
    v_energy = 0.0
    initialized = 0
    parameter_count = 0
    for name, parameter in net.named_parameters():
        parameter_count += 1
        state = optimizer.state.get(parameter)
        if not state:
            errors.append(f"{name}: missing optimizer state")
            continue
        initialized += 1
        missing = [key for key in ("step", "exp_avg", "exp_avg_sq") if key not in state]
        if missing:
            errors.append(f"{name}: missing {', '.join(missing)}")
            continue
        try:
            steps.append(_scalar_step(state["step"]))
        except (TypeError, ValueError) as exc:
            errors.append(f"{name}: invalid step ({exc})")
        for key, energy_name in (("exp_avg", "m"), ("exp_avg_sq", "v")):
            value = state[key]
            if not isinstance(value, torch.Tensor):
                errors.append(f"{name}: {key} is not a tensor")
                continue
            if value.shape != parameter.shape:
                errors.append(f"{name}: {key} shape does not match parameter")
                continue
            if not bool(torch.isfinite(value).all()):
                errors.append(f"{name}: {key} is non-finite")
                continue
            energy = float(value.detach().double().square().sum().cpu())
            if energy_name == "m":
                m_energy += energy
            else:
                v_energy += energy
    if not parameter_count:
        errors.append("network has no parameters")
    if initialized == 0:
        errors.append("optimizer moments are still zero")
    if steps:
        step_min, step_max = min(steps), max(steps)
        if step_min <= 0:
            errors.append("n_K must be > 0")
        if not step_min.is_integer() or not step_max.is_integer():
            errors.append("n_K must be an integer")
        if step_min != step_max:
            errors.append("n_K is not uniform across parameter states")
    else:
        step_min = step_max = None
    if initialized and m_energy == 0:
        errors.append("optimizer exp_avg moments are still zero")
    if initialized and v_energy == 0:
        errors.append("optimizer exp_avg_sq moments are still zero")
    return {
        "valid": not errors,
        "errors": errors,
        "parameter_count": parameter_count,
        "initialized_parameter_count": initialized,
        "n_K": step_min if step_min == step_max else None,
        "n_K_min": step_min,
        "n_K_max": step_max,
        "m_l2": math.sqrt(m_energy),
        "v_l2": math.sqrt(v_energy),
    }


def virtual_stateful_step(common_net, common_optimizer, loss_template, microbatches, *,
                          gain: float, scaler_template, amp: bool
                          ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor],
                                     dict[str, torch.Tensor], dict[str, Any]]:
    """One disposable AMP-ordered step from a copied non-zero optimizer state."""
    net = copy.deepcopy(common_net).train().requires_grad_(True)
    optimizer = torch.optim.RAdam(
        net.parameters(),
        lr=common_optimizer.defaults["lr"],
        betas=common_optimizer.defaults["betas"],
        eps=common_optimizer.defaults["eps"],
        weight_decay=common_optimizer.defaults.get("weight_decay", 0.0),
    )
    optimizer.load_state_dict(copy.deepcopy(common_optimizer.state_dict()))
    scaler = copy.deepcopy(scaler_template)
    param_before = gauge.module_state_hashes(net)
    optimizer_before = gauge.state_sha256(optimizer.state_dict())
    scaler_before = gauge.state_sha256(scaler.state_dict())
    step_before = _optimizer_step_count(optimizer)
    schedule = get_schedule(
        "global_sigmoid",
        q=float(loss_template.q), k=float(loss_template.k), b=float(loss_template.b),
        global_gap_scale=gain,
    )
    optimizer.zero_grad(set_to_none=True)
    loss_sum, loss_count = 0.0, 0
    for images, labels, t, eps, dropout_rng_state in microbatches:
        loss = gauge.fixed_ect_loss(net, loss_template, schedule, images, labels, t, eps,
                                    dropout_rng_state)
        loss_mean = loss.mean()
        scaler.scale(loss_mean).backward() if amp else loss_mean.backward()
        loss_sum += float(loss.detach().double().sum().cpu())
        loss_count += loss.numel()
    if amp:
        scaler.unscale_(optimizer)
    nonfinite_before_sanitize = False
    for parameter in net.parameters():
        if parameter.grad is not None:
            nonfinite_before_sanitize |= not bool(torch.isfinite(parameter.grad).all())
            torch.nan_to_num(parameter.grad, nan=0, posinf=1e5, neginf=-1e5, out=parameter.grad)
    grads = _grad_by_name(net)
    predicted = idealized_radam_update(net, optimizer)
    # Snapshot the branch itself so Δθ keys always match named_parameters(),
    # independent of whether common_net and the deepcopy share storage quirks.
    before_params = {
        name: parameter.detach().double().cpu().clone()
        for name, parameter in net.named_parameters()
    }
    if amp:
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        step_skipped = scale_after < scale_before
    else:
        scale_before = scale_after = None
        optimizer.step()
        step_skipped = False
    actual = {
        name: parameter.detach().double().cpu() - before_params[name]
        for name, parameter in net.named_parameters()
    }
    detail = {
        "gain": gain,
        "loss_mean": loss_sum / loss_count,
        "accumulation_rounds": len(microbatches),
        "amp_enabled": amp,
        "amp_unscale_called": amp,
        "nonfinite_before_sanitize": nonfinite_before_sanitize,
        "step_skipped": step_skipped,
        "grad_scale_before": scale_before,
        "grad_scale_after": scale_after,
        "parameter_hash_before": param_before,
        "parameter_hash_after_virtual_step": gauge.module_state_hashes(net),
        "optimizer_state_hash_before": optimizer_before,
        "optimizer_state_hash_after_virtual_step": gauge.state_sha256(optimizer.state_dict()),
        "gradscaler_hash_before": scaler_before,
        "gradscaler_hash_after_virtual_step": gauge.state_sha256(scaler.state_dict()),
        "optimizer_step_before": step_before,
        "optimizer_step_after": _optimizer_step_count(optimizer),
        "moments_nontrivial_before": True,
    }
    return grads, predicted, actual, detail


def summarize_pair(grads_1, grads_13, pred_1, pred_13, act_1, act_13) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """Assemble a_K^*, R_grad, s_K^*, c_K^*, R_opt, H_K, h_{K,i}, off-support.

    Critical residuals ``R_grad`` and ``R_opt`` share the same ``c``-convention
    algebra ``scale * probe_{1.3} ≈ reference_1``, residual ``/ ||reference||``,
    so ``R_opt - R_grad`` compares identical quantities.  ``a_K^*`` remains the
    gap-hook coefficient ``g_{1.3} ≈ a g_1`` for LR-matching interpretation.
    """
    # a-convention coefficient (gap_gradient_hook); residual kept only as telemetry.
    a_star, r_grad_a, grad_cosine = _scale_and_residual(
        grads_1, grads_13, fit_probe_to_reference=False,
    )
    # Matched c-convention residual used for R_opt - R_grad.
    _, r_grad, _ = _scale_and_residual(grads_1, grads_13, fit_probe_to_reference=True)
    s_star, r_pred, pred_cosine = _scale_and_residual(pred_1, pred_13, fit_probe_to_reference=True)
    c_star, r_opt, opt_cosine = _scale_and_residual(act_1, act_13, fit_probe_to_reference=True)

    H_actual, layers_actual = layerwise_h(act_1, act_13, c_star)
    H_predicted, layers_predicted = layerwise_h(pred_1, pred_13, s_star)
    # Identity checks (should be numerical no-ops; still recorded).
    identity_H_equals_R_opt = abs(H_actual - r_opt) <= 1e-12 * max(1.0, abs(r_opt))
    # a- and c-convention relative residuals both equal |sin θ|; record the gap.
    a_c_residual_gap = abs(r_grad_a - r_grad)

    pred_by_layer = {row["layer"]: row for row in layers_predicted}
    layer_rows = []
    h_abs_diffs = []
    for row in layers_actual:
        layer = row["layer"]
        pred_row = pred_by_layer[layer]
        h_actual = row["h_K_i"]
        h_predicted = pred_row["h_K_i"]
        abs_diff = (abs(h_actual - h_predicted)
                    if math.isfinite(h_actual) and math.isfinite(h_predicted) else math.nan)
        if math.isfinite(abs_diff):
            h_abs_diffs.append(abs_diff)
        # Layer-local coefficients for localization (same conventions as model).
        names = [name for name in act_1 if gauge.layer_name(name) == layer]
        try:
            c_layer, _, _ = _scale_and_residual(
                {name: act_1[name] for name in names},
                {name: act_13[name] for name in names},
                fit_probe_to_reference=True,
            )
        except RuntimeError:
            c_layer = math.nan
        try:
            s_layer, _, _ = _scale_and_residual(
                {name: pred_1[name] for name in names},
                {name: pred_13[name] for name in names},
                fit_probe_to_reference=True,
            )
        except RuntimeError:
            s_layer = math.nan
        layer_dot = sum(float((act_13[name] * act_1[name]).sum()) for name in names)
        n1 = _norm_sq(act_1[name] for name in names)
        n13 = _norm_sq(act_13[name] for name in names)
        cosine = layer_dot / math.sqrt(n1 * n13) if n1 and n13 else math.nan
        layer_rows.append({
            "layer": layer,
            "h_K_i_actual": h_actual,
            "h_K_i_predicted": h_predicted,
            "h_K_i_abs_diff": abs_diff,
            "update_1_l2": math.sqrt(n1),
            "update_1p3_l2": math.sqrt(n13),
            "update_cosine": cosine,
            "c_K_star_layer": c_layer,
            "predicted_1_l2": pred_row["ref_l2"],
            "predicted_1p3_l2": pred_row["probe_l2"],
            "s_K_star_layer": s_layer,
            "off_support_energy_actual": row["off_support_energy"],
            "off_support_energy_predicted": pred_row["off_support_energy"],
        })

    # Predicted-vs-actual update agreement on each gain (should be ~0 if the
    # analytical RAdam predictor matches torch.optim.RAdam).
    pred_vs_actual = {}
    for gain, predicted, actual in ((1.0, pred_1, act_1), (1.3, pred_13, act_13)):
        denom = math.sqrt(_norm_sq(actual.values()))
        if denom == 0:
            pred_vs_actual[str(gain)] = None
        else:
            err = math.sqrt(max(_norm_sq(predicted[name] - actual[name] for name in actual), 0.0))
            pred_vs_actual[str(gain)] = err / denom

    whole = {
        "gauge_defined": True,
        "gauge_error": None,
        "residual_convention": "c_star_probe_to_reference",
        "a_K_star": a_star,
        "R_grad": r_grad,
        "R_grad_a_convention": r_grad_a,
        "R_grad_a_c_abs_gap": a_c_residual_gap,
        "grad_cosine": grad_cosine,
        "s_K_star": s_star,
        "R_pred": r_pred,
        "c_K_star": c_star,
        "R_opt": r_opt,
        "update_cosine": opt_cosine,
        "predicted_update_cosine": pred_cosine,
        "H_K": H_actual,
        "H_K_predicted": H_predicted,
        "H_K_equals_R_opt_identity": identity_H_equals_R_opt,
        "R_opt_minus_R_grad": r_opt - r_grad,
        "off_support_energy": off_support_energy(act_1, act_13, c_star),
        "off_support_energy_predicted": off_support_energy(pred_1, pred_13, s_star),
        "predicted_vs_actual_relative_l2": pred_vs_actual,
        "h_K_i_abs_diff_mean": (sum(h_abs_diffs) / len(h_abs_diffs)) if h_abs_diffs else math.nan,
        "h_K_i_abs_diff_max": max(h_abs_diffs) if h_abs_diffs else math.nan,
        "update_1_l2": math.sqrt(_norm_sq(act_1.values())),
        "update_1p3_l2": math.sqrt(_norm_sq(act_13.values())),
        "grad_1_l2": math.sqrt(_norm_sq(grads_1.values())),
        "grad_1p3_l2": math.sqrt(_norm_sq(grads_13.values())),
    }
    return whole, layer_rows


def run_stateful_pair(common_net, common_optimizer, loss_template, images, labels, *,
                      gains=(1.0, 1.3), amp=True, initial_scale=65536.0,
                      scaler_state: dict[str, Any] | None = None,
                      random_seed: int | None = None,
                      microbatch_size: int | None = None
                      ) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """Fork ``z_K``, run the paired g=1.0/1.3 step, leave the source untouched."""
    if tuple(gains) != (1.0, 1.3):
        raise ValueError("this audit is defined for exactly gains (1.0, 1.3)")
    state_summary = stateful_radam_state_summary(common_net, common_optimizer)
    if not state_summary["valid"]:
        raise RuntimeError(
            "invalid stateful RAdam state; refuse to report a stateful audit:\n  - "
            + "\n  - ".join(state_summary["errors"])
            + "\n(use analysis/radam_update_gauge.py for the fresh-state sanity probe)"
        )
    device = images.device
    microbatch_size = images.shape[0] if microbatch_size is None else microbatch_size
    if microbatch_size < 1 or images.shape[0] % microbatch_size:
        raise ValueError("microbatch_size must be a positive divisor of batch size")
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state(device=device).clone() if device.type == "cuda" else None
    try:
        if random_seed is not None:
            torch.manual_seed(random_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(random_seed)
        source_before = gauge.module_state_hashes(common_net)
        source_optimizer_before = gauge.state_sha256(common_optimizer.state_dict())
        if amp and scaler_state is None:
            raise RuntimeError(
                "AMP stateful audit requires GradScaler_K from training-state "
                "(gradscaler_state); refuse a silent fresh-scaler fallback"
            )
        scaler_template = gauge._new_scaler(device, amp, initial_scale)
        if scaler_state is not None:
            scaler_template.load_state_dict(copy.deepcopy(scaler_state))
        source_scaler_before = gauge.state_sha256(scaler_template.state_dict())
        n_K = int(state_summary["n_K"])
        group0 = common_optimizer.param_groups[0]
        microbatches = []
        for start in range(0, images.shape[0], microbatch_size):
            image_micro = images[start:start + microbatch_size]
            label_micro = labels[start:start + microbatch_size]
            t = (torch.randn(image_micro.shape[0], 1, 1, 1, device=device)
                 * loss_template.P_std + loss_template.P_mean).exp()
            eps = torch.randn_like(image_micro)
            microbatches.append((image_micro, label_micro, t, eps, gauge.get_rng_state(device).clone()))
        grads, predicted, actual, branches = {}, {}, {}, []
        for gain in gains:
            g_grad, g_pred, g_act, detail = virtual_stateful_step(
                common_net, common_optimizer, loss_template, microbatches,
                gain=gain, scaler_template=scaler_template, amp=amp,
            )
            grads[gain] = g_grad
            predicted[gain] = g_pred
            actual[gain] = g_act
            branches.append(detail)
        skipped = [branch["gain"] for branch in branches if branch["step_skipped"]]
        if skipped:
            whole, layers = {
                "gauge_defined": False,
                "gauge_error": (
                    "AMP skipped optimizer.step on branch(es) "
                    f"{skipped}; refuse R_opt/R_grad and predicted-vs-actual h "
                    "(actual Δθ is zero while moment-predicted updates are not)"
                ),
                "residual_convention": "c_star_probe_to_reference",
                "a_K_star": None, "R_grad": None, "s_K_star": None,
                "c_K_star": None, "R_opt": None, "H_K": None,
                "R_opt_minus_R_grad": None,
            }, []
        else:
            try:
                whole, layers = summarize_pair(
                    grads[1.0], grads[1.3], predicted[1.0], predicted[1.3],
                    actual[1.0], actual[1.3],
                )
            except RuntimeError as exc:
                whole, layers = {
                    "gauge_defined": False,
                    "gauge_error": str(exc),
                    "residual_convention": "c_star_probe_to_reference",
                    "a_K_star": None, "R_grad": None, "s_K_star": None,
                    "c_K_star": None, "R_opt": None, "H_K": None,
                    "R_opt_minus_R_grad": None,
                }, []
        source_after = gauge.module_state_hashes(common_net)
        source_optimizer_after = gauge.state_sha256(common_optimizer.state_dict())
        source_scaler_after = gauge.state_sha256(scaler_template.state_dict())
    finally:
        torch.set_rng_state(cpu_rng_before)
        if cuda_rng_before is not None:
            torch.cuda.set_rng_state(cuda_rng_before, device=device)
    audit = {
        "gains": list(gains),
        "stateful_radam": {
            "lr": group0["lr"],
            "betas": list(group0["betas"]),
            "eps": group0["eps"],
            "n_K": n_K,
            "moments_nontrivial": True,
            "state_validation": state_summary,
            "gradscaler_restored": scaler_state is not None,
        },
        "randomness_contract": {
            "same_minibatch": True, "same_t": True, "same_noise": True,
            "same_dropout_rng_state": True,
            "minibatch_images_sha256": gauge.tensor_sha256(images),
            "minibatch_labels_sha256": gauge.tensor_sha256(labels),
            "microbatch_size": microbatch_size,
            "accumulation_rounds": len(microbatches),
            "t_sha256": gauge.state_sha256([t for _, _, t, _, _ in microbatches]),
            "noise_sha256": gauge.state_sha256([eps for _, _, _, eps, _ in microbatches]),
            "dropout_rng_state_sha256": gauge.state_sha256(
                [state for _, _, _, _, state in microbatches]),
        },
        "source_state_non_committing": {
            "parameter_hash_before": source_before, "parameter_hash_after": source_after,
            "optimizer_state_hash_before": source_optimizer_before,
            "optimizer_state_hash_after": source_optimizer_after,
            "gradscaler_hash_before": source_scaler_before,
            "gradscaler_hash_after": source_scaler_after,
            "preserved": source_before == source_after
                         and source_optimizer_before == source_optimizer_after
                         and source_scaler_before == source_scaler_after,
        },
        "branches": branches,
        "whole_model": whole,
    }
    return audit, layers


def load_loss_from_checkpoint(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if "loss_fn" not in payload:
        raise SystemExit("checkpoint must contain loss_fn")
    if payload.get("augment_pipe") is not None:
        raise SystemExit("augmentation-enabled checkpoint is unsupported: paired augmentation is not implemented")
    loss = payload["loss_fn"]
    if loss.schedule.name != "sigmoid":
        raise SystemExit(f"checkpoint schedule must be 'sigmoid', got {loss.schedule.name!r}")
    return loss


def load_training_state(path: Path, device: torch.device, *, lr: float, betas: tuple[float, float],
                        eps_opt: float):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if "net" not in data or "optimizer_state" not in data:
        raise SystemExit("training-state must contain net and optimizer_state")
    net = data["net"].to(device).train().requires_grad_(True)
    optimizer = torch.optim.RAdam(net.parameters(), lr=lr, betas=betas, eps=eps_opt)
    optimizer.load_state_dict(data["optimizer_state"])
    scaler_state = data.get("gradscaler_state")
    loss_fn_state = data.get("loss_fn_state")
    meta = {
        "cur_nimg": data.get("cur_nimg"),
        "cur_tick": data.get("cur_tick"),
        "successful_optimizer_steps": data.get("successful_optimizer_steps"),
        "attempted_iteration": data.get("attempted_iteration"),
        "has_gradscaler_state": scaler_state is not None,
        "has_loss_fn_state": loss_fn_state is not None,
    }
    return net, optimizer, scaler_state, loss_fn_state, meta


def _json_safe(value: Any) -> Any:
    """Replace NaN/Inf with null so audit JSON stays strict RFC-8259."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-state", required=True, type=Path,
                        help="training-state-*.pt with net, optimizer_state, gradscaler_state")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="network-snapshot.pkl supplying loss_fn / schedule hyperparameters")
    parser.add_argument("--data", required=True, help="EDM ImageFolderDataset zip/directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batch-gpu", type=int, default=None,
                        help="training microbatch size; reproduces gradient accumulation")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--state-kimg", type=float, default=None,
                        help="provenance label only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--initial-scale", type=float, default=65536.0,
                        help="used only for --no-amp runs; AMP requires restored gradscaler_state")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--betas", default="0.9,0.999")
    parser.add_argument("--eps", dest="eps_opt", type=float, default=1e-8)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "analysis")
    args = parser.parse_args(argv)
    try:
        args.betas = tuple(float(value) for value in args.betas.split(","))
    except ValueError as exc:
        raise SystemExit("--betas must be beta1,beta2") from exc
    if len(args.betas) != 2:
        raise SystemExit("--betas must contain exactly two values")
    if (args.batch_size < 1 or args.lr <= 0 or args.initial_scale <= 0
            or (args.batch_gpu is not None and args.batch_gpu < 1)):
        raise SystemExit("batch size, lr, and initial scale must be positive")
    if args.batch_gpu is not None and args.batch_size % args.batch_gpu:
        raise SystemExit("--batch-size must be divisible by --batch-gpu")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    if args.amp and device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--amp requires an available CUDA device; use --no-amp only for CPU test runs")
    loss = load_loss_from_checkpoint(args.checkpoint)
    net, optimizer, scaler_state, loss_fn_state, state_meta = load_training_state(
        args.training_state, device, lr=args.lr, betas=args.betas, eps_opt=args.eps_opt,
    )
    missing_z_K = []
    state_summary = stateful_radam_state_summary(net, optimizer)
    if not state_summary["valid"]:
        missing_z_K.extend(state_summary["errors"])
    if args.amp and scaler_state is None:
        missing_z_K.append("GradScaler_K (training-state key gradscaler_state)")
    if missing_z_K:
        raise SystemExit(
            "training-state is incomplete for z_K = (θ_K, m_K, v_K, n_K, GradScaler_K):\n  - "
            + "\n  - ".join(missing_z_K)
            + "\nUse a real resumed training-state-*.pt from an AMP run, or pass --no-amp "
            "only for CPU unit tests."
        )
    if loss_fn_state is not None and hasattr(loss, "load_schedule_state_dict"):
        # Prefer the live schedule stage/ratio from training-state over the
        # snapshot's possibly older loss_fn.stage.
        if not loss.load_schedule_state_dict(loss_fn_state):
            raise SystemExit(
                "training-state loss_fn_state schedule_name is incompatible with "
                "checkpoint loss_fn"
            )
    from training.dataset import ImageFolderDataset
    from torch.utils.data import DataLoader
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    dataset = ImageFolderDataset(path=args.data, use_labels=False, xflip=False, cache=True,
                                resolution=net.img_resolution)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=0, generator=generator)
    images, labels = next(iter(loader))
    images = images.to(device).to(torch.float32) / 127.5 - 1
    labels = labels.to(device)
    audit, layers = run_stateful_pair(
        net, optimizer, loss, images, labels,
        amp=args.amp, initial_scale=args.initial_scale, scaler_state=scaler_state,
        random_seed=args.seed, microbatch_size=args.batch_gpu,
    )
    data_sha256, dataset_hash_algorithm = gauge.dataset_sha256(Path(args.data))
    audit["provenance"] = {
        "training_state": str(args.training_state),
        "training_state_sha256": gauge.sha256_file(args.training_state),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": gauge.sha256_file(args.checkpoint),
        "data": str(args.data), "dataset_sha256": data_sha256,
        "dataset_hash_algorithm": dataset_hash_algorithm,
        "state_kimg": args.state_kimg, "batch_size": args.batch_size, "batch_gpu": args.batch_gpu,
        "seed": args.seed, "device": str(device),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "schedule": loss.schedule.name, "q": float(loss.q), "k": float(loss.k), "b": float(loss.b),
        "stage": int(loss.stage),
        "amp_training_order": "scale, backward, unscale, sanitize, step, update",
        "training_state_meta": state_meta,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    audit_path = args.out / "radam_update_audit_stateful.json"
    layer_path = args.out / "radam_update_stateful_layerwise.csv"
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(audit), handle, indent=2, allow_nan=False)
        handle.write("\n")
    with layer_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LAYERWISE_FIELDS)
        writer.writeheader()
        writer.writerows(layers)
    print(json.dumps(_json_safe(audit["whole_model"]), indent=2))
    print(f"source state preserved: {audit['source_state_non_committing']['preserved']}")
    print(f"wrote {audit_path} and {layer_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
