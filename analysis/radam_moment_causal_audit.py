"""Causal controls for the stateful RAdam update audit.

This wrapper keeps the canonical paired stateful audit intact while intervening
only on restored optimizer moments before each disposable branch.  It supports
first-moment attenuation and m/v reset stress controls while preserving the
model parameters, RAdam step counter, GradScaler state, minibatch, t/noise,
and dropout RNG contract from ``radam_stateful_update_audit.py``.

The additional ``R_mixed`` diagnostic measures divergence after the current
first-moment recurrence

    m_{K+1} = beta1 * m_K + (1 - beta1) * g_K

and before second-moment normalization.  It is a mechanism diagnostic, not an
additive decomposition of ``R_opt``.
"""
from __future__ import annotations

import copy
import math
import sys
from typing import Any

import torch

import radam_stateful_update_audit as base


_MOMENT_MODE = "real"
_MOMENT_SCALE = 1.0
_MIXED_BY_GAIN: dict[float, dict[str, torch.Tensor]] = {}
_ORIGINAL_VIRTUAL = base.virtual_stateful_step
_ORIGINAL_RUN_PAIR = base.run_stateful_pair


def _controlled_optimizer(common_net: torch.nn.Module,
                          common_optimizer: torch.optim.RAdam) -> torch.optim.RAdam:
    optimizer = torch.optim.RAdam(
        common_net.parameters(),
        lr=common_optimizer.defaults["lr"],
        betas=common_optimizer.defaults["betas"],
        eps=common_optimizer.defaults["eps"],
        weight_decay=common_optimizer.defaults.get("weight_decay", 0.0),
    )
    optimizer.load_state_dict(copy.deepcopy(common_optimizer.state_dict()))

    if _MOMENT_SCALE < 0 or not math.isfinite(_MOMENT_SCALE):
        raise ValueError("--moment-scale must be finite and >= 0")

    for state in optimizer.state.values():
        if _MOMENT_MODE == "real":
            if _MOMENT_SCALE != 1.0 and "exp_avg" in state:
                state["exp_avg"].mul_(_MOMENT_SCALE)
        elif _MOMENT_MODE == "reset_m":
            if "exp_avg" in state:
                state["exp_avg"].zero_()
        elif _MOMENT_MODE == "reset_v":
            if "exp_avg_sq" in state:
                state["exp_avg_sq"].zero_()
        elif _MOMENT_MODE == "reset":
            if "exp_avg" in state:
                state["exp_avg"].zero_()
            if "exp_avg_sq" in state:
                state["exp_avg_sq"].zero_()
        else:
            raise ValueError(f"unknown moment mode: {_MOMENT_MODE!r}")
    return optimizer


def _moment_geometry(common_net: torch.nn.Module,
                     optimizer: torch.optim.RAdam,
                     grads: dict[str, torch.Tensor]
                     ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    beta1 = optimizer.param_groups[0]["betas"][0]
    m_values: list[torch.Tensor] = []
    g_values: list[torch.Tensor] = []
    mixed_values: list[torch.Tensor] = []
    mixed_by_name: dict[str, torch.Tensor] = {}

    for name, parameter in common_net.named_parameters():
        if name not in grads:
            continue
        state = optimizer.state.get(parameter)
        if not state or "exp_avg" not in state:
            continue
        m = state["exp_avg"].detach().double().cpu().reshape(-1)
        g = grads[name].detach().double().cpu().reshape(-1)
        mixed = beta1 * m + (1.0 - beta1) * g
        m_values.append(m)
        g_values.append(g)
        mixed_values.append(mixed)
        mixed_by_name[name] = mixed.clone()

    if not mixed_values:
        raise RuntimeError("no initialized first-moment state for causal audit")

    m_vec = torch.cat(m_values)
    g_vec = torch.cat(g_values)
    mixed_vec = torch.cat(mixed_values)

    def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
        denom = float(left.norm() * right.norm())
        return float(torch.dot(left, right) / denom) if denom else None

    return {
        "m_l2_before": float(m_vec.norm()),
        "g_l2_current": float(g_vec.norm()),
        "mixed_m_l2": float(mixed_vec.norm()),
        "m_grad_cosine": cosine(m_vec, g_vec),
    }, mixed_by_name


def virtual_stateful_step(common_net, common_optimizer, loss_template, microbatches, *,
                          gain: float, scaler_template, amp: bool):
    controlled = _controlled_optimizer(common_net, common_optimizer)
    result = _ORIGINAL_VIRTUAL(
        common_net, controlled, loss_template, microbatches,
        gain=gain, scaler_template=scaler_template, amp=amp,
    )
    grads, predicted, actual, moments_after, detail = result
    geometry, mixed = _moment_geometry(common_net, controlled, grads)
    _MIXED_BY_GAIN[gain] = mixed
    detail["moment_mode"] = _MOMENT_MODE
    detail["moment_scale"] = _MOMENT_SCALE
    detail["moments_nontrivial_before"] = _MOMENT_MODE in {"real", "reset_v"}
    detail["momentum_geometry"] = geometry
    return grads, predicted, actual, moments_after, detail


def run_stateful_pair(*args, **kwargs):
    _MIXED_BY_GAIN.clear()
    audit, layers = _ORIGINAL_RUN_PAIR(*args, **kwargs)
    if audit["whole_model"].get("gauge_defined") and {1.0, 1.3} <= _MIXED_BY_GAIN.keys():
        a_mixed, c_mixed, r_mixed, mixed_cosine, _ = base._update_scale_and_residual(
            _MIXED_BY_GAIN[1.0], _MIXED_BY_GAIN[1.3]
        )
        whole = audit["whole_model"]
        whole["a_mixed_star"] = a_mixed
        whole["c_mixed_star"] = c_mixed
        whole["R_mixed"] = r_mixed
        whole["mixed_cosine"] = mixed_cosine
    audit["moment_control"] = {
        "mode": _MOMENT_MODE,
        "first_moment_scale": _MOMENT_SCALE,
        "optimizer_step_preserved": True,
    }
    return audit, layers


def _parse_control_args(argv: list[str]) -> tuple[str, float, list[str]]:
    mode = "real"
    scale = 1.0
    remaining: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--moment-mode":
            if index + 1 >= len(argv):
                raise SystemExit("--moment-mode requires a value")
            mode = argv[index + 1]
            index += 2
            continue
        if arg.startswith("--moment-mode="):
            mode = arg.split("=", 1)[1]
            index += 1
            continue
        if arg == "--moment-scale":
            if index + 1 >= len(argv):
                raise SystemExit("--moment-scale requires a value")
            scale = float(argv[index + 1])
            index += 2
            continue
        if arg.startswith("--moment-scale="):
            scale = float(arg.split("=", 1)[1])
            index += 1
            continue
        remaining.append(arg)
        index += 1
    if mode not in {"real", "reset_m", "reset_v", "reset"}:
        raise SystemExit("--moment-mode must be one of real, reset_m, reset_v, reset")
    return mode, scale, remaining


def main(argv: list[str] | None = None) -> int:
    global _MOMENT_MODE, _MOMENT_SCALE
    args = list(sys.argv[1:] if argv is None else argv)
    _MOMENT_MODE, _MOMENT_SCALE, remaining = _parse_control_args(args)
    base.virtual_stateful_step = virtual_stateful_step
    base.run_stateful_pair = run_stateful_pair
    return base.main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
