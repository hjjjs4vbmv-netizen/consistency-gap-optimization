#!/usr/bin/env python3
"""Exact forcing/feedback decomposition of a matched A/B/C/D continuation.

For X in {B,C,D}, this runner evaluates, at every k,

    Delta_k^X = psi(z_k^X) - psi(z_k^A)
    b_k^X = Phi_X(z_k^A, xi_k) - Phi_A(z_k^A, xi_k)
    R_k^X = Phi_X(z_k^X, xi_k) - Phi_X(z_k^A, xi_k)

and checks the telescoping identity

    z_{k+1}^X - z_{k+1}^A = b_k^X + R_k^X.

The same separation is applied to two common observables: a signed residual
vector and fixed-latent EMA features.  State-block rows additionally remove
the transition's actual carryover rule from R wherever that rule is declared.
No norm ratio is reported as a contribution percentage because vector terms
can reinforce, rotate, or cancel.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.operator_clock_gate import cli_common
from analysis.operator_clock_gate.core import (
    ARM_SPECS,
    AlgorithmicState,
    AuditBatch,
    AuditBatchGroup,
    _microbatches,
    _net_forward,
    _optimizer_discrete_signature,
    _schedule,
    preserved_rng,
    rng_sha256,
    set_device_rng_state,
    state_sha256,
    transition_step,
    write_json,
)


HERE = Path(__file__).resolve().parent
FORMAL_HASHES = {
    "training_state": "fbda746805e6614319b96653563757f9e48670339e8f275f018194ebe19c9575",
    "checkpoint": "09a41e1e7c03dcdf5ffb93bb68687390278b4b190183dfff92bacc1bf79738d9",
    "batch_file": "6751e98fcc6c91bff83fe96976453535256c2de940b3e4a5fc8b3384e7c24929",
    "batch_tensor_receipts": (
        "b1eb60e44bdd7f4e6648d2af1439cf36a3873de20b6009d963295ab3abb804e9"
    ),
}
LEGACY_ROLLOUT = (
    REPO_ROOT / "analysis/operator_clock_gate/results/raw_receipts/formal-20260826/"
    "results/matched/matched_micro_rollout.json"
)
NORM_EPSILON = 1e-30
CSV_FIELDS = (
    "arm", "k", "next_k", "space", "block", "coordinate_count",
    "delta_k_norm", "b_norm", "R_norm", "delta_norm",
    "feedback_gain_G", "cos_R_delta_k", "cos_b_R", "cos_b_delta",
    "cos_R_delta", "R_over_b_diagnostic", "carryover_rule",
    "carryover_retention_min", "carryover_retention_max",
    "carryover_norm", "corrected_R_norm", "corrected_R_over_delta_k",
    "corrected_R_over_b", "cos_corrected_R_delta_k",
    "cos_corrected_R_b", "cos_corrected_R_R", "closure_l2",
    "closure_relative", "closure_max_abs", "closure_pass",
)


def _state_tensor_blocks(state: AlgorithmicState) -> dict[str, dict[str, torch.Tensor]]:
    """Return live tensor references grouped by scientifically named block."""
    blocks: dict[str, dict[str, torch.Tensor]] = {
        "theta": {}, "EMA": {}, "m": {}, "v": {},
        "net_buffer": {}, "scaler": {},
    }
    named_parameters = dict(state.net.named_parameters())
    for name, value in named_parameters.items():
        blocks["theta"][name] = value.detach()
    for name, value in state.net.named_buffers():
        if value.is_floating_point():
            blocks["net_buffer"][name] = value.detach()
    parameter_names = {id(value): name for name, value in named_parameters.items()}
    for parameter, optimizer_state in state.optimizer.state.items():
        name = parameter_names.get(id(parameter))
        if name is None:
            raise RuntimeError("optimizer contains a parameter not owned by state.net")
        for key, value in optimizer_state.items():
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                continue
            if key == "exp_avg":
                blocks["m"][name] = value.detach()
            elif key == "exp_avg_sq":
                blocks["v"][name] = value.detach()
            elif key != "step":
                raise RuntimeError(f"unsupported continuous optimizer state {key!r}")
    for name, value in state.ema.named_parameters():
        blocks["EMA"][f"parameter.{name}"] = value.detach()
    for name, value in state.ema.named_buffers():
        if value.is_floating_point():
            blocks["EMA"][f"buffer.{name}"] = value.detach()
    if state.scaler is not None:
        scale = state.scaler.state_dict().get("scale")
        if isinstance(scale, torch.Tensor):
            blocks["scaler"]["scale"] = scale.detach()
        elif isinstance(scale, (int, float)):
            device = next(state.net.parameters()).device
            blocks["scaler"]["scale"] = torch.tensor(float(scale), device=device)
    return {key: value for key, value in blocks.items() if value}


def _safe_cos(dot: float, left_norm: float, right_norm: float) -> float | None:
    denominator = left_norm * right_norm
    if denominator == 0.0:
        return None
    return max(-1.0, min(1.0, dot / denominator))


def _clone_tensor_map(
    values: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Freeze a coordinate map before an in-place transition mutates its state."""
    return {name: value.detach().clone() for name, value in values.items()}


def _optimizer_retentions(
    state: AlgorithmicState, *, beta_index: int,
) -> dict[str, float]:
    """Read per-parameter RAdam retention from the actual optimizer groups."""
    if beta_index not in (0, 1):
        raise ValueError("beta_index must select beta1 or beta2")
    parameter_names = {id(value): name for name, value in state.net.named_parameters()}
    retentions: dict[str, float] = {}
    for group in state.optimizer.param_groups:
        betas = group.get("betas")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise RuntimeError("carryover correction requires optimizer betas")
        retention = float(betas[beta_index])
        if not 0.0 <= retention <= 1.0:
            raise RuntimeError("optimizer retention must lie in [0,1]")
        for parameter in group["params"]:
            name = parameter_names.get(id(parameter))
            if name is None:
                raise RuntimeError("optimizer parameter is not owned by state.net")
            if name in retentions:
                raise RuntimeError(f"optimizer parameter {name!r} appears in two groups")
            retentions[name] = retention
    if set(retentions) != set(parameter_names.values()):
        raise RuntimeError("optimizer groups do not cover the network parameter schema")
    return retentions


def carryover_only_map(
    block: str,
    pre_baseline: Mapping[str, torch.Tensor],
    pre_actual: Mapping[str, torch.Tensor],
    transition_state: AlgorithmicState,
    *,
    optimizer_step_skipped: bool = False,
    optimizer_skip_regime_paired: bool = True,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any]]:
    """Apply only the real transition retention to an incoming separation.

    This deliberately does not approximate EMA with one arbitrary beta.  The
    production transition lerps EMA parameters with ``ema_beta`` but leaves EMA
    buffers unchanged, so those coordinates receive different retention maps.
    """
    if set(pre_baseline) != set(pre_actual):
        raise RuntimeError("pre-transition coordinate schemas differ")
    if block == "theta":
        retentions = {name: 1.0 for name in pre_baseline}
        rule = "identity_parameter_carryover"
        source = "theta_next=theta_prev+optimizer_update"
    elif block in {"m", "v"}:
        if not optimizer_skip_regime_paired:
            return None, {
                "rule": "undefined_across_optimizer_skip_regime_mismatch",
                "retention_source": "paired transition telemetry.step_skipped",
                "retention_values": [],
            }
        beta_index = 0 if block == "m" else 1
        optimizer_retentions = (
            {name: 1.0 for name in pre_baseline}
            if optimizer_step_skipped else
            _optimizer_retentions(transition_state, beta_index=beta_index)
        )
        if set(optimizer_retentions) != set(pre_baseline):
            raise RuntimeError(f"{block} coordinates do not match optimizer groups")
        retentions = optimizer_retentions
        rule = (
            "optimizer_step_skipped_identity_carryover"
            if optimizer_step_skipped else
            f"radam_beta{beta_index + 1}_per_parameter_group"
        )
        source = (
            "paired transition telemetry.step_skipped"
            if optimizer_step_skipped else
            f"optimizer.param_groups[*].betas[{beta_index}]"
        )
    elif block == "EMA":
        ema_beta = float(transition_state.ema_beta)
        if not 0.0 <= ema_beta <= 1.0:
            raise RuntimeError("EMA retention must lie in [0,1]")
        retentions = {}
        for name in pre_baseline:
            if name.startswith("parameter."):
                retentions[name] = ema_beta
            elif name.startswith("buffer."):
                # transition_step updates EMA parameters only; buffers persist.
                retentions[name] = 1.0
            else:
                raise RuntimeError(f"unknown EMA coordinate {name!r}")
        rule = "ema_transition_carryover_counterfactual_map"
        source = (
            "transition_step parameter lerp with AlgorithmicState.ema_beta; "
            "EMA buffers unchanged"
        )
    else:
        return None, {
            "rule": "not_declared_for_this_readout",
            "retention_source": None,
            "retention_values": [],
        }
    carryover = {
        name: (pre_actual[name].detach() - pre_baseline[name].detach())
        * retentions[name]
        for name in pre_baseline
    }
    return carryover, {
        "rule": rule,
        "retention_source": source,
        "retention_values": sorted(set(retentions.values())),
    }


def exact_three_point_metrics(
    baseline: Mapping[str, torch.Tensor],
    counterfactual: Mapping[str, torch.Tensor],
    actual: Mapping[str, torch.Tensor],
    *,
    pre_baseline: Mapping[str, torch.Tensor] | None = None,
    pre_actual: Mapping[str, torch.Tensor] | None = None,
    carryover: Mapping[str, torch.Tensor] | None = None,
    norm_epsilon: float = NORM_EPSILON,
    closure_atol: float = 1e-12,
    closure_rtol: float = 1e-12,
) -> dict[str, Any]:
    """Compute forcing, propagation and corrected-feedback vector metrics."""
    if set(baseline) != set(counterfactual) or set(baseline) != set(actual):
        raise RuntimeError("three-point coordinate schemas differ")
    if (pre_baseline is None) != (pre_actual is None):
        raise ValueError("pre_baseline and pre_actual must be supplied together")
    if pre_baseline is not None and (
        set(pre_baseline) != set(baseline) or set(pre_actual) != set(baseline)
    ):
        raise RuntimeError("pre/post coordinate schemas differ")
    if carryover is not None and set(carryover) != set(baseline):
        raise RuntimeError("carryover coordinate schema differs")
    if not math.isfinite(float(norm_epsilon)) or norm_epsilon <= 0:
        raise ValueError("norm_epsilon must be positive and finite")
    sums = {key: 0.0 for key in (
        "b2", "R2", "d2", "dk2", "bR", "bd", "Rd", "Rdk", "c2",
        "carry2", "corrected2", "corrected_dk", "corrected_b", "corrected_R",
    )}
    closure_max = 0.0
    coordinate_count = 0
    with torch.no_grad():
        for name in sorted(baseline):
            a = baseline[name].detach().double()
            c = counterfactual[name].detach().double()
            x = actual[name].detach().double()
            if a.shape != c.shape or a.shape != x.shape:
                raise RuntimeError(f"three-point shape mismatch for {name}")
            b = c - a
            feedback = x - c
            delta = x - a
            closure = b + feedback - delta
            delta_k = None
            if pre_baseline is not None and pre_actual is not None:
                pre_a = pre_baseline[name].detach().double()
                pre_x = pre_actual[name].detach().double()
                if pre_a.shape != a.shape or pre_x.shape != a.shape:
                    raise RuntimeError(f"pre/post shape mismatch for {name}")
                delta_k = pre_x - pre_a
            sums["b2"] += float(b.square().sum())
            sums["R2"] += float(feedback.square().sum())
            sums["d2"] += float(delta.square().sum())
            sums["bR"] += float((b * feedback).sum())
            sums["bd"] += float((b * delta).sum())
            sums["Rd"] += float((feedback * delta).sum())
            sums["c2"] += float(closure.square().sum())
            if delta_k is not None:
                sums["dk2"] += float(delta_k.square().sum())
                sums["Rdk"] += float((feedback * delta_k).sum())
            if carryover is not None:
                retained = carryover[name].detach().double()
                if retained.shape != a.shape:
                    raise RuntimeError(f"carryover shape mismatch for {name}")
                corrected = feedback - retained
                sums["carry2"] += float(retained.square().sum())
                sums["corrected2"] += float(corrected.square().sum())
                sums["corrected_b"] += float((corrected * b).sum())
                sums["corrected_R"] += float((corrected * feedback).sum())
                if delta_k is None:
                    raise ValueError("carryover metrics require pre-transition separation")
                sums["corrected_dk"] += float((corrected * delta_k).sum())
            if closure.numel():
                closure_max = max(closure_max, float(closure.abs().max()))
            coordinate_count += int(a.numel())
    b_norm = math.sqrt(sums["b2"])
    feedback_norm = math.sqrt(sums["R2"])
    delta_norm = math.sqrt(sums["d2"])
    delta_k_norm = math.sqrt(sums["dk2"]) if pre_baseline is not None else None
    closure_l2 = math.sqrt(sums["c2"])
    closure_limit = float(closure_atol) + float(closure_rtol) * delta_norm
    carryover_norm = math.sqrt(sums["carry2"]) if carryover is not None else None
    corrected_norm = math.sqrt(sums["corrected2"]) if carryover is not None else None
    return {
        "coordinate_count": coordinate_count,
        "delta_k_norm": delta_k_norm,
        "b_norm": b_norm,
        "R_norm": feedback_norm,
        "delta_norm": delta_norm,
        "feedback_gain_G": (
            feedback_norm / max(float(delta_k_norm), float(norm_epsilon))
            if delta_k_norm is not None else None
        ),
        "cos_R_delta_k": (
            _safe_cos(sums["Rdk"], feedback_norm, delta_k_norm)
            if delta_k_norm is not None else None
        ),
        "cos_b_R": _safe_cos(sums["bR"], b_norm, feedback_norm),
        "cos_b_delta": _safe_cos(sums["bd"], b_norm, delta_norm),
        "cos_R_delta": _safe_cos(sums["Rd"], feedback_norm, delta_norm),
        "R_over_b_diagnostic": (feedback_norm / b_norm if b_norm else None),
        "carryover_norm": carryover_norm,
        "corrected_R_norm": corrected_norm,
        "corrected_R_over_delta_k": (
            corrected_norm / max(float(delta_k_norm), float(norm_epsilon))
            if corrected_norm is not None and delta_k_norm is not None else None
        ),
        "corrected_R_over_b": (
            corrected_norm / max(b_norm, float(norm_epsilon))
            if corrected_norm is not None else None
        ),
        "cos_corrected_R_delta_k": (
            _safe_cos(sums["corrected_dk"], corrected_norm, delta_k_norm)
            if corrected_norm is not None and delta_k_norm is not None else None
        ),
        "cos_corrected_R_b": (
            _safe_cos(sums["corrected_b"], corrected_norm, b_norm)
            if corrected_norm is not None else None
        ),
        "cos_corrected_R_R": (
            _safe_cos(sums["corrected_R"], corrected_norm, feedback_norm)
            if corrected_norm is not None else None
        ),
        "closure_l2": closure_l2,
        "closure_relative": closure_l2 / max(delta_norm, torch.finfo(torch.float64).tiny),
        "closure_max_abs": closure_max,
        "closure_pass": closure_l2 <= closure_limit,
    }


def _signed_residual_vector(
    state: AlgorithmicState,
    batch: AuditBatch | AuditBatchGroup,
    *,
    arm: str = "A",
) -> torch.Tensor:
    """Common signed validation residual; arm is fixed across all three states."""
    values = []
    state.net.eval()
    with torch.no_grad(), preserved_rng(batch.audit_id):
        for micro in _microbatches(batch):
            schedule = _schedule(state.loss_fn, ARM_SPECS[arm]["target_scale"])
            r_target = schedule.compute_r(micro.t, stage=int(state.loss_fn.stage))
            set_device_rng_state(micro.dropout_rng_state, micro.images.device)
            online = _net_forward(
                state.net, micro.images + micro.noise * micro.t,
                micro.t, micro.labels,
            )
            set_device_rng_state(micro.dropout_rng_state, micro.images.device)
            target = _net_forward(
                state.net, micro.images + micro.noise * r_target,
                r_target, micro.labels,
            )
            target = torch.nan_to_num(target)
            target = (r_target > 0) * target + (r_target <= 0) * micro.images
            values.append((online - target).detach())
    return torch.cat([value.reshape(-1) for value in values])


def _fixed_feature_vector(
    state: AlgorithmicState,
    batch: AuditBatch | AuditBatchGroup,
    latent: torch.Tensor | None,
) -> torch.Tensor:
    micro = _microbatches(batch)[0]
    state.ema.eval()
    if latent is None:
        latent = micro.noise
    latent = latent.to(device=micro.images.device, dtype=micro.images.dtype)
    sigma = micro.t[:latent.shape[0]]
    labels = micro.labels[:latent.shape[0]]
    if sigma.shape[0] != latent.shape[0]:
        sigma = micro.t[:1].expand(latent.shape[0], -1, -1, -1)
        labels = micro.labels[:1].expand(latent.shape[0], *micro.labels.shape[1:])
    with torch.no_grad(), preserved_rng(batch.audit_id):
        set_device_rng_state(micro.dropout_rng_state, micro.images.device)
        value = _net_forward(state.ema, latent * sigma, sigma, labels)
    return value.detach().reshape(-1)


def observable_vectors(
    state: AlgorithmicState,
    batch: AuditBatch | AuditBatchGroup,
    latent: torch.Tensor | None,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "residual": {"value": _signed_residual_vector(state, batch, arm="A")},
        "feature": {"value": _fixed_feature_vector(state, batch, latent)},
    }


def _batch_receipt(batch: AuditBatch | AuditBatchGroup) -> dict[str, Any]:
    micros = _microbatches(batch)
    return {
        "audit_id": int(batch.audit_id),
        "microbatch_count": len(micros),
        "microbatches": [
            {
                "images": state_sha256(item.images),
                "labels": state_sha256(item.labels),
                "t": state_sha256(item.t),
                "noise": state_sha256(item.noise),
                "dropout_rng_state": state_sha256(item.dropout_rng_state),
            }
            for item in micros
        ],
    }


def _forcing_input_receipt(state: AlgorithmicState) -> dict[str, Any]:
    return {
        "state_sha256": state.sha256(),
        "optimizer_and_amp_discrete_state": _optimizer_discrete_signature(state),
        "scaler_state_sha256": state_sha256(
            state.scaler.state_dict() if state.scaler is not None else None),
    }


def _transition(
    state: AlgorithmicState,
    batch: AuditBatch | AuditBatchGroup,
    arm: str,
    pairing_seed: int,
    *,
    clone_input: bool,
) -> tuple[AlgorithmicState, dict[str, Any]]:
    input_receipt = _forcing_input_receipt(state)
    rng_before = rng_sha256()
    with preserved_rng(pairing_seed):
        result, telemetry = transition_step(
            state, batch, arm=arm, clone_input=clone_input)
    rng_after = rng_sha256()
    if rng_before != rng_after:
        raise RuntimeError("paired transition polluted process RNG")
    telemetry["pairing_seed"] = int(pairing_seed)
    telemetry["process_rng_preserved"] = True
    telemetry["input_augmented_state"] = input_receipt
    return result, telemetry


def _row(
    arm: str,
    k: int,
    space: str,
    block: str,
    baseline: Mapping[str, torch.Tensor],
    counterfactual: Mapping[str, torch.Tensor],
    actual: Mapping[str, torch.Tensor],
    *,
    pre_baseline: Mapping[str, torch.Tensor],
    pre_actual: Mapping[str, torch.Tensor],
    carryover: Mapping[str, torch.Tensor] | None = None,
    carryover_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(carryover_metadata or {})
    retentions = [float(value) for value in metadata.get("retention_values", [])]
    return {
        "arm": arm, "k": k, "next_k": k + 1,
        "space": space, "block": block,
        "carryover_rule": metadata.get("rule"),
        "carryover_retention_min": min(retentions) if retentions else None,
        "carryover_retention_max": max(retentions) if retentions else None,
        **exact_three_point_metrics(
            baseline, counterfactual, actual,
            pre_baseline=pre_baseline, pre_actual=pre_actual,
            carryover=carryover,
        ),
    }


def audit_legacy_rollout(path: Path = LEGACY_ROLLOUT) -> dict[str, Any]:
    """Fail closed unless the old rollout has recoverable state for all k=0..64."""
    if not path.is_file():
        return {"path": str(path), "recoverable": False, "reason": "receipt_missing"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    horizons = payload.get("horizons", [])
    required = set(range(65))
    per_arm = {}
    for arm in "ABCD":
        branch = payload.get("branches", {}).get(arm, {})
        state_steps = branch.get("augmented_states", {})
        present = {int(item) for item in state_steps if str(item).isdigit()}
        per_arm[arm] = {
            "complete_k_0_through_64": required.issubset(present),
            "present_state_steps": sorted(present),
        }
    recoverable = all(item["complete_k_0_through_64"] for item in per_arm.values())
    return {
        "path": str(path.resolve()),
        "sha256": cli_common.sha256_file(path),
        "legacy_horizons": horizons,
        "per_arm": per_arm,
        "recoverable": recoverable,
        "reason": ("complete_augmented_states_present" if recoverable else
                   "only projections/summaries/hashes are stored; full per-k augmented state absent"),
    }


def run_exact_decomposition(
    source: AlgorithmicState,
    batches: Sequence[AuditBatch | AuditBatchGroup],
    *,
    steps: int = 64,
    latent: torch.Tensor | None = None,
    include_observables: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if steps < 1:
        raise ValueError("steps must be positive")
    if not batches:
        raise ValueError("at least one frozen batch is required")
    source_hash = source.sha256()
    source_rng = rng_sha256()
    branches = {arm: source.clone() for arm in "ABCD"}
    rows: list[dict[str, Any]] = []
    step_receipts = []
    carryover_rules: dict[str, list[dict[str, Any]]] = {}
    state_hashes = [{"k": 0, "by_arm": {arm: branches[arm].sha256() for arm in "ABCD"}}]
    with preserved_rng():
        for k in range(steps):
            batch = batches[k % len(batches)]
            pairing_seed = int(batch.audit_id) + k
            z_a = branches["A"]
            a_pre_blocks = {
                block: _clone_tensor_map(values)
                for block, values in _state_tensor_blocks(z_a).items()
            }
            a_pre_observables = (
                observable_vectors(z_a, batches[0], latent)
                if include_observables else {}
            )
            a_after, a_telemetry = _transition(
                z_a, batch, "A", pairing_seed, clone_input=True)
            a_blocks = _state_tensor_blocks(a_after)
            a_observables = (observable_vectors(a_after, batches[0], latent)
                             if include_observables else {})
            arm_receipts = {}
            for arm in "BCD":
                x_pre_blocks = {
                    block: _clone_tensor_map(values)
                    for block, values in _state_tensor_blocks(branches[arm]).items()
                }
                x_pre_observables = (
                    observable_vectors(branches[arm], batches[0], latent)
                    if include_observables else {}
                )
                # Both Phi_A(zA) and Phi_X(zA) receive clones of exactly zA,
                # so optimizer moments, optimizer step and AMP state are paired.
                counterfactual, cf_telemetry = _transition(
                    z_a, batch, arm, pairing_seed, clone_input=True)
                actual_after, actual_telemetry = _transition(
                    branches[arm], batch, arm, pairing_seed, clone_input=False)
                cf_blocks = _state_tensor_blocks(counterfactual)
                actual_blocks = _state_tensor_blocks(actual_after)
                if (set(a_blocks) != set(cf_blocks)
                        or set(a_blocks) != set(actual_blocks)
                        or set(a_blocks) != set(a_pre_blocks)
                        or set(a_blocks) != set(x_pre_blocks)):
                    raise RuntimeError("augmented state block schemas differ")
                for block in a_blocks:
                    carryover, carryover_metadata = carryover_only_map(
                        block, a_pre_blocks[block], x_pre_blocks[block], actual_after,
                        optimizer_step_skipped=bool(
                            actual_telemetry["step_skipped"]),
                        optimizer_skip_regime_paired=(
                            bool(actual_telemetry["step_skipped"])
                            == bool(cf_telemetry["step_skipped"])),
                    )
                    observed_rules = carryover_rules.setdefault(block, [])
                    if carryover_metadata not in observed_rules:
                        observed_rules.append(dict(carryover_metadata))
                    rows.append(_row(
                        arm, k, "state", block, a_blocks[block],
                        cf_blocks[block], actual_blocks[block],
                        pre_baseline=a_pre_blocks[block],
                        pre_actual=x_pre_blocks[block],
                        carryover=carryover,
                        carryover_metadata=carryover_metadata))
                if include_observables:
                    cf_observables = observable_vectors(counterfactual, batches[0], latent)
                    actual_observables = observable_vectors(actual_after, batches[0], latent)
                    for block in ("residual", "feature"):
                        rows.append(_row(
                            arm, k, "observable", block, a_observables[block],
                            cf_observables[block], actual_observables[block],
                            pre_baseline=a_pre_observables[block],
                            pre_actual=x_pre_observables[block],
                            carryover_metadata={
                                "rule": "not_declared_for_this_readout",
                                "retention_source": None,
                                "retention_values": [],
                            }))
                forcing_input = a_telemetry["input_augmented_state"]
                forcing_input_cf = cf_telemetry["input_augmented_state"]
                if forcing_input_cf != forcing_input:
                    raise RuntimeError("forcing pair did not start from identical augmented state")
                arm_receipts[arm] = {
                    "forcing_input": forcing_input,
                    "forcing_input_rechecked": forcing_input_cf,
                    "phi_A_from_zA": a_telemetry,
                    "phi_X_from_zA": cf_telemetry,
                    "phi_X_from_zX": actual_telemetry,
                    "counterfactual_state_sha256": counterfactual.sha256(),
                    "actual_next_state_sha256": actual_after.sha256(),
                }
                branches[arm] = actual_after
                del counterfactual
            branches["A"] = a_after
            state_hashes.append({
                "k": k + 1,
                "by_arm": {arm: branches[arm].sha256() for arm in "ABCD"},
            })
            step_receipts.append({
                "k": k, "next_k": k + 1, "pairing_seed": pairing_seed,
                "batch": _batch_receipt(batch), "arms": arm_receipts,
            })
    source_after = source.sha256()
    rng_after = rng_sha256()
    required_blocks = {"theta", "EMA", "m", "v"}
    observed_blocks = {row["block"] for row in rows if row["space"] == "state"}
    closure_pass = all(row["closure_pass"] for row in rows)
    receipt = {
        "schema_version": 2,
        "kind": "exact_nonlinear_forcing_feedback",
        "definition": {
            "pre_transition_separation": "psi(z_k^X)-psi(z_k^A)",
            "forcing": "Phi_X(z_k^A,xi_k)-Phi_A(z_k^A,xi_k)",
            "feedback": "Phi_X(z_k^X,xi_k)-Phi_X(z_k^A,xi_k)",
            "closure": "z_{k+1}^X-z_{k+1}^A=b_k^X+R_k^X",
            "feedback_gain": "||R_k||/max(||Delta_k||,epsilon)",
            "feedback_alignment": "cos(R_k,Delta_k)",
            "incremental_feedback": "R_tilde_k=R_k-carryover_only(Delta_k)",
        },
        "norm_epsilon": NORM_EPSILON,
        "carryover_correction": {
            "rules_by_state_block": carryover_rules,
            "theta_formula": "R_tilde_k^theta=R_k^theta-Delta_k^theta",
            "m_formula": "R_tilde_k^m=R_k^m-beta1*Delta_k^m",
            "v_formula": "R_tilde_k^v=R_k^v-beta2*Delta_k^v",
            "ema_formula": (
                "R_tilde_k^EMA=R_k^EMA-"
                "carryover_only_transition_map(Delta_k^EMA)"
            ),
        },
        "steps": steps,
        "arms": {arm: ARM_SPECS[arm] for arm in "ABCD"},
        "state_blocks": sorted(observed_blocks),
        "required_state_blocks_present": required_blocks.issubset(observed_blocks),
        "observable_definition": {
            "residual": "signed online-minus-target tensor under common arm-A validation map",
            "feature": "fixed-latent EMA output tensor under a common validation map",
        },
        "state_hashes_k_0_through_horizon": state_hashes,
        "step_replay_receipts": step_receipts,
        "source_state_sha256_before": source_hash,
        "source_state_sha256_after": source_after,
        "source_rng_sha256_before": source_rng,
        "source_rng_sha256_after": rng_after,
        "source_preserved": source_hash == source_after and source_rng == rng_after,
        "all_exact_closures_pass": closure_pass,
        "max_closure_l2": max(row["closure_l2"] for row in rows),
        "max_closure_relative": max(row["closure_relative"] for row in rows),
    }
    receipt["status"] = (
        "PASS" if receipt["source_preserved"] and closure_pass
        and receipt["required_state_blocks_present"] else "FAIL_CLOSED"
    )
    return rows, receipt


def _median(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values
              if value is not None and math.isfinite(float(value))]
    return statistics.median(finite) if finite else None


def _mechanism_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["arm"]), str(row["space"]), str(row["block"]))].append(row)
    output = {}
    for (arm, space, block), items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: int(item["k"]))
        split = max(1, len(items) // 4)
        late = items[-split:]
        ratios = [item["R_over_b_diagnostic"] for item in items]
        forcing_fraction = sum(
            value is not None and value <= 0.25 for value in ratios) / len(items)
        late_feedback_fraction = sum(
            item["R_norm"] > item["b_norm"] for item in late) / len(late)
        cancellation_fraction = sum(
            item["cos_b_R"] is not None and item["cos_b_R"] <= -0.9
            and item["b_norm"] >= item["delta_norm"]
            and item["R_norm"] >= item["delta_norm"]
            for item in items) / len(items)
        early_ratio = _median(item["R_over_b_diagnostic"] for item in items[:split])
        late_ratio = _median(item["R_over_b_diagnostic"] for item in late)
        late_alignment = _median(item["cos_R_delta"] for item in late)
        gains = [item.get("feedback_gain_G") for item in items]
        aligned_expansion_fraction = sum(
            item.get("feedback_gain_G") is not None
            and item["feedback_gain_G"] > 1.0
            and item.get("cos_R_delta_k") is not None
            and item["cos_R_delta_k"] >= 0.8
            for item in items
        ) / len(items)
        contraction_fraction = sum(
            value is not None and value < 1.0 for value in gains
        ) / len(items)
        low_alignment_fraction = sum(
            item.get("cos_R_delta_k") is not None
            and item["cos_R_delta_k"] < 0.8
            for item in items
        ) / len(items)
        if cancellation_fraction >= 0.5:
            label = "forcing_feedback_cancellation"
        elif forcing_fraction >= 0.8:
            label = "persistent_forcing_accumulation"
        elif (late_feedback_fraction >= 0.75 and late_ratio is not None
              and late_ratio > 1.0 and late_alignment is not None
              and late_alignment >= 0.8
              and (early_ratio is None or late_ratio > early_ratio)):
            label = "persistent_state_feedback_dominance"
        else:
            label = "mixed_or_inconclusive"
        output[f"{arm}:{space}:{block}"] = {
            "arm": arm, "space": space, "block": block,
            "classification": label,
            "forcing_dominant_fraction_R_le_0p25_b": forcing_fraction,
            "late_feedback_dominant_fraction_R_gt_b": late_feedback_fraction,
            "cancellation_fraction": cancellation_fraction,
            "early_quarter_median_R_over_b": early_ratio,
            "late_quarter_median_R_over_b": late_ratio,
            "late_quarter_median_cos_R_delta": late_alignment,
            "early_quarter_median_feedback_gain_G": _median(
                item.get("feedback_gain_G") for item in items[:split]),
            "late_quarter_median_feedback_gain_G": _median(
                item.get("feedback_gain_G") for item in late),
            "late_quarter_median_cos_R_delta_k": _median(
                item.get("cos_R_delta_k") for item in late),
            "contraction_fraction_G_lt_1": contraction_fraction,
            "aligned_expansion_fraction_G_gt_1_cos_ge_0p8": (
                aligned_expansion_fraction),
            "low_alignment_fraction_cos_R_delta_k_lt_0p8": (
                low_alignment_fraction),
            "median_corrected_R_over_delta_k": _median(
                item.get("corrected_R_over_delta_k") for item in items),
            "median_corrected_R_over_b": _median(
                item.get("corrected_R_over_b") for item in items),
            "median_cos_corrected_R_delta_k": _median(
                item.get("cos_corrected_R_delta_k") for item in items),
            "median_cos_corrected_R_b": _median(
                item.get("cos_corrected_R_b") for item in items),
            "strong_expansion_claim_allowed": False,
            "strong_expansion_gate_reason": (
                "second independent state replication is not part of this audit"
            ),
        }
    return output


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _compact_run_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    compact = {
        key: value for key, value in receipt.items()
        if key != "step_replay_receipts"
    }
    step_receipts = receipt["step_replay_receipts"]
    compact["step_replay_receipt_count"] = len(step_receipts)
    compact["step_replay_receipts_sha256"] = _canonical_sha256(step_receipts)
    return compact


def build_summary(
    rows: Sequence[Mapping[str, Any]], receipt: Mapping[str, Any],
    legacy_audit: Mapping[str, Any], assets: Mapping[str, Any],
    *, include_full_receipt: bool = False,
) -> dict[str, Any]:
    step_receipts = receipt["step_replay_receipts"]
    replay_receipt = {
        "state_hashes_k_0_through_horizon": receipt[
            "state_hashes_k_0_through_horizon"],
        "step_replay_receipt_count": len(step_receipts),
        "step_replay_receipts_sha256": _canonical_sha256(step_receipts),
    }
    run_receipt = (
        dict(receipt) if include_full_receipt else _compact_run_receipt(receipt)
    )
    if include_full_receipt:
        replay_receipt["step_replay_receipts"] = step_receipts
    return {
        "schema_version": 3,
        "status": receipt["status"],
        "task_1_existing_rollout_state_audit": dict(legacy_audit),
        "instrumentation": {
            "kind": "64-step matched continuation from frozen source",
            "scientific_scope": "local counterfactual instrumentation; not a new experiment",
            "assets": dict(assets),
            "replay_receipt": replay_receipt,
        },
        "exact_closure": {
            "all_pass": receipt["all_exact_closures_pass"],
            "max_l2": receipt["max_closure_l2"],
            "max_relative": receipt["max_closure_relative"],
        },
        "carryover_correction": receipt["carryover_correction"],
        "mechanism_by_arm_and_block": _mechanism_summary(rows),
        "mechanism_decision_rules": {
            "persistent_forcing_accumulation": (
                "R/b <= 0.25 on at least 80% of steps"
            ),
            "persistent_state_feedback_dominance": (
                "in the last quarter R>b on at least 75% of steps, median R/b>1, "
                "median cos(R,Delta_next)>=0.8, and late median R/b exceeds early median; "
                "this is a propagation/persistence label, not a stronger expansion claim"
            ),
            "forcing_feedback_cancellation": (
                "cos(b,R)<=-0.9 and both ||b|| and ||R|| >= ||delta|| on at least "
                "50% of steps"
            ),
        },
        "pre_transition_propagation_interpretation": {
            "G_approximately_1_and_alignment_approximately_1": (
                "state persistence is the main observed propagation component"
            ),
            "G_lt_1": "contractive propagation",
            "G_gt_1_and_alignment_approximately_1": (
                "possible same-direction expansion, subject to the stronger causal claim gate"
            ),
            "low_alignment": (
                "rotation or complex deformation rather than simple same-direction expansion"
            ),
        },
        "strong_expansion_claim_gate": {
            "required_facts": [
                "G_k>1",
                "R_k highly aligned with Delta_k",
                "carryover-corrected R_tilde_k is non-trivial",
                "direction is consistent in a second independent state replication",
            ],
            "second_state_replication_available": False,
            "status": "WITHHELD",
            "allowed_wording": "propagation / persistence",
        },
        "interpretation_guard": (
            "R_over_b is a scale diagnostic, never a contribution percentage; "
            "large state-block values identify history-dominated state "
            "propagation, not a quality mechanism. G is a measured propagation "
            "gain only when paired with its direction cosine."
        ),
        "run_receipt": run_receipt,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=CSV_FIELDS, extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    legacy = summary["task_1_existing_rollout_state_audit"]
    closure = summary["exact_closure"]
    lines = [
        "# Forcing–feedback decomposition v2",
        "",
        f"Status: **{summary['status']}**",
        "",
        "## State recoverability gate",
        "",
        ("The PR #87 matched-rollout receipt is recoverable." if legacy["recoverable"] else
         "The PR #87 matched-rollout receipt is **not** recoverable at every k=0,…,64. "
         "It stores projections, summaries and hashes, not full augmented states."),
        "The decomposition therefore comes from a matched 64-step replay of the same "
        "hash-pinned frozen source and is instrumentation, not a new scientific experiment.",
        "",
        "## Exact identity",
        "",
        "For each X∈{B,C,D} and k, the run evaluates `Phi_A(zA)`, `Phi_X(zA)` "
        "and `Phi_X(zX)` using the same frozen batch/noise/dropout receipt and pairing seed.",
        f"All block/observable closures pass: **{closure['all_pass']}**; maximum relative "
        f"closure error: `{closure['max_relative']:.6g}`.",
        "",
        "The CSV also records the pre-transition separation `Delta_k`, measured "
        "propagation gain `G_k=||R_k||/max(||Delta_k||,epsilon)`, and "
        "`cos(R_k,Delta_k)` for every state block and observable.",
        "",
        "## Propagation and incremental-feedback diagnostics",
        "",
        "| arm | space | block | classification | late G | late cos(R,Delta_k) | median corrected R/Delta_k | median corrected R/b | median cos(corrected R,Delta_k) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["mechanism_by_arm_and_block"].values():
        def fmt(value: Any) -> str:
            return "NA" if value is None else f"{float(value):.4g}"
        lines.append(
            f"| {item['arm']} | {item['space']} | {item['block']} | "
            f"{item['classification']} | "
            f"{fmt(item['late_quarter_median_feedback_gain_G'])} | "
            f"{fmt(item['late_quarter_median_cos_R_delta_k'])} | "
            f"{fmt(item['median_corrected_R_over_delta_k'])} | "
            f"{fmt(item['median_corrected_R_over_b'])} | "
            f"{fmt(item['median_cos_corrected_R_delta_k'])} |"
        )
    lines += [
        "",
        "`G≈1` with alignment near one indicates persistence; `G<1` indicates "
        "contractive propagation. `G>1` with high alignment is only possible "
        "same-direction expansion. Low alignment indicates rotation or more "
        "complex deformation.",
        "",
        "For `theta`, corrected feedback subtracts `Delta_k`. For RAdam `m` and "
        "`v`, it subtracts the actual per-parameter-group `beta1*Delta_k` and "
        "`beta2*Delta_k`. EMA uses the implemented transition map: parameters "
        "retain `AlgorithmicState.ema_beta`, while EMA buffers are unchanged and "
        "therefore retain their full incoming separation. No arbitrary EMA beta "
        "is introduced. The CSV additionally records corrected alignment against "
        "the incoming separation, current forcing, and raw propagation term.",
        "",
        "No mechanism winner is selected. Only propagation/persistence wording is "
        "used: this audit has no second independent state replication, so the "
        "stronger causal claim gate is not satisfied.",
        "",
        "## Interpretation boundary",
        "",
        "`R/b` is reported only as a scale diagnostic. It is not a contribution "
        "percentage: large forcing and feedback terms can be nearly antiparallel and cancel.",
        "For persistent state blocks, the dominance label means that accumulated state "
        "history is larger than the current common-state forcing; it is a propagation label.",
        "Residual closure uses a common signed arm-A validation residual map for all three "
        "post-transition states; feature closure uses the same fixed-latent EMA map.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    cli_common.add_common_args(parser)
    parser.set_defaults(
        expected_training_state_sha256=FORMAL_HASHES["training_state"],
        expected_checkpoint_sha256=FORMAL_HASHES["checkpoint"],
        expected_batch_file_sha256=FORMAL_HASHES["batch_file"],
        out=HERE,
    )
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--fixed-latent-file", type=Path)
    parser.add_argument("--skip-observables", action="store_true",
                        help="Debug only; formal runs must include observables")
    parser.add_argument("--legacy-rollout", type=Path, default=LEGACY_ROLLOUT)
    parser.add_argument(
        "--expected-batch-tensor-receipts-sha256",
        default=FORMAL_HASHES["batch_tensor_receipts"],
        help=(
            "Canonical SHA256 of all frozen microbatch image/label/t/noise/"
            "dropout tensor receipts; validates content across container rebuilds"
        ),
    )
    parser.add_argument(
        "--full-summary", action="store_true",
        help="Include full per-step replay telemetry; intended for external archives",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    determinism = cli_common.configure_determinism()
    legacy = audit_legacy_rollout(args.legacy_rollout)
    assets = cli_common.source_assets(args)
    assets["runtime_determinism"] = determinism
    assets["implementation"][Path(__file__).name] = cli_common.sha256_file(Path(__file__))
    source = cli_common.load_algorithmic_state(args)
    batches = cli_common.load_frozen_batches(args, source.loss_fn)
    batch_tensor_receipts = [_batch_receipt(item) for item in batches]
    batch_tensor_receipts_sha256 = _canonical_sha256(batch_tensor_receipts)
    if (args.expected_batch_tensor_receipts_sha256 is not None
            and batch_tensor_receipts_sha256
            != args.expected_batch_tensor_receipts_sha256):
        raise RuntimeError(
            "frozen batch tensor receipt SHA256 mismatch: "
            f"{batch_tensor_receipts_sha256} != "
            f"{args.expected_batch_tensor_receipts_sha256}")
    assets["batch_tensor_receipts"] = {
        "microbatch_count": sum(
            item["microbatch_count"] for item in batch_tensor_receipts),
        "canonical_sha256": batch_tensor_receipts_sha256,
        "expected_canonical_sha256": (
            args.expected_batch_tensor_receipts_sha256),
        "matched": (
            args.expected_batch_tensor_receipts_sha256 is None
            or batch_tensor_receipts_sha256
            == args.expected_batch_tensor_receipts_sha256),
        "scope": "images, labels, t, noise, and dropout RNG state",
    }
    latent = None
    if args.fixed_latent_file is not None:
        latent = torch.load(args.fixed_latent_file, map_location=args.device,
                            weights_only=False)
        if isinstance(latent, Mapping):
            latent = latent.get("latents")
        if not isinstance(latent, torch.Tensor):
            raise RuntimeError("fixed latent must be a tensor or {'latents': tensor}")
        assets["fixed_latent"] = {
            "path": str(args.fixed_latent_file.resolve()),
            "sha256": cli_common.sha256_file(args.fixed_latent_file),
        }
    rows, receipt = run_exact_decomposition(
        source, batches, steps=args.steps, latent=latent,
        include_observables=not args.skip_observables,
    )
    if args.steps != 64 or args.skip_observables:
        receipt["status"] = "DEBUG_ONLY"
    summary = build_summary(
        rows, receipt, legacy, assets,
        include_full_receipt=args.full_summary,
    )
    write_csv(args.out / "forcing_feedback_per_step_v2.csv", rows)
    write_json(args.out / "forcing_feedback_summary_v2.json", summary)
    write_report(args.out / "FORCING_FEEDBACK_REPORT_V2.md", summary)
    return 0 if receipt["status"] in {"PASS", "DEBUG_ONLY"} else 3


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
