"""Core machinery for field and full-state ECT Jacobian audits.

The important distinction in this module is between three maps:

``squared_gn_operator_jvp``
    A deliberately simplified squared-pair baseline.  It is never labelled as
    the ECT training Jacobian.

``field_jvp``
    A central finite difference of the *recompute-and-detach* gradient map.
    Both the online and target forwards are rerun at every perturbed parameter
    value, and the newly computed target is detached inside that forward.

``algorithmic_jvp``
    A central finite difference of a complete optimizer step, including model
    parameters, optimizer moments, GradScaler state and EMA parameters.

All public audits are non-committing: source objects and process RNG streams
are hashed/snapshotted before the audit and verified after it.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

from training.schedules import get_schedule


ARM_SPECS: dict[str, dict[str, Any]] = {
    "A": {"target_scale": 1.0, "denominator_scale": 1.0,
          "meaning": "native_baseline"},
    "B": {"target_scale": 1.1, "denominator_scale": 1.1,
          "meaning": "native_probe"},
    "C": {"target_scale": 1.1, "denominator_scale": 1.0,
          "meaning": "target_geometry_only"},
    "D": {"target_scale": 1.0, "denominator_scale": 1.1,
          "meaning": "loss_weighting_only"},
}
DEFAULT_EPSILONS = (1e-2, 3e-3, 1e-3, 3e-4)


def _tensor_bytes(value: torch.Tensor) -> bytes:
    value = value.detach().cpu().contiguous()
    return (str(value.dtype).encode() + str(tuple(value.shape)).encode()
            + value.reshape(-1).view(torch.uint8).numpy().tobytes())


def state_sha256(value: Any) -> str:
    """Stable content hash for nested state containing tensors."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            digest.update(b"tensor\0")
            digest.update(_tensor_bytes(item))
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda x: repr(x)):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + b"\0")
            for child in item:
                visit(child)
        elif isinstance(item, (str, int, float, bool, type(None))):
            digest.update(repr(item).encode() + b"\0")
        else:
            # State dicts should have reduced custom objects already.  This
            # fallback is useful for simple protocol metadata only.
            digest.update(type(item).__qualname__.encode() + b"\0")
            digest.update(repr(item).encode() + b"\0")

    visit(value)
    return digest.hexdigest()


def tensor_map_sha256(values: Mapping[str, torch.Tensor]) -> str:
    return state_sha256({key: values[key] for key in sorted(values)})


def _capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": ([item.clone() for item in torch.cuda.get_rng_state_all()]
                       if torch.cuda.is_available() else None),
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def rng_sha256(state: Mapping[str, Any] | None = None) -> str:
    state = _capture_rng() if state is None else state
    # numpy/python states contain arrays and tuples; torch.save is used only in
    # memory and gives a deterministic encoding for a fixed PyTorch runtime.
    stream = io.BytesIO()
    torch.save(dict(state), stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


@contextlib.contextmanager
def preserved_rng(seed: int | None = None):
    """Run with a paired RNG stream and restore every process RNG afterward."""
    original = _capture_rng()
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        yield
    finally:
        _restore_rng(original)


def get_device_rng_state(device: torch.device) -> torch.Tensor:
    return (torch.cuda.get_rng_state(device) if device.type == "cuda"
            else torch.get_rng_state()).clone()


def set_device_rng_state(state: torch.Tensor, device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    else:
        torch.set_rng_state(state)


@dataclass(frozen=True)
class AuditBatch:
    """A minibatch with all stochastic ECT draws frozen in advance."""

    images: torch.Tensor
    labels: torch.Tensor
    t: torch.Tensor
    noise: torch.Tensor
    dropout_rng_state: torch.Tensor
    audit_id: int

    def clone(self) -> "AuditBatch":
        return AuditBatch(
            images=self.images.detach().clone(),
            labels=self.labels.detach().clone(),
            t=self.t.detach().clone(),
            noise=self.noise.detach().clone(),
            dropout_rng_state=self.dropout_rng_state.detach().clone(),
            audit_id=self.audit_id,
        )


@dataclass(frozen=True)
class AuditBatchGroup:
    """One optimizer batch represented by production-sized microbatches."""

    microbatches: tuple[AuditBatch, ...]
    audit_id: int

    def __post_init__(self):
        if not self.microbatches:
            raise ValueError("an audit batch group must contain microbatches")
        if any(item.audit_id != self.audit_id for item in self.microbatches):
            raise ValueError("all microbatches must share the group audit_id")


def _microbatches(batch: AuditBatch | AuditBatchGroup) -> tuple[AuditBatch, ...]:
    return batch.microbatches if isinstance(batch, AuditBatchGroup) else (batch,)


def _flatten_batches(
    batches: Sequence[AuditBatch | AuditBatchGroup],
) -> list[AuditBatch]:
    return [micro for batch in batches for micro in _microbatches(batch)]


def freeze_batches(
    raw_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    loss_template: Any,
    audit_ids: Sequence[int],
) -> list[AuditBatch]:
    if len(raw_batches) != len(audit_ids) or not raw_batches:
        raise ValueError("raw_batches and audit_ids must have the same non-zero length")
    frozen = []
    original = _capture_rng()
    try:
        for (images, labels), audit_id in zip(raw_batches, audit_ids):
            torch.manual_seed(int(audit_id))
            if images.device.type == "cuda":
                torch.cuda.manual_seed_all(int(audit_id))
            t = (torch.randn(
                images.shape[0], 1, 1, 1, device=images.device,
                dtype=images.dtype,
            ) * float(loss_template.P_std) + float(loss_template.P_mean)).exp()
            noise = torch.randn_like(images)
            frozen.append(AuditBatch(
                images=images.detach().clone(),
                labels=labels.detach().clone(),
                t=t.detach().clone(),
                noise=noise.detach().clone(),
                dropout_rng_state=get_device_rng_state(images.device),
                audit_id=int(audit_id),
            ))
    finally:
        _restore_rng(original)
    return frozen


def freeze_batch_groups(
    raw_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    loss_template: Any,
    audit_ids: Sequence[int],
    *,
    microbatch_size: int,
) -> list[AuditBatchGroup]:
    """Freeze four optimizer batches as sequential accumulation microbatches."""
    if microbatch_size < 1:
        raise ValueError("microbatch_size must be positive")
    groups = []
    original = _capture_rng()
    try:
        for (images, labels), audit_id in zip(raw_batches, audit_ids):
            if images.shape[0] % microbatch_size:
                raise ValueError("batch size must be divisible by microbatch_size")
            torch.manual_seed(int(audit_id))
            if images.device.type == "cuda":
                torch.cuda.manual_seed_all(int(audit_id))
            micros = []
            for start in range(0, images.shape[0], microbatch_size):
                image_micro = images[start:start + microbatch_size]
                label_micro = labels[start:start + microbatch_size]
                t = (torch.randn(
                    image_micro.shape[0], 1, 1, 1, device=images.device,
                    dtype=images.dtype,
                ) * float(loss_template.P_std) + float(loss_template.P_mean)).exp()
                noise = torch.randn_like(image_micro)
                micros.append(AuditBatch(
                    images=image_micro.detach().clone(),
                    labels=label_micro.detach().clone(),
                    t=t.detach().clone(), noise=noise.detach().clone(),
                    dropout_rng_state=get_device_rng_state(images.device),
                    audit_id=int(audit_id),
                ))
            groups.append(AuditBatchGroup(tuple(micros), int(audit_id)))
    finally:
        _restore_rng(original)
    return groups


def _schedule(loss_template: Any, scale: float):
    return get_schedule(
        "global_sigmoid", q=float(loss_template.q), k=float(loss_template.k),
        b=float(loss_template.b), global_gap_scale=float(scale),
    )


def _net_forward(net: torch.nn.Module, x: torch.Tensor, sigma: torch.Tensor,
                 labels: torch.Tensor) -> torch.Tensor:
    try:
        return net(x, sigma, labels, augment_labels=None)
    except TypeError as exc:
        if "augment_labels" not in str(exc):
            raise
        return net(x, sigma, labels)


def ect_pair(
    net: torch.nn.Module,
    loss_template: Any,
    batch: AuditBatch,
    arm: str,
    *,
    detach_target: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Recompute both branches for one frozen draw.

    ``detach_target=True`` is the actual ECT gradient-field definition.  The
    target is computed anew at the current parameters and detached only inside
    this invocation; no detached scalar loss or cached target is reused.
    """
    if arm not in ARM_SPECS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {tuple(ARM_SPECS)}")
    spec = ARM_SPECS[arm]
    target_schedule = _schedule(loss_template, spec["target_scale"])
    denominator_schedule = _schedule(loss_template, spec["denominator_scale"])
    r_target = target_schedule.compute_r(t=batch.t, stage=int(loss_template.stage))
    r_denominator = denominator_schedule.compute_r(
        t=batch.t, stage=int(loss_template.stage))
    denominator = batch.t - r_denominator
    if not bool(torch.isfinite(denominator).all()) or not bool((denominator > 0).all()):
        raise RuntimeError("ECT denominator must be positive and finite")
    device = batch.images.device
    set_device_rng_state(batch.dropout_rng_state, device)
    online = _net_forward(
        net, batch.images + batch.noise * batch.t, batch.t, batch.labels)
    target_input = batch.images + batch.noise * r_target
    if detach_target:
        with torch.no_grad():
            set_device_rng_state(batch.dropout_rng_state, device)
            target = _net_forward(net, target_input, r_target, batch.labels)
    else:
        set_device_rng_state(batch.dropout_rng_state, device)
        target = _net_forward(net, target_input, r_target, batch.labels)
    target = torch.nan_to_num(target)
    positive = r_target > 0
    target = positive * target + (~positive) * batch.images
    raw_sq = (online - target).square().reshape(batch.images.shape[0], -1).sum(dim=1)
    c = float(loss_template.c)
    numerator = torch.sqrt(raw_sq + c * c) - c if c > 0 else torch.sqrt(raw_sq)
    loss = numerator / denominator.flatten()
    detail = {
        "audit_id": batch.audit_id,
        "arm": arm,
        "target_recomputed": True,
        "target_detached_in_this_forward": detach_target,
        "target_requires_grad": bool(target.requires_grad),
        "target_sha256": state_sha256(target),
        "online_sha256": state_sha256(online),
        "residual_l2": raw_sq.detach().sqrt().double().cpu(),
        "loss": loss.detach().double().cpu(),
        "denominator": denominator.detach().double().cpu().flatten(),
    }
    return loss, detail


def _named_parameters(net: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return dict(net.named_parameters())


def _gradient_field(
    net: torch.nn.Module,
    loss_template: Any,
    batches: Sequence[AuditBatch | AuditBatchGroup],
    arm: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Evaluate g_G(theta) by complete recomputation on a disposable model."""
    net.zero_grad(set_to_none=True)
    details = []
    microbatches = _flatten_batches(batches)
    total_samples = sum(batch.images.shape[0] for batch in microbatches)
    for batch in microbatches:
        loss, detail = ect_pair(net, loss_template, batch, arm, detach_target=True)
        (loss.sum() / total_samples).backward()
        details.append(detail)
    field = {}
    for name, parameter in net.named_parameters():
        value = (torch.zeros_like(parameter) if parameter.grad is None else parameter.grad)
        field[name] = value.detach().double().cpu().clone()
    if not field or not all(bool(torch.isfinite(value).all()) for value in field.values()):
        raise RuntimeError("gradient field is empty or non-finite")
    return field, {
        "forward_count": 2 * len(microbatches),
        "target_recompute_count": len(microbatches),
        "all_targets_detached": all(not item["target_requires_grad"] for item in details),
        "target_hashes": [item["target_sha256"] for item in details],
    }


def parameter_vector(net: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().double().cpu().clone()
            for name, value in net.named_parameters()}


def random_direction_like(
    values: Mapping[str, torch.Tensor], seed: int, *, normalize: bool = True,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    direction = {
        name: torch.randn(value.shape, dtype=torch.float64, generator=generator)
        for name, value in sorted(values.items())
    }
    if normalize:
        norm = math.sqrt(sum(float(item.square().sum()) for item in direction.values()))
        if norm <= 0:
            raise RuntimeError("random direction unexpectedly has zero norm")
        direction = {name: item / norm for name, item in direction.items()}
    return direction


def state_relative_direction_like(
    values: Mapping[str, torch.Tensor], seed: int, *, scale_floor: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """Draw a reproducible direction with numerically resolvable coordinates.

    A globally unit-normalized vector is unusable for million-parameter FP32
    networks: ``epsilon*u_i`` falls below one ULP.  This preregistered
    convention gives each tensor a Gaussian direction whose RMS equals that
    tensor's state RMS (with a small floor for zero-initialized coordinates).
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    direction = {}
    for name, value in sorted(values.items()):
        draw = torch.randn(value.shape, dtype=torch.float64, generator=generator)
        draw_rms = math.sqrt(float(draw.square().mean())) if draw.numel() else 0.0
        value_rms = math.sqrt(float(value.detach().double().square().mean()))
        scale = max(value_rms, float(scale_floor))
        if draw_rms <= 0:
            draw = torch.ones_like(draw)
            draw_rms = 1.0
        direction[name] = draw * (scale / draw_rms)
    return direction


def safe_algorithmic_direction_like(
    values: Mapping[str, torch.Tensor], seed: int, *, max_epsilon: float,
) -> dict[str, torch.Tensor]:
    """Draw a unit direction without leaving positive optimizer coordinates.

    RAdam second moments and GradScaler scale live in a positive state space.
    Zero second-moment coordinates therefore receive a zero two-sided tangent;
    the remaining global direction is shrunk only when needed so every frozen
    ``S +/- epsilon*u`` point is admissible.
    """
    direction = state_relative_direction_like(values, seed)
    constrained = [key for key in values
                   if key.endswith(".exp_avg_sq") or key == "scaler.scale"]
    for key in constrained:
        value = values[key].detach().double()
        proposal = direction[key]
        zero = value <= 0
        proposal = proposal.masked_fill(zero, 0.0)
        bound = 0.5 * value / abs(float(max_epsilon))
        proposal = proposal.sign() * torch.minimum(proposal.abs(), bound)
        direction[key] = proposal
    if vector_l2(direction) <= 0:
        raise RuntimeError("algorithmic direction has zero norm")
    return direction


def _validate_vector_pair(values: Mapping[str, torch.Tensor],
                          direction: Mapping[str, torch.Tensor]) -> None:
    if set(values) != set(direction) or not values:
        raise ValueError("direction keys must exactly match the state-vector keys")
    for key in values:
        if values[key].shape != direction[key].shape:
            raise ValueError(f"direction shape mismatch for {key}")


def _perturb_parameters(net: torch.nn.Module,
                        direction: Mapping[str, torch.Tensor], scale: float) -> None:
    params = _named_parameters(net)
    if set(params) != set(direction):
        raise ValueError("parameter direction does not match model parameters")
    with torch.no_grad():
        for name, parameter in params.items():
            parameter.add_(direction[name].to(parameter), alpha=float(scale))


def _difference(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor],
                denominator: float) -> dict[str, torch.Tensor]:
    if set(left) != set(right):
        missing = sorted(set(left) ^ set(right))
        raise RuntimeError(f"state coordinate mismatch across FD branches: {missing[:8]}")
    return {key: (left[key].double() - right[key].double()) / denominator
            for key in sorted(left)}


def vector_l2(values: Mapping[str, torch.Tensor]) -> float:
    return math.sqrt(sum(float(value.detach().double().square().sum())
                         for value in values.values()))


def relative_difference(left: Mapping[str, torch.Tensor],
                        right: Mapping[str, torch.Tensor]) -> float:
    delta = _difference(left, right, 1.0)
    return vector_l2(delta) / max(vector_l2(right), torch.finfo(torch.float64).tiny)


def fd_convergence(estimates: Mapping[float, Mapping[str, torch.Tensor]],
                   *, tolerance: float = 5e-2) -> dict[str, Any]:
    if len(estimates) < 3:
        raise ValueError("finite-difference convergence requires at least three epsilons")
    epsilons = sorted((float(item) for item in estimates), reverse=True)
    if any(not math.isfinite(item) or item <= 0 for item in epsilons):
        raise ValueError("epsilons must be positive finite numbers")
    comparisons = []
    for coarse, fine in zip(epsilons[:-1], epsilons[1:]):
        change = relative_difference(estimates[fine], estimates[coarse])
        comparisons.append({
            "coarse_epsilon": coarse,
            "fine_epsilon": fine,
            "relative_change": change,
        })
    best = min(comparisons, key=lambda item: item["relative_change"])
    finest = comparisons[-1]
    return {
        "epsilons": epsilons,
        "comparisons": comparisons,
        "best_adjacent_pair": best,
        "finest_adjacent_pair": finest,
        "tolerance": float(tolerance),
        "passed": bool(math.isfinite(finest["relative_change"])
                       and finest["relative_change"] <= tolerance),
        "selection_rule": (
            "smallest preregistered epsilon; convergence is the adjacent "
            "change from the next-smallest epsilon"),
    }


def field_jvp(
    source_net: torch.nn.Module,
    loss_template: Any,
    batches: Sequence[AuditBatch | AuditBatchGroup],
    direction: Mapping[str, torch.Tensor],
    *,
    arm: str = "A",
    epsilons: Sequence[float] = DEFAULT_EPSILONS,
    learning_rate: float | None = None,
    convergence_tolerance: float = 5e-2,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Central FD of the true recompute-and-detach field Jacobian."""
    source_hash = state_sha256(source_net.state_dict())
    source_rng = rng_sha256()
    base = parameter_vector(source_net)
    _validate_vector_pair(base, direction)
    estimates: dict[float, dict[str, torch.Tensor]] = {}
    branch_receipts = []
    with preserved_rng():
        for epsilon in epsilons:
            epsilon = float(epsilon)
            plus = copy.deepcopy(source_net).train().requires_grad_(True)
            minus = copy.deepcopy(source_net).train().requires_grad_(True)
            _perturb_parameters(plus, direction, epsilon)
            _perturb_parameters(minus, direction, -epsilon)
            plus_field, plus_detail = _gradient_field(
                plus, copy.deepcopy(loss_template), batches, arm)
            minus_field, minus_detail = _gradient_field(
                minus, copy.deepcopy(loss_template), batches, arm)
            estimates[epsilon] = _difference(plus_field, minus_field, 2 * epsilon)
            branch_receipts.append({
                "epsilon": epsilon,
                "plus": plus_detail,
                "minus": minus_detail,
                "paired_target_recomputation": (
                    plus_detail["target_recompute_count"]
                    == minus_detail["target_recompute_count"]
                    == len(_flatten_batches(batches))),
            })
    convergence = fd_convergence(estimates, tolerance=convergence_tolerance)
    selected_epsilon = min(float(item) for item in epsilons)
    selected = estimates[selected_epsilon]
    operator = None
    if learning_rate is not None:
        operator = {name: direction[name].double() - float(learning_rate) * selected[name]
                    for name in selected}
    after_hash = state_sha256(source_net.state_dict())
    after_rng = rng_sha256()
    receipt = {
        "schema_version": 1,
        "predictor": "recompute_and_detach_field_jacobian",
        "arm": arm,
        "arm_spec": ARM_SPECS[arm],
        "selected_epsilon": selected_epsilon,
        "convergence": convergence,
        "branch_receipts": branch_receipts,
        "field_jvp_l2": vector_l2(selected),
        "operator_jvp_l2": vector_l2(operator) if operator is not None else None,
        "learning_rate": learning_rate,
        "source_state_sha256_before": source_hash,
        "source_state_sha256_after": after_hash,
        "source_rng_sha256_before": source_rng,
        "source_rng_sha256_after": after_rng,
        "source_preserved": source_hash == after_hash and source_rng == after_rng,
        "definition_guard": (
            "Every theta+/-epsilon branch reruns online and target forwards and "
            "detaches the freshly recomputed target inside that branch."),
    }
    receipt["status"] = ("PASS" if receipt["source_preserved"]
                         and convergence["passed"] else "FAIL_CLOSED")
    return selected if operator is None else operator, receipt


def squared_gn_operator_jvp(
    source_net: torch.nn.Module,
    loss_template: Any,
    batches: Sequence[AuditBatch | AuditBatchGroup],
    direction: Mapping[str, torch.Tensor],
    *,
    arm: str = "A",
    learning_rate: float,
    output_fd_epsilon: float = 1e-3,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Compute the simplified ``I-eta sum w J_i^T(J_i-J_j)`` baseline."""
    source_hash = state_sha256(source_net.state_dict())
    source_rng = rng_sha256()
    base = parameter_vector(source_net)
    _validate_vector_pair(base, direction)
    accumulated = {name: torch.zeros_like(value) for name, value in base.items()}
    with preserved_rng():
        plus = copy.deepcopy(source_net).train().requires_grad_(True)
        minus = copy.deepcopy(source_net).train().requires_grad_(True)
        _perturb_parameters(plus, direction, output_fd_epsilon)
        _perturb_parameters(minus, direction, -output_fd_epsilon)
        live = copy.deepcopy(source_net).train().requires_grad_(True)
        microbatches = _flatten_batches(batches)
        total_samples = sum(batch.images.shape[0] for batch in microbatches)
        for batch in microbatches:
            # Live target here is intentional: only its output directional
            # derivative enters (J_i-J_j)u.  The VJP below is through J_i only.
            _, plus_detail = ect_pair(
                plus, loss_template, batch, arm, detach_target=False)
            _, minus_detail = ect_pair(
                minus, loss_template, batch, arm, detach_target=False)
            # Recompute actual tensors because receipts deliberately store only
            # detached diagnostics.
            spec = ARM_SPECS[arm]
            r = _schedule(loss_template, spec["target_scale"]).compute_r(
                batch.t, stage=int(loss_template.stage))
            set_device_rng_state(batch.dropout_rng_state, batch.images.device)
            online_plus = _net_forward(
                plus, batch.images + batch.noise * batch.t, batch.t, batch.labels)
            set_device_rng_state(batch.dropout_rng_state, batch.images.device)
            target_plus = _net_forward(
                plus, batch.images + batch.noise * r, r, batch.labels)
            set_device_rng_state(batch.dropout_rng_state, batch.images.device)
            online_minus = _net_forward(
                minus, batch.images + batch.noise * batch.t, batch.t, batch.labels)
            set_device_rng_state(batch.dropout_rng_state, batch.images.device)
            target_minus = _net_forward(
                minus, batch.images + batch.noise * r, r, batch.labels)
            positive = r > 0
            target_plus = positive * target_plus + (~positive) * batch.images
            target_minus = positive * target_minus + (~positive) * batch.images
            jdiff_u = ((online_plus - target_plus) - (online_minus - target_minus))
            jdiff_u = jdiff_u.detach() / (2 * float(output_fd_epsilon))
            set_device_rng_state(batch.dropout_rng_state, batch.images.device)
            online_live = _net_forward(
                live, batch.images + batch.noise * batch.t, batch.t, batch.labels)
            denominator = batch.t - _schedule(
                loss_template, spec["denominator_scale"]).compute_r(
                    batch.t, stage=int(loss_template.stage))
            weights = (1.0 / denominator).reshape(
                (batch.images.shape[0],) + (1,) * (online_live.ndim - 1))
            scalar = (online_live * jdiff_u * weights).sum() / total_samples
            grads = torch.autograd.grad(
                scalar, tuple(live.parameters()), allow_unused=True)
            for (name, parameter), grad in zip(live.named_parameters(), grads):
                if grad is not None:
                    accumulated[name].add_(grad.detach().double().cpu())
            del plus_detail, minus_detail
    result = {name: direction[name].double() - float(learning_rate) * accumulated[name]
              for name in accumulated}
    after_hash = state_sha256(source_net.state_dict())
    after_rng = rng_sha256()
    receipt = {
        "schema_version": 1,
        "predictor": "squared_loss_simplified_operator",
        "formula": "I - eta * sum_e p_e w_e J_i^T (J_i-J_j)",
        "arm": arm,
        "learning_rate": float(learning_rate),
        "output_fd_epsilon": float(output_fd_epsilon),
        "jvp_l2": vector_l2(result),
        "source_state_sha256_before": source_hash,
        "source_state_sha256_after": after_hash,
        "source_rng_sha256_before": source_rng,
        "source_rng_sha256_after": after_rng,
        "source_preserved": source_hash == after_hash and source_rng == after_rng,
        "claim_boundary": (
            "Theoretical squared-pair baseline only; it is not the true ECT "
            "recompute-and-detach field or the optimizer state transition."),
    }
    receipt["status"] = "PASS" if receipt["source_preserved"] else "FAIL_CLOSED"
    return result, receipt


@dataclass
class AlgorithmicState:
    net: torch.nn.Module
    optimizer: torch.optim.Optimizer
    ema: torch.nn.Module
    loss_fn: Any
    scaler: Any | None = None
    ema_beta: float = 0.9993

    def clone(self) -> "AlgorithmicState":
        # One deepcopy call preserves optimizer -> parameter aliasing.
        return copy.deepcopy(self)

    def content_state(self) -> dict[str, Any]:
        return {
            "net": self.net.state_dict(),
            "net_gradient_buffers": {
                name: (None if parameter.grad is None
                       else parameter.grad.detach().clone())
                for name, parameter in self.net.named_parameters()
            },
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "loss": (self.loss_fn.schedule_state_dict()
                     if hasattr(self.loss_fn, "schedule_state_dict") else repr(self.loss_fn)),
            "ema_beta": float(self.ema_beta),
        }

    def sha256(self) -> str:
        return state_sha256(self.content_state())

    def continuous_vector(self) -> dict[str, torch.Tensor]:
        """Flatten continuous coordinates of (theta,m,v,scaler,EMA,...)."""
        values: dict[str, torch.Tensor] = {}
        named = dict(self.net.named_parameters())
        for name, parameter in named.items():
            values[f"theta.{name}"] = parameter.detach().double().cpu().clone()
        for name, buffer in self.net.named_buffers():
            if buffer.is_floating_point():
                values[f"net_buffer.{name}"] = buffer.detach().double().cpu().clone()
        param_name = {id(parameter): name for name, parameter in named.items()}
        for parameter, state in self.optimizer.state.items():
            name = param_name.get(id(parameter))
            if name is None:
                raise RuntimeError("optimizer parameter is not owned by state.net")
            for key, value in sorted(state.items()):
                if (isinstance(value, torch.Tensor) and value.is_floating_point()
                        and key != "step"):
                    values[f"optimizer.{name}.{key}"] = value.detach().double().cpu().clone()
        for name, parameter in self.ema.named_parameters():
            values[f"ema.{name}"] = parameter.detach().double().cpu().clone()
        for name, buffer in self.ema.named_buffers():
            if buffer.is_floating_point():
                values[f"ema_buffer.{name}"] = buffer.detach().double().cpu().clone()
        if self.scaler is not None:
            for key, value in sorted(self.scaler.state_dict().items()):
                if key == "scale" and isinstance(value, (float, int)):
                    values[f"scaler.{key}"] = torch.tensor(float(value), dtype=torch.float64)
                elif (key == "scale" and isinstance(value, torch.Tensor)
                      and value.is_floating_point()):
                    values[f"scaler.{key}"] = value.detach().double().cpu().clone()
        return values

    def add_direction_(self, direction: Mapping[str, torch.Tensor], scale: float) -> None:
        current = self.continuous_vector()
        _validate_vector_pair(current, direction)
        named = dict(self.net.named_parameters())
        named_buffers = dict(self.net.named_buffers())
        ema_named = dict(self.ema.named_parameters())
        ema_buffers = dict(self.ema.named_buffers())
        param_name = {id(parameter): name for name, parameter in named.items()}
        with torch.no_grad():
            for name, parameter in named.items():
                parameter.add_(direction[f"theta.{name}"].to(parameter), alpha=float(scale))
            for name, buffer in named_buffers.items():
                if buffer.is_floating_point():
                    buffer.add_(direction[f"net_buffer.{name}"].to(buffer), alpha=float(scale))
            for parameter, opt_state in self.optimizer.state.items():
                name = param_name[id(parameter)]
                for key, value in opt_state.items():
                    vector_key = f"optimizer.{name}.{key}"
                    if vector_key in direction:
                        value.add_(direction[vector_key].to(value), alpha=float(scale))
            for name, parameter in ema_named.items():
                parameter.add_(direction[f"ema.{name}"].to(parameter), alpha=float(scale))
            for name, buffer in ema_buffers.items():
                if buffer.is_floating_point():
                    buffer.add_(direction[f"ema_buffer.{name}"].to(buffer), alpha=float(scale))
        scaler_key = "scaler.scale"
        if scaler_key in direction:
            scaler_state = copy.deepcopy(self.scaler.state_dict())
            original = scaler_state["scale"]
            delta = float(direction[scaler_key]) * float(scale)
            if isinstance(original, torch.Tensor):
                scaler_state["scale"] = original + delta
            else:
                scaler_state["scale"] = float(original) + delta
            if float(scaler_state["scale"]) <= 0:
                raise RuntimeError("GradScaler scale perturbation crossed zero")
            self.scaler.load_state_dict(scaler_state)


def _optimizer_discrete_signature(state: AlgorithmicState) -> dict[str, Any]:
    steps = []
    for item in state.optimizer.state.values():
        if "step" in item:
            step = item["step"]
            steps.append(int(step.item()) if isinstance(step, torch.Tensor) else int(step))
    scaler = state.scaler.state_dict() if state.scaler is not None else {}
    tracker = scaler.get("_growth_tracker")
    if isinstance(tracker, torch.Tensor):
        tracker = int(tracker.item())
    return {
        "optimizer_steps": sorted(steps),
        "scaler_growth_tracker": tracker,
    }


def transition_step(
    state: AlgorithmicState,
    batch: AuditBatch | AuditBatchGroup,
    *,
    arm: str,
    clone_input: bool = True,
) -> tuple[AlgorithmicState, dict[str, Any]]:
    """Apply one production-shaped step.

    The public-safe default clones its input. Internal FD/rollout callers pass
    ``clone_input=False`` only for disposable branches that have already been
    cloned from the immutable source, reducing peak GPU memory substantially.
    """
    result = state.clone() if clone_input else state
    result.net.train().requires_grad_(True)
    result.optimizer.zero_grad(set_to_none=True)
    before_params = parameter_vector(result.net)
    micros = _microbatches(batch)
    total_samples = sum(item.images.shape[0] for item in micros)
    loss_sum = 0.0
    details = []
    scale_before = (float(result.scaler.get_scale())
                    if result.scaler is not None else 1.0)
    for micro in micros:
        loss, detail = ect_pair(
            result.net, result.loss_fn, micro, arm, detach_target=True)
        objective = loss.sum() / total_samples
        if result.scaler is not None:
            result.scaler.scale(objective).backward()
        else:
            objective.backward()
        loss_sum += float(loss.detach().double().sum().cpu())
        details.append(detail)
    if result.scaler is not None:
        result.scaler.unscale_(result.optimizer)
    raw_grad_sq = 0.0
    for parameter in result.net.parameters():
        if parameter.grad is not None:
            raw_grad_sq += float(parameter.grad.detach().double().square().sum())
            torch.nan_to_num(parameter.grad, nan=0.0, posinf=1e5, neginf=-1e5,
                             out=parameter.grad)
    if result.scaler is not None:
        result.scaler.step(result.optimizer)
        result.scaler.update()
        scale_after = float(result.scaler.get_scale())
        step_skipped = scale_after < scale_before
    else:
        result.optimizer.step()
        scale_after = scale_before
        step_skipped = False
    after_params = parameter_vector(result.net)
    update = _difference(after_params, before_params, 1.0)
    with torch.no_grad():
        for ema_parameter, parameter in zip(result.ema.parameters(), result.net.parameters()):
            ema_parameter.copy_(parameter.detach().lerp(
                ema_parameter, float(result.ema_beta)))
    return result, {
        "audit_id": batch.audit_id,
        "arm": arm,
        "loss_mean": loss_sum / total_samples,
        "microbatch_count": len(micros),
        "optimizer_batch_size": total_samples,
        "raw_gradient_l2": math.sqrt(raw_grad_sq),
        "raw_update_l2": vector_l2(update),
        "amp_enabled": result.scaler is not None,
        "grad_scale_before": scale_before,
        "grad_scale_after": scale_after,
        "step_skipped": bool(step_skipped),
        "target_recomputed": all(item["target_recomputed"] for item in details),
        "target_recompute_count": len(details),
        "target_detached_in_this_forward": all(
            item["target_detached_in_this_forward"] for item in details),
    }


def algorithmic_jvp(
    source: AlgorithmicState,
    batch: AuditBatch | AuditBatchGroup,
    direction: Mapping[str, torch.Tensor],
    *,
    arm: str = "A",
    epsilons: Sequence[float] = DEFAULT_EPSILONS,
    convergence_tolerance: float = 5e-2,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Central FD of the complete state transition ``D_S Phi_G(S;xi)``."""
    source_hash = source.sha256()
    source_rng = rng_sha256()
    base = source.continuous_vector()
    _validate_vector_pair(base, direction)
    estimates: dict[float, dict[str, torch.Tensor]] = {}
    branches = []
    amp_pairing = True
    discrete_pairing = True
    with preserved_rng():
        for epsilon in epsilons:
            epsilon = float(epsilon)
            plus = source.clone()
            plus.add_direction_(direction, epsilon)
            # The frozen batch stores the exact stochastic draw, while this
            # context also prevents unrelated process RNG from leaking in.
            with preserved_rng(batch.audit_id):
                plus_after, plus_detail = transition_step(
                    plus, batch, arm=arm, clone_input=False)
            plus_vector = plus_after.continuous_vector()
            plus_discrete = _optimizer_discrete_signature(plus_after)
            del plus, plus_after

            minus = source.clone()
            minus.add_direction_(direction, -epsilon)
            with preserved_rng(batch.audit_id):
                minus_after, minus_detail = transition_step(
                    minus, batch, arm=arm, clone_input=False)
            minus_vector = minus_after.continuous_vector()
            minus_discrete = _optimizer_discrete_signature(minus_after)
            del minus, minus_after
            estimates[epsilon] = _difference(plus_vector, minus_vector, 2 * epsilon)
            paired = plus_detail["step_skipped"] == minus_detail["step_skipped"]
            amp_pairing &= paired
            discrete_equal = plus_discrete == minus_discrete
            discrete_pairing &= discrete_equal
            branches.append({
                "epsilon": epsilon,
                "plus": plus_detail,
                "minus": minus_detail,
                "amp_skip_behavior_identical": paired,
                "discrete_state_plus": plus_discrete,
                "discrete_state_minus": minus_discrete,
                "discrete_state_identical": discrete_equal,
            })
    convergence = fd_convergence(estimates, tolerance=convergence_tolerance)
    selected_epsilon = min(float(item) for item in epsilons)
    selected = estimates[selected_epsilon]
    after_hash = source.sha256()
    after_rng = rng_sha256()
    skip_patterns = [
        (item["plus"]["step_skipped"], item["minus"]["step_skipped"])
        for item in branches
    ]
    amp_regime_identical = len(set(skip_patterns)) == 1
    receipt = {
        "schema_version": 1,
        "predictor": "full_algorithmic_state_transition_jacobian",
        "state_coordinates": sorted(base),
        "state_coordinate_count": sum(value.numel() for value in base.values()),
        "arm": arm,
        "arm_spec": ARM_SPECS[arm],
        "selected_epsilon": selected_epsilon,
        "convergence": convergence,
        "branches": branches,
        "jvp_l2": vector_l2(selected),
        "amp_skip_behavior_identical_all_eps": amp_pairing,
        "amp_regime_identical_across_eps": amp_regime_identical,
        "discrete_state_behavior_identical_all_eps": discrete_pairing,
        "source_state_sha256_before": source_hash,
        "source_state_sha256_after": after_hash,
        "source_rng_sha256_before": source_rng,
        "source_rng_sha256_after": after_rng,
        "source_preserved": source_hash == after_hash and source_rng == after_rng,
        "no_in_place_source_pollution": source_hash == after_hash,
    }
    receipt["status"] = (
        "PASS" if receipt["source_preserved"] and amp_pairing
        and amp_regime_identical and discrete_pairing
        and convergence["passed"] else "FAIL_CLOSED")
    return selected, receipt


def _distribution(values: torch.Tensor, *,
                  max_quantile_elements: int = 1_000_000) -> dict[str, Any]:
    values = values.detach().double().cpu().flatten()
    if max_quantile_elements < 1:
        raise ValueError("max_quantile_elements must be positive")
    stride = max(1, math.ceil(values.numel() / max_quantile_elements))
    quantile_values = values[::stride][:max_quantile_elements]
    quantiles = torch.quantile(quantile_values, torch.tensor(
        [0.05, 0.5, 0.95], dtype=torch.float64))
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()), "std": float(values.std(unbiased=False)),
        "p05": float(quantiles[0]), "p50": float(quantiles[1]),
        "p95": float(quantiles[2]), "l2": float(values.norm()),
        "quantile_method": ("exact" if stride == 1
                            else "deterministic_stride_sample"),
        "quantile_sample_count": int(quantile_values.numel()),
        "quantile_stride": stride,
    }


def _projection_summary(values: Mapping[str, torch.Tensor], seeds: Sequence[int]) -> list[float]:
    flat_values = {key: value.detach().double().cpu() for key, value in values.items()}
    result = []
    for seed in seeds:
        direction = random_direction_like(flat_values, seed)
        result.append(sum(float((flat_values[key] * direction[key]).sum())
                          for key in flat_values))
    return result


def _moment_summary(state: AlgorithmicState) -> dict[str, Any]:
    by_key: dict[str, list[torch.Tensor]] = {}
    for item in state.optimizer.state.values():
        for key, value in item.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point() and key != "step":
                by_key.setdefault(key, []).append(value.detach().double().cpu().flatten())
    return {key: _distribution(torch.cat(values)) for key, values in sorted(by_key.items())}


def _fixed_outputs(state: AlgorithmicState, batch: AuditBatch | AuditBatchGroup,
                   latent: torch.Tensor | None) -> dict[str, Any]:
    batch = _microbatches(batch)[0]
    state.ema.eval()
    with torch.no_grad(), preserved_rng(batch.audit_id):
        set_device_rng_state(batch.dropout_rng_state, batch.images.device)
        validation = _net_forward(
            state.ema, batch.images + batch.noise * batch.t, batch.t, batch.labels)
        if latent is None:
            latent = batch.noise
        sigma = batch.t[:latent.shape[0]]
        labels = batch.labels[:latent.shape[0]]
        if sigma.shape[0] != latent.shape[0]:
            sigma = batch.t[:1].expand(latent.shape[0], -1, -1, -1)
            labels = batch.labels[:1].expand(latent.shape[0], *batch.labels.shape[1:])
        features = _net_forward(state.ema, latent * sigma, sigma, labels)
    return {
        "validation_output": _distribution(validation),
        "validation_output_sha256": state_sha256(validation),
        "fixed_latent_sample_features": _distribution(features),
        "fixed_latent_sample_features_sha256": state_sha256(features),
    }


def _residual_profile(state: AlgorithmicState, loss_template: Any,
                      batch: AuditBatch | AuditBatchGroup, arm: str) -> dict[str, float]:
    residuals = []
    with preserved_rng(batch.audit_id), torch.no_grad():
        for micro in _microbatches(batch):
            _, detail = ect_pair(
                state.net, loss_template, micro, arm, detach_target=True)
            residuals.append(detail["residual_l2"])
    return _distribution(torch.cat(residuals))


def matched_micro_rollout(
    source: AlgorithmicState,
    batches: Sequence[AuditBatch | AuditBatchGroup],
    *,
    horizons: Sequence[int] = (1, 4, 16, 64),
    projection_seeds: Sequence[int] = tuple(range(2026082101, 2026082109)),
    arms: Sequence[str] = ("A", "B", "C", "D"),
    fixed_latent: torch.Tensor | None = None,
) -> dict[str, Any]:
    if sorted(set(horizons)) != list(horizons) or not horizons or horizons[0] < 1:
        raise ValueError("horizons must be unique, sorted, positive integers")
    if len(projection_seeds) != 8 or len(set(projection_seeds)) != 8:
        raise ValueError("the formal protocol requires exactly eight fixed projection seeds")
    if not batches:
        raise ValueError("at least one frozen audit batch is required")
    for arm in arms:
        if arm not in ARM_SPECS:
            raise ValueError(f"unknown arm {arm}")
    source_hash = source.sha256()
    source_rng = rng_sha256()
    branches = {arm: source.clone() for arm in arms}
    receipts = {arm: {"steps": [], "horizons": {}} for arm in arms}
    with preserved_rng():
        for step in range(1, max(horizons) + 1):
            batch = batches[(step - 1) % len(batches)]
            for arm in arms:
                with preserved_rng(batch.audit_id + step - 1):
                    branches[arm], telemetry = transition_step(
                        branches[arm], batch, arm=arm, clone_input=False)
                receipts[arm]["steps"].append(telemetry)
            if step in horizons:
                for arm in arms:
                    state = branches[arm]
                    batch0 = batches[0]
                    snapshot = {
                        "parameter_random_projections": _projection_summary(
                            {f"theta.{key}": value for key, value in parameter_vector(state.net).items()},
                            projection_seeds),
                        "ema_parameter_random_projections": _projection_summary(
                            {f"ema.{key}": value for key, value in parameter_vector(state.ema).items()},
                            projection_seeds),
                        "residual_profile": _residual_profile(
                            state, state.loss_fn, batch0, arm),
                        "optimizer_moment_summaries": _moment_summary(state),
                        **_fixed_outputs(state, batch0, fixed_latent),
                        "state_sha256": state.sha256(),
                    }
                    receipts[arm]["horizons"][str(step)] = snapshot
    after_hash = source.sha256()
    after_rng = rng_sha256()
    skip_pairing = []
    for step in range(max(horizons)):
        skips = {arm: receipts[arm]["steps"][step]["step_skipped"] for arm in arms}
        skip_pairing.append({"step": step + 1, "by_arm": skips,
                             "all_identical": len(set(skips.values())) == 1})
    return {
        "schema_version": 1,
        "kind": "matched_counterfactual_micro_rollout",
        "arms": {arm: ARM_SPECS[arm] for arm in arms},
        "audit_minibatch_ids": [batch.audit_id for batch in batches],
        "projection_seeds": list(projection_seeds),
        "horizons": list(horizons),
        "branches": receipts,
        "amp_skip_pairing": skip_pairing,
        "amp_skip_all_arms_identical": all(item["all_identical"] for item in skip_pairing),
        "source_state_sha256_before": source_hash,
        "source_state_sha256_after": after_hash,
        "source_rng_sha256_before": source_rng,
        "source_rng_sha256_after": after_rng,
        "source_preserved": source_hash == after_hash and source_rng == after_rng,
        # Cross-arm skips are an observed consequence of the counterfactual,
        # not an FD-pairing violation.  The +/- pairing gate lives in
        # algorithmic_jvp; rollouts record arm divergence without erasing it.
        "status": ("PASS" if source_hash == after_hash and source_rng == after_rng
                   else "FAIL_CLOSED"),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().double().cpu())
        return value.detach().double().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def central_difference_map(
    function: Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]],
    point: tuple[torch.Tensor, ...],
    direction: tuple[torch.Tensor, ...],
    epsilon: float,
) -> tuple[torch.Tensor, ...]:
    """Small functional helper used by the differentiable-optimizer oracle test."""
    plus = tuple(x + float(epsilon) * u for x, u in zip(point, direction))
    minus = tuple(x - float(epsilon) * u for x, u in zip(point, direction))
    return tuple((a - b) / (2 * float(epsilon))
                 for a, b in zip(function(plus), function(minus)))
