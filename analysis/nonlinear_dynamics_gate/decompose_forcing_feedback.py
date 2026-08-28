#!/usr/bin/env python3
"""Exact forcing/feedback decomposition of a matched A/B/C/D continuation.

For X in {B,C,D}, this runner evaluates, at every k,

    b_k^X = Phi_X(z_k^A, xi_k) - Phi_A(z_k^A, xi_k)
    R_k^X = Phi_X(z_k^X, xi_k) - Phi_X(z_k^A, xi_k)

and checks the telescoping identity

    z_{k+1}^X - z_{k+1}^A = b_k^X + R_k^X.

The same three-point subtraction is applied to two common observables: a
signed residual vector and fixed-latent EMA features.  No norm ratio is
reported as a contribution percentage because b and R can cancel.
"""
from __future__ import annotations

import argparse
import csv
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
}
LEGACY_ROLLOUT = (
    REPO_ROOT / "analysis/operator_clock_gate/results/raw_receipts/formal-20260826/"
    "results/matched/matched_micro_rollout.json"
)
CSV_FIELDS = (
    "arm", "k", "next_k", "space", "block", "coordinate_count",
    "b_norm", "R_norm", "delta_norm", "cos_b_R", "cos_b_delta",
    "cos_R_delta", "R_over_b_diagnostic", "closure_l2",
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


def exact_three_point_metrics(
    baseline: Mapping[str, torch.Tensor],
    counterfactual: Mapping[str, torch.Tensor],
    actual: Mapping[str, torch.Tensor],
    *,
    closure_atol: float = 1e-12,
    closure_rtol: float = 1e-12,
) -> dict[str, Any]:
    """Compute b, R and delta metrics without materializing a flat vector."""
    if set(baseline) != set(counterfactual) or set(baseline) != set(actual):
        raise RuntimeError("three-point coordinate schemas differ")
    sums = {key: 0.0 for key in ("b2", "R2", "d2", "bR", "bd", "Rd", "c2")}
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
            sums["b2"] += float(b.square().sum())
            sums["R2"] += float(feedback.square().sum())
            sums["d2"] += float(delta.square().sum())
            sums["bR"] += float((b * feedback).sum())
            sums["bd"] += float((b * delta).sum())
            sums["Rd"] += float((feedback * delta).sum())
            sums["c2"] += float(closure.square().sum())
            if closure.numel():
                closure_max = max(closure_max, float(closure.abs().max()))
            coordinate_count += int(a.numel())
    b_norm = math.sqrt(sums["b2"])
    feedback_norm = math.sqrt(sums["R2"])
    delta_norm = math.sqrt(sums["d2"])
    closure_l2 = math.sqrt(sums["c2"])
    closure_limit = float(closure_atol) + float(closure_rtol) * delta_norm
    return {
        "coordinate_count": coordinate_count,
        "b_norm": b_norm,
        "R_norm": feedback_norm,
        "delta_norm": delta_norm,
        "cos_b_R": _safe_cos(sums["bR"], b_norm, feedback_norm),
        "cos_b_delta": _safe_cos(sums["bd"], b_norm, delta_norm),
        "cos_R_delta": _safe_cos(sums["Rd"], feedback_norm, delta_norm),
        "R_over_b_diagnostic": (feedback_norm / b_norm if b_norm else None),
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
) -> dict[str, Any]:
    return {
        "arm": arm, "k": k, "next_k": k + 1,
        "space": space, "block": block,
        **exact_three_point_metrics(baseline, counterfactual, actual),
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
    state_hashes = [{"k": 0, "by_arm": {arm: branches[arm].sha256() for arm in "ABCD"}}]
    with preserved_rng():
        for k in range(steps):
            batch = batches[k % len(batches)]
            pairing_seed = int(batch.audit_id) + k
            z_a = branches["A"]
            a_after, a_telemetry = _transition(
                z_a, batch, "A", pairing_seed, clone_input=True)
            a_blocks = _state_tensor_blocks(a_after)
            a_observables = (observable_vectors(a_after, batches[0], latent)
                             if include_observables else {})
            arm_receipts = {}
            for arm in "BCD":
                # Both Phi_A(zA) and Phi_X(zA) receive clones of exactly zA,
                # so optimizer moments, optimizer step and AMP state are paired.
                counterfactual, cf_telemetry = _transition(
                    z_a, batch, arm, pairing_seed, clone_input=True)
                actual_after, actual_telemetry = _transition(
                    branches[arm], batch, arm, pairing_seed, clone_input=False)
                cf_blocks = _state_tensor_blocks(counterfactual)
                actual_blocks = _state_tensor_blocks(actual_after)
                if set(a_blocks) != set(cf_blocks) or set(a_blocks) != set(actual_blocks):
                    raise RuntimeError("augmented state block schemas differ")
                for block in a_blocks:
                    rows.append(_row(
                        arm, k, "state", block, a_blocks[block],
                        cf_blocks[block], actual_blocks[block]))
                if include_observables:
                    cf_observables = observable_vectors(counterfactual, batches[0], latent)
                    actual_observables = observable_vectors(actual_after, batches[0], latent)
                    for block in ("residual", "feature"):
                        rows.append(_row(
                            arm, k, "observable", block, a_observables[block],
                            cf_observables[block], actual_observables[block]))
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
        "schema_version": 1,
        "kind": "exact_nonlinear_forcing_feedback",
        "definition": {
            "forcing": "Phi_X(z_k^A,xi_k)-Phi_A(z_k^A,xi_k)",
            "feedback": "Phi_X(z_k^X,xi_k)-Phi_X(z_k^A,xi_k)",
            "closure": "z_{k+1}^X-z_{k+1}^A=b_k^X+R_k^X",
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
        if cancellation_fraction >= 0.5:
            label = "forcing_feedback_cancellation"
        elif forcing_fraction >= 0.8:
            label = "persistent_forcing_accumulation"
        elif (late_feedback_fraction >= 0.75 and late_ratio is not None
              and late_ratio > 1.0 and late_alignment is not None
              and late_alignment >= 0.8
              and (early_ratio is None or late_ratio > early_ratio)):
            label = "trajectory_feedback_amplification"
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
        }
    return output


def build_summary(
    rows: Sequence[Mapping[str, Any]], receipt: Mapping[str, Any],
    legacy_audit: Mapping[str, Any], assets: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": receipt["status"],
        "task_1_existing_rollout_state_audit": dict(legacy_audit),
        "instrumentation": {
            "kind": "64-step matched continuation from frozen source",
            "scientific_scope": "local counterfactual instrumentation; not a new experiment",
            "assets": dict(assets),
            "replay_receipt": {
                "state_hashes_k_0_through_horizon": receipt[
                    "state_hashes_k_0_through_horizon"],
                "step_replay_receipts": receipt["step_replay_receipts"],
            },
        },
        "exact_closure": {
            "all_pass": receipt["all_exact_closures_pass"],
            "max_l2": receipt["max_closure_l2"],
            "max_relative": receipt["max_closure_relative"],
        },
        "mechanism_by_arm_and_block": _mechanism_summary(rows),
        "mechanism_decision_rules": {
            "persistent_forcing_accumulation": (
                "R/b <= 0.25 on at least 80% of steps"
            ),
            "trajectory_feedback_amplification": (
                "in the last quarter R>b on at least 75% of steps, median R/b>1, "
                "median cos(R,delta)>=0.8, and late median R/b exceeds early median"
            ),
            "forcing_feedback_cancellation": (
                "cos(b,R)<=-0.9 and both ||b|| and ||R|| >= ||delta|| on at least "
                "50% of steps"
            ),
        },
        "interpretation_guard": (
            "R_over_b is a scale diagnostic, never a contribution percentage; "
            "forcing and feedback may cancel."
        ),
        "run_receipt": dict(receipt),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    legacy = summary["task_1_existing_rollout_state_audit"]
    closure = summary["exact_closure"]
    lines = [
        "# Exact nonlinear forcing–feedback decomposition",
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
        "## Mechanism diagnostics",
        "",
        "| arm | space | block | classification | early median R/b | late median R/b | late cos(R,Δ) |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for item in summary["mechanism_by_arm_and_block"].values():
        def fmt(value: Any) -> str:
            return "NA" if value is None else f"{float(value):.4g}"
        lines.append(
            f"| {item['arm']} | {item['space']} | {item['block']} | "
            f"{item['classification']} | {fmt(item['early_quarter_median_R_over_b'])} | "
            f"{fmt(item['late_quarter_median_R_over_b'])} | "
            f"{fmt(item['late_quarter_median_cos_R_delta'])} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "`R/b` is reported only as a scale diagnostic. It is not a contribution "
        "percentage: large forcing and feedback terms can be nearly antiparallel and cancel.",
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
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    determinism = cli_common.configure_determinism()
    legacy = audit_legacy_rollout(args.legacy_rollout)
    assets = cli_common.source_assets(args)
    assets["runtime_determinism"] = determinism
    assets["implementation"][Path(__file__).name] = cli_common.sha256_file(Path(__file__))
    source = cli_common.load_algorithmic_state(args)
    batches = cli_common.load_frozen_batches(args, source.loss_fn)
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
    summary = build_summary(rows, receipt, legacy, assets)
    write_csv(args.out / "forcing_feedback_per_step.csv", rows)
    write_json(args.out / "forcing_feedback_summary.json", summary)
    write_report(args.out / "FORCING_FEEDBACK_REPORT.md", summary)
    return 0 if receipt["status"] in {"PASS", "DEBUG_ONLY"} else 3


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
