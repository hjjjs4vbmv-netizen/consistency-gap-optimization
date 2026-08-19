"""Calibrate and preflight the q256 g=1.10 RAdam moment transport.

The eight canonical factorial-audit batches are split by sorted rank before
any aggregate is computed.  Ranks 0/2/4/6 calibrate one whole-model scalar
per training seed; ranks 1/3/5/7 are replayed as held-out virtual optimizer
steps.  Every optimizer step occurs on a disposable deep clone.

This program performs no continuation training, sampling, FID, or KID.  It
fails closed when a receipt field, source hash, replay hash, or pre-registered
gate is unavailable.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "analysis"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch  # noqa: E402

import radam_stateful_update_audit as audit_lib  # noqa: E402
import radam_update_gauge as gauge  # noqa: E402


FORMAL_TRAINING_SEEDS = (3, 4, 5)
EXPECTED_BATCHES_PER_SEED = 8
FORMAL_AUDIT_BATCH_IDS = (
    2026081101,
    2026081102,
    2026081103,
    2026081104,
    2026081105,
    2026081106,
    2026081107,
    2026081108,
)
CALIBRATION_RANKS = (0, 2, 4, 6)
HELDOUT_RANKS = (1, 3, 5, 7)
FACTORIAL_CELLS = (
    "observed_real",
    "observed_reset",
    "exact_scalar_real",
    "exact_scalar_reset",
)
ARM_ORDER = ("F", "G", "T", "T_exact")
MOMENT_FIELDS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
A_STAR_DEFINITIONS = {
    "dot(G_1.00,G_1.10)/dot(G_1.00,G_1.00)",
    "dot(G110,G100)/dot(G100,G100)",
    "dot(G110,G100)/||G100||^2",
}

CALIBRATION_FIELDS = (
    "training_seed",
    "audit_batch_id",
    "canonical_rank",
    "split",
    "selected_for_calibration",
    "a_s_batch",
    "a_s_seed",
    "a_source",
    "a_fit_numerator",
    "a_fit_denominator",
    "control_gradient_sha256",
    "treatment_gradient_sha256",
    "observed_pair_sha256",
    "source_state_hash",
    "source_state_sha256",
    "checkpoint_sha256",
    "dataset_sha256",
    "code_commit",
    "factorial_runner_sha256",
    "audit_library_sha256",
    "batch_size",
    "batch_gpu",
    "support_atol",
    "batch_correctness_pass",
    "source_preserved",
    "branch_order_pass",
    "rerun_hash_pass",
)

PREFLIGHT_BATCH_FIELDS = (
    "training_seed",
    "audit_batch_id",
    "canonical_rank",
    "split",
    "a_s_seed",
    "R_opt_G",
    "R_opt_T",
    "R_opt_T_exact",
    "s_star_G",
    "s_star_T",
    "s_star_T_exact",
    "c_star_G",
    "c_star_T",
    "c_star_T_exact",
    "update_norm_U_F",
    "update_norm_U_G",
    "update_norm_U_T",
    "update_norm_U_T_exact",
    "update_norm_ratio_G",
    "update_norm_ratio_T",
    "update_norm_ratio_T_exact",
    "update_cosine_G",
    "update_cosine_T",
    "update_cosine_T_exact",
    "G_layer_residuals_json",
    "T_layer_residuals_json",
    "T_exact_layer_residuals_json",
    "control_gradient_sha256",
    "treatment_gradient_sha256",
    "exact_gradient_sha256",
    "U_F_sha256",
    "U_G_sha256",
    "U_T_sha256",
    "U_T_exact_sha256",
    "runtime_source_state_hash",
    "receipt_source_state_hash",
    "source_hash_match",
    "gradient_hash_match",
    "randomness_hash_match",
    "all_outputs_finite",
    "no_branch_skipped",
    "amp_check_pass",
    "source_preserved",
    "deterministic_rerun_pass",
    "branch_order_pass",
    "source_checkpoint_unchanged",
    "transport_contract_pass",
)

PREFLIGHT_SEED_FIELDS = (
    "training_seed",
    "a_s_seed",
    "heldout_batch_count",
    "median_R_opt_G",
    "median_R_opt_T",
    "median_R_opt_T_exact",
    "suppression",
    "median_update_norm_ratio_T",
    "all_outputs_finite",
    "no_branch_skipped",
    "amp_check_pass",
    "source_preserved",
    "deterministic_rerun_pass",
    "branch_order_pass",
    "source_checkpoint_unchanged",
    "positive_suppression_pass",
    "exact_residual_pass",
    "norm_ratio_pass",
    "seed_gate_pass",
)


class PreflightError(RuntimeError):
    """A fail-closed receipt, replay, or gate validation error."""


def _require(mapping: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = mapping
    traversed: list[str] = []
    for key in dotted_path.split("."):
        traversed.append(key)
        if not isinstance(value, Mapping) or key not in value:
            raise PreflightError(f"missing required field: {'.'.join(traversed)}")
        value = value[key]
    return value


def _finite_float(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreflightError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "finite and > 0" if positive else "finite"
        raise PreflightError(f"{label} must be {qualifier}")
    return result


def _positive_int(value: Any, label: str) -> int:
    number = _finite_float(value, label, positive=True)
    if not number.is_integer():
        raise PreflightError(f"{label} must be an integer")
    return int(number)


def _nonnegative_float(value: Any, label: str) -> float:
    number = _finite_float(value, label)
    if number < 0:
        raise PreflightError(f"{label} must be >= 0")
    return number


def _required_true(mapping: Mapping[str, Any], dotted_path: str) -> bool:
    value = _require(mapping, dotted_path)
    if value is not True:
        raise PreflightError(f"required gate is not true: {dotted_path}")
    return True


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PreflightError(f"{label} must be a lowercase SHA256")
    return value


def canonical_batch_id(value: Any) -> int:
    """Return one unambiguous sortable audit batch id.

    Integral JSON values and canonical decimal strings are accepted.  Floats,
    signs, whitespace, leading zeroes, and booleans are rejected so that two
    spellings can never silently name the same formal batch.
    """
    if isinstance(value, bool):
        raise PreflightError("audit batch id cannot be boolean")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value):
        result = int(value)
    else:
        raise PreflightError(f"non-canonical audit batch id: {value!r}")
    if result < 0:
        raise PreflightError("audit batch id must be non-negative")
    return result


def _close(left: float, right: float, *, label: str) -> None:
    if not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15):
        raise PreflightError(f"{label} mismatch: {left!r} != {right!r}")


def derive_batch_scale(
    receipt: Mapping[str, Any],
) -> tuple[float, str, float | None, float | None]:
    """Validate and recover dot(G110,G100)/||G100||^2 from one receipt."""
    _required_true(receipt, "correctness_gate.valid")
    fit = _require(receipt, "a_star_fit")
    if not isinstance(fit, Mapping):
        raise PreflightError("a_star_fit must be an object")
    definition = _require(fit, "definition")
    if definition not in A_STAR_DEFINITIONS:
        raise PreflightError(f"unsupported a_star definition: {definition!r}")
    if _require(fit, "float_accumulation") != "float64":
        raise PreflightError("a_star_fit.float_accumulation must be float64")
    if _require(fit, "unclamped") is not True:
        raise PreflightError("a_star_fit.unclamped must be true")

    top_level = _finite_float(_require(receipt, "a_star"), "a_star", positive=True)
    numerator_present = "numerator" in fit
    denominator_present = "denominator" in fit
    if numerator_present != denominator_present:
        raise PreflightError(
            "a_star_fit numerator and denominator must appear together"
        )
    numerator: float | None = None
    denominator: float | None = None
    if numerator_present:
        numerator = _finite_float(fit["numerator"], "a_star_fit.numerator")
        denominator = _finite_float(
            fit["denominator"], "a_star_fit.denominator", positive=True
        )
        denominator_atol = _finite_float(
            fit.get("denominator_atol", 0.0), "a_star_fit.denominator_atol"
        )
        if denominator_atol < 0 or denominator <= denominator_atol:
            raise PreflightError("a_star_fit denominator is at or below its tolerance")
        derived = numerator / denominator
        if not math.isfinite(derived) or derived <= 0:
            raise PreflightError("derived a_star must be finite and > 0")
        _close(derived, top_level, label="top-level/derived a_star")
        source = "numerator_denominator"
    else:
        derived = top_level
        source = "top_level_validated_fit"

    cells = _require(receipt, "cells")
    if not isinstance(cells, Mapping) or set(cells) != set(FACTORIAL_CELLS):
        raise PreflightError("receipt must contain exactly the four factorial cells")
    for cell_name in FACTORIAL_CELLS:
        cell = cells[cell_name]
        _close(
            _finite_float(_require(cell, "a_star"), f"{cell_name}.a_star"),
            derived,
            label=f"{cell_name}.a_star",
        )
        _close(
            _finite_float(
                _require(cell, "pair_fitted_a_star"), f"{cell_name}.pair_fitted_a_star"
            ),
            derived,
            label=f"{cell_name}.pair_fitted_a_star",
        )
        _close(
            _finite_float(
                _require(cell, "whole_model.a_star"), f"{cell_name}.whole_model.a_star"
            ),
            derived,
            label=f"{cell_name}.whole_model.a_star",
        )
    return derived, source, numerator, denominator


def _validate_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    seed_value = _require(receipt, "training_seed")
    if isinstance(seed_value, bool) or not isinstance(seed_value, int):
        raise PreflightError("training_seed must be an integer")
    audit_batch_id = canonical_batch_id(_require(receipt, "audit_batch_id"))
    if canonical_batch_id(_require(receipt, "audit_seed")) != audit_batch_id:
        raise PreflightError("audit_seed and audit_batch_id differ")
    a_star, a_source, numerator, denominator = derive_batch_scale(receipt)

    _required_true(receipt, "source_state_non_committing.preserved")
    for cell_name in FACTORIAL_CELLS:
        cell = _require(receipt, f"cells.{cell_name}")
        _required_true(cell, "finite_gate")
        if _require(cell, "branch_skipped_flag") is not False:
            raise PreflightError(f"{cell_name} recorded a skipped branch")
        for branch in ("control", "treatment"):
            _required_true(
                cell, f"branches.{branch}.optimizer_step_advanced_exactly_once"
            )
            _required_true(cell, f"branches.{branch}.clone_contract.independent")
            if _require(cell, f"branches.{branch}.step_skipped") is not False:
                raise PreflightError(f"{cell_name}/{branch} step was skipped")
        order = _require(receipt, f"order_invariance_and_rerun.{cell_name}")
        _required_true(order, "numerically_invariant")
        _required_true(order, "result_hash_identical")

    gradient_contract = _require(receipt, "gradient_contract")
    _required_true(gradient_contract, "observed_computed_once")
    _required_true(gradient_contract, "observed_control.finite")
    _required_true(gradient_contract, "observed_treatment.finite")
    _required_true(gradient_contract, "observed_control.unscaled")
    _required_true(gradient_contract, "observed_treatment.unscaled")
    control_gradient = _sha256(
        _require(gradient_contract, "observed_control.gradient_sha256"),
        "observed control gradient hash",
    )
    treatment_gradient = _sha256(
        _require(gradient_contract, "observed_treatment.gradient_sha256"),
        "observed treatment gradient hash",
    )
    observed_pair = _sha256(
        _require(gradient_contract, "observed_pair_sha256_before"), "observed pair hash"
    )
    if (
        _sha256(
            _require(gradient_contract, "observed_pair_sha256_after"),
            "observed pair hash after",
        )
        != observed_pair
    ):
        raise PreflightError("observed gradient pair changed during factorial audit")
    observed_cell = _require(receipt, "cells.observed_real")
    if (
        _sha256(
            _require(observed_cell, "control_gradient_sha256"),
            "cell control gradient hash",
        )
        != control_gradient
    ):
        raise PreflightError("control gradient hashes disagree within receipt")
    if (
        _sha256(
            _require(observed_cell, "treatment_gradient_sha256"),
            "cell treatment gradient hash",
        )
        != treatment_gradient
    ):
        raise PreflightError("treatment gradient hashes disagree within receipt")

    source_state_hash = _sha256(
        _require(receipt, "source_state_non_committing.source_state_hash"),
        "source state hash",
    )
    for cell_name in FACTORIAL_CELLS:
        if (
            _sha256(
                _require(receipt, f"cells.{cell_name}.source_state_hash"),
                f"{cell_name} source state hash",
            )
            != source_state_hash
        ):
            raise PreflightError("factorial cells do not share one source state hash")
    provenance = _require(receipt, "provenance")
    source_state_sha256 = _sha256(
        _require(provenance, "source_state_sha256"), "source state file hash"
    )
    checkpoint_sha256 = _sha256(
        _require(provenance, "checkpoint_sha256"), "checkpoint hash"
    )
    dataset_sha256 = _sha256(_require(provenance, "dataset_sha256"), "dataset hash")
    code_commit = _require(provenance, "code_commit")
    if (
        not isinstance(code_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", code_commit) is None
    ):
        raise PreflightError("provenance.code_commit must be a 40-hex commit")
    factorial_runner_sha256 = _sha256(
        _require(provenance, "runner_sha256"), "factorial runner hash"
    )
    audit_library_sha256 = _sha256(
        _require(provenance, "audit_library_sha256"), "audit library hash"
    )
    if (
        _positive_int(_require(provenance, "training_seed"), "provenance.training_seed")
        != seed_value
    ):
        raise PreflightError("receipt/provenance training seed mismatch")
    if _finite_float(_require(provenance, "q"), "provenance.q") != 256.0:
        raise PreflightError("formal preflight requires q=256")
    if (
        _finite_float(
            _require(provenance, "reference_gap_scale"), "reference gap scale"
        )
        != 1.0
    ):
        raise PreflightError("formal preflight requires reference gap scale 1.0")
    if _finite_float(_require(provenance, "probe_gap_scale"), "probe gap scale") != 1.1:
        raise PreflightError("formal preflight requires probe gap scale 1.1")
    batch_size = _positive_int(_require(provenance, "batch_size"), "batch_size")
    batch_gpu = _positive_int(_require(provenance, "batch_gpu"), "batch_gpu")
    if batch_size % batch_gpu:
        raise PreflightError("receipt batch_gpu must divide batch_size")
    support_atol = _nonnegative_float(
        _require(provenance, "support_atol"), "support_atol"
    )

    return {
        "training_seed": seed_value,
        "audit_batch_id": audit_batch_id,
        "a_s_batch": a_star,
        "a_source": a_source,
        "a_fit_numerator": numerator,
        "a_fit_denominator": denominator,
        "control_gradient_sha256": control_gradient,
        "treatment_gradient_sha256": treatment_gradient,
        "observed_pair_sha256": observed_pair,
        "source_state_hash": source_state_hash,
        "source_state_sha256": source_state_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_sha256": dataset_sha256,
        "code_commit": code_commit,
        "factorial_runner_sha256": factorial_runner_sha256,
        "audit_library_sha256": audit_library_sha256,
        "batch_size": batch_size,
        "batch_gpu": batch_gpu,
        "support_atol": support_atol,
        "batch_correctness_pass": True,
        "source_preserved": True,
        "branch_order_pass": True,
        "rerun_hash_pass": True,
        "receipt": receipt,
    }


def build_calibration(
    receipts: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int] = FORMAL_TRAINING_SEEDS,
) -> tuple[
    list[dict[str, Any]], dict[int, float], dict[tuple[int, int], Mapping[str, Any]]
]:
    """Validate receipts, assign canonical ranks, and freeze seed medians."""
    expected = tuple(expected_seeds)
    if len(set(expected)) != len(expected):
        raise PreflightError("expected training seeds must be unique")
    validated = [_validate_receipt(receipt) for receipt in receipts]
    by_seed: dict[int, list[dict[str, Any]]] = {seed: [] for seed in expected}
    for item in validated:
        if item["training_seed"] not in by_seed:
            raise PreflightError(f"unexpected training seed: {item['training_seed']}")
        by_seed[item["training_seed"]].append(item)

    rows: list[dict[str, Any]] = []
    frozen: dict[int, float] = {}
    index: dict[tuple[int, int], Mapping[str, Any]] = {}
    invariant_fields = (
        "source_state_hash",
        "source_state_sha256",
        "checkpoint_sha256",
        "dataset_sha256",
        "code_commit",
        "factorial_runner_sha256",
        "audit_library_sha256",
        "batch_size",
        "batch_gpu",
        "support_atol",
    )
    for seed in expected:
        items = sorted(by_seed[seed], key=lambda item: item["audit_batch_id"])
        if len(items) != EXPECTED_BATCHES_PER_SEED:
            raise PreflightError(
                f"seed {seed} requires exactly {EXPECTED_BATCHES_PER_SEED} receipts, got {len(items)}"
            )
        ids = [item["audit_batch_id"] for item in items]
        if len(set(ids)) != len(ids):
            raise PreflightError(f"seed {seed} has duplicate audit batch ids")
        if tuple(ids) != FORMAL_AUDIT_BATCH_IDS:
            raise PreflightError(
                f"seed {seed} does not contain the frozen formal audit ids"
            )
        for field in invariant_fields:
            if len({item[field] for item in items}) != 1:
                raise PreflightError(f"seed {seed} has inconsistent {field}")
        calibration_values = [items[rank]["a_s_batch"] for rank in CALIBRATION_RANKS]
        a_s = float(statistics.median(calibration_values))
        if not math.isfinite(a_s) or a_s <= 0:
            raise PreflightError(f"seed {seed} frozen transport scalar is invalid")
        frozen[seed] = a_s
        for rank, item in enumerate(items):
            split = "calibration" if rank in CALIBRATION_RANKS else "heldout"
            row = {key: item[key] for key in CALIBRATION_FIELDS if key in item}
            row.update(
                {
                    "canonical_rank": rank,
                    "split": split,
                    "selected_for_calibration": rank in CALIBRATION_RANKS,
                    "a_s_seed": a_s,
                }
            )
            rows.append(row)
            index[(seed, item["audit_batch_id"])] = item["receipt"]
    rows.sort(key=lambda row: (row["training_seed"], row["canonical_rank"]))
    return rows, frozen, index


def load_receipts(receipt_root: Path) -> list[dict[str, Any]]:
    paths = sorted(receipt_root.glob("receipts/seed*/audit*/batch_receipt.json"))
    if not paths:
        paths = sorted(receipt_root.glob("seed*/audit*/batch_receipt.json"))
    if not paths:
        raise PreflightError(f"no batch_receipt.json files below {receipt_root}")
    receipts: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PreflightError(f"cannot load receipt {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise PreflightError(f"receipt must be a JSON object: {path}")
        payload["_receipt_path"] = str(path.resolve())
        receipts.append(payload)
    return receipts


def exact_update_metrics(
    reference: Mapping[str, torch.Tensor], candidate: Mapping[str, torch.Tensor]
) -> dict[str, float]:
    """Expose the audit library's exact whole-model R_opt convention."""
    if set(reference) != set(candidate) or not reference:
        raise PreflightError("update maps must have identical non-empty keys")
    s_star, c_star, r_opt, cosine, _ = audit_lib._update_scale_and_residual(
        dict(reference), dict(candidate)
    )
    reference_l2 = math.sqrt(audit_lib._norm_sq(reference.values()))
    candidate_l2 = math.sqrt(audit_lib._norm_sq(candidate.values()))
    result = {
        "s_star": s_star,
        "c_star": c_star,
        "R_opt": r_opt,
        "update_cosine": cosine,
        "update_reference_l2": reference_l2,
        "update_probe_l2": candidate_l2,
        "update_norm_ratio": candidate_l2 / reference_l2,
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise PreflightError("non-finite update metric")
    return result


def _optimizer_nonmoment_snapshot(
    net: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    return {
        "state": {
            name: {
                key: copy.deepcopy(value)
                for key, value in optimizer.state.get(parameter, {}).items()
                if key not in MOMENT_FIELDS
            }
            for name, parameter in net.named_parameters()
        },
        "param_groups": audit_lib._optimizer_hyperparameter_contract(optimizer),
    }


def _moment_l2_by_field(
    net: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for field in MOMENT_FIELDS:
        tensors = [
            optimizer.state[parameter][field]
            for parameter in net.parameters()
            if parameter in optimizer.state and field in optimizer.state[parameter]
        ]
        result[field] = (
            math.sqrt(
                sum(float(value.detach().double().square().sum()) for value in tensors)
            )
            if tensors
            else None
        )
    return result


def transport_optimizer_moments_(
    net: torch.nn.Module, optimizer: torch.optim.Optimizer, a_s: float
) -> dict[str, Any]:
    """Scale only supported moment tensors on an already-disposable optimizer."""
    a_s = _finite_float(a_s, "a_s", positive=True)
    a_squared = a_s * a_s
    if not math.isfinite(a_squared):
        raise PreflightError("a_s squared must be finite")
    model_parameters = list(net.parameters())
    grouped_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    model_ids = {id(parameter) for parameter in model_parameters}
    grouped_ids = [id(parameter) for parameter in grouped_parameters]
    state_ids = {id(parameter) for parameter in optimizer.state}
    if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != model_ids:
        raise PreflightError(
            "optimizer parameter groups do not map one-to-one to the model"
        )
    if state_ids != model_ids:
        raise PreflightError(
            "optimizer state keys do not map exactly to model parameters"
        )
    optimizer_before = gauge.state_sha256(optimizer.state_dict())
    nonmoment_before = gauge.state_sha256(_optimizer_nonmoment_snapshot(net, optimizer))
    before_norms = _moment_l2_by_field(net, optimizer)
    parameter_count = 0
    max_field_count = 0
    for name, parameter in net.named_parameters():
        state = optimizer.state.get(parameter)
        if not state:
            raise PreflightError(
                f"{name}: missing optimizer state during moment transport"
            )
        missing = [key for key in ("step", "exp_avg", "exp_avg_sq") if key not in state]
        if missing:
            raise PreflightError(f"{name}: missing optimizer fields {missing}")
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and key not in {
                "step",
                "exp_avg",
                "exp_avg_sq",
                "max_exp_avg_sq",
            }:
                raise PreflightError(
                    f"{name}: unsupported tensor optimizer field {key!r}"
                )
        for field in ("exp_avg", "exp_avg_sq"):
            value = state[field]
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                raise PreflightError(f"{name}: invalid {field} tensor")
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise PreflightError(f"{name}: non-finite or non-floating {field}")
        if "max_exp_avg_sq" in state:
            maximum = state["max_exp_avg_sq"]
            if (
                not isinstance(maximum, torch.Tensor)
                or maximum.shape != parameter.shape
                or not maximum.is_floating_point()
                or not bool(torch.isfinite(maximum).all())
            ):
                raise PreflightError(f"{name}: invalid max_exp_avg_sq tensor")
            max_field_count += 1
        if a_s != 1.0:
            state["exp_avg"].mul_(a_s)
            state["exp_avg_sq"].mul_(a_squared)
            if "max_exp_avg_sq" in state:
                state["max_exp_avg_sq"].mul_(a_squared)
        parameter_count += 1
    if parameter_count == 0:
        raise PreflightError("optimizer has no initialized parameter states")
    if max_field_count not in (0, parameter_count):
        raise PreflightError("max_exp_avg_sq is present for only part of the optimizer")

    after_norms = _moment_l2_by_field(net, optimizer)
    nonmoment_after = gauge.state_sha256(_optimizer_nonmoment_snapshot(net, optimizer))
    ratios: dict[str, float | None] = {}
    expected = {
        "exp_avg": a_s,
        "exp_avg_sq": a_squared,
        "max_exp_avg_sq": a_squared,
    }
    ratio_checks: dict[str, bool] = {}
    for field in MOMENT_FIELDS:
        before = before_norms[field]
        after = after_norms[field]
        if before is None:
            ratios[field] = None
            ratio_checks[field] = field == "max_exp_avg_sq" and after is None
        elif before == 0:
            ratios[field] = None
            ratio_checks[field] = after == 0
        else:
            assert after is not None
            ratios[field] = after / before
            ratio_checks[field] = math.isclose(
                ratios[field], expected[field], rel_tol=2e-6, abs_tol=1e-12
            )
    contract = {
        "a_s": a_s,
        "parameter_count": parameter_count,
        "max_exp_avg_sq_parameter_count": max_field_count,
        "optimizer_hash_before": optimizer_before,
        "optimizer_hash_after": gauge.state_sha256(optimizer.state_dict()),
        "nonmoment_hash_before": nonmoment_before,
        "nonmoment_hash_after": nonmoment_after,
        "nonmoment_state_preserved": nonmoment_before == nonmoment_after,
        "moment_l2_before": before_norms,
        "moment_l2_after": after_norms,
        "observed_norm_ratios": ratios,
        "expected_norm_ratios": expected,
        "norm_ratio_checks": ratio_checks,
    }
    contract["valid"] = contract["nonmoment_state_preserved"] and all(
        ratio_checks.values()
    )
    if not contract["valid"]:
        raise PreflightError("in-memory moment transport contract failed")
    return contract


def _clone_transported_source(
    common_net: torch.nn.Module, common_optimizer: torch.optim.Optimizer, a_s: float
):
    net, optimizer, clone_contract = audit_lib._clone_radam_branch(
        common_net, common_optimizer
    )
    contract = transport_optimizer_moments_(net, optimizer, a_s)
    contract["clone_contract"] = clone_contract
    contract["valid"] = contract["valid"] and clone_contract["independent"]
    return net, optimizer, contract


def _rng_hash(device: torch.device) -> dict[str, str | None]:
    return {
        "cpu": gauge.tensor_sha256(torch.get_rng_state()),
        "cuda_all": (
            gauge.state_sha256(torch.cuda.get_rng_state_all())
            if device.type == "cuda"
            else None
        ),
    }


def _make_microbatches(
    loss_template,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    audit_seed: int,
    microbatch_size: int,
):
    device = images.device
    if microbatch_size < 1 or images.shape[0] % microbatch_size:
        raise PreflightError("microbatch size must divide the held-out batch size")
    torch.manual_seed(audit_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(audit_seed)
    result = []
    for start in range(0, images.shape[0], microbatch_size):
        image_micro = images[start : start + microbatch_size]
        label_micro = labels[start : start + microbatch_size]
        t = (
            torch.randn(image_micro.shape[0], 1, 1, 1, device=device)
            * loss_template.P_std
            + loss_template.P_mean
        ).exp()
        eps = torch.randn_like(image_micro)
        result.append(
            (image_micro, label_micro, t, eps, gauge.get_rng_state(device).clone())
        )
    return result


def _randomness_record(
    images: torch.Tensor, labels: torch.Tensor, microbatches: Sequence[tuple[Any, ...]]
) -> dict[str, Any]:
    return {
        "minibatch_images_sha256": gauge.tensor_sha256(images),
        "minibatch_labels_sha256": gauge.tensor_sha256(labels),
        "microbatch_size": int(microbatches[0][0].shape[0]),
        "accumulation_rounds": len(microbatches),
        "t_sha256": gauge.state_sha256([item[2] for item in microbatches]),
        "noise_sha256": gauge.state_sha256([item[3] for item in microbatches]),
        "dropout_rng_state_sha256": gauge.state_sha256(
            [item[4] for item in microbatches]
        ),
    }


def _validate_replay_hashes(
    receipt: Mapping[str, Any],
    gradients: Mapping[str, Mapping[str, torch.Tensor]],
    randomness: Mapping[str, Any],
) -> tuple[bool, bool]:
    control_hash = gauge.state_sha256(gradients["control"])
    treatment_hash = gauge.state_sha256(gradients["treatment"])
    gradient_match = (
        control_hash
        == _require(receipt, "gradient_contract.observed_control.gradient_sha256")
        and treatment_hash
        == _require(receipt, "gradient_contract.observed_treatment.gradient_sha256")
        and gauge.state_sha256(dict(gradients))
        == _require(receipt, "gradient_contract.observed_pair_sha256_before")
    )
    expected_randomness = _require(receipt, "randomness_contract")
    randomness_match = all(
        randomness[key] == _require(expected_randomness, key) for key in randomness
    )
    return gradient_match, randomness_match


def _execute_arms(
    common_net,
    common_optimizer,
    transported_net,
    transported_optimizer,
    gradients,
    exact_gradient,
    scaler_template,
    order: Sequence[str],
) -> dict[str, dict[str, Any]]:
    configs = {
        "F": (common_net, common_optimizer, gradients["control"]),
        "G": (common_net, common_optimizer, gradients["treatment"]),
        "T": (transported_net, transported_optimizer, gradients["treatment"]),
        "T_exact": (transported_net, transported_optimizer, exact_gradient),
    }
    if set(order) != set(ARM_ORDER) or len(order) != len(ARM_ORDER):
        raise PreflightError(
            "virtual arm order must contain F/G/T/T_exact exactly once"
        )
    result: dict[str, dict[str, Any]] = {}
    for arm in order:
        net, optimizer, gradient = configs[arm]
        predicted, actual, moments, detail = audit_lib.virtual_step_from_unscaled_grads(
            net, optimizer, gradient, condition="real", scaler_template=scaler_template
        )
        result[arm] = {
            "predicted": predicted,
            "actual": actual,
            "moments": moments,
            "detail": detail,
        }
    return result


def _summarize_arm_pair(
    arms: Mapping[str, Mapping[str, Any]],
    candidate_arm: str,
    gradients: Mapping[str, Mapping[str, torch.Tensor]],
    exact_gradient: Mapping[str, torch.Tensor],
    *,
    eps: float,
    support_atol: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate_gradient = (
        exact_gradient if candidate_arm == "T_exact" else gradients["treatment"]
    )
    return audit_lib.summarize_pair(
        gradients["control"],
        candidate_gradient,
        arms["F"]["predicted"],
        arms[candidate_arm]["predicted"],
        arms["F"]["actual"],
        arms[candidate_arm]["actual"],
        moments_reference=arms["F"]["moments"],
        moments_probe=arms[candidate_arm]["moments"],
        eps=eps,
        support_atol=support_atol,
    )


def _layer_json(rows: Sequence[Mapping[str, Any]]) -> str:
    keys = (
        "layer",
        "update_reference_l2",
        "update_probe_l2",
        "update_cosine",
        "s_K_star_layer",
        "c_K_star_layer",
        "R_opt_layer",
        "layer_residual_with_global_c_star",
        "support_coordinate_count",
        "coordinate_count",
        "off_support_candidate_energy_exact",
    )
    payload = [
        {key: audit_lib._json_safe(row.get(key)) for key in keys} for row in rows
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _required_metrics_finite(
    whole: Mapping[str, Any], layers: Sequence[Mapping[str, Any]]
) -> bool:
    whole_fields = (
        "R_opt",
        "s_star",
        "c_star",
        "update_reference_l2",
        "update_probe_l2",
        "update_norm_ratio",
        "update_cosine",
    )
    if not all(
        isinstance(whole.get(key), (int, float))
        and not isinstance(whole.get(key), bool)
        and math.isfinite(float(whole[key]))
        for key in whole_fields
    ):
        return False
    layer_fields = (
        "update_reference_l2",
        "update_probe_l2",
        "update_cosine",
        "s_K_star_layer",
        "c_K_star_layer",
        "R_opt_layer",
    )
    return bool(layers) and all(
        isinstance(row.get(key), (int, float))
        and not isinstance(row.get(key), bool)
        and math.isfinite(float(row[key]))
        for row in layers
        for key in layer_fields
    )


def run_heldout_virtual_batch(
    common_net,
    common_optimizer,
    loss_template,
    images,
    labels,
    *,
    receipt: Mapping[str, Any],
    a_s: float,
    scaler_state: Mapping[str, Any] | None,
    amp: bool,
    initial_scale: float,
    microbatch_size: int,
    support_atol: float,
) -> dict[str, Any]:
    """Replay one held-out batch and return a fully gated CSV row."""
    seed = int(_require(receipt, "training_seed"))
    audit_batch_id = canonical_batch_id(_require(receipt, "audit_batch_id"))
    device = images.device
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if device.type == "cuda"
        else None
    )
    source_parameter_before = gauge.module_state_hashes(common_net)
    source_optimizer_before = gauge.state_sha256(common_optimizer.state_dict())
    source_gradient_before = audit_lib._source_gradient_buffers_hash(common_net)
    runtime_source_state_hash = gauge.state_sha256(
        {
            "parameters": source_parameter_before,
            "optimizer": source_optimizer_before,
        }
    )
    receipt_source_state_hash = _require(
        receipt, "source_state_non_committing.source_state_hash"
    )
    try:
        microbatches = _make_microbatches(
            loss_template,
            images,
            labels,
            audit_seed=audit_batch_id,
            microbatch_size=microbatch_size,
        )
        randomness = _randomness_record(images, labels, microbatches)
        scaler_template = gauge._new_scaler(device, amp, initial_scale)
        if scaler_state is not None:
            scaler_template.load_state_dict(copy.deepcopy(dict(scaler_state)))
        gradients: dict[str, Mapping[str, torch.Tensor]] = {}
        gradient_details: dict[str, Mapping[str, Any]] = {}
        for label, gap_scale in (("control", 1.0), ("treatment", 1.1)):
            gradients[label], gradient_details[label] = (
                audit_lib.compute_unscaled_gradient(
                    common_net,
                    common_optimizer,
                    loss_template,
                    microbatches,
                    gap_scale=gap_scale,
                    scaler_template=scaler_template,
                    amp=amp,
                )
            )
        gradient_hash_match, randomness_hash_match = _validate_replay_hashes(
            receipt, gradients, randomness
        )
        if not gradient_hash_match:
            raise PreflightError(
                f"seed {seed} audit {audit_batch_id}: held-out gradient hashes differ"
            )
        if not randomness_hash_match:
            raise PreflightError(
                f"seed {seed} audit {audit_batch_id}: held-out randomness hashes differ"
            )

        exact_gradient = audit_lib.construct_exact_scalar_gradient(
            dict(gradients["control"]), a_s
        )
        transported_net, transported_optimizer, transport_contract = (
            _clone_transported_source(common_net, common_optimizer, a_s)
        )
        primary = _execute_arms(
            common_net,
            common_optimizer,
            transported_net,
            transported_optimizer,
            gradients,
            exact_gradient,
            scaler_template,
            ARM_ORDER,
        )
        reverse = _execute_arms(
            common_net,
            common_optimizer,
            transported_net,
            transported_optimizer,
            gradients,
            exact_gradient,
            scaler_template,
            tuple(reversed(ARM_ORDER)),
        )
        order_checks = {
            arm: all(
                primary[arm]["detail"][key] == reverse[arm]["detail"][key]
                for key in (
                    "actual_update_sha256",
                    "predicted_update_sha256",
                    "moments_after_sha256",
                )
            )
            for arm in ARM_ORDER
        }
        branch_order_pass = all(order_checks.values())
        deterministic_rerun_pass = branch_order_pass

        eps_values = {float(group["eps"]) for group in common_optimizer.param_groups}
        if len(eps_values) != 1:
            raise PreflightError(
                "all optimizer parameter groups must share one epsilon"
            )
        eps = next(iter(eps_values))
        summaries: dict[str, dict[str, Any]] = {}
        layers: dict[str, list[dict[str, Any]]] = {}
        for arm in ("G", "T", "T_exact"):
            summaries[arm], layers[arm] = _summarize_arm_pair(
                primary,
                arm,
                gradients,
                exact_gradient,
                eps=eps,
                support_atol=support_atol,
            )
        all_outputs_finite = all(
            _required_metrics_finite(summaries[arm], layers[arm])
            for arm in ("G", "T", "T_exact")
        )
        no_branch_skipped = all(
            detail["detail"]["step_skipped"] is False
            and detail["detail"]["optimizer_step_advanced_exactly_once"] is True
            for detail in primary.values()
        )
        amp_check_pass = (
            (not amp or scaler_state is not None)
            and all(item["finite"] for item in gradient_details.values())
            and all(item["detail"]["gradscaler_preserved"] for item in primary.values())
        )
    finally:
        torch.set_rng_state(cpu_rng_before)
        if cuda_rng_before is not None:
            torch.cuda.set_rng_state_all(cuda_rng_before)

    source_parameter_after = gauge.module_state_hashes(common_net)
    source_optimizer_after = gauge.state_sha256(common_optimizer.state_dict())
    source_gradient_after = audit_lib._source_gradient_buffers_hash(common_net)
    source_preserved = (
        source_parameter_before == source_parameter_after
        and source_optimizer_before == source_optimizer_after
        and source_gradient_before == source_gradient_after
        and _rng_hash(device)
        == {
            "cpu": gauge.tensor_sha256(cpu_rng_before),
            "cuda_all": (
                gauge.state_sha256(cuda_rng_before)
                if cuda_rng_before is not None
                else None
            ),
        }
    )
    if not source_preserved:
        raise PreflightError(
            f"seed {seed} audit {audit_batch_id}: source state changed"
        )

    arm_hash = {
        arm: primary[arm]["detail"]["actual_update_sha256"] for arm in ARM_ORDER
    }
    return {
        "training_seed": seed,
        "audit_batch_id": audit_batch_id,
        "a_s_seed": a_s,
        "R_opt_G": summaries["G"]["R_opt"],
        "R_opt_T": summaries["T"]["R_opt"],
        "R_opt_T_exact": summaries["T_exact"]["R_opt"],
        "s_star_G": summaries["G"]["s_star"],
        "s_star_T": summaries["T"]["s_star"],
        "s_star_T_exact": summaries["T_exact"]["s_star"],
        "c_star_G": summaries["G"]["c_star"],
        "c_star_T": summaries["T"]["c_star"],
        "c_star_T_exact": summaries["T_exact"]["c_star"],
        "update_norm_U_F": summaries["G"]["update_reference_l2"],
        "update_norm_U_G": summaries["G"]["update_probe_l2"],
        "update_norm_U_T": summaries["T"]["update_probe_l2"],
        "update_norm_U_T_exact": summaries["T_exact"]["update_probe_l2"],
        "update_norm_ratio_G": summaries["G"]["update_norm_ratio"],
        "update_norm_ratio_T": summaries["T"]["update_norm_ratio"],
        "update_norm_ratio_T_exact": summaries["T_exact"]["update_norm_ratio"],
        "update_cosine_G": summaries["G"]["update_cosine"],
        "update_cosine_T": summaries["T"]["update_cosine"],
        "update_cosine_T_exact": summaries["T_exact"]["update_cosine"],
        "G_layer_residuals_json": _layer_json(layers["G"]),
        "T_layer_residuals_json": _layer_json(layers["T"]),
        "T_exact_layer_residuals_json": _layer_json(layers["T_exact"]),
        "control_gradient_sha256": gauge.state_sha256(gradients["control"]),
        "treatment_gradient_sha256": gauge.state_sha256(gradients["treatment"]),
        "exact_gradient_sha256": gauge.state_sha256(exact_gradient),
        "U_F_sha256": arm_hash["F"],
        "U_G_sha256": arm_hash["G"],
        "U_T_sha256": arm_hash["T"],
        "U_T_exact_sha256": arm_hash["T_exact"],
        "runtime_source_state_hash": runtime_source_state_hash,
        "receipt_source_state_hash": receipt_source_state_hash,
        "source_hash_match": runtime_source_state_hash == receipt_source_state_hash,
        "gradient_hash_match": gradient_hash_match,
        "randomness_hash_match": randomness_hash_match,
        "all_outputs_finite": all_outputs_finite,
        "no_branch_skipped": no_branch_skipped,
        "amp_check_pass": amp_check_pass,
        "source_preserved": source_preserved,
        "deterministic_rerun_pass": deterministic_rerun_pass,
        "branch_order_pass": branch_order_pass,
        "source_checkpoint_unchanged": True,
        "transport_contract_pass": bool(transport_contract["valid"]),
        "_transport_contract": transport_contract,
        "_order_checks": order_checks,
    }


def evaluate_go_gate(
    batch_rows: Sequence[Mapping[str, Any]],
    frozen_a: Mapping[int, float],
    *,
    expected_seeds: Sequence[int] = FORMAL_TRAINING_SEEDS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate held-out rows and apply the immutable pre-registered GO gate."""
    seed_rows: list[dict[str, Any]] = []
    expected = tuple(expected_seeds)
    if set(frozen_a) != set(expected):
        raise PreflightError(
            "frozen a_s keys do not exactly match expected training seeds"
        )
    row_seeds = []
    for row in batch_rows:
        seed_value = _require(row, "training_seed")
        if isinstance(seed_value, bool) or not isinstance(seed_value, int):
            raise PreflightError("preflight row training_seed must be an integer")
        row_seeds.append(seed_value)
    if len(batch_rows) != len(expected) * len(HELDOUT_RANKS):
        raise PreflightError("preflight table does not have the exact formal row count")
    if set(row_seeds) != set(expected):
        raise PreflightError(
            "preflight table seeds do not exactly match expected seeds"
        )
    for seed in expected:
        rows = sorted(
            (row for row in batch_rows if row.get("training_seed") == seed),
            key=lambda row: canonical_batch_id(row["audit_batch_id"]),
        )
        if len(rows) != len(HELDOUT_RANKS):
            raise PreflightError(
                f"seed {seed} requires {len(HELDOUT_RANKS)} held-out rows, got {len(rows)}"
            )
        ranks = [_require(row, "canonical_rank") for row in rows]
        if any(isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks):
            raise PreflightError(f"seed {seed} canonical ranks must be integers")
        if set(ranks) != set(HELDOUT_RANKS) or len(set(ranks)) != len(ranks):
            raise PreflightError(f"seed {seed} rows are not the frozen held-out ranks")
        if any(_require(row, "split") != "heldout" for row in rows):
            raise PreflightError(f"seed {seed} contains a non-heldout preflight row")
        batch_ids = [canonical_batch_id(row["audit_batch_id"]) for row in rows]
        if len(set(batch_ids)) != len(batch_ids):
            raise PreflightError(f"seed {seed} has duplicate held-out audit batch ids")
        if any(
            canonical_batch_id(row["audit_batch_id"])
            != FORMAL_AUDIT_BATCH_IDS[row["canonical_rank"]]
            for row in rows
        ):
            raise PreflightError(
                f"seed {seed} held-out ranks are mapped to wrong audit ids"
            )
        frozen_seed_a = _finite_float(frozen_a[seed], "a_s_seed", positive=True)
        for row in rows:
            _close(
                _finite_float(_require(row, "a_s_seed"), "row.a_s_seed", positive=True),
                frozen_seed_a,
                label=f"seed {seed} row/frozen a_s",
            )
        r_g = [_finite_float(row["R_opt_G"], "R_opt_G", positive=True) for row in rows]
        r_t = [_nonnegative_float(row["R_opt_T"], "R_opt_T") for row in rows]
        r_exact = [
            _nonnegative_float(row["R_opt_T_exact"], "R_opt_T_exact") for row in rows
        ]
        norm_t = [
            _finite_float(
                row["update_norm_ratio_T"], "update_norm_ratio_T", positive=True
            )
            for row in rows
        ]
        median_g = float(statistics.median(r_g))
        median_t = float(statistics.median(r_t))
        median_exact = float(statistics.median(r_exact))
        median_norm_t = float(statistics.median(norm_t))
        suppression = 1.0 - median_t / median_g
        boolean_fields = (
            "all_outputs_finite",
            "no_branch_skipped",
            "amp_check_pass",
            "source_preserved",
            "deterministic_rerun_pass",
            "branch_order_pass",
            "source_checkpoint_unchanged",
            "source_hash_match",
            "gradient_hash_match",
            "randomness_hash_match",
            "transport_contract_pass",
        )
        booleans = {
            field: all(row.get(field) is True for row in rows)
            for field in boolean_fields
        }
        seed_row = {
            "training_seed": seed,
            "a_s_seed": frozen_seed_a,
            "heldout_batch_count": len(rows),
            "median_R_opt_G": median_g,
            "median_R_opt_T": median_t,
            "median_R_opt_T_exact": median_exact,
            "suppression": suppression,
            "median_update_norm_ratio_T": median_norm_t,
            "all_outputs_finite": booleans["all_outputs_finite"],
            "no_branch_skipped": booleans["no_branch_skipped"],
            "amp_check_pass": booleans["amp_check_pass"],
            "source_preserved": (
                booleans["source_preserved"]
                and booleans["source_hash_match"]
                and booleans["gradient_hash_match"]
                and booleans["randomness_hash_match"]
                and booleans["transport_contract_pass"]
            ),
            "deterministic_rerun_pass": booleans["deterministic_rerun_pass"],
            "branch_order_pass": booleans["branch_order_pass"],
            "source_checkpoint_unchanged": booleans["source_checkpoint_unchanged"],
            "positive_suppression_pass": suppression > 0.0,
            "exact_residual_pass": median_exact <= 0.01,
            "norm_ratio_pass": 0.90 <= median_norm_t <= 1.10,
        }
        seed_row["seed_gate_pass"] = all(
            seed_row[key] is True
            for key in (
                "all_outputs_finite",
                "no_branch_skipped",
                "amp_check_pass",
                "source_preserved",
                "deterministic_rerun_pass",
                "branch_order_pass",
                "source_checkpoint_unchanged",
                "positive_suppression_pass",
                "exact_residual_pass",
                "norm_ratio_pass",
            )
        )
        seed_rows.append(seed_row)

    cross_seed_median = float(
        statistics.median(row["suppression"] for row in seed_rows)
    )
    gates = {
        "all_outputs_finite": all(row["all_outputs_finite"] for row in seed_rows),
        "no_heldout_batch_skipped": all(row["no_branch_skipped"] for row in seed_rows),
        "source_preservation_and_deterministic_rerun": all(
            row["source_preserved"] and row["deterministic_rerun_pass"]
            for row in seed_rows
        ),
        "every_seed_positive_suppression": all(
            row["positive_suppression_pass"] for row in seed_rows
        ),
        "cross_seed_median_suppression": cross_seed_median,
        "cross_seed_median_suppression_at_least_0p50": cross_seed_median >= 0.50,
        "every_seed_exact_residual_at_most_0p01": all(
            row["exact_residual_pass"] for row in seed_rows
        ),
        "every_seed_median_norm_ratio_in_0p90_1p10": all(
            row["norm_ratio_pass"] for row in seed_rows
        ),
        "branch_order_invariant": all(row["branch_order_pass"] for row in seed_rows),
        "source_checkpoint_unchanged": all(
            row["source_checkpoint_unchanged"] for row in seed_rows
        ),
        "amp_checks_pass": all(row["amp_check_pass"] for row in seed_rows),
    }
    boolean_gates = [value for value in gates.values() if isinstance(value, bool)]
    verdict = {
        "schema_version": 1,
        "status": "GO" if all(boolean_gates) else "NO_GO",
        "formal_training_authorized": all(boolean_gates),
        "gates": gates,
        "calibration_ranks": list(CALIBRATION_RANKS),
        "heldout_ranks": list(HELDOUT_RANKS),
        "frozen_a_s": {str(seed): frozen_a[seed] for seed in expected},
        "seed_results": seed_rows,
        "operations_excluded": ["training", "sample_generation", "FID", "KID"],
    }
    return seed_rows, verdict


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return value


def _write_csv(
    path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _strict_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            audit_lib._json_safe(payload),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def emit_outputs(
    out: Path,
    calibration_rows: Sequence[Mapping[str, Any]],
    batch_rows: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    verdict: Mapping[str, Any],
) -> None:
    _write_csv(out / "calibration.csv", CALIBRATION_FIELDS, calibration_rows)
    _write_csv(out / "preflight_batch.csv", PREFLIGHT_BATCH_FIELDS, batch_rows)
    _write_csv(out / "preflight_seed.csv", PREFLIGHT_SEED_FIELDS, seed_rows)
    _strict_dump(out / "preflight_verdict.json", verdict)


def _parse_expected_seeds(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected seeds must be comma-separated integers"
        ) from exc
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("expected seeds must be non-empty and unique")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument(
        "--inputs",
        type=Path,
        required=True,
        help="JSON with a seeds list and training_state/checkpoint/data paths",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--expected-seeds", type=_parse_expected_seeds, default=FORMAL_TRAINING_SEEDS
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--initial-scale", type=float, default=65536.0)
    args = parser.parse_args(argv)
    if args.expected_seeds != FORMAL_TRAINING_SEEDS:
        parser.error("formal preflight requires exactly training seeds 3,4,5")
    if not math.isfinite(args.initial_scale) or args.initial_scale <= 0:
        parser.error("--initial-scale must be finite and > 0")
    return args


def _resolve_manifest_path(manifest_path: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PreflightError(f"{label} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def load_inputs(path: Path, expected_seeds: Sequence[int]) -> dict[int, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot load input manifest {path}: {exc}") from exc
    seeds = _require(payload, "seeds")
    if not isinstance(seeds, list):
        raise PreflightError("inputs.seeds must be a list")
    result: dict[int, dict[str, Any]] = {}
    for item in seeds:
        if not isinstance(item, Mapping):
            raise PreflightError("each seed input must be an object")
        seed = _require(item, "training_seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise PreflightError("input training_seed must be an integer")
        if seed in result:
            raise PreflightError(f"duplicate input for seed {seed}")
        config = dict(item)
        for field in ("training_state", "checkpoint", "data"):
            config[field] = _resolve_manifest_path(path, _require(item, field), field)
        config["lr"] = _finite_float(item.get("lr", 1e-4), "lr", positive=True)
        betas = item.get("betas", [0.9, 0.999])
        if (
            not isinstance(betas, list)
            or len(betas) != 2
            or any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                for value in betas
            )
        ):
            raise PreflightError("betas must be a two-number list")
        config["betas"] = tuple(
            _finite_float(value, f"betas[{index}]") for index, value in enumerate(betas)
        )
        if any(value < 0 or value >= 1 for value in config["betas"]):
            raise PreflightError("betas must lie in [0, 1)")
        config["eps_opt"] = _finite_float(
            item.get("eps_opt", 1e-8), "eps_opt", positive=True
        )
        result[seed] = config
    if set(result) != set(expected_seeds):
        raise PreflightError(
            f"input manifest seeds {sorted(result)} != expected {sorted(expected_seeds)}"
        )
    return result


def _asset_hashes(config: Mapping[str, Any]) -> dict[str, Any]:
    for field in ("training_state", "checkpoint", "data"):
        path = config[field]
        if field == "data":
            if not path.exists():
                raise PreflightError(f"missing data asset: {path}")
        elif not path.is_file():
            raise PreflightError(f"missing {field} asset: {path}")
    dataset_hash, dataset_algorithm = gauge.dataset_sha256(config["data"])
    return {
        "source_state_sha256": gauge.sha256_file(config["training_state"]),
        "checkpoint_sha256": gauge.sha256_file(config["checkpoint"]),
        "dataset_sha256": dataset_hash,
        "dataset_hash_algorithm": dataset_algorithm,
    }


def _next_batch(dataset, *, batch_size: int, seed: int, device: torch.device):
    from torch.utils.data import DataLoader

    generator = torch.Generator(device="cpu").manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )
    images, labels = next(iter(loader))
    return images.to(device).to(torch.float32) / 127.5 - 1, labels.to(device)


def run(args) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    receipts = load_receipts(args.receipt_root)
    calibration_rows, frozen_a, receipt_index = build_calibration(
        receipts, expected_seeds=args.expected_seeds
    )
    _write_csv(args.out / "calibration.csv", CALIBRATION_FIELDS, calibration_rows)
    expected_audit_hashes = {row["audit_library_sha256"] for row in calibration_rows}
    expected_runner_hashes = {
        row["factorial_runner_sha256"] for row in calibration_rows
    }
    current_audit_hash = gauge.sha256_file(Path(audit_lib.__file__))
    current_runner_hash = gauge.sha256_file(
        REPO_ROOT / "analysis" / "q256_gradient_state_factorial.py"
    )
    if expected_audit_hashes != {current_audit_hash}:
        raise PreflightError(
            "current R_opt audit implementation hash differs from factorial receipts"
        )
    if expected_runner_hashes != {current_runner_hash}:
        raise PreflightError(
            "current factorial runner hash differs from formal receipts"
        )
    inputs = load_inputs(args.inputs, args.expected_seeds)
    device = torch.device(args.device)
    if args.amp and (device.type != "cuda" or not torch.cuda.is_available()):
        raise PreflightError("formal AMP preflight requires an available CUDA device")

    batch_rows: list[dict[str, Any]] = []
    source_assets: dict[str, Any] = {}
    for seed in args.expected_seeds:
        seed_calibration = [
            row for row in calibration_rows if row["training_seed"] == seed
        ]
        receipt_meta = seed_calibration[0]
        config = inputs[seed]
        hashes_before = _asset_hashes(config)
        for field in ("source_state_sha256", "checkpoint_sha256", "dataset_sha256"):
            if hashes_before[field] != receipt_meta[field]:
                raise PreflightError(
                    f"seed {seed}: input {field} differs from factorial receipt"
                )
        for optional, field in (
            ("expected_training_state_sha256", "source_state_sha256"),
            ("expected_checkpoint_sha256", "checkpoint_sha256"),
            ("expected_data_sha256", "dataset_sha256"),
        ):
            if optional in config:
                expected_hash = _sha256(config[optional], optional)
                if expected_hash != hashes_before[field]:
                    raise PreflightError(
                        f"seed {seed}: {optional} does not match actual asset"
                    )

        loss = audit_lib.load_loss_from_checkpoint(config["checkpoint"])
        if float(loss.q) != 256.0:
            raise PreflightError(f"seed {seed}: expected q=256, got {loss.q}")
        net, optimizer, scaler_state, loss_fn_state, state_meta = (
            audit_lib.load_training_state(
                config["training_state"],
                device,
                lr=config["lr"],
                betas=config["betas"],
                eps_opt=config["eps_opt"],
            )
        )
        if args.amp and scaler_state is None:
            raise PreflightError(f"seed {seed}: AMP source lacks gradscaler_state")
        if loss_fn_state is not None and hasattr(loss, "load_schedule_state_dict"):
            if not loss.load_schedule_state_dict(copy.deepcopy(loss_fn_state)):
                raise PreflightError(f"seed {seed}: incompatible loss_fn_state")
        if state_meta["cur_nimg"] is None or int(state_meta["cur_nimg"]) != 256000:
            raise PreflightError(f"seed {seed}: formal source cur_nimg must be 256000")
        source_state_hash = gauge.state_sha256(
            {
                "parameters": gauge.module_state_hashes(net),
                "optimizer": gauge.state_sha256(optimizer.state_dict()),
            }
        )
        if source_state_hash != receipt_meta["source_state_hash"]:
            raise PreflightError(f"seed {seed}: runtime source-state hash mismatch")

        first_receipt = receipt_index[(seed, seed_calibration[0]["audit_batch_id"])]
        provenance = _require(first_receipt, "provenance")
        batch_size = _positive_int(_require(provenance, "batch_size"), "batch_size")
        microbatch_size = _positive_int(_require(provenance, "batch_gpu"), "batch_gpu")
        support_atol = _finite_float(
            _require(provenance, "support_atol"), "support_atol"
        )
        if support_atol < 0:
            raise PreflightError("support_atol must be >= 0")
        from training.dataset import ImageFolderDataset

        dataset = ImageFolderDataset(
            path=str(config["data"]),
            use_labels=False,
            xflip=False,
            cache=True,
            resolution=net.img_resolution,
        )
        heldout = [row for row in seed_calibration if row["split"] == "heldout"]
        for calibration_row in heldout:
            audit_batch_id = calibration_row["audit_batch_id"]
            receipt = receipt_index[(seed, audit_batch_id)]
            images, labels = _next_batch(
                dataset, batch_size=batch_size, seed=audit_batch_id, device=device
            )
            row = run_heldout_virtual_batch(
                net,
                optimizer,
                loss,
                images,
                labels,
                receipt=receipt,
                a_s=frozen_a[seed],
                scaler_state=scaler_state,
                amp=args.amp,
                initial_scale=args.initial_scale,
                microbatch_size=microbatch_size,
                support_atol=support_atol,
            )
            row.update(
                {
                    "canonical_rank": calibration_row["canonical_rank"],
                    "split": "heldout",
                }
            )
            batch_rows.append(row)
            _write_csv(
                args.out / "preflight_batch.csv",
                PREFLIGHT_BATCH_FIELDS,
                sorted(
                    batch_rows,
                    key=lambda item: (item["training_seed"], item["canonical_rank"]),
                ),
            )
        hashes_after = _asset_hashes(config)
        source_unchanged = hashes_before == hashes_after
        if not source_unchanged:
            raise PreflightError(
                f"seed {seed}: source asset hash changed during preflight"
            )
        for row in batch_rows:
            if row["training_seed"] == seed:
                row["source_checkpoint_unchanged"] = source_unchanged
        source_assets[str(seed)] = {
            "training_state": str(config["training_state"]),
            "checkpoint": str(config["checkpoint"]),
            "data": str(config["data"]),
            **hashes_after,
            "runtime_source_state_hash": source_state_hash,
        }

    batch_rows.sort(key=lambda row: (row["training_seed"], row["canonical_rank"]))
    seed_rows, verdict = evaluate_go_gate(
        batch_rows, frozen_a, expected_seeds=args.expected_seeds
    )
    verdict = dict(verdict)
    verdict.update(
        {
            "source_assets": source_assets,
            "batch_checks": [
                {
                    "training_seed": row["training_seed"],
                    "audit_batch_id": row["audit_batch_id"],
                    "canonical_rank": row["canonical_rank"],
                    "source_hash_match": row["source_hash_match"],
                    "gradient_hash_match": row["gradient_hash_match"],
                    "randomness_hash_match": row["randomness_hash_match"],
                    "source_preserved": row["source_preserved"],
                    "transport_contract": row["_transport_contract"],
                    "order_checks": row["_order_checks"],
                }
                for row in batch_rows
            ],
            "batch_membership": {
                str(seed): {
                    "calibration": [
                        row["audit_batch_id"]
                        for row in calibration_rows
                        if row["training_seed"] == seed
                        and row["split"] == "calibration"
                    ],
                    "heldout": [
                        row["audit_batch_id"]
                        for row in calibration_rows
                        if row["training_seed"] == seed and row["split"] == "heldout"
                    ],
                }
                for seed in args.expected_seeds
            },
            "factorial_code_commits": sorted(
                {row["code_commit"] for row in calibration_rows}
            ),
            "receipt_root": str(args.receipt_root.resolve()),
            "input_manifest": str(args.inputs.resolve()),
            "preflight_code_commit": audit_lib._source_commit(),
            "preflight_runner_sha256": gauge.sha256_file(Path(__file__)),
            "audit_library_sha256": gauge.sha256_file(Path(audit_lib.__file__)),
            "command_line": list(sys.argv),
            "environment": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": str(device),
                "amp": args.amp,
            },
        }
    )
    emit_outputs(args.out, calibration_rows, batch_rows, seed_rows, verdict)
    print(
        json.dumps(
            {
                "status": verdict["status"],
                "formal_training_authorized": verdict["formal_training_authorized"],
                "frozen_a_s": verdict["frozen_a_s"],
            },
            sort_keys=True,
        )
    )
    return 0 if verdict["status"] == "GO" else 4


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (Exception, SystemExit) as exc:
        failure = {
            "schema_version": 1,
            "status": "FAIL_CLOSED",
            "formal_training_authorized": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "operations_excluded": ["training", "sample_generation", "FID", "KID"],
        }
        _strict_dump(args.out / "preflight_verdict.json", failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
