"""Pure contracts for q256 moment-transport calibration and GO gating."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import tempfile
from pathlib import Path
import unittest

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "analysis" / "q256_moment_transport_preflight.py"
SPEC = importlib.util.spec_from_file_location(
    "q256_moment_transport_preflight", MODULE_PATH
)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PREFLIGHT)


def _sha(character: str) -> str:
    return character * 64


def _receipt(seed: int, batch_id: int, a_star: float) -> dict:
    control_hash = _sha("1")
    treatment_hash = _sha("2")
    source_hash = _sha(str(seed))
    branch = {
        "step_skipped": False,
        "optimizer_step_advanced_exactly_once": True,
        "clone_contract": {"independent": True},
    }
    cells = {}
    order = {}
    for name in PREFLIGHT.FACTORIAL_CELLS:
        cells[name] = {
            "a_star": a_star,
            "pair_fitted_a_star": a_star,
            "finite_gate": True,
            "branch_skipped_flag": False,
            "source_state_hash": source_hash,
            "control_gradient_sha256": control_hash,
            "treatment_gradient_sha256": treatment_hash,
            "branches": {
                "control": copy.deepcopy(branch),
                "treatment": copy.deepcopy(branch),
            },
            "whole_model": {"a_star": a_star},
        }
        order[name] = {"numerically_invariant": True, "result_hash_identical": True}
    numerator = 10.0 * a_star
    return {
        "schema_version": 1,
        "training_seed": seed,
        "audit_batch_id": batch_id,
        "audit_seed": batch_id,
        "a_star": a_star,
        "a_star_fit": {
            "definition": "dot(G_1.00,G_1.10)/dot(G_1.00,G_1.00)",
            "float_accumulation": "float64",
            "numerator": numerator,
            "denominator": 10.0,
            "denominator_atol": 1e-30,
            "unclamped": True,
        },
        "correctness_gate": {"valid": True},
        "source_state_non_committing": {
            "preserved": True,
            "source_state_hash": source_hash,
        },
        "cells": cells,
        "order_invariance_and_rerun": order,
        "gradient_contract": {
            "observed_computed_once": True,
            "observed_control": {
                "finite": True,
                "unscaled": True,
                "gradient_sha256": control_hash,
            },
            "observed_treatment": {
                "finite": True,
                "unscaled": True,
                "gradient_sha256": treatment_hash,
            },
            "observed_pair_sha256_before": _sha("4"),
            "observed_pair_sha256_after": _sha("4"),
        },
        "provenance": {
            "training_seed": seed,
            "q": 256.0,
            "reference_gap_scale": 1.0,
            "probe_gap_scale": 1.1,
            "source_state_sha256": _sha(str(seed)),
            "checkpoint_sha256": _sha("a"),
            "dataset_sha256": _sha("b"),
            "code_commit": "c" * 40,
            "runner_sha256": _sha("d"),
            "audit_library_sha256": _sha("e"),
            "batch_size": 128,
            "batch_gpu": 16,
            "support_atol": 0.0,
        },
    }


def _gate_rows(
    *, r_g: float = 0.4, r_t: float = 0.1, r_exact: float = 0.005, norm_t: float = 1.0
):
    rows = []
    frozen = {3: 0.9, 4: 0.91, 5: 0.92}
    true_fields = (
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
    for seed in (3, 4, 5):
        for rank in PREFLIGHT.HELDOUT_RANKS:
            row = {
                "training_seed": seed,
                "audit_batch_id": PREFLIGHT.FORMAL_AUDIT_BATCH_IDS[rank],
                "canonical_rank": rank,
                "split": "heldout",
                "a_s_seed": frozen[seed],
                "R_opt_G": r_g,
                "R_opt_T": r_t,
                "R_opt_T_exact": r_exact,
                "update_norm_ratio_T": norm_t,
            }
            row.update({field: True for field in true_fields})
            rows.append(row)
    return rows


class MomentTransportPreflightTests(unittest.TestCase):
    def test_canonical_sort_split_and_calibration_median_are_frozen(self):
        receipts = []
        values = (0.80, 1.90, 1.00, 2.00, 1.20, 2.10, 1.40, 2.20)
        for seed in (3, 4, 5):
            for offset, value in reversed(tuple(enumerate(values))):
                receipts.append(
                    _receipt(seed, 2026081101 + offset, value + seed * 0.001)
                )
        rows, frozen, index = PREFLIGHT.build_calibration(receipts)
        self.assertEqual(len(rows), 24)
        self.assertEqual(len(index), 24)
        for seed in (3, 4, 5):
            seed_rows = [row for row in rows if row["training_seed"] == seed]
            self.assertEqual(
                [row["canonical_rank"] for row in seed_rows], list(range(8))
            )
            self.assertEqual(
                [
                    row["canonical_rank"]
                    for row in seed_rows
                    if row["split"] == "calibration"
                ],
                list(PREFLIGHT.CALIBRATION_RANKS),
            )
            expected = statistics_median(
                [values[index] + seed * 0.001 for index in PREFLIGHT.CALIBRATION_RANKS]
            )
            self.assertAlmostEqual(frozen[seed], expected)
            self.assertTrue(all(row["a_s_seed"] == frozen[seed] for row in seed_rows))
        wrong_ids = copy.deepcopy(receipts)
        wrong_ids[-1]["audit_batch_id"] = 2026081199
        wrong_ids[-1]["audit_seed"] = 2026081199
        with self.assertRaisesRegex(
            PREFLIGHT.PreflightError, "frozen formal audit ids"
        ):
            PREFLIGHT.build_calibration(wrong_ids)

    def test_batch_id_rejects_ambiguous_spellings(self):
        self.assertEqual(PREFLIGHT.canonical_batch_id("2026081101"), 2026081101)
        for value in (True, 1.0, "01", "+1", " 1"):
            with self.subTest(value=value):
                with self.assertRaises(PREFLIGHT.PreflightError):
                    PREFLIGHT.canonical_batch_id(value)

    def test_scale_can_fall_back_to_validated_top_level(self):
        receipt = _receipt(3, 1, 0.925)
        del receipt["a_star_fit"]["numerator"]
        del receipt["a_star_fit"]["denominator"]
        value, source, numerator, denominator = PREFLIGHT.derive_batch_scale(receipt)
        self.assertEqual(value, 0.925)
        self.assertEqual(source, "top_level_validated_fit")
        self.assertIsNone(numerator)
        self.assertIsNone(denominator)

    def test_scale_cross_checks_whole_model_and_fails_on_missing_gate(self):
        inconsistent = _receipt(3, 1, 0.925)
        inconsistent["cells"]["observed_real"]["whole_model"]["a_star"] = 0.8
        with self.assertRaisesRegex(PREFLIGHT.PreflightError, "mismatch"):
            PREFLIGHT.derive_batch_scale(inconsistent)
        missing = _receipt(3, 1, 0.925)
        del missing["correctness_gate"]["valid"]
        with self.assertRaisesRegex(PREFLIGHT.PreflightError, "missing required field"):
            PREFLIGHT.derive_batch_scale(missing)

    def test_exact_update_metrics_use_reference_normalized_projection(self):
        reference = {
            "b": torch.tensor([2.0], dtype=torch.float64),
            "a": torch.tensor([1.0, 0.0], dtype=torch.float64),
        }
        candidate = {
            "b": torch.tensor([4.0], dtype=torch.float64),
            "a": torch.tensor([2.0, 1.0], dtype=torch.float64),
        }
        metrics = PREFLIGHT.exact_update_metrics(reference, candidate)
        self.assertEqual(metrics["s_star"], 2.0)
        self.assertAlmostEqual(metrics["c_star"], 10.0 / 21.0)
        self.assertAlmostEqual(metrics["R_opt"], 1.0 / math.sqrt(5.0))
        self.assertAlmostEqual(metrics["update_norm_ratio"], math.sqrt(21.0 / 5.0))

    def test_go_gate_passes_only_all_preregistered_thresholds(self):
        frozen = {3: 0.9, 4: 0.91, 5: 0.92}
        seed_rows, verdict = PREFLIGHT.evaluate_go_gate(_gate_rows(), frozen)
        self.assertEqual(verdict["status"], "GO")
        self.assertTrue(verdict["formal_training_authorized"])
        self.assertEqual(verdict["gates"]["cross_seed_median_suppression"], 0.75)
        self.assertTrue(all(row["seed_gate_pass"] for row in seed_rows))

        no_go_rows = _gate_rows(r_t=0.24)
        _, no_go = PREFLIGHT.evaluate_go_gate(no_go_rows, frozen)
        self.assertEqual(no_go["status"], "NO_GO")
        self.assertFalse(no_go["gates"]["cross_seed_median_suppression_at_least_0p50"])
        no_go_rows[0]["branch_order_pass"] = False
        _, no_go = PREFLIGHT.evaluate_go_gate(no_go_rows, frozen)
        self.assertFalse(no_go["gates"]["branch_order_invariant"])

    def test_go_gate_rejects_wrong_membership_or_coefficient(self):
        frozen = {3: 0.9, 4: 0.91, 5: 0.92}
        wrong_rank = _gate_rows()
        wrong_rank[0]["canonical_rank"] = 0
        with self.assertRaisesRegex(PREFLIGHT.PreflightError, "held-out ranks"):
            PREFLIGHT.evaluate_go_gate(wrong_rank, frozen)
        wrong_split = _gate_rows()
        wrong_split[0]["split"] = "calibration"
        with self.assertRaisesRegex(PREFLIGHT.PreflightError, "non-heldout"):
            PREFLIGHT.evaluate_go_gate(wrong_split, frozen)
        wrong_id = _gate_rows()
        wrong_id[0]["audit_batch_id"] = PREFLIGHT.FORMAL_AUDIT_BATCH_IDS[0]
        with self.assertRaisesRegex(PREFLIGHT.PreflightError, "wrong audit ids"):
            PREFLIGHT.evaluate_go_gate(wrong_id, frozen)
        wrong_a = _gate_rows()
        wrong_a[0]["a_s_seed"] = 0.8
        with self.assertRaisesRegex(PREFLIGHT.PreflightError, "mismatch"):
            PREFLIGHT.evaluate_go_gate(wrong_a, frozen)

    def test_emit_is_deterministic_and_strict_json(self):
        frozen = {3: 0.9, 4: 0.91, 5: 0.92}
        batch_rows = _gate_rows()
        seed_rows, verdict = PREFLIGHT.evaluate_go_gate(batch_rows, frozen)
        receipts = [
            _receipt(seed, batch_id, 0.9 + seed * 0.001)
            for seed in (3, 4, 5)
            for batch_id in PREFLIGHT.FORMAL_AUDIT_BATCH_IDS
        ]
        calibration, _, _ = PREFLIGHT.build_calibration(receipts)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            PREFLIGHT.emit_outputs(first, calibration, batch_rows, seed_rows, verdict)
            PREFLIGHT.emit_outputs(second, calibration, batch_rows, seed_rows, verdict)
            for name in (
                "calibration.csv",
                "preflight_batch.csv",
                "preflight_seed.csv",
                "preflight_verdict.json",
            ):
                self.assertEqual(
                    (first / name).read_bytes(), (second / name).read_bytes()
                )
            parsed = json.loads(
                (first / "preflight_verdict.json").read_text(encoding="utf-8")
            )
            self.assertEqual(parsed["status"], "GO")


def statistics_median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (ordered[middle - 1] + ordered[middle]) / 2


if __name__ == "__main__":
    unittest.main()
