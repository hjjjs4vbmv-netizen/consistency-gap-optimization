"""Carryover-corrected feedback uses the transition's real retention rules."""
from __future__ import annotations

import copy
import csv
import json
import unittest
from pathlib import Path

import torch

from analysis.nonlinear_dynamics_gate.decompose_forcing_feedback import (
    carryover_only_map,
    exact_three_point_metrics,
)
from analysis.operator_clock_gate.core import AlgorithmicState


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "analysis" / "nonlinear_dynamics_gate"


class TwoParameterNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.right = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def forward(self, value):
        return value * self.left + self.right


def state_with_nonuniform_radam_betas() -> AlgorithmicState:
    net = TwoParameterNet()
    optimizer = torch.optim.RAdam([
        {"params": [net.left], "betas": (0.8, 0.95)},
        {"params": [net.right], "betas": (0.9, 0.99)},
    ], lr=1e-3)
    return AlgorithmicState(
        net=net,
        optimizer=optimizer,
        ema=copy.deepcopy(net).eval().requires_grad_(False),
        loss_fn=object(),
        ema_beta=0.7,
    )


class CarryoverCorrectedFeedbackTests(unittest.TestCase):
    def test_theta_subtracts_the_incoming_parameter_separation(self):
        state = state_with_nonuniform_radam_betas()
        pre_a = {"left": torch.tensor(1.0), "right": torch.tensor(2.0)}
        pre_x = {"left": torch.tensor(3.0), "right": torch.tensor(6.0)}
        carryover, metadata = carryover_only_map("theta", pre_a, pre_x, state)
        self.assertEqual(metadata["retention_values"], [1.0])
        metrics = exact_three_point_metrics(
            {"left": torch.tensor(10.0), "right": torch.tensor(20.0)},
            {"left": torch.tensor(11.0), "right": torch.tensor(22.0)},
            {"left": torch.tensor(14.0), "right": torch.tensor(27.0)},
            pre_baseline=pre_a,
            pre_actual=pre_x,
            carryover=carryover,
        )
        # R=(3,5), Delta_k=(2,4), so R_tilde=(1,1).
        self.assertAlmostEqual(metrics["corrected_R_norm"], 2.0 ** 0.5)
        self.assertAlmostEqual(metrics["delta_k_norm"], 20.0 ** 0.5)
        self.assertAlmostEqual(
            metrics["corrected_R_over_delta_k"], (2.0 / 20.0) ** 0.5)

    def test_radam_m_and_v_use_each_parameter_groups_actual_betas(self):
        state = state_with_nonuniform_radam_betas()
        pre_a = {
            "left": torch.tensor(0.0, dtype=torch.float64),
            "right": torch.tensor(0.0, dtype=torch.float64),
        }
        pre_x = {
            "left": torch.tensor(2.0, dtype=torch.float64),
            "right": torch.tensor(4.0, dtype=torch.float64),
        }
        carry_m, metadata_m = carryover_only_map("m", pre_a, pre_x, state)
        carry_v, metadata_v = carryover_only_map("v", pre_a, pre_x, state)
        self.assertEqual(metadata_m["retention_values"], [0.8, 0.9])
        self.assertEqual(metadata_v["retention_values"], [0.95, 0.99])
        self.assertAlmostEqual(float(carry_m["left"]), 1.6)
        self.assertAlmostEqual(float(carry_m["right"]), 3.6)
        self.assertAlmostEqual(float(carry_v["left"]), 1.9)
        self.assertAlmostEqual(float(carry_v["right"]), 3.96)

    def test_ema_uses_parameter_lerp_and_identity_for_untouched_buffers(self):
        state = state_with_nonuniform_radam_betas()
        pre_a = {
            "parameter.left": torch.tensor(1.0),
            "buffer.running": torch.tensor(2.0),
        }
        pre_x = {
            "parameter.left": torch.tensor(5.0),
            "buffer.running": torch.tensor(8.0),
        }
        carryover, metadata = carryover_only_map("EMA", pre_a, pre_x, state)
        self.assertEqual(
            metadata["rule"], "ema_transition_carryover_counterfactual_map")
        self.assertEqual(metadata["retention_values"], [0.7, 1.0])
        self.assertAlmostEqual(float(carryover["parameter.left"]), 2.8)
        self.assertAlmostEqual(float(carryover["buffer.running"]), 6.0)

    def test_skipped_radam_step_uses_identity_or_fails_closed_on_mismatch(self):
        state = state_with_nonuniform_radam_betas()
        pre_a = {
            "left": torch.tensor(0.0, dtype=torch.float64),
            "right": torch.tensor(0.0, dtype=torch.float64),
        }
        pre_x = {
            "left": torch.tensor(2.0, dtype=torch.float64),
            "right": torch.tensor(4.0, dtype=torch.float64),
        }
        carryover, metadata = carryover_only_map(
            "m", pre_a, pre_x, state, optimizer_step_skipped=True)
        self.assertEqual(
            metadata["rule"], "optimizer_step_skipped_identity_carryover")
        self.assertEqual(metadata["retention_values"], [1.0])
        self.assertAlmostEqual(float(carryover["left"]), 2.0)
        self.assertAlmostEqual(float(carryover["right"]), 4.0)
        undefined, mismatch = carryover_only_map(
            "m", pre_a, pre_x, state, optimizer_skip_regime_paired=False)
        self.assertIsNone(undefined)
        self.assertEqual(
            mismatch["rule"],
            "undefined_across_optimizer_skip_regime_mismatch")

    def test_readouts_without_a_declared_carryover_map_fail_open_to_na(self):
        state = state_with_nonuniform_radam_betas()
        carryover, metadata = carryover_only_map(
            "residual", {"value": torch.tensor(0.0)},
            {"value": torch.tensor(1.0)}, state)
        self.assertIsNone(carryover)
        self.assertEqual(metadata["rule"], "not_declared_for_this_readout")

    def test_compact_formal_outputs_obey_v2_schema(self):
        summary = json.loads(
            (RESULT_ROOT / "forcing_feedback_summary_v2.json").read_text(
                encoding="utf-8"))
        self.assertEqual(summary["schema_version"], 3)
        self.assertEqual(summary["status"], "PASS")
        self.assertTrue(summary["exact_closure"]["all_pass"])
        self.assertEqual(
            summary["strong_expansion_claim_gate"]["status"], "WITHHELD")
        self.assertFalse(summary["strong_expansion_claim_gate"][
            "second_state_replication_available"])
        self.assertNotIn("step_replay_receipts", summary["run_receipt"])
        self.assertEqual(summary["run_receipt"][
            "step_replay_receipt_count"], 64)
        classifications = {
            item["classification"]
            for item in summary["mechanism_by_arm_and_block"].values()
        }
        self.assertNotIn("trajectory_feedback_amplification", classifications)

    def test_formal_csv_has_pretransition_and_corrected_metrics(self):
        with (RESULT_ROOT / "forcing_feedback_per_step_v2.csv").open(
                newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1536)
        self.assertTrue(all(row["delta_k_norm"] for row in rows))
        self.assertTrue(all(row["feedback_gain_G"] for row in rows))
        corrected = [
            row for row in rows
            if row["space"] == "state"
            and row["block"] in {"theta", "EMA", "m", "v"}
        ]
        self.assertEqual(len(corrected), 768)
        self.assertTrue(all(row["corrected_R_norm"] for row in corrected))
        self.assertTrue(all(row["corrected_R_over_delta_k"]
                            for row in corrected))
        self.assertTrue(all(row["corrected_R_over_b"] for row in corrected))
        self.assertTrue(all(row["closure_pass"] == "True" for row in rows))


if __name__ == "__main__":
    unittest.main()
