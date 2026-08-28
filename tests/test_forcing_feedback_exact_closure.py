"""Exact closure and block separation for nonlinear forcing/feedback."""
from __future__ import annotations

import copy
import unittest

import torch

from analysis.nonlinear_dynamics_gate.decompose_forcing_feedback import (
    _state_tensor_blocks,
    audit_legacy_rollout,
    exact_three_point_metrics,
    run_exact_decomposition,
)
from analysis.operator_clock_gate.core import AlgorithmicState, freeze_batches


class TinyLoss:
    P_mean = -0.5
    P_std = 0.2
    q = 8.0
    k = 2.0
    b = 1.0
    c = 0.1
    stage = 0

    def schedule_state_dict(self):
        return {"q": self.q, "k": self.k, "b": self.b, "stage": self.stage}


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))

    def forward(self, x, sigma, labels, augment_labels=None):
        del labels, augment_labels
        return self.weight * x + self.bias * sigma


def fixture():
    torch.manual_seed(17)
    net = TinyNet().train()
    optimizer = torch.optim.RAdam(net.parameters(), lr=1e-2)
    for parameter in net.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state = AlgorithmicState(
        net, optimizer, copy.deepcopy(net).eval().requires_grad_(False),
        TinyLoss(), ema_beta=0.9,
    )
    images = torch.linspace(-0.9, 0.9, 8, dtype=torch.float64).reshape(8, 1, 1, 1)
    labels = torch.empty(8, 0, dtype=torch.float64)
    return state, freeze_batches([(images, labels)], state.loss_fn, (2026082701,))


class ExactClosureTests(unittest.TestCase):
    def test_legacy_projection_receipt_fails_recoverability_gate(self):
        audit = audit_legacy_rollout()
        self.assertFalse(audit["recoverable"])
        self.assertIn(audit["reason"], {
            "receipt_missing",
            "only projections/summaries/hashes are stored; full per-k augmented state absent",
        })

    def test_three_point_identity_with_cancellation(self):
        a = {"x": torch.tensor([1.0, 2.0], dtype=torch.float64)}
        c = {"x": torch.tensor([4.0, 2.0], dtype=torch.float64)}
        x = {"x": torch.tensor([2.0, 2.0], dtype=torch.float64)}
        result = exact_three_point_metrics(a, c, x)
        self.assertTrue(result["closure_pass"])
        self.assertEqual(result["closure_l2"], 0.0)
        self.assertAlmostEqual(result["cos_b_R"], -1.0)
        self.assertGreater(result["b_norm"], result["delta_norm"])

    def test_rollout_closes_for_each_required_block_and_observable(self):
        state, batches = fixture()
        source_hash = state.sha256()
        rows, receipt = run_exact_decomposition(state, batches, steps=2)
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(state.sha256(), source_hash)
        self.assertTrue(receipt["all_exact_closures_pass"])
        self.assertEqual(len(receipt["state_hashes_k_0_through_horizon"]), 3)
        for arm in "BCD":
            arm_rows = [row for row in rows if row["arm"] == arm]
            blocks = {(row["space"], row["block"]) for row in arm_rows}
            self.assertTrue({("state", item) for item in ("theta", "EMA", "m", "v")}
                            .issubset(blocks))
            self.assertTrue({("observable", "residual"), ("observable", "feature")}
                            .issubset(blocks))
            self.assertTrue(all(row["closure_pass"] for row in arm_rows))

    def test_state_blocks_are_not_collapsed_to_one_augmented_norm(self):
        state, _ = fixture()
        blocks = _state_tensor_blocks(state)
        self.assertTrue({"theta", "EMA", "m", "v"}.issubset(blocks))
        self.assertNotIn("augmented_state", blocks)


if __name__ == "__main__":
    unittest.main()
