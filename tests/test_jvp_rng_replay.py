"""RNG replay contracts for all paired FP32 branches."""
from __future__ import annotations

import unittest

from analysis.jacobian_failure_factorial.core import gauss_newton_convergence
from analysis.operator_clock_gate import core as gate
from tests.test_operator_gate_state_preservation import fixture


class JVPRNGReplayTests(unittest.TestCase):
    def test_squared_gn_is_reproducible_and_restores_entry_rng(self):
        state, batches = fixture()
        direction = gate.state_relative_direction_like(
            gate.parameter_vector(state.net), 2026082611)
        before = gate.rng_sha256()
        first, first_receipt = gauss_newton_convergence(
            state.net, state.loss_fn, [batches[0]], direction, arm="A",
            epsilons=(1e-2, 3e-3, 1e-3), tolerance=1.0,
            residual_geometry=False)
        middle = gate.rng_sha256()
        second, second_receipt = gauss_newton_convergence(
            state.net, state.loss_fn, [batches[0]], direction, arm="A",
            epsilons=(1e-2, 3e-3, 1e-3), tolerance=1.0,
            residual_geometry=False)
        self.assertEqual(before, middle)
        self.assertEqual(before, gate.rng_sha256())
        self.assertEqual(gate.tensor_map_sha256(first), gate.tensor_map_sha256(second))
        self.assertTrue(first_receipt["source_preserved"])
        self.assertTrue(second_receipt["source_preserved"])


if __name__ == "__main__":
    unittest.main()
