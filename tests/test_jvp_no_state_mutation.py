"""Every diagnostic is read-only with respect to the source state."""
from __future__ import annotations

import unittest

from analysis.jacobian_failure_factorial import core
from analysis.operator_clock_gate import core as gate
from tests.test_operator_gate_state_preservation import fixture


class JVPNoStateMutationTests(unittest.TestCase):
    def test_fp32_and_algorithmic_regimes_preserve_source_and_rng(self):
        state, batches = fixture()
        direction = gate.state_relative_direction_like(
            gate.parameter_vector(state.net), 2026082613)
        state_before = state.sha256()
        rng_before = gate.rng_sha256()
        _, a = core.gauss_newton_convergence(
            state.net, state.loss_fn, [batches[0]], direction, arm="A",
            epsilons=(1e-2, 3e-3, 1e-3), tolerance=1.0,
            residual_geometry=False)
        _, c = core.full_field_fp32_jvp(
            state.net, state.loss_fn, [batches[0]], direction, arm="A",
            epsilons=(1e-2, 3e-3, 1e-3), tolerance=1.0,
            c_override=0.0)
        _, d = core.parameter_partial_algorithmic_jvp(
            state, batches[0], direction, arm="A",
            epsilons=(1e-3, 3e-4, 1e-4), tolerance=1.0)
        self.assertTrue(a["source_preserved"])
        self.assertTrue(c["source_preserved"])
        self.assertTrue(d["source_preserved"])
        self.assertTrue(d["no_in_place_source_pollution"])
        self.assertEqual(state.sha256(), state_before)
        self.assertEqual(gate.rng_sha256(), rng_before)


if __name__ == "__main__":
    unittest.main()
