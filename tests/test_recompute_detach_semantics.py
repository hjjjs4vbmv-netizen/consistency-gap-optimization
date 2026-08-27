"""Fresh target recomputation and one-sided stop-gradient contracts."""
from __future__ import annotations

import unittest

from analysis.jacobian_failure_factorial.core import full_field_fp32_jvp
from analysis.operator_clock_gate import core as gate
from tests.test_operator_gate_state_preservation import fixture


class RecomputeDetachSemanticsTests(unittest.TestCase):
    def test_every_perturbed_field_recomputes_and_detaches_target(self):
        state, batches = fixture()
        direction = gate.state_relative_direction_like(
            gate.parameter_vector(state.net), 2026082612)
        _, receipt = full_field_fp32_jvp(
            state.net, state.loss_fn, [batches[0]], direction, arm="B",
            epsilons=(1e-2, 3e-3, 1e-3), tolerance=1.0,
            c_override=0.0)
        self.assertIn("fresh target", receipt["definition_guard"])
        for branch in receipt["branches"]:
            self.assertTrue(branch["paired_target_recomputation"])
            for side in ("plus", "minus"):
                self.assertEqual(branch[side]["target_recompute_count"], 1)
                self.assertTrue(branch[side]["all_targets_detached"])
                self.assertEqual(len(branch[side]["target_hashes"]), 1)


if __name__ == "__main__":
    unittest.main()
