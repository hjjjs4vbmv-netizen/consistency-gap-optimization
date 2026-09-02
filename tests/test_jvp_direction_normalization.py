"""Direction construction contracts for the Jacobian failure factorial."""
from __future__ import annotations

import math
import unittest

import torch

from analysis.jacobian_failure_factorial.core import parameter_direction_in_full_state
from analysis.operator_clock_gate.core import state_relative_direction_like


class JVPDirectionNormalizationTests(unittest.TestCase):
    def test_per_tensor_rms_is_frozen_and_reproducible(self):
        values = {
            "large": torch.tensor([3.0, 4.0], dtype=torch.float32),
            "zero": torch.zeros(7, dtype=torch.float32),
        }
        first = state_relative_direction_like(values, 2026082611)
        second = state_relative_direction_like(values, 2026082611)
        for name, value in values.items():
            self.assertTrue(torch.equal(first[name], second[name]))
            observed = math.sqrt(float(first[name].square().mean()))
            expected = max(math.sqrt(float(value.double().square().mean())), 1e-3)
            self.assertAlmostEqual(observed, expected, places=12)

    def test_parameter_direction_embeds_with_zero_nonparameter_inputs(self):
        continuous = {
            "theta.weight": torch.tensor([1.0], dtype=torch.float64),
            "optimizer.weight.exp_avg": torch.tensor([0.2], dtype=torch.float64),
            "ema.weight": torch.tensor([0.9], dtype=torch.float64),
        }
        parameter = {"weight": torch.tensor([0.3], dtype=torch.float64)}
        result = parameter_direction_in_full_state(continuous, parameter)
        self.assertTrue(torch.equal(result["theta.weight"], parameter["weight"]))
        self.assertEqual(float(result["optimizer.weight.exp_avg"]), 0.0)
        self.assertEqual(float(result["ema.weight"]), 0.0)


if __name__ == "__main__":
    unittest.main()
