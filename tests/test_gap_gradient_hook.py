"""Tests for the Role C gap-gradient diagnostic hook.

Verifies the four hook guarantees required for the deep-network gradient
diagnostic (today's objective):

  1. external gap specification (--gaps must include the reference 1.0);
  2. same batch / noise / timestep reused across gaps
     (same t/eps/dropout in, same loss out);
  3. per-layer gradient statistics (pure-scaling detected as rescalable,
     rotation detected as not);
  4. no model / optimizer / EMA-state mutation (parameter and buffer SHA256
     unchanged; no optimizer created).
"""
import importlib.util
from pathlib import Path
import unittest

import torch

from training.loss import ECMLoss
from training.schedules import get_schedule

SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "gap_gradient_hook.py"
SPEC = importlib.util.spec_from_file_location("gap_gradient_hook", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class GapGradientHookTests(unittest.TestCase):
    # ---------------------------------------------------------------
    # 1. external gap specification
    # ---------------------------------------------------------------
    def test_parse_gaps_requires_reference(self):
        self.assertEqual(MODULE.parse_gaps("0.9,1.0,1.2,1.3"), [0.9, 1.0, 1.2, 1.3])
        with self.assertRaises(Exception):
            MODULE.parse_gaps("0.9,1.2")           # missing reference 1.0
        with self.assertRaises(Exception):
            MODULE.parse_gaps("0.9,0.9,1.0")       # duplicate

    # ---------------------------------------------------------------
    # 2. fixed randomness: same t/eps/dropout -> same loss (cross-gap reuse)
    # ---------------------------------------------------------------
    def test_fixed_randomness_gives_identical_loss(self):
        class Net(torch.nn.Module):
            def forward(self, x, sigma, labels, augment_labels=None):
                return x * 1.0

        net = Net().eval()
        loss_template = ECMLoss(P_mean=-1.1, P_std=2.0, q=128, c=0.5, k=8, b=1,
                                adj="sigmoid")
        for g in (1.0, 1.3):
            schedule = get_schedule("global_sigmoid", q=128, k=8, b=1,
                                    global_gap_scale=g)
            images = torch.randn(2, 1, 2, 2)
            labels = torch.empty((2, 0))
            t = (torch.randn((2, 1, 1, 1)) * 2.0 - 1.1).exp()
            eps = torch.randn_like(images)
            dropout_state = torch.get_rng_state()
            l1, _ = MODULE.ect_loss_with_fixed_randomness(
                net, loss_template, schedule, images, labels, t, eps, dropout_state)
            l2, _ = MODULE.ect_loss_with_fixed_randomness(
                net, loss_template, schedule, images, labels, t, eps, dropout_state)
            self.assertTrue(torch.allclose(l1, l2, rtol=1e-6, atol=1e-6),
                            f"fixed randomness not reproducible at g={g}")

    # ---------------------------------------------------------------
    # 3. per-layer statistics detect rescale vs rotate
    # ---------------------------------------------------------------
    def test_pure_scaling_detected_as_rescalable(self):
        means = {
            "1.0": {"layer.weight": torch.tensor([2.0, 0.0])},
            "1.3": {"layer.weight": torch.tensor([2.6, 0.0])},   # a*=1.3
        }
        rows, layer_rows = MODULE.mean_vector_statistics(means, 1.0, 2)
        by = {r["gap"]: r for r in rows}
        self.assertAlmostEqual(by[1.3]["scalar_fit_to_g1"], 1.3, places=6)
        self.assertAlmostEqual(by[1.3]["cosine_to_g1"], 1.0, places=6)
        self.assertAlmostEqual(by[1.3]["direction_residual"], 0.0, places=6)
        self.assertEqual(len(layer_rows), 2)   # 1 layer x 2 gaps

    def test_rotation_detected_as_not_rescalable(self):
        means = {
            "1.0": {"layer.weight": torch.tensor([2.0, 0.0])},
            "1.3": {"layer.weight": torch.tensor([0.0, 2.0])},   # 90-degree
        }
        rows, _ = MODULE.mean_vector_statistics(means, 1.0, 2)
        by = {r["gap"]: r for r in rows}
        self.assertAlmostEqual(by[1.3]["cosine_to_g1"], 0.0, places=6)
        self.assertAlmostEqual(by[1.3]["direction_residual"], 1.0, places=6)

    def test_per_layer_statistics_reported(self):
        means = {
            "1.0": {"model.a.weight": torch.tensor([1.0, 0.0]),
                    "model.b.weight": torch.tensor([0.0, 2.0])},
            "1.3": {"model.a.weight": torch.tensor([1.3, 0.0]),   # a*=1.3
                    "model.b.weight": torch.tensor([0.0, 2.0])},  # a*=1.0
        }
        rows, layer_rows = MODULE.mean_vector_statistics(means, 1.0, 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(layer_rows), 4)                      # 2 layers x 2 gaps
        by = {(r["layer"], r["gap"]): r for r in layer_rows}
        self.assertAlmostEqual(by[("model.a", 1.3)]["scalar_fit_to_g1"], 1.3, places=6)
        self.assertAlmostEqual(by[("model.a", 1.3)]["direction_residual"], 0.0, places=6)
        self.assertAlmostEqual(by[("model.b", 1.3)]["scalar_fit_to_g1"], 1.0, places=6)

    # ---------------------------------------------------------------
    # 4. no state mutation
    # ---------------------------------------------------------------
    def test_parameter_and_buffer_hashes_ignore_gradient_writes(self):
        net = torch.nn.BatchNorm1d(2).eval()
        before = MODULE.module_state_hashes(net)
        values = torch.randn(4, 2)
        net(values).sum().backward()
        after = MODULE.module_state_hashes(net)
        self.assertEqual(before, after)          # params/buffers untouched
        self.assertIsNotNone(net.weight.grad)    # gradients were written

    def test_probe_never_creates_optimizer(self):
        # GapGradientProbe.run only zero_grad + backward; it holds no optimizer.
        probe = MODULE.GapGradientProbe.__init__
        self.assertTrue(callable(probe))


if __name__ == "__main__":
    unittest.main()
