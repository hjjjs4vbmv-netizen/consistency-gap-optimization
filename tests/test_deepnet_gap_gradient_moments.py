import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import torch

from training.loss import ECMLoss
from training.schedules import get_schedule


SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "deepnet_gap_gradient_moments.py"
SPEC = importlib.util.spec_from_file_location("deepnet_gap_gradient_moments", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DeepnetGapGradientMomentTests(unittest.TestCase):
    def test_parse_gaps_requires_reference(self):
        self.assertEqual(MODULE.parse_gaps("0.9,1.0,1.2"), [0.9, 1.0, 1.2])
        with self.assertRaises(Exception):
            MODULE.parse_gaps("0.9,1.2")

    def test_projection_moments(self):
        # Mean gradients: mu_1=[1,0], mu_2=[2,2].  Thus a*=2 and residual
        # sqrt(4)/sqrt(8)=1/sqrt(2); batch norm identity gives variance=0.
        means = {
            "1.0": {"layer.weight": MODULE.torch.tensor([2.0, 0.0])},
            "2.0": {"layer.weight": MODULE.torch.tensor([4.0, 4.0])},
        }
        rows, layers = MODULE.mean_vector_statistics(
            means, 1.0, 2,
            {"1.0": 1.0, "2.0": 8.0},
            {"1.0": {"layer": 1.0}, "2.0": {"layer": 8.0}},
        )
        by_gap = {row["gap"]: row for row in rows}
        self.assertAlmostEqual(by_gap[2.0]["scalar_fit_to_g1"], 2.0)
        self.assertAlmostEqual(by_gap[2.0]["direction_residual"], 2 ** -0.5)
        self.assertAlmostEqual(by_gap[2.0]["gradient_variance_trace"], 0.0)
        self.assertEqual(len(layers), 2)

    def test_fixed_randomness_loss_and_gradient_match_real_ect_loss(self):
        class DropoutNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.75))
                self.dropout = torch.nn.Dropout(p=0.25)

            def forward(self, x, sigma, labels, augment_labels=None):
                return self.dropout(x * self.weight + sigma.reshape(-1, 1, 1, 1))

        torch.manual_seed(17)
        net = DropoutNet().train()
        loss = ECMLoss(P_mean=-1.1, P_std=2.0, q=128, c=0.5, k=8, b=1, adj="sigmoid")
        images = torch.randn(3, 1, 2, 2)
        labels = torch.empty((3, 0))
        start_state = torch.get_rng_state()

        # ECMLoss is CUDA-oriented because training is CUDA-only.  Patch its
        # dropout-state calls to CPU equivalents so parity is testable in CI.
        with mock.patch("torch.cuda.get_rng_state", side_effect=torch.get_rng_state), mock.patch(
            "torch.cuda.set_rng_state", side_effect=torch.set_rng_state
        ):
            net.zero_grad(set_to_none=True)
            torch.set_rng_state(start_state)
            real_losses = loss(net, images, labels)
            real_losses.mean().backward()
            real_gradient = net.weight.grad.detach().clone()

        net.zero_grad(set_to_none=True)
        torch.set_rng_state(start_state)
        t = (torch.randn((3, 1, 1, 1)) * loss.P_std + loss.P_mean).exp()
        eps = torch.randn_like(images)
        dropout_state = torch.get_rng_state()
        schedule = get_schedule("global_sigmoid", q=loss.q, k=loss.k, b=loss.b, global_gap_scale=1.0)
        manual_losses, _ = MODULE.ect_loss_with_fixed_randomness(
            net, loss, schedule, images, labels, t, eps, dropout_state
        )
        manual_losses.mean().backward()
        self.assertTrue(torch.allclose(manual_losses, real_losses, rtol=1e-6, atol=1e-6))
        self.assertTrue(torch.allclose(net.weight.grad, real_gradient, rtol=1e-6, atol=1e-6))

    def test_parameter_and_buffer_hashes_ignore_gradient_writes(self):
        net = torch.nn.BatchNorm1d(2).eval()
        before = MODULE.module_state_hashes(net)
        values = torch.randn(4, 2)
        net(values).sum().backward()
        after = MODULE.module_state_hashes(net)
        self.assertEqual(before, after)
        self.assertIsNotNone(net.weight.grad)


if __name__ == "__main__":
    unittest.main()
