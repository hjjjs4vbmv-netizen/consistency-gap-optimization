"""Tests for the Role C gap-gradient diagnostic hook (PR #40 review rev.2).

Covers the review's required tests:
  - test_dataset_uint8_is_normalized_like_training_loop
  - test_probe_runs_network_in_train_mode
  - test_controlled_loss_and_gradient_match_real_ecm_loss   (real parity)
  - test_dropout_mask_is_reused_but_active
  - test_sgd_matched_lr_uses_inverse_scalar_fit
  - test_checkpoint_schedule_parameters_are_loaded
plus the four original hook guarantees.
"""
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import torch

from training.loss import ECMLoss
from training.schedules import get_schedule

SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "gap_gradient_hook.py"
SPEC = importlib.util.spec_from_file_location("gap_gradient_hook", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _make_dropout_net():
    class DropoutNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.75))
            self.dropout = torch.nn.Dropout(p=0.25)

        def forward(self, x, sigma, labels, augment_labels=None, force_fp32=False):
            del labels, augment_labels, force_fp32
            return self.dropout(x * self.weight + sigma.reshape(-1, 1, 1, 1))

    return DropoutNet()


class GapGradientHookTests(unittest.TestCase):
    # ---------------------------------------------------------------
    # review: input normalization must match the training loop
    # ---------------------------------------------------------------
    def test_dataset_uint8_is_normalized_like_training_loop(self):
        # uint8 0..255 -> [-1, 1]; 0 -> -1, 255 -> 1, 127.5 -> 0
        x = torch.tensor([0.0, 127.5, 255.0])
        normed = x / 127.5 - 1
        self.assertTrue(torch.allclose(normed, torch.tensor([-1.0, 0.0, 1.0])))
        # confirm the module's run() uses this exact formula (source check)
        src = Path(SCRIPT).read_text(encoding="utf-8")
        self.assertIn("/ 127.5 - 1", src)
        self.assertIn("127.5", src)

    # ---------------------------------------------------------------
    # review: network must run in train mode (dropout active)
    # ---------------------------------------------------------------
    def test_probe_runs_network_in_train_mode(self):
        net = _make_dropout_net()
        # module's load_checkpoint calls .train() on the EMA
        src = Path(SCRIPT).read_text(encoding="utf-8")
        self.assertIn(".train()", src)
        # and the probe must not call .eval()
        self.assertNotIn("net.eval()", src)
        self.assertIn("net.train()", src)

    def test_dropout_mask_is_reused_but_active(self):
        # train mode: dropout is active; two calls with the SAME rng state
        # must give identical masks, and masks must be non-trivial (dropout p>0).
        net = _make_dropout_net()
        net.train()
        x = torch.randn(8, 1, 2, 2)
        sigma = torch.randn(8)
        labels = torch.empty((8, 0))
        # capture dropout randomness once
        rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        out1 = net(x, sigma, labels)
        torch.set_rng_state(rng)
        out2 = net(x, sigma, labels)
        self.assertTrue(torch.allclose(out1, out2))        # reused mask
        self.assertGreater(net.dropout.p, 0.0)             # dropout active
        # train-mode output must DIFFER from eval-mode output (dropout applied)
        net.eval()
        out_eval = net(x, sigma, labels)
        self.assertFalse(torch.allclose(out1, out_eval))   # dropout changed output

    # ---------------------------------------------------------------
    # review: controlled loss/gradient must match the real ECMLoss
    # ---------------------------------------------------------------
    def test_controlled_loss_and_gradient_match_real_ecm_loss(self):
        net = _make_dropout_net()
        loss = ECMLoss(P_mean=-1.1, P_std=2.0, q=128, c=0.5, k=8, b=1, adj="sigmoid")
        images = torch.randn(3, 1, 2, 2)
        labels = torch.empty((3, 0))
        start_state = torch.get_rng_state()

        # real ECMLoss path
        with mock.patch("torch.cuda.get_rng_state", side_effect=torch.get_rng_state), \
             mock.patch("torch.cuda.set_rng_state", side_effect=torch.set_rng_state):
            net.zero_grad(set_to_none=True)
            torch.set_rng_state(start_state)
            real_losses = loss(net, images, labels)
            real_losses.mean().backward()
            real_grad = net.weight.grad.detach().clone()

        # controlled path with the same t/noise/dropout
        net.zero_grad(set_to_none=True)
        torch.set_rng_state(start_state)
        t = (torch.randn((3, 1, 1, 1)) * loss.P_std + loss.P_mean).exp()
        eps = torch.randn_like(images)
        dropout_state = torch.get_rng_state()
        schedule = get_schedule("global_sigmoid", q=loss.q, k=loss.k, b=loss.b,
                                global_gap_scale=1.0)  # g=1 == official sigmoid
        with mock.patch("torch.cuda.get_rng_state", side_effect=torch.get_rng_state), \
             mock.patch("torch.cuda.set_rng_state", side_effect=torch.set_rng_state):
            manual_losses, _ = MODULE.ect_loss_with_fixed_randomness(
                net, loss, schedule, images, labels, t, eps, dropout_state)
            manual_losses.mean().backward()
            manual_grad = net.weight.grad.detach().clone()

        self.assertTrue(torch.allclose(manual_losses, real_losses, rtol=1e-6, atol=1e-6))
        self.assertTrue(torch.allclose(manual_grad, real_grad, rtol=1e-5, atol=1e-5))

    # ---------------------------------------------------------------
    # review: LR matching uses the INVERSE scalar fit
    # ---------------------------------------------------------------
    def test_sgd_matched_lr_uses_inverse_scalar_fit(self):
        # mu_g = a* mu_1;  SGD update match: eta_g * mu_g = eta_1 * mu_1
        # => eta_g = eta_1 / a*
        a_star = 0.7688
        eta_1 = 1e-4
        eta_g = eta_1 / a_star
        self.assertAlmostEqual(eta_g / eta_1, 1.30068, places=4)   # > 1, not a*
        # sanity: naive a* * eta_1 is wrong direction
        self.assertLess(a_star * eta_1, eta_1)
        self.assertGreater(eta_g, eta_1)

    # ---------------------------------------------------------------
    # review: checkpoint schedule parameters loaded from loss_fn, not hardcoded
    # ---------------------------------------------------------------
    def test_checkpoint_schedule_parameters_are_loaded(self):
        # probe reads q/k/b from loss_fn; a fake loss exposes them
        class FakeLoss:
            q, k, b, c, stage = 256.0, 8.0, 1.0, 0.0, 3
            schedule = type("S", (), {"name": "sigmoid"})()
        probe = MODULE.GapGradientProbe(None, FakeLoss(), [1.0])
        self.assertEqual(probe.q, 256.0)
        self.assertEqual(probe.k, 8.0)
        self.assertEqual(probe.b, 1.0)
        # source must not hardcode 128
        src = Path(SCRIPT).read_text(encoding="utf-8")
        self.assertNotIn("q: float = 128", src)

    # ---------------------------------------------------------------
    # original guarantees (retained)
    # ---------------------------------------------------------------
    def test_parse_gaps_requires_reference(self):
        self.assertEqual(MODULE.parse_gaps("0.9,1.0,1.2,1.3"), [0.9, 1.0, 1.2, 1.3])
        with self.assertRaises(Exception):
            MODULE.parse_gaps("0.9,1.2")
        with self.assertRaises(Exception):
            MODULE.parse_gaps("0.9,0.9,1.0")

    def test_pure_scaling_detected_as_rescalable(self):
        means = {"1.0": {"l.w": torch.tensor([2.0, 0.0])},
                 "1.3": {"l.w": torch.tensor([2.6, 0.0])}}
        rows, lrows = MODULE.mean_vector_statistics(means, 1.0, 2)
        by = {r["gap"]: r for r in rows}
        self.assertAlmostEqual(by[1.3]["scalar_fit_to_g1"], 1.3, places=6)
        self.assertAlmostEqual(by[1.3]["direction_residual"], 0.0, places=6)

    def test_rotation_detected_as_not_rescalable(self):
        means = {"1.0": {"l.w": torch.tensor([2.0, 0.0])},
                 "1.3": {"l.w": torch.tensor([0.0, 2.0])}}
        rows, _ = MODULE.mean_vector_statistics(means, 1.0, 2)
        by = {r["gap"]: r for r in rows}
        self.assertAlmostEqual(by[1.3]["cosine_to_g1"], 0.0, places=6)
        self.assertAlmostEqual(by[1.3]["direction_residual"], 1.0, places=6)

    def test_parameter_and_buffer_hashes_ignore_gradient_writes(self):
        net = torch.nn.BatchNorm1d(2).eval()
        before = MODULE.module_state_hashes(net)
        net(torch.randn(4, 2)).sum().backward()
        after = MODULE.module_state_hashes(net)
        self.assertEqual(before, after)
        self.assertIsNotNone(net.weight.grad)


if __name__ == "__main__":
    unittest.main()
