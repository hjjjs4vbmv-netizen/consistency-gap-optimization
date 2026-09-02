"""Finite-difference convergence and differentiable optimizer oracle tests."""
from __future__ import annotations

import unittest

import torch

from analysis.operator_clock_gate.core import (
    AuditBatch,
    _distribution,
    central_difference_map,
    fd_convergence,
    field_jvp,
    get_device_rng_state,
)


def differentiable_momentum_step(state):
    """Toy smooth state transition (theta, momentum, EMA)."""
    theta, momentum, ema = state
    gradient = theta.sin() + 0.3 * theta.square()
    next_momentum = 0.8 * momentum + 0.2 * gradient
    next_theta = theta - 0.05 * next_momentum
    next_ema = 0.9 * ema + 0.1 * next_theta
    return next_theta, next_momentum, next_ema


class OperatorGateFDConvergenceTests(unittest.TestCase):
    def test_large_distribution_quantiles_use_declared_deterministic_stride(self):
        values = torch.arange(100, dtype=torch.float64)
        first = _distribution(values, max_quantile_elements=10)
        second = _distribution(values, max_quantile_elements=10)
        self.assertEqual(first, second)
        self.assertEqual(first["count"], 100)
        self.assertEqual(first["quantile_sample_count"], 10)
        self.assertEqual(first["quantile_stride"], 10)
        self.assertEqual(first["quantile_method"], "deterministic_stride_sample")
        self.assertEqual(first["mean"], 49.5)

    def test_toy_differentiable_optimizer_fd_matches_autograd_jvp(self):
        point = tuple(torch.tensor(value, dtype=torch.float64, requires_grad=True)
                      for value in (0.4, -0.2, 0.1))
        direction = tuple(torch.tensor(value, dtype=torch.float64)
                          for value in (0.3, -0.5, 0.7))
        _, autograd_jvp = torch.autograd.functional.jvp(
            lambda *state: differentiable_momentum_step(state),
            point, direction, create_graph=False)
        estimates = {
            epsilon: central_difference_map(
                differentiable_momentum_step, point, direction, epsilon)
            for epsilon in (1e-2, 3e-3, 1e-3, 3e-4)
        }
        report = fd_convergence({
            epsilon: {str(index): value.detach().reshape(())
                      for index, value in enumerate(estimate)}
            for epsilon, estimate in estimates.items()
        }, tolerance=1e-4)
        self.assertTrue(report["passed"])
        finest = estimates[3e-4]
        for finite, exact in zip(finest, autograd_jvp):
            self.assertTrue(torch.allclose(finite, exact, atol=1e-8, rtol=1e-7))

    def test_convergence_requires_multiple_frozen_epsilons(self):
        with self.assertRaisesRegex(ValueError, "at least three"):
            fd_convergence({
                1e-2: {"x": torch.tensor(1.0)},
                1e-3: {"x": torch.tensor(1.0)},
            })

    def test_nonlinear_map_improves_over_epsilon_sweep(self):
        point = (torch.tensor(0.7, dtype=torch.float64),)
        direction = (torch.tensor(0.4, dtype=torch.float64),)

        def cubic(state):
            return (state[0] ** 3,)

        estimates = {
            epsilon: central_difference_map(cubic, point, direction, epsilon)[0]
            for epsilon in (1e-1, 3e-2, 1e-2, 3e-3)
        }
        exact = 3 * point[0].square() * direction[0]
        errors = [abs(float(estimates[epsilon] - exact))
                  for epsilon in (1e-1, 3e-2, 1e-2, 3e-3)]
        self.assertGreater(errors[0], errors[-1] * 100)

    def test_recompute_detach_field_is_not_cached_target_hessian(self):
        class Loss:
            q, k, b, c, stage = 8.0, 2.0, 1.0, 0.5, 0

        class ScalarNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))

            def forward(self, x, sigma, labels, augment_labels=None):
                del sigma, labels, augment_labels
                return self.weight * x

        net = ScalarNet()
        images = torch.tensor([[[[0.2]]], [[[0.8]]]], dtype=torch.float64)
        labels = torch.empty(2, 0, dtype=torch.float64)
        t = torch.tensor([0.6, 1.1], dtype=torch.float64).reshape(2, 1, 1, 1)
        noise = torch.tensor([[[[0.9]]], [[[-0.4]]]], dtype=torch.float64)
        batch = AuditBatch(images, labels, t, noise, get_device_rng_state(images.device), 1)
        direction = {"weight": torch.ones((), dtype=torch.float64)}
        true_jvp, receipt = field_jvp(
            net, Loss(), [batch], direction,
            epsilons=(1e-3, 3e-4, 1e-4), convergence_tolerance=1e-4)
        self.assertTrue(receipt["convergence"]["passed"])

        # The incorrect construction caches the target at theta_0 and then
        # differentiates one scalar loss twice.  It must disagree because it
        # omits target-recomputation feedback.
        from analysis.operator_clock_gate.core import ARM_SPECS, _schedule
        r = _schedule(Loss(), ARM_SPECS["A"]["target_scale"]).compute_r(t, stage=0)
        denominator = t - _schedule(
            Loss(), ARM_SPECS["A"]["denominator_scale"]).compute_r(t, stage=0)
        online_input = images + noise * t
        target_input = images + noise * r
        target_cached = (net.weight.detach() * target_input).detach()
        residual = net.weight * online_input - target_cached
        raw_sq = residual.square().reshape(2, -1).sum(dim=1)
        cached_loss = (torch.sqrt(raw_sq + Loss.c ** 2) - Loss.c) / denominator.flatten()
        cached_grad = torch.autograd.grad(cached_loss.mean(), net.weight, create_graph=True)[0]
        cached_hessian_u = torch.autograd.grad(cached_grad, net.weight)[0]
        self.assertGreater(
            abs(float(true_jvp["weight"] - cached_hessian_u.detach())), 1e-3)


if __name__ == "__main__":
    unittest.main()
