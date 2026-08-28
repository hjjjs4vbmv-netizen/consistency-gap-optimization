"""Tests for the separately frozen squared-GN calibration oracle."""
from __future__ import annotations

import torch

from analysis.jvp_harness_calibration import core
from analysis.operator_clock_gate.core import AuditBatch, get_device_rng_state


class Loss:
    q, k, b, c, stage = 8.0, 2.0, 1.0, 0.0, 0


class ScalarNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))

    def forward(self, x, sigma, labels, augment_labels=None, force_fp32=False):
        del sigma, labels, augment_labels, force_fp32
        return self.weight * x


def batch() -> AuditBatch:
    images = torch.tensor([[[[0.2]]], [[[0.8]]]], dtype=torch.float64)
    labels = torch.empty(2, 0, dtype=torch.float64)
    t = torch.tensor([0.6, 1.1], dtype=torch.float64).reshape(2, 1, 1, 1)
    noise = torch.tensor([[[[0.9]]], [[[-0.4]]]], dtype=torch.float64)
    return AuditBatch(
        images, labels, t, noise, get_device_rng_state(images.device), 1)


def test_exact_oracle_matches_central_difference_for_linear_network():
    net = ScalarNet()
    direction = {"weight": torch.ones((), dtype=torch.float64)}
    oracle_tangent, oracle_action, oracle_detail = core.exact_oracle(
        net, Loss(), [batch()], direction, arm="A")
    tangent, action, detail = core.finite_difference_estimate(
        net, Loss(), [batch()], direction, arm="A", epsilon=1e-3)
    assert oracle_detail["finite"] and oracle_detail["source_preserved"]
    assert detail["finite"] and detail["source_preserved"]
    assert core.comparison(tangent, oracle_tangent)["relative_error"] < 1e-10
    assert core.comparison(
        list(action.values()), list(oracle_action.values()))["relative_error"] < 1e-10


def test_plateau_requires_frozen_number_of_consecutive_scales():
    def row(epsilon, error):
        return {
            "epsilon": epsilon,
            "finite": True,
            "source_preserved": True,
            "tangent_vs_oracle": {"relative_error": error},
            "action_vs_oracle": {"relative_error": error},
        }

    rows = [row(0.08, 0.2), row(0.04, 0.04), row(0.02, 0.03),
            row(0.01, 0.049), row(0.005, 0.08)]
    decision = core.classify(
        rows, tolerance=0.05, minimum_consecutive=3, oracle_ok=True)
    assert decision["verdict"] == "PASS_CALIBRATED"
    assert decision["plateaus"][0]["epsilons"] == [0.04, 0.02, 0.01]


def test_two_good_scales_do_not_form_a_plateau():
    rows = [{
        "epsilon": value,
        "finite": True,
        "source_preserved": True,
        "tangent_vs_oracle": {"relative_error": 0.01},
        "action_vs_oracle": {"relative_error": 0.01},
    } for value in (0.02, 0.01)]
    decision = core.classify(
        rows, tolerance=0.05, minimum_consecutive=3, oracle_ok=True)
    assert decision["verdict"] == "NO_PLATEAU"
