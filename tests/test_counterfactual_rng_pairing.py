"""RNG, frozen draw, optimizer-step and AMP pairing contracts."""
from __future__ import annotations

import copy
import random
import unittest

import numpy as np
import torch

from analysis.nonlinear_dynamics_gate.decompose_forcing_feedback import (
    _batch_receipt,
    _forcing_input_receipt,
    _transition,
)
from analysis.operator_clock_gate.core import AlgorithmicState, freeze_batches, rng_sha256


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


class DropoutNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
        self.dropout = torch.nn.Dropout(0.4)

    def forward(self, x, sigma, labels, augment_labels=None):
        del sigma, labels, augment_labels
        return self.weight * self.dropout(x)


def fixture():
    torch.manual_seed(23)
    net = DropoutNet().train()
    optimizer = torch.optim.RAdam(net.parameters(), lr=1e-2)
    for parameter in net.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state = AlgorithmicState(
        net, optimizer, copy.deepcopy(net).eval().requires_grad_(False),
        TinyLoss(), ema_beta=0.9,
    )
    images = torch.linspace(-1.0, 1.0, 8, dtype=torch.float64).reshape(8, 1, 1, 1)
    labels = torch.empty(8, 0, dtype=torch.float64)
    batch = freeze_batches([(images, labels)], state.loss_fn, (2026082702,))[0]
    return state, batch


class CounterfactualRngPairingTests(unittest.TestCase):
    def test_replay_is_independent_of_ambient_rng_and_preserves_it(self):
        state, batch = fixture()
        input_receipt = _forcing_input_receipt(state)
        batch_receipt = _batch_receipt(batch)
        rng_before = rng_sha256()
        first, first_telemetry = _transition(
            state, batch, "B", 2026082702, clone_input=True)
        self.assertEqual(rng_sha256(), rng_before)
        random.random()
        np.random.random()
        torch.rand(5)
        rng_perturbed = rng_sha256()
        second, second_telemetry = _transition(
            state, batch, "B", 2026082702, clone_input=True)
        self.assertEqual(rng_sha256(), rng_perturbed)
        self.assertEqual(first.sha256(), second.sha256())
        self.assertEqual(first_telemetry, second_telemetry)
        self.assertEqual(_forcing_input_receipt(state), input_receipt)
        self.assertEqual(_batch_receipt(batch), batch_receipt)

    def test_forcing_arms_share_optimizer_step_and_amp_input(self):
        state, batch = fixture()
        before = _forcing_input_receipt(state)
        a_after, _ = _transition(state, batch, "A", 7, clone_input=True)
        b_after, _ = _transition(state, batch, "B", 7, clone_input=True)
        self.assertEqual(_forcing_input_receipt(state), before)
        self.assertEqual(before["optimizer_and_amp_discrete_state"],
                         _forcing_input_receipt(state)["optimizer_and_amp_discrete_state"])
        self.assertNotEqual(a_after.sha256(), b_after.sha256())


if __name__ == "__main__":
    unittest.main()
