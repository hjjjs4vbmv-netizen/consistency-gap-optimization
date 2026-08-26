"""Frozen noise/dropout pairing contracts for the operator gate."""
from __future__ import annotations

import copy
import unittest

import torch

from analysis.operator_clock_gate.core import (
    ARM_SPECS,
    ect_pair,
    field_jvp,
    freeze_batches,
    parameter_vector,
    random_direction_like,
    rng_sha256,
)
from tests.test_operator_gate_state_preservation import TinyDropoutNet, TinyLoss


class OperatorGateRNGPairingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.net = TinyDropoutNet().train()
        images = torch.linspace(-1, 1, 8, dtype=torch.float64).reshape(8, 1, 1, 1)
        labels = torch.empty(8, 0, dtype=torch.float64)
        self.batch = freeze_batches(
            [(images, labels)], TinyLoss(), (2026082601,))[0]

    def test_rerun_and_factorial_target_pairing_are_exact(self):
        before = rng_sha256()
        first_loss, first = ect_pair(
            self.net, TinyLoss(), self.batch, "A", detach_target=True)
        second_loss, second = ect_pair(
            self.net, TinyLoss(), self.batch, "A", detach_target=True)
        self.assertTrue(torch.equal(first_loss, second_loss))
        self.assertEqual(first["online_sha256"], second["online_sha256"])
        self.assertEqual(first["target_sha256"], second["target_sha256"])
        self.assertFalse(first["target_requires_grad"])

        _, a = ect_pair(self.net, TinyLoss(), self.batch, "A", detach_target=True)
        _, d = ect_pair(self.net, TinyLoss(), self.batch, "D", detach_target=True)
        _, b = ect_pair(self.net, TinyLoss(), self.batch, "B", detach_target=True)
        _, c = ect_pair(self.net, TinyLoss(), self.batch, "C", detach_target=True)
        self.assertEqual(a["target_sha256"], d["target_sha256"])
        self.assertEqual(b["target_sha256"], c["target_sha256"])
        self.assertEqual(ARM_SPECS["C"]["denominator_scale"], 1.0)
        self.assertEqual(ARM_SPECS["D"]["target_scale"], 1.0)
        # ect_pair itself leaves the stream at the paired forward endpoint;
        # public audits must restore the caller stream.
        direction = random_direction_like(parameter_vector(self.net), 4)
        _, receipt = field_jvp(
            self.net, TinyLoss(), [self.batch], direction,
            epsilons=(1e-2, 3e-3, 1e-3), convergence_tolerance=1.0)
        self.assertTrue(receipt["source_preserved"])
        self.assertNotEqual(rng_sha256(), before)

    def test_public_field_audit_pairs_every_epsilon_and_preserves_entry_rng(self):
        direction = random_direction_like(parameter_vector(self.net), 5)
        before = rng_sha256()
        _, receipt = field_jvp(
            self.net, TinyLoss(), [self.batch], direction,
            epsilons=(1e-2, 3e-3, 1e-3), convergence_tolerance=1.0)
        self.assertEqual(rng_sha256(), before)
        self.assertTrue(receipt["source_preserved"])
        for item in receipt["branch_receipts"]:
            self.assertTrue(item["paired_target_recomputation"])
            self.assertEqual(item["plus"]["target_recompute_count"], 1)
            self.assertEqual(item["minus"]["target_recompute_count"], 1)
            self.assertTrue(item["plus"]["all_targets_detached"])
            self.assertTrue(item["minus"]["all_targets_detached"])


if __name__ == "__main__":
    unittest.main()
