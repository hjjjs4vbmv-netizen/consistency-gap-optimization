"""State-preservation contracts for operator-clock audits."""
from __future__ import annotations

import copy
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from analysis.operator_clock_gate.core import (
    AlgorithmicState,
    algorithmic_jvp,
    field_jvp,
    freeze_batches,
    matched_micro_rollout,
    parameter_vector,
    random_direction_like,
    rng_sha256,
    squared_gn_operator_jvp,
)
from analysis.operator_clock_gate import cli_common


class TinyLoss:
    P_mean = -0.5
    P_std = 0.2
    q = 8.0
    k = 2.0
    b = 1.0
    c = 0.1
    stage = 0


class TinyDropoutNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x, sigma, labels, augment_labels=None):
        del labels, augment_labels
        return self.weight * self.dropout(x) + self.bias * sigma


def fixture():
    torch.manual_seed(7)
    net = TinyDropoutNet().train()
    optimizer = torch.optim.RAdam(net.parameters(), lr=1e-2)
    for parameter in net.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    state = AlgorithmicState(net, optimizer, ema, TinyLoss(), ema_beta=0.9)
    images = torch.linspace(-0.9, 0.9, 8, dtype=torch.float64).reshape(8, 1, 1, 1)
    labels = torch.empty(8, 0, dtype=torch.float64)
    batches = freeze_batches([(images, labels)] * 4, state.loss_fn,
                             (2026082601, 2026082602, 2026082603, 2026082604))
    return state, batches


class OperatorGateStatePreservationTests(unittest.TestCase):
    def test_trusted_artifact_loader_reconstructs_complete_state(self):
        state, batches = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "training-state.pt"
            checkpoint_path = root / "network-snapshot.pkl"
            batch_path = root / "batches.pt"
            torch.save({
                "net": state.net,
                "optimizer_state": state.optimizer.state_dict(),
            }, state_path)
            with checkpoint_path.open("wb") as handle:
                pickle.dump({"ema": state.ema, "loss_fn": state.loss_fn}, handle)
            torch.save({
                "batches": [(batch.images, batch.labels) for batch in batches],
            }, batch_path)
            args = SimpleNamespace(
                training_state=state_path, checkpoint=checkpoint_path,
                batch_file=batch_path, expected_training_state_sha256=None,
                expected_checkpoint_sha256=None,
                expected_batch_file_sha256=None, device="cpu", ema_beta=0.9,
            )
            assets = cli_common.source_assets(args)
            loaded = cli_common.load_algorithmic_state(args)
            frozen = cli_common.load_frozen_batches(args, loaded.loss_fn)
            self.assertEqual(len(frozen), 4)
            self.assertEqual(set(loaded.continuous_vector()),
                             set(state.continuous_vector()))
            self.assertTrue(assets["training_state"]["matched"])

    def test_field_and_algorithmic_jvps_preserve_source(self):
        state, batches = fixture()
        state_before = state.sha256()
        rng_before = rng_sha256()
        parameter_direction = random_direction_like(parameter_vector(state.net), 11)
        _, square_receipt = squared_gn_operator_jvp(
            state.net, state.loss_fn, [batches[0]], parameter_direction,
            learning_rate=1e-2)
        self.assertTrue(square_receipt["source_preserved"])
        self.assertIn("not the true ECT", square_receipt["claim_boundary"])
        _, field_receipt = field_jvp(
            state.net, state.loss_fn, [batches[0]], parameter_direction,
            epsilons=(1e-2, 3e-3, 1e-3), convergence_tolerance=1.0)
        self.assertTrue(field_receipt["source_preserved"])
        self.assertEqual(state.sha256(), state_before)
        self.assertEqual(rng_sha256(), rng_before)

        full_direction = random_direction_like(state.continuous_vector(), 12)
        # This fixture has positive v; use small eps so both branches remain in
        # the RAdam state domain.
        _, algorithmic_receipt = algorithmic_jvp(
            state, batches[0], full_direction,
            epsilons=(1e-5, 3e-6, 1e-6), convergence_tolerance=1.0)
        self.assertTrue(algorithmic_receipt["source_preserved"])
        self.assertTrue(algorithmic_receipt["no_in_place_source_pollution"])
        self.assertEqual(state.sha256(), state_before)
        self.assertEqual(rng_sha256(), rng_before)

    def test_matched_rollout_forks_full_state_without_committing(self):
        state, batches = fixture()
        before = state.sha256()
        receipt = matched_micro_rollout(
            state, batches, horizons=(1, 2),
            projection_seeds=tuple(range(100, 108)))
        self.assertTrue(receipt["source_preserved"])
        self.assertEqual(state.sha256(), before)
        self.assertEqual(set(receipt["branches"]), set("ABCD"))
        for arm in "ABCD":
            self.assertEqual(len(receipt["branches"][arm]["steps"]), 2)
            horizon = receipt["branches"][arm]["horizons"]["2"]
            self.assertEqual(len(horizon["parameter_random_projections"]), 8)
            self.assertIn("optimizer_moment_summaries", horizon)
            self.assertIn("validation_output", horizon)
            self.assertIn("residual_profile", horizon)
            self.assertIn("fixed_latent_sample_features", horizon)


if __name__ == "__main__":
    unittest.main()
