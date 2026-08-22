"""Regression contracts for Role-D frozen gap diagnostics."""
from __future__ import annotations

import pytest
import torch

from analysis.frozen_gap_diagnostic import (
    a_pred_realized,
    batch_decomposition,
    batch_optimal_scalar,
    four_arm_interaction,
    global_sigmoid_upper_clip_free,
    realized_gap_pair,
    scaling_exponent,
)
from training.schedules import get_schedule


def test_q128_global130_is_unclipped_and_uses_the_requested_ratio():
    assert global_sigmoid_upper_clip_free(
        q=128, k=8, b=1, stage=0, requested_gain=1.3,
    )
    assert global_sigmoid_upper_clip_free(
        q=256, k=8, b=1, stage=0, requested_gain=1.1,
    )
    t = torch.logspace(-6, 3, steps=257, dtype=torch.float64)
    baseline = get_schedule("global_sigmoid", q=128.0, k=8.0, b=1.0, global_gap_scale=1.0)
    probe = get_schedule("global_sigmoid", q=128.0, k=8.0, b=1.0, global_gap_scale=1.3)
    pair = realized_gap_pair(
        t, baseline.compute_r(t, stage=0), probe.compute_r(t, stage=0),
        requested_gain=1.3,
    )
    assert pair.upper_clip_rate == 0.0
    assert bool(pair.theorem_valid.all())
    assert torch.allclose(pair.ratio[pair.valid], torch.full_like(pair.ratio[pair.valid], 1.3))


def test_clipped_samples_report_realized_ratio_but_are_outside_theorem_scope():
    assert not global_sigmoid_upper_clip_free(
        q=2, k=8, b=1, stage=0, requested_gain=2.0,
    )
    t = torch.tensor([1e-6], dtype=torch.float64)
    baseline = get_schedule("global_sigmoid", q=2.0, k=8.0, b=1.0, global_gap_scale=1.0)
    probe = get_schedule("global_sigmoid", q=2.0, k=8.0, b=1.0, global_gap_scale=2.0)
    prediction, pair = a_pred_realized(
        t, baseline.compute_r(t, stage=0), probe.compute_r(t, stage=0),
        requested_gain=2.0, nu=1.0, p=1.0, alpha=1.0,
    )
    assert bool(pair.upper_clipped.item())
    assert pair.ratio.item() == 1.0
    assert pair.ratio.item() != 2.0
    assert not bool(pair.theorem_valid.item())
    assert torch.isnan(prediction).item()


def test_upper_clip_certificate_rejects_negative_sigmoid_rate():
    with pytest.raises(ValueError, match="b >= 0"):
        global_sigmoid_upper_clip_free(
            q=128, k=8, b=-1, stage=0, requested_gain=1.3,
        )


def test_batch_decomposition_is_an_identity_but_not_a_batch_scalarity_heuristic():
    # Orthogonal exact sample scalars: local residual is zero, yet the optimal
    # whole-batch scalar leaves nonzero residual when sample scalars differ.
    reference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    scalars = torch.tensor([1.0, 2.0])
    local_residual = torch.zeros_like(reference)
    probe = scalars[:, None] * reference + local_residual

    mean_scalar, heterogeneity, local = batch_decomposition(
        reference, scalars, local_residual,
    )
    assert torch.allclose(probe.mean(0), mean_scalar * reference.mean(0) + heterogeneity + local)
    a_batch = batch_optimal_scalar(reference.mean(0), probe.mean(0))
    assert a_batch.item() == 1.5
    assert torch.linalg.vector_norm(probe.mean(0) - a_batch * reference.mean(0)).item() > 0


def test_kappa_retains_nu_outside_degree_one_and_cancels_it_at_degree_one():
    assert scaling_exponent(nu=3, p=2, alpha=1) == 2
    assert scaling_exponent(nu=1, p=1, alpha=1) == -1
    assert scaling_exponent(nu=7, p=1, alpha=1) == -1


def test_four_arm_interaction_is_available_only_as_a_post_training_readout():
    gradients = {
        "A": torch.tensor([1.0, 2.0]),
        "B": torch.tensor([4.0, 8.0]),
        "C": torch.tensor([2.0, 3.0]),
        "D": torch.tensor([3.0, 5.0]),
    }
    assert torch.equal(four_arm_interaction(gradients), torch.tensor([0.0, 2.0]))
