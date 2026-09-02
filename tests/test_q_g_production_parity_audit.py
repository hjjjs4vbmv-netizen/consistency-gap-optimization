from __future__ import annotations

import torch

from analysis.q_g_production_parity_audit import (
    CANDIDATE_G,
    CANDIDATE_Q,
    REFERENCE_G,
    REFERENCE_Q,
    dense_grid_parity,
)


def test_q_eff_is_exact_real_arithmetic_reparameterization():
    assert CANDIDATE_Q == REFERENCE_Q / REFERENCE_G
    assert CANDIDATE_G == 1.0
    assert CANDIDATE_G / CANDIDATE_Q == REFERENCE_G / REFERENCE_Q


def test_dense_grid_parity_passes_frozen_coordinate_gate_float32():
    result = dense_grid_parity(
        grid_size=4096,
        dtype=torch.float32,
        eps_multiplier=32.0,
    )
    assert result["reference_zero_or_clipped_count"] == 0
    assert result["candidate_zero_or_clipped_count"] == 0
    assert result["coordinate_gate_passed"]


def test_dense_grid_parity_passes_frozen_coordinate_gate_float64():
    result = dense_grid_parity(
        grid_size=4096,
        dtype=torch.float64,
        eps_multiplier=32.0,
    )
    assert result["reference_zero_or_clipped_count"] == 0
    assert result["candidate_zero_or_clipped_count"] == 0
    assert result["coordinate_gate_passed"]
