from __future__ import annotations

import unittest

import torch

from training import reproducibility
from training.ct_training_loop import (
    _FACTORIAL_TELEMETRY_FIELDS,
    _SAME_STATE_TELEMETRY_FIELDS,
    aggregate_factorial_runtime_metrics,
    normalize_immutable_checkpoint_attempts,
)


class SameStateProtocolTest(unittest.TestCase):
    def test_attempt_milestones_are_exact_and_bounded(self):
        values = normalize_immutable_checkpoint_attempts(
            (3001, 3004, 3016, 3064, 3128, 3256, 3500),
            total_kimg=448, batch_size=128,
        )
        self.assertEqual(values[-1], 3500)
        with self.assertRaises(ValueError):
            normalize_immutable_checkpoint_attempts(
                (3004, 3001), total_kimg=448, batch_size=128,
            )
        with self.assertRaises(ValueError):
            normalize_immutable_checkpoint_attempts(
                (3501,), total_kimg=448, batch_size=128,
            )

    def test_same_state_telemetry_extends_v1_without_reordering_v1(self):
        for field in _FACTORIAL_TELEMETRY_FIELDS:
            self.assertIn(field, _SAME_STATE_TELEMETRY_FIELDS)
        self.assertEqual(len(_SAME_STATE_TELEMETRY_FIELDS), len(_FACTORIAL_TELEMETRY_FIELDS) + 3)
        self.assertLess(_SAME_STATE_TELEMETRY_FIELDS.index("base_r_sha256"),
                        _SAME_STATE_TELEMETRY_FIELDS.index("input_noise_sha256"))

    def test_exogenous_hashes_aggregate_in_microbatch_order(self):
        base = {
            "schema": "x", "protocol": "q256_target_weight_v1", "arm": "A",
            "target_gap_scale": 1.0, "denominator_gap_scale": 1.0,
            "sample_count": 16, "base_r_zero_count": 0, "target_r_zero_count": 0,
            "target_r_equal_t_count": 0, "target_scaled_to_zero_count": 0,
            "denominator_r_zero_count": 0, "denominator_r_equal_t_count": 0,
            "denominator_scaled_to_zero_count": 0, "nonfinite_count": 0,
            "nonpositive_denominator_count": 0, "target_delta_min": 1.0,
            "target_delta_max": 1.0, "target_delta_mean": 1.0,
            "denominator_delta_min": 1.0, "denominator_delta_max": 1.0,
            "denominator_delta_mean": 1.0,
        }
        hashes = (
            "t_sha256", "base_r_sha256", "target_r_sha256",
            "denominator_r_sha256", "target_delta_sha256",
            "denominator_delta_sha256", "input_noise_sha256",
            "dropout_rng_sha256", "augmentation_rng_sha256",
        )
        batches = []
        for index in range(2):
            item = dict(base)
            item.update({field: f"{field}-{index}" for field in hashes})
            batches.append(item)
        result = aggregate_factorial_runtime_metrics(batches)
        for field in hashes:
            self.assertEqual(
                result[field],
                reproducibility.state_sha256([f"{field}-0", f"{field}-1"]),
            )


if __name__ == "__main__":
    unittest.main()
