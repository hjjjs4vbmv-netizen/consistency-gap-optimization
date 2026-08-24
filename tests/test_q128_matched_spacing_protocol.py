import unittest

import torch

from training.loss import (
    Q128_MATCHED_SPACING_ARMS,
    Q128_MATCHED_SPACING_GAP_SCALE,
    Q128_MATCHED_SPACING_PROTOCOL,
    compute_target_weight_times,
    resolve_target_weight_factorial,
)
from training.schedules import get_schedule


class Q128MatchedSpacingProtocolTest(unittest.TestCase):
    def test_exactly_five_frozen_arms_are_derived_from_factors(self):
        self.assertEqual(
            set(Q128_MATCHED_SPACING_ARMS.values()),
            {"A", "Bsame", "Bmatch", "Cmatch", "Dmatch"},
        )
        for factors, expected_arm in Q128_MATCHED_SPACING_ARMS.items():
            with self.subTest(arm=expected_arm):
                resolved = resolve_target_weight_factorial(
                    Q128_MATCHED_SPACING_PROTOCOL,
                    factors[0],
                    factors[1],
                    adj="sigmoid",
                    global_gap_scale=1.0,
                    q=128,
                    c=0,
                )
                self.assertEqual(resolved["arm"], expected_arm)

    def test_protocol_fails_closed_on_wrong_q_or_unfrozen_factor(self):
        invalid = [
            (256, 1.0, 1.0),
            (128, 0.56, 0.56),
            (128, 1.1, 1.0),
        ]
        for q, target, denominator in invalid:
            with self.subTest(q=q, target=target, denominator=denominator):
                with self.assertRaises(ValueError):
                    resolve_target_weight_factorial(
                        Q128_MATCHED_SPACING_PROTOCOL,
                        target,
                        denominator,
                        adj="sigmoid",
                        global_gap_scale=1.0,
                        q=q,
                        c=0,
                    )

    def test_matched_gap_tracks_q256_reference_at_stage_zero(self):
        t = torch.logspace(-5, 3, 10000, dtype=torch.float64).reshape(-1, 1, 1, 1)
        q256_r = get_schedule("sigmoid", q=256, k=8, b=1).compute_r(t, stage=0)
        q128_r = get_schedule("sigmoid", q=128, k=8, b=1).compute_r(t, stage=0)
        _, _, q256_gap, _ = compute_target_weight_times(
            t,
            q256_r,
            target_gap_scale=1.1,
            denominator_gap_scale=1.1,
        )
        _, _, q128_gap, _ = compute_target_weight_times(
            t,
            q128_r,
            target_gap_scale=Q128_MATCHED_SPACING_GAP_SCALE,
            denominator_gap_scale=Q128_MATCHED_SPACING_GAP_SCALE,
        )
        self.assertTrue(torch.allclose(q128_gap, q256_gap, rtol=1e-11, atol=1e-15))


if __name__ == "__main__":
    unittest.main()
