import math
import unittest

import torch

from training.ct_training_loop import (
    LocalTBinSignalWindow,
    gather_adaptive_signal_window_state,
    globally_average_local_tbin_loss,
    local_adaptive_signal_window_state,
)
from training.schedules import get_schedule


class LocalTBinScheduleTest(unittest.TestCase):
    def test_global_sigmoid_one_is_bitwise_official(self):
        t = torch.logspace(-3, 2, 1024)
        baseline = get_schedule('sigmoid', q=256.0)
        global_only = get_schedule(
            'global_sigmoid', q=256.0, global_gap_scale=1.0
        )
        self.assertTrue(torch.equal(
            global_only.compute_r(t=t, stage=0),
            baseline.compute_r(t=t, stage=0),
        ))

    def test_global_sigmoid_scales_every_official_gap(self):
        t = torch.logspace(-3, 1, 1024, dtype=torch.float64)
        baseline_r = get_schedule('sigmoid', q=256.0).compute_r(t=t, stage=0)
        global_only = get_schedule(
            'global_sigmoid', q=256.0, global_gap_scale=1.032
        )
        scaled_r = global_only.compute_r(t=t, stage=0)
        expected_gap = torch.minimum((t - baseline_r) * 1.032, t)
        self.assertTrue(torch.allclose(
            t - scaled_r, expected_gap, rtol=1e-12, atol=1e-12
        ))
        self.assertTrue(torch.equal(
            global_only.preclip_gap_scale(t),
            torch.full_like(t, 1.032),
        ))

    def test_invalid_global_gap_scale_is_rejected(self):
        for value in [0, -1, float('nan'), float('inf')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                get_schedule('global_sigmoid', global_gap_scale=value)

    def test_quantile_bins_are_balanced_under_training_distribution(self):
        generator = torch.Generator().manual_seed(123)
        z = torch.randn(100_000, generator=generator, dtype=torch.float64)
        t = (z * 2.0 - 1.1).exp()
        schedule = get_schedule('local_tbin_v1', p_mean=-1.1, p_std=2.0)
        counts = torch.bincount(schedule.bin_indices(t), minlength=4)
        fractions = counts.to(torch.float64) / counts.sum()
        self.assertTrue(torch.all(torch.abs(fractions - 0.25) < 0.01), fractions)

    def test_inactive_controller_is_bitwise_official_sigmoid(self):
        t = torch.logspace(-3, 2, 1024)
        baseline = get_schedule('sigmoid', q=256.0)
        local = get_schedule('local_tbin_v1', q=256.0)
        self.assertTrue(torch.equal(
            local.compute_r(t=t, stage=0),
            baseline.compute_r(t=t, stage=0),
        ))

    def test_local_raw_loss_trends_scale_only_their_baseline_gaps(self):
        schedule = get_schedule(
            'local_tbin_v1', q=256.0, short_beta=0.0, long_beta=0.9,
            warmup_updates=0, gain=0.5, deadband=0.0,
        )
        schedule.update_training_signal([1.0, 1.0, 1.0, 1.0])
        schedule.update_training_signal([1.0, 4.0, 1.0, 0.25])
        scales = schedule.gap_scales()
        self.assertAlmostEqual(scales[0], 1.0)
        self.assertGreater(scales[1], 1.0)  # Worsening bin widens its gap.
        self.assertAlmostEqual(scales[2], 1.0)
        self.assertLess(scales[3], 1.0)     # Improving bin tightens its gap.

        log_t = torch.tensor([-3.0, -2.0, -0.5, 1.0], dtype=torch.float64)
        t = log_t.exp()
        baseline_r = get_schedule('sigmoid', q=256.0).compute_r(t=t, stage=0)
        local_r = schedule.compute_r(t=t, stage=0)
        baseline_gap = (t - baseline_r) / t
        local_gap = (t - local_r) / t
        expected = baseline_gap * torch.tensor(scales, dtype=torch.float64)
        self.assertTrue(torch.allclose(local_gap, expected, rtol=1e-12, atol=1e-12))
        self.assertTrue(torch.allclose(
            schedule.preclip_gap_scale(t),
            torch.tensor(scales, dtype=torch.float64),
            rtol=0,
            atol=0,
        ))

    def test_state_round_trip_preserves_bin_controller(self):
        source = get_schedule('local_tbin_v1', warmup_updates=0)
        source.update_training_signal([1.0, 2.0, 3.0, 4.0])
        source.update_training_signal([0.5, 2.5, 2.0, 5.0])
        clone = get_schedule('local_tbin_v1', warmup_updates=0)
        clone.load_state_dict(source.state_dict())
        self.assertEqual(clone.state_dict(), source.state_dict())
        self.assertEqual(clone.gap_scales(), source.gap_scales())

    def test_v2_is_official_sigmoid_until_every_bin_finishes_warmup(self):
        t = torch.logspace(-3, 2, 1024)
        baseline = get_schedule('sigmoid', q=256.0)
        local = get_schedule('local_tbin_v2', q=256.0, warmup_updates=1)
        local.update_training_signal([1.0, 1.0, 1.0, None])
        local.update_training_signal([0.5, 2.0, 1.0, None])
        self.assertEqual(local.gap_scales(), [1.0] * 4)
        self.assertTrue(torch.equal(
            local.compute_r(t=t, stage=0),
            baseline.compute_r(t=t, stage=0),
        ))

    def test_v2_partial_warmup_does_not_apply_min_gap_clamp(self):
        t = torch.logspace(-3, 2, 1024, dtype=torch.float64)
        baseline = get_schedule('sigmoid', q=2.0)
        local = get_schedule('local_tbin_v2', q=2.0, warmup_updates=0)
        local.update_training_signal([1.0, None, None, None])
        self.assertTrue(any(local.bin_is_active(index) for index in range(4)))
        self.assertFalse(local.correction_is_active())
        self.assertTrue(torch.equal(
            local.compute_r(t=t, stage=20),
            baseline.compute_r(t=t, stage=20),
        ))

    def test_v2_scales_are_bounded_and_geometrically_neutral(self):
        schedule = get_schedule(
            'local_tbin_v2',
            short_beta=0.0,
            long_beta=0.9,
            warmup_updates=0,
            gain=0.25,
            min_scale=0.85,
            max_scale=1.25,
            deadband=0.0,
        )
        schedule.update_training_signal([1.0, 1.0, 1.0, 1.0])
        schedule.update_training_signal([0.01, 0.25, 4.0, 100.0])
        scales = schedule.gap_scales()
        self.assertTrue(all(0.85 <= scale <= 1.25 for scale in scales))
        self.assertAlmostEqual(
            sum(math.log(scale) for scale in scales) / len(scales),
            0.0,
            places=14,
        )
        self.assertTrue(any(scale < 1 for scale in scales))
        self.assertTrue(any(scale > 1 for scale in scales))

    def test_v2_local_gap_is_only_normalized_rescaling_of_sigmoid_gap(self):
        schedule = get_schedule(
            'local_tbin_v2',
            q=256.0,
            short_beta=0.0,
            long_beta=0.9,
            warmup_updates=0,
            deadband=0.0,
        )
        schedule.update_training_signal([1.0, 1.0, 1.0, 1.0])
        schedule.update_training_signal([0.5, 2.0, 4.0, 0.25])
        scales = schedule.gap_scales()
        t = torch.tensor([0.05, 0.2, 0.7, 3.0], dtype=torch.float64)
        baseline_r = get_schedule('sigmoid', q=256.0).compute_r(t=t, stage=0)
        local_r = schedule.compute_r(t=t, stage=0)
        baseline_gap = (t - baseline_r) / t
        local_gap = (t - local_r) / t
        expected = baseline_gap * torch.tensor(scales, dtype=torch.float64)
        self.assertTrue(torch.allclose(local_gap, expected, rtol=1e-12, atol=1e-12))

    def test_v3_factorizes_global_and_neutral_local_scales(self):
        schedule = get_schedule(
            'local_tbin_v3',
            q=256.0,
            short_beta=0.0,
            long_beta=0.9,
            warmup_updates=0,
            deadband=0.0,
            global_gap_scale=1.032,
        )
        schedule.update_training_signal([1.0, 1.0, 1.0, 1.0])
        schedule.update_training_signal([0.5, 2.0, 4.0, 0.25])
        local_scales = schedule.gap_scales()
        self.assertAlmostEqual(
            sum(math.log(scale) for scale in local_scales) / len(local_scales),
            0.0,
            places=14,
        )
        t = torch.tensor([0.05, 0.2, 0.7, 3.0], dtype=torch.float64)
        baseline_r = get_schedule('sigmoid', q=256.0).compute_r(t=t, stage=0)
        combined_r = schedule.compute_r(t=t, stage=0)
        expected_gap = (
            (t - baseline_r)
            * torch.tensor(local_scales, dtype=torch.float64)
            * 1.032
        )
        self.assertTrue(torch.allclose(
            t - combined_r, expected_gap, rtol=1e-12, atol=1e-12
        ))
        self.assertTrue(torch.allclose(
            schedule.preclip_gap_scale(t),
            torch.tensor(local_scales, dtype=torch.float64) * 1.032,
            rtol=1e-12,
            atol=1e-12,
        ))


class LocalTBinSignalWindowTest(unittest.TestCase):
    def test_window_aggregates_and_resumes_per_bin_raw_stats(self):
        window = LocalTBinSignalWindow(update_kimg=0.5, num_bins=4)
        window.add([1.0, 2.0, 3.0, 4.0], [1, 1, 1, 1])
        self.assertIsNone(window.pop_if_due(384))
        state = gather_adaptive_signal_window_state(window, torch.device('cpu'))

        resumed = LocalTBinSignalWindow(update_kimg=0.5, num_bins=4, start_nimg=384)
        resumed.load_state_dict(local_adaptive_signal_window_state(state))
        self.assertEqual(resumed.state_dict(), window.state_dict())

        resumed.add([2.0, 4.0, 6.0, 8.0], [1, 2, 3, 4])
        sums, counts = resumed.pop_if_due(512)
        means = globally_average_local_tbin_loss(sums, counts, torch.device('cpu'))
        self.assertEqual(counts, [2, 3, 4, 5])
        self.assertEqual(means, [1.5, 2.0, 2.25, 2.4])


if __name__ == '__main__':
    unittest.main()
