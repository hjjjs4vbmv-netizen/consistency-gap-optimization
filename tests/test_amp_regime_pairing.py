"""AMP scale, skip, and optimizer-step pairing gates."""
from __future__ import annotations

import unittest

from analysis.jacobian_failure_factorial.core import (
    amp_regime_signature,
    optimizer_step_executed,
)


class AMPRegimePairingTests(unittest.TestCase):
    def test_optimizer_step_parity_uses_all_tracked_steps(self):
        before = {"optimizer_steps": [4, 4], "scaler_growth_tracker": 7}
        self.assertTrue(optimizer_step_executed(
            before, {"optimizer_steps": [5, 5], "scaler_growth_tracker": 8}))
        self.assertFalse(optimizer_step_executed(
            before, {"optimizer_steps": [5, 4], "scaler_growth_tracker": 8}))

    def test_amp_signature_detects_scale_skip_and_step_mismatch(self):
        base = {
            "amp_enabled": True, "grad_scale_before": 1024.0,
            "grad_scale_after": 1024.0, "step_skipped": False,
        }
        reference = amp_regime_signature(base, step_executed=True)
        self.assertEqual(reference, amp_regime_signature(base, step_executed=True))
        changed_scale = dict(base, grad_scale_after=512.0, step_skipped=True)
        self.assertNotEqual(
            reference, amp_regime_signature(changed_scale, step_executed=False))
        self.assertNotEqual(
            reference, amp_regime_signature(base, step_executed=False))


if __name__ == "__main__":
    unittest.main()
