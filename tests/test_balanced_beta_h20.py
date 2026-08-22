"""Regression test: Git-committed balanced-β h=20 raw h_i arrays recompute the
h_i statistics in analysis/balanced_beta/summary.json.

The raw_h20 files store h_actual = ug/u1 on the effective support at h=20 for
each β config and each K. This test reloads them and asserts the recomputed
Disp(h), h_actual_mean, h_actual_std, and effective_coords match summary.json.

R_opt itself is computed over the full 55M-dim vector (not committed — it is
covered by the raw-input SHA256 locators in analysis/crossk_scalar_history/
raw_manifest.json); the h_i statistics are the committable audit trail.
"""
import json
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "analysis" / "balanced_beta"
LABELS = ["k32", "k64", "k128", "k256"]
CONFIGS = ["standard", "balanced_0.9", "balanced_0.99", "balanced_0.999"]


class BalancedBetaRecomputeTests(unittest.TestCase):
    def test_h20_h_i_stats_recompute_from_committed_raw(self):
        summary = json.loads((ART / "summary.json").read_text())
        for label in LABELS:
            for cfg in CONFIGS:
                with self.subTest(K=label, config=cfg):
                    row = next(h for h in summary[label]["configs"][cfg]["horizons"]
                               if h["horizon_steps"] == 20)
                    h = np.load(ART / label / "raw_h20" / f"{cfg}_h_actual.npy")
                    self.assertGreater(h.shape[0], 0)
                    self.assertEqual(int(h.shape[0]), row["effective_coords"])
                    self.assertAlmostEqual(float(np.mean(h)), row["h_actual_mean"],
                                           places=5)
                    self.assertAlmostEqual(float(np.std(h)), row["h_actual_std"],
                                           places=5)
                    self.assertAlmostEqual(float(np.std(h)), row["Disp_h"], places=5)


if __name__ == "__main__":
    unittest.main()
