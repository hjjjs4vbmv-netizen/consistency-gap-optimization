"""Regression test: Git-committed h=20 raw predictions recompute the headline
R²/Corr in analysis/crossk_scalar_history/summary.json.

PR #58 review P0-2: the h=20 headline metrics must be end-to-end self-contained
from the arrays committed to the repo. This test reloads the raw per-coordinate
predictions (h_pred, h_actual, weights) for each K and asserts the weighted R²
and Corr reproduce summary.json within float tolerance — the same metric
functions used by the sweep (analysis/crossk_horizon_sweep.weighted_r2/corr),
applied to the same saved arrays, so any mismatch is a persistence error.

The full R²(K,h) matrix for h < 20 still requires the external raw histories
(see analysis/crossk_scalar_history/raw_manifest.json); this test covers the
headline h=20 horizon that IS committed.
"""
import json
import unittest
from pathlib import Path

import numpy as np

from analysis.crossk_horizon_sweep import corr, weighted_r2

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "analysis" / "crossk_scalar_history"
LABELS = ["k32", "k64", "k128", "k256"]


class CrossKRecomputeTests(unittest.TestCase):
    def test_h20_metrics_recompute_from_committed_raw(self):
        summary = json.loads((ART / "summary.json").read_text())
        for label in LABELS:
            with self.subTest(K=label):
                row = next(h for h in summary[label]["horizons"]
                           if h["horizon_steps"] == 20)
                raw = ART / label / "raw_predictions"
                h_pred = np.load(raw / "h_pred_scalar_h20.npy")
                h_act = np.load(raw / "h_actual_h20.npy")
                w = np.load(raw / "weights_h20.npy")
                # committed arrays are float64; shapes must agree
                self.assertEqual(h_pred.shape, h_act.shape)
                self.assertEqual(h_pred.shape, w.shape)
                self.assertGreater(h_pred.shape[0], 0)
                # weights are squared update magnitudes -> non-negative
                self.assertGreaterEqual(w.min(), 0.0)
                r2 = weighted_r2(h_pred, h_act, w)
                r = corr(h_pred, h_act, w)
                self.assertAlmostEqual(r2, row["weighted_R2"], places=5,
                                       msg=f"{label}: h=20 weighted R² mismatch")
                self.assertAlmostEqual(r, row["corr"], places=5,
                                       msg=f"{label}: h=20 corr mismatch")
                # honest consistency: raw arrays carry the effective-coord count
                self.assertEqual(int(h_pred.shape[0]), row["effective_coords"])

    def test_h20_a_star_series_is_committed_and_finite(self):
        # a* series is part of the committed raw audit trail; must be finite
        # and near the reported range (~0.77) at every stage.
        summary = json.loads((ART / "summary.json").read_text())
        for label in LABELS:
            with self.subTest(K=label):
                series = np.load(ART / label / "raw_predictions" /
                                 "a_star_series.npy")
                self.assertTrue(np.isfinite(series).all())
                self.assertEqual(series.shape[0], summary[label]["T_steps"])
                self.assertAlmostEqual(float(series.mean()),
                                       summary[label]["a_star_mean"], places=4)


if __name__ == "__main__":
    unittest.main()
