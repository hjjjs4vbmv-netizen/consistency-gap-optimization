"""Regression test: predictor-comparison summary structure and the exact-replay
identity.

The finite-history replay is the exact moment recursion, so its weighted R^2 and
Corr against the actual h must be 1.0 (a mathematical identity, not a result).
This test guards the summary structure and that identity; it does not assert the
scientific finding (scalar/first-order R^2 < 0), which is data-dependent.
"""
import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "analysis" / "predictor_comparison"
LABELS = ["k32", "k64", "k128", "k256"]
PREDS = ["scalar", "firstorder", "replay"]


class PredictorComparisonTests(unittest.TestCase):
    def test_summary_structure_and_replay_exact(self):
        summary = json.loads((ART / "summary.json").read_text())
        self.assertEqual(set(summary.keys()), set(LABELS))
        for label in LABELS:
            with self.subTest(K=label):
                self.assertEqual(summary[label]["T_steps"], 20)
                horizons = summary[label]["horizons"]
                self.assertEqual(len(horizons), 20)
                for h in horizons:
                    self.assertEqual(set(h.keys()),
                                     {"horizon_steps", "eval_step", "effective_coords",
                                      "scalar", "firstorder", "replay"})
                    for p in PREDS:
                        self.assertEqual(set(h[p].keys()),
                                         {"weighted_R2", "corr", "wRMSE"})
                    # exact-replay identity: R^2 = Corr = 1.0, wRMSE = 0
                    self.assertAlmostEqual(h["replay"]["weighted_R2"], 1.0, places=6)
                    self.assertAlmostEqual(h["replay"]["corr"], 1.0, places=6)
                    self.assertAlmostEqual(h["replay"]["wRMSE"], 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
