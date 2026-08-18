"""Regression test: predictor-comparison summary structure (revised, no oracle).

Verifies the summary has the two-regime, three-predictor (same information
budget) structure and that Var_w(h_actual) is recorded (contextualizes R^2).
Does NOT assert the scientific finding (which predictor wins) — that is
data-dependent. Does NOT assert any oracle identity (the old h_replay=h_actual
predictor is removed).
"""
import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "analysis" / "predictor_comparison"
LABELS = ["k32", "k64", "k128", "k256"]
PREDS = ["global_scalar", "local_continuous", "discrete_replay"]
REGIMES = ["fresh", "real"]


class PredictorComparisonTests(unittest.TestCase):
    def test_summary_structure(self):
        summary = json.loads((ART / "summary.json").read_text())
        self.assertEqual(set(summary.keys()), set(LABELS))
        for label in LABELS:
            with self.subTest(K=label):
                self.assertEqual(summary[label]["T_steps"], 20)
                self.assertEqual(set(summary[label]["regimes"].keys()), set(REGIMES))
                for regime in REGIMES:
                    horizons = summary[label]["regimes"][regime]["horizons"]
                    self.assertEqual(len(horizons), 20)
                    for h in horizons:
                        self.assertEqual(
                            set(h.keys()),
                            {"horizon_steps", "eval_step", "effective_coords",
                             "global_scalar", "local_continuous", "discrete_replay",
                             "Var_w_h_actual", "h_actual_mean"})
                        for p in PREDS:
                            self.assertEqual(set(h[p].keys()),
                                             {"weighted_R2", "corr", "wRMSE"})
                        # Var_w must be finite and non-negative (a variance)
                        self.assertTrue(h["Var_w_h_actual"] >= 0,
                                        f"{label}/{regime} h{h['horizon_steps']}: Var_w<0")
                        self.assertTrue(h["Var_w_h_actual"] == h["Var_w_h_actual"],
                                        f"{label}/{regime}: Var_w is NaN")

    def test_fresh_var_smaller_than_real(self):
        # the target variance should be much smaller in fresh (near-constant h)
        # than in real — this is why fresh-start R^2 is uninformative.
        summary = json.loads((ART / "summary.json").read_text())
        for label in LABELS:
            with self.subTest(K=label):
                vf = summary[label]["regimes"]["fresh"]["horizons"][-1]["Var_w_h_actual"]
                vr = summary[label]["regimes"]["real"]["horizons"][-1]["Var_w_h_actual"]
                self.assertLess(vf, vr,
                                f"{label}: fresh Var_w should be < real Var_w")


if __name__ == "__main__":
    unittest.main()
