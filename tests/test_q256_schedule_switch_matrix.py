import csv
import unittest
from pathlib import Path

from analysis.q256_schedule_switch_v1 import analyze_results
from analysis.q256_schedule_switch_v1 import prepare_evaluation_manifest


class ScheduleSwitchMatrixTests(unittest.TestCase):
    RESULT_ROOT = (
        Path(__file__).resolve().parents[1]
        / "results" / "q256_schedule_switch_seed3_7"
    )

    def test_exact_80_job_matrix(self):
        cells = {
            (seed, branch, budget, nfe)
            for seed in prepare_evaluation_manifest.SEEDS
            for branch in prepare_evaluation_manifest.BRANCHES
            for budget in prepare_evaluation_manifest.BUDGETS
            for nfe in (1, 2)
        }
        self.assertEqual(len(cells), 80)
        self.assertEqual(
            prepare_evaluation_manifest.EVALUATOR_COMMIT,
            "d6aba02fb88e9db0993623895eb2228ed717d810",
        )

    def test_normalized_trapezoidal_aulc(self):
        points = [(budget, 7.5) for budget in analyze_results.BUDGETS]
        self.assertEqual(analyze_results.normalized_aulc(points), 7.5)
        linear = [(budget, float(budget)) for budget in analyze_results.BUDGETS]
        self.assertEqual(analyze_results.normalized_aulc(linear), 768.0)

    def test_frozen_contrast_algebra(self):
        aa, ab, ba, bb = 1.0, 3.0, 2.0, 7.0
        self.assertEqual(ab - aa, 2.0)  # S_A
        self.assertEqual(bb - ba, 5.0)  # S_B
        self.assertEqual(ba - aa, 1.0)  # H_A
        self.assertEqual(bb - ab, 4.0)  # H_B
        self.assertEqual(bb - ba - ab + aa, 3.0)  # I_switch

    def test_post_unblind_outputs_are_labeled_and_scoped(self):
        delayed_path = self.RESULT_ROOT / "per_seed_delayed_reversal.csv"
        log_path = self.RESULT_ROOT / "contrast_summaries_logfid.csv"
        for path in (delayed_path, log_path):
            with path.open(encoding="utf-8") as handle:
                self.assertEqual(
                    handle.readline().rstrip("\n"),
                    "# post-unblind descriptive",
                )

        with delayed_path.open(encoding="utf-8") as handle:
            next(handle)
            delayed = list(csv.DictReader(handle))
        self.assertEqual(len(delayed), 5)
        self.assertEqual(
            {
                int(row["seed"])
                for row in delayed
                if row["descriptive_delayed_reversal"] == "True"
            },
            {5, 6, 7},
        )
        self.assertTrue(all(
            row["endpoint_B_history_better_under_both_current_policies"]
            == "True"
            for row in delayed
        ))

        with log_path.open(encoding="utf-8") as handle:
            next(handle)
            log_rows = {row["contrast"]: row for row in csv.DictReader(handle)}
        self.assertEqual(set(log_rows), {"S_A", "S_B", "H_A", "H_B", "I_switch"})
        self.assertEqual(log_rows["H_A"]["n_negative"], "5")
        self.assertEqual(log_rows["H_B"]["n_negative"], "5")
        self.assertEqual(log_rows["I_switch"]["n_negative"], "3")
        self.assertEqual(log_rows["I_switch"]["n_positive"], "2")

        report = (self.RESULT_ROOT / "REPORT.md").read_text(encoding="utf-8")
        self.assertIn("EXECUTION AND ANALYSIS PIPELINE PASS", report)
        self.assertIn("not a scientific-hypothesis verdict", report)
        self.assertIn("do not support a strong interaction claim", report)


if __name__ == "__main__":
    unittest.main()
