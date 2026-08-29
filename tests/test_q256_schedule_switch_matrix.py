import unittest

from analysis.q256_schedule_switch_v1 import analyze_results
from analysis.q256_schedule_switch_v1 import prepare_evaluation_manifest


class ScheduleSwitchMatrixTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
