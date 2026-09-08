import math
import unittest

from scripts import state_intervention_contrasts as analysis


class ContrastTests(unittest.TestCase):
    def rows(self):
        return [dict(seed=55, branch='L_A', budget_kimg=1024, readout='E_512', block=b,
                     status='PASS', fid50k_full=f, kid50k_full=k)
                for b, f, k in [('B0', 1, -0.1), ('B1', 4, 0), ('B2', 16, 0.1)]]

    def test_blocks_are_mean_logs_and_kid_is_raw(self):
        key = (55, 'L_A', 1024)
        fid = analysis.aggregate_cells(self.rows(), 'fid50k_full')[key]
        self.assertAlmostEqual(fid, math.log(4))
        self.assertNotAlmostEqual(fid, math.log(7))
        self.assertEqual(analysis.aggregate_cells(self.rows(), 'kid50k_full')[key], 0)
        self.assertIsNone(analysis.aggregate_cells(self.rows()[:2], 'fid50k_full')[key])
        with self.assertRaises(ValueError):
            analysis.aggregate_cells(self.rows() + self.rows()[:1], 'fid50k_full')
        failed = self.rows()
        failed[2]['status'] = 'EVAL_NONFINITE'
        self.assertIsNone(analysis.aggregate_cells(failed, 'fid50k_full')[key])

    def cells(self):
        values = {('K_A', 1024): 1, ('K_B', 1024): 2, ('R_A', 1024): 3,
                  ('R_B', 1024): 7, ('L_A', 1024): 4, ('L_B', 1024): 5,
                  ('K_A', 768): 10, ('K_B', 768): 20, ('R_A', 768): 12,
                  ('R_B', 768): 23, ('X_A_from_B', 1024): 0, ('X_B_from_A', 1024): 6}
        return {(seed, *key): value for seed in analysis.SEEDS for key, value in values.items()}

    def test_m2_components_and_same_chase_use_correct_baselines(self):
        result = analysis.summarize_cells(self.cells())
        se = result['m2_same_endpoint']['per_seed'][0]
        self.assertEqual([se[k] for k in ('d_A_512', 'd_B_512', 'd_A_768', 'd_B_768')], [2, 5, 3, 3])
        self.assertEqual(se['Theta'], (5 - 4) - (7 - 3))
        sc = result['m2_same_chase']['per_seed'][0]
        self.assertEqual(sc['Theta'], -1)
        self.assertEqual(sc['d_B_768'], 5 - 2)

    def test_missingness_keeps_analyses_and_single_sides_separate(self):
        cells = self.cells()
        cells[(55, 'L_A', 1024)] = None
        cells[(56, 'R_A', 768)] = None
        cells[(59, 'X_B_from_A', 1024)] = None
        cells[(61, 'R_A', 1024)] = None
        result = analysis.summarize_cells(cells)
        self.assertEqual(result['m2_same_endpoint']['seeds'], [56, 59, 60, 62])
        self.assertEqual(result['m2_same_chase']['seeds'], [59, 60, 61, 62])
        self.assertEqual(result['m2_time_intersection']['seeds'], [59, 60, 62])
        self.assertEqual(result['swap_common']['seeds'], [55, 56, 60, 61, 62])
        self.assertIn(59, result['swap_A_all_pairs']['seeds'])
        self.assertNotIn(59, result['swap_B_all_pairs']['seeds'])
        # Even where K cancels algebraically, its required component must remain available.
        cells[(60, 'K_B', 1024)] = None
        self.assertNotIn(60, analysis.summarize_cells(cells)['m2_same_endpoint']['seeds'])

    def test_swap_sign_and_no_joint_verdict(self):
        swap = analysis.summarize_cells(self.cells())['swap_common']
        self.assertEqual(swap['per_seed'][0], dict(seed=55, D_A=-1, D_B=-4, P_match=1.5, T=1))
        self.assertNotIn('verdict', swap)
        self.assertEqual(swap['estimates']['D_A']['n'], 6)

    def test_seed_level_t_interval_and_degenerate_cases(self):
        result = analysis.paired_statistics([1, 2, 3])
        self.assertEqual(result['n'], 3)
        self.assertEqual(result['sample_sd'], 1)
        self.assertAlmostEqual(result['ci95'][0], -0.4841377117, places=8)
        self.assertEqual(result['ci_direction'], 'INCONCLUSIVE')
        for values in ([], [2]):
            self.assertEqual(analysis.paired_statistics(values)['ci_direction'], 'INSUFFICIENT_COMPLETE_PAIRS')
        self.assertEqual(analysis.paired_statistics([2, 2])['ci_direction'], 'DEGENERATE_SD')
        self.assertEqual(analysis.paired_statistics([2, 2])['ci95'], None)


if __name__ == '__main__':
    unittest.main()
