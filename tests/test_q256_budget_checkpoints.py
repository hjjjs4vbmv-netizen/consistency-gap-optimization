import unittest

from training import q256_budget_checkpoints as schedule


class Q256BudgetCheckpointScheduleTest(unittest.TestCase):
    def test_exact_learning_curve_budgets(self):
        observed = [
            kimg
            for kimg in range(0, 1100)
            if schedule.checkpoint_budget_kimg(
                kimg * 1000,
                interval_kimg=128,
                start_kimg=384,
                total_kimg=1024,
            )
            is not None
        ]
        self.assertEqual(observed, list(schedule.BUDGETS_KIMG))
        self.assertEqual(len(observed), 6)

    def test_non_kimg_and_adjacent_counts_do_not_trigger(self):
        for cur_nimg in (383_999, 384_001, 511_999, 512_001, 1_024_001):
            self.assertIsNone(
                schedule.checkpoint_budget_kimg(
                    cur_nimg,
                    interval_kimg=128,
                    start_kimg=384,
                    total_kimg=1024,
                )
            )

    def test_contract_is_frozen(self):
        for field, value in (
            ("interval_kimg", 64),
            ("start_kimg", 320),
            ("total_kimg", 2048),
        ):
            kwargs = {
                "interval_kimg": 128,
                "start_kimg": 384,
                "total_kimg": 1024,
            }
            kwargs[field] = value
            with self.assertRaises(ValueError):
                schedule.checkpoint_budget_kimg(384_000, **kwargs)


if __name__ == "__main__":
    unittest.main()
