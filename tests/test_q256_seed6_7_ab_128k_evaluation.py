import tempfile
import unittest
from pathlib import Path

from scripts import collect_q256_seed6_7_ab_128k_learning_curve as collector
from scripts import run_q256_seed6_7_ab_128k_frozen_evaluation as evaluation


class FakeEvaluator:
    REPO_ROOT = Path("/frozen/evaluator")
    METRICS = ("kid50k_full", "fid50k_full")
    SAMPLE_COUNT = 50_000
    SAMPLE_SEEDS = "0-49999"
    METRIC_SEED = 20_260_730


class Q256Seed67AB128kEvaluationTest(unittest.TestCase):
    def cells(self):
        return [
            {
                "seed": seed,
                "arm": arm,
                "budget_kimg": budget,
                "checkpoint": f"/checkpoints/seed{seed}/arm{arm}/{budget}k/network-snapshot.pkl",
                "checkpoint_sha256": "a" * 64,
                "training_state": f"/checkpoints/seed{seed}/arm{arm}/{budget}k/training-state.pt",
                "training_state_sha256": "b" * 64,
                "run_dir": f"/checkpoints/seed{seed}/arm{arm}/{budget}k",
                "training_validation_receipt": "/receipts/cell.json",
                "training_validation_receipt_sha256": "c" * 64,
                "training_hash_receipt": "/receipts/cell.json",
                "training_hash_receipt_sha256": "c" * 64,
            }
            for seed in evaluation.SEEDS
            for arm in evaluation.ARMS
            for budget in evaluation.BUDGETS_KIMG
        ]

    def test_exact_24_job_nfe1_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = evaluation.build_jobs(
                FakeEvaluator, self.cells(), Path(tmp), 33_880
            )
        self.assertEqual(len(jobs), 24)
        self.assertEqual(len({job["job_id"] for job in jobs}), 24)
        self.assertEqual(
            {(job["seed"], job["arm"], job["budget_kimg"]) for job in jobs},
            {
                (seed, arm, budget)
                for seed in (6, 7)
                for arm in ("A", "B")
                for budget in (384, 512, 640, 768, 896, 1024)
            },
        )
        for job in jobs:
            command = job["command_argv_template"]
            self.assertEqual(job["nfe"], 1)
            self.assertEqual(job["mid_t"], [])
            self.assertIn("--nfe=1", command)
            self.assertNotIn("--nfe=2", command)
            self.assertFalse(any(item.startswith("--mid_t") for item in command))
            self.assertIn("--metrics=kid50k_full,fid50k_full", command)
            self.assertIn("--sample-seeds=0-49999", command)
            self.assertIn("--seed=20260730", command)
            self.assertFalse(any("5k" in item and "50k" not in item for item in command))

    def test_contrast_and_threshold_summary_stays_on_observed_budgets(self):
        rows = []
        for seed in (6, 7):
            for arm in ("A", "B"):
                for index, budget in enumerate(collector.BUDGETS):
                    rows.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "budget_kimg": budget,
                            "fid50k_full": 325 - index * 25 + (5 if arm == "B" else 0),
                            "kid50k_full": 0.32 - index * 0.03 + (0.01 if arm == "B" else 0),
                        }
                    )
        contrasts, summaries, thresholds = collector.build_summary(rows)
        self.assertEqual(len(contrasts), 14)
        self.assertEqual(len(summaries), 2)
        self.assertTrue(thresholds)
        for row in contrasts:
            self.assertIn(row["budget_kimg"], collector.BUDGETS)
            self.assertAlmostEqual(row["B_minus_A_fid"], 5.0)
            self.assertAlmostEqual(row["B_minus_A_kid"], 0.01)
        for row in thresholds:
            self.assertTrue(row["within_observed_curve_range"])
            for field in ("tau_A_kimg", "tau_B_kimg"):
                self.assertTrue(row[field] is None or row[field] in collector.BUDGETS)


if __name__ == "__main__":
    unittest.main()
