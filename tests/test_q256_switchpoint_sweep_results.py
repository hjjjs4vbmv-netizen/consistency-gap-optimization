import csv
import itertools
import json
import math
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "results/q256_switchpoint_sweep/full_cohort"


class PublishedResultsTest(unittest.TestCase):
    def assert_same_numbers(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_same_numbers(actual[key], expected[key])
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_same_numbers(left, right)
        elif isinstance(expected, float):
            # statistics.stdev can differ in its final bits across Python versions.
            self.assertAlmostEqual(actual, expected, delta=1e-12)
        else:
            self.assertEqual(actual, expected)

    def test_report_reproduces_published_outputs_without_rewriting_verification(self):
        verification = (BUNDLE / "verification.json").read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [sys.executable, "-m", "analysis.q256_switchpoint_sweep.summarize_results",
                 "--output-dir", directory], cwd=REPO, check=True, stdout=subprocess.DEVNULL,
            )
            for name in (
                "REPORT_ZH.md", "raw_metrics.csv", "fixed_chase_seed_results.csv",
                "summary.csv", "frozen_calculation.json", "common_endpoint_descriptive.json",
                "preliminary_comparison.json", "companion_status.json",
            ):
                with self.subTest(name=name):
                    actual, expected = Path(directory) / name, BUNDLE / name
                    if name.endswith(".json"):
                        self.assert_same_numbers(json.loads(actual.read_text()), json.loads(expected.read_text()))
                    elif name == "summary.csv":
                        with actual.open(newline="") as a, expected.open(newline="") as b:
                            left, right = list(csv.DictReader(a)), list(csv.DictReader(b))
                        self.assertEqual(len(left), len(right))
                        for rows in (left, right):
                            for row in rows:
                                for key in row:
                                    if key != "comparison":
                                        row[key] = float(row[key])
                        self.assert_same_numbers(left, right)
                    else:
                        self.assertEqual(actual.read_bytes(), expected.read_bytes())
            self.assertFalse((Path(directory) / "verification.json").exists())
        self.assertEqual((BUNDLE / "verification.json").read_bytes(), verification)

    def test_page_value_from_raw_metrics_and_independent_exact_distribution(self):
        with (BUNDLE / "raw_metrics.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        cells = {(int(r["seed"]), r["trajectory"], int(r["kimg"])): float(r["fid50k_full"]) for r in rows}
        self.assertEqual(len(rows), 132)
        self.assertEqual(len(cells), 132)
        observed = 0
        for seed in range(81, 93):
            z = [-math.log(cells[seed, f"BA{s}", s + 512] / cells[seed, "CTRL", s + 512])
                 for s in (128, 256, 384, 512)]
            self.assertEqual(len(set(z)), 4)
            ranks = [sorted(z).index(value) + 1 for value in z]
            observed += sum(weight * rank for weight, rank in zip((1, 2, 3, 4), ranks))
        block = Counter(sum(w * r for w, r in zip((1, 2, 3, 4), p))
                        for p in itertools.permutations((1, 2, 3, 4)))
        distribution = Counter({0: 1})
        for _ in range(12):
            updated = Counter()
            for score, count in distribution.items():
                for increment, frequency in block.items():
                    updated[score + increment] += count * frequency
            distribution = updated
        p_exact = sum(count for score, count in distribution.items() if score >= observed) / 24 ** 12
        result = json.loads((BUNDLE / "frozen_calculation.json").read_text())["page_test"]
        self.assertEqual(result["L_observed"], observed)
        self.assertEqual(result["p_exact"], p_exact)


if __name__ == "__main__":
    unittest.main()
