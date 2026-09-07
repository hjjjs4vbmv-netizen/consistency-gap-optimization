"""The published result bundle must reproduce its saved seed-level statistics."""
import json
import subprocess
import sys
import unittest
from pathlib import Path


class PublishedM1ResultsTests(unittest.TestCase):
    def assert_statistics_equal(self, actual, expected):
        self.assertEqual(type(actual), type(expected))
        if isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_statistics_equal(actual[key], expected[key])
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))
            for a, e in zip(actual, expected):
                self.assert_statistics_equal(a, e)
        elif isinstance(expected, float):
            # Recomputed SD/CI values differ by a few ULPs across math runtimes.
            self.assertAlmostEqual(actual, expected, delta=1e-12)
        else:
            self.assertEqual(actual, expected)

    def test_recompute_published_statistics(self):
        root = Path(__file__).resolve().parents[1]
        bundle = root / 'analysis/q256_optimizer_restart_ema_rebuild_v1/results'
        result = subprocess.run([sys.executable, str(bundle / 'analyze.py')],
                                check=True, capture_output=True, text=True, cwd=root)
        actual = json.loads(result.stdout)
        self.assert_statistics_equal(actual, json.loads((bundle / 'statistics.json').read_text()))
        self.assertEqual(actual['primary']['status'], 'INCONCLUSIVE')
        self.assertEqual(actual['primary']['n'], 7)
        self.assertEqual(actual['interaction']['n'], 6)


if __name__ == '__main__':
    unittest.main()
