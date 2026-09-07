"""The published result bundle must reproduce its saved seed-level statistics."""
import json
import subprocess
import sys
import unittest
from pathlib import Path


class PublishedM1ResultsTests(unittest.TestCase):
    def test_recompute_published_statistics(self):
        root = Path(__file__).resolve().parents[1]
        bundle = root / 'analysis/q256_optimizer_restart_ema_rebuild_v1/results'
        result = subprocess.run([sys.executable, str(bundle / 'analyze.py')],
                                check=True, capture_output=True, text=True, cwd=root)
        actual = json.loads(result.stdout)
        self.assertEqual(actual, json.loads((bundle / 'statistics.json').read_text()))
        self.assertEqual(actual['primary']['status'], 'INCONCLUSIVE')
        self.assertEqual(actual['primary']['n'], 7)
        self.assertEqual(actual['interaction']['n'], 6)


if __name__ == '__main__':
    unittest.main()
