import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class RetrospectivePackageTest(unittest.TestCase):
    def test_repository_package_is_path_consistent_and_not_formal(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_q128_1024k_retrospective_package.py",
                "--package",
                "results/q128_1024k_retrospective",
            ],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("provenance remains incomplete", result.stdout)


if __name__ == "__main__":
    unittest.main()
