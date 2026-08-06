import csv
import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class RetrospectivePackageTest(unittest.TestCase):
    def test_deprecated_q128_256k_path_has_no_misclassified_1024k_rows(self):
        legacy = REPOSITORY_ROOT / "results" / "q128_256k_formal"
        self.assertFalse((legacy / "evaluation_results.csv").exists())
        self.assertFalse((legacy / "paired_differences.csv").exists())
        self.assertIn("# Deprecated", (legacy / "README.md").read_text(encoding="utf-8"))
        for csv_path in legacy.rglob("*.csv"):
            with csv_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    self.assertNotEqual(row.get("budget_kimg"), "1024", csv_path)
                    self.assertFalse(
                        str(row.get("checkpoint_id", "")).startswith("q128-1024k-"),
                        csv_path,
                    )

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
