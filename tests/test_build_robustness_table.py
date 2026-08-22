import csv
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_robustness_table.py"
TABLE = ROOT / "results/robustness/robustness_table.csv"
REPORT = ROOT / "results/robustness/ROBUSTNESS_TABLE.md"


class RobustnessTableTests(unittest.TestCase):
    def test_table_preserves_cross_seed_heterogeneity(self):
        subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, check=True)
        with TABLE.open(newline="") as handle:
            rows = {row["row"]: row for row in csv.DictReader(handle)}

        self.assertEqual(rows["seed 3"]["control_effect"], "regresses")
        self.assertEqual(rows["seed 4"]["control_effect"], "regresses")
        self.assertEqual(rows["seed 5"]["control_effect"], "improves")
        self.assertEqual(rows["sign agreement"]["gap_effect"], "Robust: FID −3/3; KID −3/3")
        self.assertEqual(
            rows["sign agreement"]["control_effect"],
            "Seed-dependent: FID +2/−1; KID +2/−1",
        )
        self.assertEqual(rows["seed 4"]["R_opt"], "not measured")
        self.assertEqual(rows["seed 5"]["R_opt"], "not measured")
        self.assertEqual(rows["seed 3"]["fid_delta_gap_B_minus_A"], "-107.909610640")
        self.assertEqual(rows["seed 3"]["fid_delta_control_C_minus_B"], "90.494417313")
        self.assertEqual(rows["seed 5"]["kid_delta_gap_B_minus_A"], "-0.021834542")
        self.assertEqual(rows["seed 5"]["kid_delta_control_C_minus_B"], "-0.017868804")
        self.assertEqual(rows["mean"]["fid_delta_control_C_minus_B"], "26.974464717")
        self.assertEqual(rows["std"]["fid_delta_control_C_minus_B"], "55.864025241")
        self.assertEqual(rows["seed 3"]["absorption_ratio_fid"], "0.838613139")
        self.assertEqual(rows["seed 5"]["absorption_ratio_fid"], "-0.806500387")

        text = REPORT.read_text()
        self.assertIn("**Seed-dependent / sign-unstable**", text)
        self.assertIn("**Not reproduced**", text)
        self.assertIn("The sign is stable within each seed's three disjoint sampling blocks", text)


if __name__ == "__main__":
    unittest.main()
