import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts import render_paper_asset_c as asset_c


class PaperAssetCTest(unittest.TestCase):
    @staticmethod
    def make_rows():
        rows = []
        base_values = {
            256: {"arm_a": 300, "arm_b": 180, "arm_c": 130, "arm_d": 100},
            512: {"arm_a": 180, "arm_b": 120, "arm_c": 100, "arm_d": 80},
            768: {"arm_a": 90, "arm_b": 65, "arm_c": 55, "arm_d": 45},
            1024: {"arm_a": 45, "arm_b": 35, "arm_c": 32, "arm_d": 28},
        }
        for budget, arms in base_values.items():
            for seed in (3, 4, 5):
                for method, value in arms.items():
                    rows.append({
                        "method": method, "training_seed": seed, "budget_kimg": budget,
                        "nfe": 2, "metric_name": "fid5k_full", "metric_value": value + seed,
                        "checkpoint_sha256": "{}-{}-{}".format(method, seed, budget),
                        "sample_count": 5000, "generation_seed_range": "0-4999", "metric_seed": 20260730,
                        "evidence_class": "auxiliary" if budget == 1024 else "quick",
                        "evaluation_contract": "q256-common-5k-v1", "analysis_track": "budget_curve",
                    })
        return rows

    @staticmethod
    def config():
        return {
            "schema_version": 1, "asset": "C", "metric_name": "fid5k_full", "nfe": 2,
            "evaluation_contract": "q256-common-5k-v1", "analysis_track": "budget_curve",
            "budgets_kimg": [256, 512, 768, 1024],
            "arms": {"A": "arm_a", "B": "arm_b", "C": "arm_c", "D": "arm_d"},
        }

    def write_inputs(self, root, rows=None):
        rows = self.make_rows() if rows is None else rows
        source = root / "input.csv"
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        config = root / "arms.json"
        config.write_text(json.dumps(self.config(), indent=2) + "\n", encoding="utf-8")
        return source, config

    def test_writes_seed_resolved_dispersion_and_summary_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config = self.write_inputs(root)
            output = root / "asset_c"
            asset_c.main(["--input-csv", str(source), "--arm-config", str(config), "--outdir", str(output)])
            with (output / "asset_c_dispersion_by_seed.csv").open(newline="", encoding="utf-8") as handle:
                per_seed = list(csv.DictReader(handle))
            self.assertEqual(len(per_seed), 12)
            self.assertEqual({row["dispersion_fid"] for row in per_seed if row["budget_kimg"] == "256"}, {"200"})
            with (output / "asset_c_contraction_by_seed.csv").open(newline="", encoding="utf-8") as handle:
                contraction = list(csv.DictReader(handle))
            self.assertEqual({row["contracts"] for row in contraction}, {"true"})
            manifest = json.loads((output / "asset_c_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["all_seed_dispersions_contract"])
            for extension in ("svg", "png", "pdf"):
                self.assertGreater((output / "asset_c_arm_dispersion.{}".format(extension)).stat().st_size, 0)

    def test_rejects_missing_four_arm_seed_budget_cell(self):
        rows = [
            row for row in self.make_rows()
            if not (row["method"] == "arm_d" and row["training_seed"] == 5 and row["budget_kimg"] == 1024)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config = self.write_inputs(root, rows=rows)
            with self.assertRaisesRegex(SystemExit, "matrix incomplete"):
                asset_c.main(["--input-csv", str(source), "--arm-config", str(config), "--outdir", str(root / "asset_c")])

    def test_rejects_single_seed_mean_only_plot(self):
        rows = [row for row in self.make_rows() if row["training_seed"] == 3]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config = self.write_inputs(root, rows=rows)
            with self.assertRaisesRegex(SystemExit, "at least two seeds"):
                asset_c.main(["--input-csv", str(source), "--arm-config", str(config), "--outdir", str(root / "asset_c")])


if __name__ == "__main__":
    unittest.main()
