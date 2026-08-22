import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts import render_paper_asset_a as asset_a


class PaperAssetATest(unittest.TestCase):
    @staticmethod
    def make_rows():
        rows = []
        for budget in (256, 512, 768, 1024):
            for seed in (3, 4, 5, 6):
                for method in ("fixed", "global110"):
                    rows.append({
                        "method": method,
                        "training_seed": seed,
                        "budget_kimg": budget,
                        "nfe": 2,
                        "metric_name": "fid5k_full",
                        "metric_value": 450 - budget / 3 + seed + (0 if method == "fixed" else -7),
                        "checkpoint_sha256": "{}-{}-{}".format(method, seed, budget),
                        "sample_count": 5000,
                        "generation_seed_range": "0-4999",
                        "metric_seed": 20260730,
                        "evidence_class": "auxiliary" if budget == 1024 else "quick",
                        "evaluation_contract": "q256-common-5k-v1",
                        "analysis_track": "budget_curve",
                    })
        return rows

    def write_source(self, root: Path, rows: list[dict]) -> Path:
        source = root / "input.csv"
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        return source

    def test_renders_complete_protocol_matched_four_budget_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_source(root, self.make_rows())
            output = root / "asset_a"
            asset_a.main([
                "--input-csv", str(source), "--outdir", str(output), "--nfe", "2",
                "--primary-seeds", "3,4", "--seed-labels", "3=A,4=B,5=C,6=D",
            ])
            for extension in ("svg", "png", "pdf"):
                self.assertGreater((output / "asset_a_fid_vs_budget.{}".format(extension)).stat().st_size, 0)
            with (output / "asset_a_fid_vs_budget.csv").open(newline="", encoding="utf-8") as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual(len(records), 32)
            self.assertEqual({row["visibility"] for row in records if row["seed_label"] in {"A", "B"}}, {"primary"})
            self.assertEqual({row["visibility"] for row in records if row["seed_label"] in {"C", "D"}}, {"context"})
            manifest = json.loads((output / "asset_a_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["budgets_kimg"], [256, 512, 768, 1024])
            self.assertEqual(manifest["protocol"]["evaluation_contract"], "q256-common-5k-v1")

    def test_rejects_a_curve_with_mismatched_protocol(self):
        rows = self.make_rows()
        rows[-1]["sample_count"] = 50000
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_source(root, rows)
            with self.assertRaisesRegex(SystemExit, "one explicit evaluation_contract"):
                asset_a.main(["--input-csv", str(source), "--outdir", str(root / "asset_a"), "--nfe", "2"])

    def test_rejects_a_missing_budget_checkpoint(self):
        rows = [
            row for row in self.make_rows()
            if not (row["method"] == "global110" and row["training_seed"] == 6 and row["budget_kimg"] == 1024)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_source(root, rows)
            with self.assertRaisesRegex(SystemExit, "matrix incomplete"):
                asset_a.main(["--input-csv", str(source), "--outdir", str(root / "asset_a"), "--nfe", "2"])

    def test_rejects_two_budget_endpoint_as_a_learning_curve(self):
        rows = [row for row in self.make_rows() if row["budget_kimg"] in {256, 1024}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_source(root, rows)
            with self.assertRaisesRegex(SystemExit, "reserved for the complete protocol-matched"):
                asset_a.main([
                    "--input-csv", str(source), "--outdir", str(root / "asset_a"), "--nfe", "2",
                    "--budgets", "256,1024",
                ])


if __name__ == "__main__":
    unittest.main()
