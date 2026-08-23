import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts import render_paper_asset_b as asset_b


class PaperAssetBTest(unittest.TestCase):
    @staticmethod
    def make_rows():
        rows = []
        for budget, fixed, global110 in (
            (256, 220, 180), (512, 150, 90), (768, 80, 55), (1024, 45, 30),
        ):
            for seed in (3, 4, 5):
                for method, value in (("fixed", fixed + seed), ("global110", global110 + seed)):
                    rows.append({
                        "method": method, "training_seed": seed, "budget_kimg": budget,
                        "nfe": 2, "metric_name": "fid5k_full", "metric_value": value,
                        "checkpoint_sha256": "{}-{}-{}".format(method, seed, budget),
                        "sample_count": 5000, "generation_seed_range": "0-4999", "metric_seed": 20260730,
                        "evidence_class": "auxiliary" if budget == 1024 else "quick",
                        "evaluation_contract": "q256-common-5k-v1", "analysis_track": "budget_curve",
                    })
        return rows

    @staticmethod
    def config(mode="first_observed"):
        return {
            "schema_version": 1, "asset": "B", "threshold_id": "fid5k-eta-100",
            "metric_name": "fid5k_full", "nfe": 2, "threshold": 100,
            "crossing_mode": mode, "evaluation_contract": "q256-common-5k-v1",
            "analysis_track": "budget_curve", "budgets_kimg": [256, 512, 768, 1024],
            "arms": {"A": "fixed", "B": "global110"},
        }

    def write_inputs(self, root, rows=None, config=None):
        rows = self.make_rows() if rows is None else rows
        config = self.config() if config is None else config
        source = root / "input.csv"
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        frozen = root / "threshold.json"
        frozen.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        return source, frozen

    def test_first_observed_crossings_keep_per_seed_deltas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, frozen = self.write_inputs(root)
            output = root / "asset_b"
            asset_b.main(["--input-csv", str(source), "--threshold-config", str(frozen), "--outdir", str(output)])
            with (output / "asset_b_delta_tau_by_seed.csv").open(newline="", encoding="utf-8") as handle:
                deltas = list(csv.DictReader(handle))
            self.assertEqual([row["delta_tau_B_minus_A_kimg"] for row in deltas], ["-256", "-256", "-256"])
            self.assertEqual({row["A_crossing_status"] for row in deltas}, {"first_observed"})
            self.assertEqual({row["B_crossing_status"] for row in deltas}, {"first_observed"})
            manifest = json.loads((output / "asset_b_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["all_paired_deltas_negative"])
            for extension in ("svg", "png", "pdf"):
                self.assertGreater((output / "asset_b_compute_to_quality.{}".format(extension)).stat().st_size, 0)

    def test_linear_mode_is_explicitly_descriptive_and_never_uses_discrete_tau(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, frozen = self.write_inputs(root, config=self.config("linear_interpolation_descriptive"))
            output = root / "asset_b"
            asset_b.main(["--input-csv", str(source), "--threshold-config", str(frozen), "--outdir", str(output)])
            with (output / "asset_b_tau_by_seed.csv").open(newline="", encoding="utf-8") as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual({row["crossing_status"] for row in records}, {"linear_interpolation_descriptive"})
            self.assertTrue(all(float(row["tau_kimg"]) not in {512, 768} for row in records))

    def test_linear_mode_rejects_unbracketed_left_censoring(self):
        rows = self.make_rows()
        for row in rows:
            if row["method"] == "global110" and row["budget_kimg"] == 256:
                row["metric_value"] = 80
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, frozen = self.write_inputs(root, rows=rows, config=self.config("linear_interpolation_descriptive"))
            with self.assertRaisesRegex(SystemExit, "first observed checkpoint already meets"):
                asset_b.main(["--input-csv", str(source), "--threshold-config", str(frozen), "--outdir", str(root / "asset_b")])


if __name__ == "__main__":
    unittest.main()
