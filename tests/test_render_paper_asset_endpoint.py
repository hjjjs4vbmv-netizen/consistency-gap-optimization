import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts import render_paper_asset_endpoint as endpoint


class PaperAssetEndpointTest(unittest.TestCase):
    @staticmethod
    def config():
        return {
            "schema_version": 1, "asset": "two_budget_endpoint", "metric_name": "fid50k_full",
            "nfe": 1, "evaluation_contract": "q256-fid50k-50k-v1",
            "analysis_track": "two_budget_endpoint", "budgets_kimg": [256, 1024],
            "arms": {
                "A": "arm_a_g100_weight100", "B": "arm_b_g110_weight110",
                "C": "arm_c_g110_weight100", "D": "arm_d_g100_weight110",
            },
        }

    @staticmethod
    def rows():
        rows = []
        for budget, base in ((256, 300), (1024, 10)):
            for seed in (3, 4, 5):
                for arm_index, method in enumerate((
                        "arm_a_g100_weight100", "arm_b_g110_weight110",
                        "arm_c_g110_weight100", "arm_d_g100_weight110",
                )):
                    rows.append({
                        "method": method, "training_seed": seed, "budget_kimg": budget, "nfe": 1,
                        "metric_name": "fid50k_full", "metric_value": base + seed + arm_index,
                        "checkpoint_sha256": "{}-{}-{}".format(method, seed, budget),
                        "sample_count": 50000, "generation_seed_range": "0-49999", "metric_seed": 20260730,
                        "evidence_class": "formal_preregistered", "evaluation_contract": "q256-fid50k-50k-v1",
                        "analysis_track": "two_budget_endpoint",
                    })
        return rows

    def write_inputs(self, root, rows=None):
        source = root / "source.csv"
        rows = self.rows() if rows is None else rows
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        config = root / "endpoint.json"
        config.write_text(json.dumps(self.config()) + "\n", encoding="utf-8")
        return source, config

    def test_renders_two_budget_endpoint_without_claiming_a_complete_curve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config = self.write_inputs(root)
            output = root / "asset"
            endpoint.main(["--input-csv", str(source), "--endpoint-config", str(config), "--outdir", str(output)])
            for extension in ("svg", "png", "pdf"):
                self.assertGreater((output / "asset_endpoint_two_budget.{}".format(extension)).stat().st_size, 0)
            manifest = json.loads((output / "asset_endpoint_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["not_a_complete_learning_curve"])
            self.assertEqual(manifest["publication_qa"]["preview_dpi"], 600)
            self.assertTrue((output / manifest["publication_qa"]["grayscale_preview"]).is_file())

    def test_rejects_mixed_sampling_protocols(self):
        rows = self.rows()
        rows[-1]["sample_count"] = 5000
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config = self.write_inputs(root, rows)
            with self.assertRaisesRegex(SystemExit, "one explicit evaluation_contract"):
                endpoint.main([
                    "--input-csv", str(source), "--endpoint-config", str(config),
                    "--outdir", str(root / "asset"),
                ])


if __name__ == "__main__":
    unittest.main()
