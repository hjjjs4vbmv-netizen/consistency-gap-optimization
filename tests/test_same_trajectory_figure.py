import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts import plot_same_trajectory_longitudinal as longitudinal_plot


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = REPO_ROOT / "analysis" / "same_trajectory_longitudinal"
SUMMARY = BUNDLE / "longitudinal_summary.csv"


class SameTrajectoryFigureTest(unittest.TestCase):
    def test_summary_matches_receipts_and_locked_display_values(self):
        rows = longitudinal_plot.read_summary(SUMMARY)
        self.assertEqual(
            [longitudinal_plot.format_percent(row.r_opt) for row in rows],
            ["8.22%", "8.96%", "9.38%", "9.90%"],
        )
        self.assertEqual(
            [f"{row.c_k_star:.3f}" for row in rows],
            ["1.033", "1.030", "1.035", "1.033"],
        )
        self.assertTrue(all(1.029 <= row.c_k_star <= 1.035 for row in rows))

        for row in rows:
            state_id = f"k{round(row.k_kimg * 1000):06d}"
            receipt = json.loads(
                (BUNDLE / f"radam_update_audit_stateful_{state_id}.json").read_text(
                    encoding="utf-8"
                )
            )
            whole_model = receipt["whole_model"]
            self.assertTrue(whole_model["gauge_defined"])
            self.assertTrue(math.isclose(row.r_grad, whole_model["R_grad"], abs_tol=1e-15))
            self.assertTrue(math.isclose(row.r_opt, whole_model["R_opt"], abs_tol=1e-15))
            self.assertTrue(
                math.isclose(row.c_k_star, whole_model["c_K_star"], abs_tol=1e-15)
            )
            self.assertEqual(row.n_k, receipt["stateful_radam"]["n_K"])
            self.assertTrue(receipt["stateful_radam"]["moments_nontrivial"])
            self.assertTrue(receipt["stateful_radam"]["gradscaler_restored"])
            self.assertTrue(
                all(
                    receipt["randomness_contract"][field]
                    for field in (
                        "same_minibatch",
                        "same_t",
                        "same_noise",
                        "same_dropout_rng_state",
                    )
                )
            )
            self.assertTrue(receipt["source_state_non_committing"]["preserved"])
            self.assertTrue(all(not branch["step_skipped"] for branch in receipt["branches"]))
            layerwise = BUNDLE / f"radam_update_stateful_layerwise_{state_id}.csv"
            with layerwise.open(encoding="utf-8") as handle:
                self.assertEqual(sum(1 for _ in handle) - 1, 208)

    def test_bundle_manifest_matches_checked_in_artifacts(self):
        manifest = json.loads((BUNDLE / "artifact_sha256.json").read_text(encoding="utf-8"))
        self.assertIn("same_trajectory_residuals.pdf", manifest)
        self.assertIn("same_trajectory_residuals.svg", manifest)
        for filename, expected_digest in manifest.items():
            payload = (BUNDLE / filename).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected_digest, filename)

    def test_publication_figure_renders_in_all_formats(self):
        rows = longitudinal_plot.read_summary(SUMMARY)
        with tempfile.TemporaryDirectory() as directory:
            outputs = longitudinal_plot.render_main_figure(
                rows, Path(directory), png_dpi=600
            )
            self.assertEqual(
                {path.suffix for path in outputs}, {".pdf", ".svg", ".png"}
            )
            for output in outputs:
                self.assertGreater(output.stat().st_size, 1000)
            svg = (Path(directory) / "same_trajectory_residuals.svg").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("<image", svg)
            with Image.open(Path(directory) / "same_trajectory_residuals.png") as image:
                self.assertEqual(image.size, (4200, 1680))


if __name__ == "__main__":
    unittest.main()
