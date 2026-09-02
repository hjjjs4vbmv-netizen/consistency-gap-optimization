import csv
import tempfile
import unittest
from pathlib import Path

from scripts import build_imagenet91_paper_figures as figures


class ImageNet91PaperFigureTests(unittest.TestCase):
    def test_uses_canonical_pr91_table(self):
        expected = (
            figures.ROOT
            / "results/imagenet64_gap_ab_full120_20260829/per_trajectory.csv"
        )
        self.assertEqual(figures.DEFAULT_IMAGENET, expected)
        self.assertTrue(expected.is_file())
        self.assertFalse(
            (
                figures.ROOT
                / "results/imagenet91_paper_summary/imagenet_per_trajectory_source.csv"
            ).exists()
        )
        self.assertEqual(len(figures.read_imagenet(expected)), 120)

    def test_contraction_table_keeps_all_three_seeds(self):
        rows = figures.read_imagenet(figures.DEFAULT_IMAGENET)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "contraction.csv"
            figures.write_contraction(rows, output)
            with output.open(newline="", encoding="utf-8") as handle:
                summaries = list(csv.DictReader(handle))
        self.assertEqual(len(summaries), 6)
        self.assertEqual({int(row["seed"]) for row in summaries}, {101, 102, 103})
        self.assertTrue(all(
            row["interpretation"] == "not interpretable after trajectory instability"
            for row in summaries
            if int(row["seed"]) == 103
        ))

    def test_readme_marks_post_hoc_and_replay_boundaries(self):
        readme = (
            figures.ROOT / "results/imagenet91_paper_summary/README.md"
        ).read_text(encoding="utf-8")
        self.assertIn("结果已经存在后选取，未预注册", readme)
        self.assertIn("seed3–5 replay illustration", readme)
        self.assertIn("not a substitute for the planned balanced q256 cohort", readme)
        self.assertNotIn("effects converge, reverse, or destabilize", readme)


if __name__ == "__main__":
    unittest.main()
