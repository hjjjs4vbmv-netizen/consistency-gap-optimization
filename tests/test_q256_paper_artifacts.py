import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import plot_q256_main_results as q256_plots


REPO_ROOT = Path(__file__).resolve().parents[1]
PAIR_CSV = REPO_ROOT / "results/q256_256k_formal/paired_differences.csv"
MAIN_TABLE = REPO_ROOT / "tables/q256_main_table.tex"
APPENDIX_TABLE = REPO_ROOT / "tables/q256_per_seed_appendix.tex"


class Q256PaperArtifactsTest(unittest.TestCase):
    def test_q256_paper_pdfs_are_reproducible_and_nonempty(self):
        indexed = q256_plots.data_index(q256_plots.read_pairs(PAIR_CSV))
        with tempfile.TemporaryDirectory() as directory:
            outdir = Path(directory)
            q256_plots.render_figure1(indexed, outdir, "q256_paired_seed_plot", ("pdf",))
            q256_plots.render_figure2(indexed, outdir, "q256_effect_heterogeneity", ("pdf",))
            for name in ("q256_paired_seed_plot.pdf", "q256_effect_heterogeneity.pdf"):
                self.assertGreater((outdir / name).stat().st_size, 0)

    @unittest.skipUnless(shutil.which("pdflatex"), "pdflatex is not available")
    def test_q256_latex_tables_compile(self):
        self.assertTrue(MAIN_TABLE.is_file())
        self.assertTrue(APPENDIX_TABLE.is_file())
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            source = build / "paper_artifacts_smoke_test.tex"
            source.write_text(
                "\\documentclass{article}\n"
                "\\usepackage[margin=0.7in]{geometry}\n"
                "\\usepackage{booktabs}\n"
                "\\begin{document}\n"
                "\\input{tables/q256_main_table}\n"
                "\\input{tables/q256_per_seed_appendix}\n"
                "\\end{document}\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(build), str(source)],
                cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertGreater((build / "paper_artifacts_smoke_test.pdf").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
