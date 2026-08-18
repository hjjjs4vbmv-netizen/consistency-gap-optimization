#!/usr/bin/env python3
"""Build an allowlisted, history-free anonymous code artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


FILES = (
    "ct_eval.py",
    "dataset_tool.py",
    "environment-publication.yml",
    "analysis/crossk_horizon_sweep.py",
    "analysis/numpy_radam.py",
    "analysis/plot_crossk.py",
    "analysis/balanced_beta.py",
    "analysis/plot_balanced_beta.py",
    "scripts/build_robustness_table.py",
    "scripts/plot_same_trajectory_longitudinal.py",
    "scripts/reconstruct_headline_tables.py",
    "scripts/verify_submission_export.py",
    "tests/test_crossk_h20_recompute.py",
    "tests/test_same_trajectory_figure.py",
    "tests/test_metric_artifact_retention.py",
    "tests/test_balanced_beta_h20.py",
    "tests/test_build_robustness_table.py",
    "evidence/disjoint_5k_cell_manifest_v1.json",
    "evidence/disjoint_5k_cell_manifest_v1.sha256",
)

TREES = (
    "dnnlib",
    "metrics",
    "training",
    "torch_utils",
    "analysis/same_trajectory_longitudinal",
    "analysis/balanced_beta",
    "figures/cross_k_scalar_history",
    "figures/balanced_beta",
    "results/robustness",
    "results/publication_v2_regenerated",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source_root: Path, output: Path, relative: str) -> None:
    source = source_root / relative
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"missing, non-file, or symlinked allowlist entry: {relative}")
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(source_root: Path, output: Path, relative: str) -> None:
    source = source_root / relative
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlink forbidden in release: {path}")
        if path.is_file() and "__pycache__" not in path.parts:
            copy_file(source_root, output, path.relative_to(source_root).as_posix())


def write_release_docs(output: Path) -> None:
    (output / "README.md").write_text(
        "# Anonymous publication artifact\n\n"
        "This history-free export contains the self-contained scientific evidence "
        "and reconstruction code used for the submission gate. It contains no "
        "development Git history or collaboration metadata.\n\n"
        "## Bootstrap\n\n"
        "```bash\n"
        "conda env create -f environment-publication.yml\n"
        "conda run -n anonymous-publication-audit python scripts/verify_submission_export.py --root .\n"
        "conda run -n anonymous-publication-audit pytest -q -p no:cacheprovider "
        "tests/test_crossk_h20_recompute.py tests/test_same_trajectory_figure.py "
        "tests/test_balanced_beta_h20.py tests/test_build_robustness_table.py "
        "tests/test_metric_artifact_retention.py tests/test_submission_export.py\n"
        "```\n\n"
        "To verify the separately distributed data payload, add "
        "`--data-root PATH --require-data` to the verifier command.\n\n"
        "## Reconstruction\n\n"
        "```bash\n"
        "python scripts/reconstruct_headline_tables.py "
        "--manifest results/publication_v2_regenerated/publication_v2_cell_manifest.json "
        "--outdir rebuilt/tables --verify-against results/publication_v2_regenerated\n"
        "python scripts/plot_same_trajectory_longitudinal.py --outdir rebuilt/same_trajectory\n"
        "python analysis/plot_crossk.py --out rebuilt/cross_k\n"
        "python analysis/plot_balanced_beta.py --out rebuilt/balanced_beta\n"
        "python scripts/build_robustness_table.py\n"
        "```\n\n"
        "The same-trajectory vector and raster artifacts are byte-locked. Cross-K "
        "acceptance is numerical (the committed raw h=20 arrays recompute all four "
        "headline values) plus successful rendering, because PDF metadata and a "
        "one-pixel canvas variation differ across Matplotlib backends.\n",
        encoding="utf-8",
    )
    (output / "LIMITATIONS.md").write_text(
        "# Publication limitations\n\n"
        "B002 is an appendix reproducibility limitation. Only the self-contained "
        "h=20 Cross-K finding is eligible for headline use. The exploratory full "
        "R2(K,h) matrix outside h=20 is not included and must not support a "
        "headline claim. Generic Adam scale sensitivity and the beta1=beta2 "
        "first-order invariance are outside the novelty claim. The balanced-beta "
        "result is a controlled replay, not a training intervention: it does not "
        "establish FID/KID causality, scalar-history-only causality, or a uniformly "
        "better balanced-beta configuration. Full-vector R_opt retains its declared "
        "external-input reproducibility boundary.\n",
        encoding="utf-8",
    )
    (output / "PROVENANCE_DECISIONS.md").write_text(
        "# Provenance decisions\n\n"
        "- B003: resolved by regenerated evaluation provenance. The data payload "
        "retains 27 exact sample arrays and 54 exact feature arrays.\n"
        "- B005: resolved by recovery and independent hashing of the exact original "
        "seed-3 Arm B/C EMA bytes; no checkpoint was regenerated.\n"
        "- B006: resolved at 54/54 checkpoint-hash-bound metric receipts.\n"
        "- The older evaluation table is archival only. Publication claims use the "
        "regenerated-v2 table without mixing lineages.\n",
        encoding="utf-8",
    )


TEST_SOURCE = '''import tempfile
import unittest
from pathlib import Path

from scripts.reconstruct_headline_tables import reconstruct
from scripts.verify_submission_export import verify_lightweight


ROOT = Path(__file__).resolve().parents[1]


class SubmissionExportTests(unittest.TestCase):
    def test_self_contained_manifest_gate(self):
        result = verify_lightweight(ROOT)
        self.assertGreater(result["release_files"], 0)

    def test_headline_tables_reconstruct_byte_exactly(self):
        bundle = ROOT / "results" / "publication_v2_regenerated"
        rendered = reconstruct(bundle / "publication_v2_cell_manifest.json")
        for name, content in rendered.items():
            self.assertEqual(content.encode(), (bundle / name).read_bytes(), name)


if __name__ == "__main__":
    unittest.main()
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    for relative in FILES:
        copy_file(source, output, relative)
    for relative in TREES:
        copy_tree(source, output, relative)
    # Preserve the scientific protocol while removing the private repository mapping
    # from the one-way anonymous export. The private source receipt is not rewritten.
    balanced_provenance = output / "analysis" / "balanced_beta" / "provenance.json"
    provenance = json.loads(balanced_provenance.read_text(encoding="utf-8"))
    provenance["git"]["repository"] = "anonymous-submission-repository"
    provenance["git"]["branch"] = "anonymous-release"
    balanced_provenance.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # Copy only the h=20 self-contained Cross-K evidence, not private raw locators.
    copy_file(source, output, "analysis/crossk_scalar_history/summary.json")
    for label in ("k32", "k64", "k128", "k256"):
        for name in ("a_star_series.npy", "h_actual_h20.npy", "h_pred_scalar_h20.npy", "weights_h20.npy"):
            copy_file(source, output, f"analysis/crossk_scalar_history/{label}/raw_predictions/{name}")
    copy_file(source, output, "evidence/b005_recovery_receipt_2026_08_18.json")
    copied_receipt = output / "evidence" / "b005_recovery_receipt_2026_08_18.json"
    copied_receipt.replace(output / "evidence" / "b005_recovery_receipt.json")
    write_release_docs(output)
    (output / "tests" / "test_submission_export.py").write_text(TEST_SOURCE, encoding="utf-8")
    files = sorted(path for path in output.rglob("*") if path.is_file())
    (output / "RELEASE_SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "files": len(files)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
