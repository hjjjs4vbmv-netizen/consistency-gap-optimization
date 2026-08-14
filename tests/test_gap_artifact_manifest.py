import json
import tempfile
import unittest
from pathlib import Path

from scripts import rebuild_disjoint_5k_summary
from scripts import verify_gap_artifact_manifest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evidence" / "gap_artifact_manifest_v1.json"


class GapArtifactManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = verify_gap_artifact_manifest.run_audit(ROOT, MANIFEST)

    def test_full_structural_audit_passes_with_explicit_blockers(self):
        self.assertEqual(self.report["structural_checks"], "PASS")
        self.assertEqual(self.report["pull_requests_checked"], 6)
        self.assertEqual(self.report["checkpoint_records_checked"], 12)
        self.assertEqual(self.report["disjoint_evaluation_cells_checked"], 27)
        self.assertFalse(self.report["publication_ready"])
        self.assertEqual(
            {item["id"] for item in self.report["blocking_findings"]},
            {"B001", "B002", "B003", "B005", "B006"},
        )

    def test_pr53_csvs_rebuild_exactly(self):
        rebuild_disjoint_5k_summary.verify_committed_tables(
            ROOT, rebuild_disjoint_5k_summary.DEFAULT_SOURCE_REF
        )

    def test_state_hash_tampering_fails_closed(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["checkpoint_records"][0]["training_state"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                verify_gap_artifact_manifest.AuditFailure,
                "handoff state hash mismatch",
            ):
                verify_gap_artifact_manifest.run_audit(ROOT, path)

    def test_mutable_latest_duplicate_is_not_a_canonical_plot_input(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        bundle = manifest["evidence_bundles"]["prospective_scalar_history"]
        forbidden = manifest["noncanonical_artifacts"][0]["path"]
        self.assertNotIn(forbidden, bundle["canonical_result_paths"])
        self.assertNotIn(forbidden, bundle["plotting_script"]["reads_only"])


if __name__ == "__main__":
    unittest.main()
