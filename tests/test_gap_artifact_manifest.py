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

    def test_full_structural_audit_passes_publication_gate(self):
        self.assertEqual(self.report["structural_checks"], "PASS")
        self.assertEqual(self.report["pull_requests_checked"], 7)
        self.assertEqual(self.report["checkpoint_records_checked"], 12)
        self.assertEqual(self.report["crossk_h20_raw_arrays_checked"], 16)
        self.assertEqual(self.report["crossk_external_records_checked"], 30)
        self.assertEqual(self.report["disjoint_evaluation_cells_checked"], 27)
        self.assertEqual(self.report["disjoint_cell_bindings_checked"], 27)
        self.assertEqual(self.report["sample_range_bound_receipts_checked"], 54)
        self.assertEqual(self.report["checkpoint_hash_bound_receipts_checked"], 54)
        self.assertEqual(self.report["checkpoint_hash_unbound_receipts"], 0)
        self.assertEqual(self.report["publication_v2_cells_checked"], 27)
        self.assertEqual(self.report["publication_v2_metric_receipts_checked"], 54)
        self.assertEqual(self.report["publication_v2_retained_arrays_checked"], 81)
        self.assertTrue(self.report["publication_ready"])
        self.assertEqual(self.report["blocking_findings"], [])

    def test_b002_is_downgraded_to_appendix_limitation(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        resolved = {item["id"]: item for item in manifest["resolved_findings"]}
        blockers = {item["id"]: item for item in manifest["blocking_findings"]}
        limitations = {
            item["id"]: item for item in manifest["publication_limitations"]
        }
        self.assertIn("B002-H20", resolved)
        self.assertNotIn("B002", blockers)
        self.assertIn("B002", limitations)
        self.assertIn("full R2(K,h)", limitations["B002"]["scope"])
        self.assertIn("must be disclosed", limitations["B002"]["detail"])
        bundle = manifest["evidence_bundles"]["crossk_h20_scalar_history"]
        self.assertEqual(bundle["status"], "canonical_git_self_contained_headline")
        self.assertIsNone(
            bundle["full_matrix_external_raw"]["durable_external_locator"]
        )

    def test_b003_b005_b006_are_resolved_and_hash_bound(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        resolved = {item["id"]: item for item in manifest["resolved_findings"]}
        blockers = {item["id"]: item for item in manifest["blocking_findings"]}
        self.assertIn("B006-SAMPLE-RANGE", resolved)
        self.assertIn("B003", resolved)
        self.assertIn("B005", resolved)
        self.assertIn("B006", resolved)
        self.assertNotIn("B003", blockers)
        self.assertNotIn("B005", blockers)
        self.assertNotIn("B006", blockers)
        bundle = manifest["evidence_bundles"]["disjoint_fid_kid_5k"]
        self.assertEqual(
            bundle["cell_binding_manifest"]["checkpoint_hash_bound_receipts"], 54
        )
        self.assertEqual(
            bundle["regenerated_publication_v2"]["retained_feature_arrays"], 54
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

    def test_mutable_latest_path_is_a_fail_closed_tombstone(self):
        tombstone = json.loads(
            (ROOT / "analysis" / "real_history" / "scalar_prediction.json").read_text()
        )
        self.assertEqual(tombstone["status"], "NON_CANONICAL_TOMBSTONE")
        self.assertEqual(tombstone["claim_use"], "FORBIDDEN")
        self.assertEqual(
            tombstone["canonical_result"],
            "analysis/real_history/k256/scalar_prediction.json",
        )


if __name__ == "__main__":
    unittest.main()
