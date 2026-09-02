import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/q128_matched_spacing_20260824"
PROVENANCE = RESULTS / "provenance"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Q128MatchedSpacingEvidenceTest(unittest.TestCase):
    def test_result_package_sha256_manifest(self):
        for line in (RESULTS / "SHA256SUMS.txt").read_text().splitlines():
            expected, relative = line.split("  ", 1)
            self.assertEqual(sha256(RESULTS / relative), expected, relative)

    def test_calibration_is_content_bound_and_analytic_exact(self):
        manifest_path = PROVENANCE / "calibration_manifest.json"
        exact = json.loads((PROVENANCE / "calibration_exact_match.json").read_text())
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(
            sha256(manifest_path),
            "08b84aa61c3cf9949944f038f40fda66ccc34f5dce182b5493b49ae653908825",
        )
        self.assertEqual(
            manifest["calibration_source_sha256"],
            "edea5685ba58f4c8d8f69d9c464e372bd047d8c920f103397827d48fb3bc34fc",
        )
        self.assertEqual(manifest["selected_g128_star"], 0.55)
        self.assertEqual(manifest["selected_objective"], 0.0)
        self.assertEqual(manifest["clipping_statistics"]["reference_fraction"], 0.0)
        self.assertEqual(manifest["clipping_statistics"]["candidate_fraction"], 0.0)
        self.assertAlmostEqual(0.55 / 128, 1.10 / 256, places=18)
        self.assertEqual(exact["status"], "PASS")
        self.assertEqual(exact["evidence_status"], "analytic_exact_match_confirmed_by_quality_blind_numerical_diagnostics")

    def test_blindness_chronology_distinguishes_historical_and_fresh_results(self):
        chronology = json.loads((PROVENANCE / "blindness_chronology.json").read_text())
        scope = chronology["blindness_scope"]
        self.assertTrue(scope["historical_a_bsame_quality_known_before_calibration_and_protocol_freeze"])
        self.assertFalse(scope["matched_arm_quality_known_before_freeze"])
        self.assertTrue(scope["fresh_210_job_matrix_unblinded_after_completion"])
        unblind = next(event for event in chronology["events"] if event["event"] == "first_metric_unblind")
        self.assertEqual(unblind["timestamp_status"], "bounded_not_exact")

    def test_training_integrity_and_pairing_gate(self):
        root = PROVENANCE / "training"
        summary = json.loads((root / "training_integrity_summary.json").read_text())
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["trajectories_completed"], 15)
        self.assertEqual(summary["checkpoint_receipts"], 105)
        self.assertTrue(summary["artifact_file_hashes_recomputed"])
        self.assertTrue(summary["stage0_only"])
        self.assertEqual(summary["stages_observed"], [0])
        self.assertEqual(summary["fresh_a_cells"], 3)
        self.assertEqual(summary["fresh_bsame_cells"], 3)
        self.assertEqual(summary["hardware_models"], ["NVIDIA A100-PCIE-40GB, 40960 MiB"])
        self.assertEqual(summary["gpu_uuid_status"], "not_recorded_in_preflight_v1")
        self.assertTrue(all(seed["status"] == "PASS" for seed in summary["pairing"].values()))
        with (root / "training_artifact_hashes.csv").open(newline="") as handle:
            artifacts = list(csv.DictReader(handle))
        self.assertEqual(len(artifacts), 105)
        self.assertEqual({row["state_hash_verified"] for row in artifacts}, {"True"})
        self.assertEqual({row["snapshot_hash_verified"] for row in artifacts}, {"True"})

    def test_report_and_audit_do_not_overstate_aulc_or_patch_identity(self):
        report = (RESULTS / "REPORT.md").read_text()
        self.assertNotIn("Frozen AULC", report)
        self.assertNotIn("primary `Bmatch-Bsame` AULC", report)
        self.assertIn("post-unblind", report.lower())
        audit = json.loads((RESULTS / "audit.json").read_text())
        protocol = audit["protocol"]
        self.assertNotIn("shared_feature_reuse_patch", protocol)
        self.assertEqual(
            set(protocol["deployed_evaluator_source_sha256"]),
            {
                "ct_eval.py",
                "metrics/frechet_inception_distance.py",
                "metrics/kernel_inception_distance.py",
                "metrics/metric_main.py",
                "metrics/metric_utils.py",
                "scripts/run_q128_stream_eval_worker.sh",
            },
        )


if __name__ == "__main__":
    unittest.main()
