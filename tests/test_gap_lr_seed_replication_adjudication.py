"""CPU-only tests for the post-run blind adjudication layer."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts import adjudicate_gap_lr_seed_replication as adjudicator


try:
    import torch

    from scripts import reconstruct_gap_lr_seed_initialization as reconstruction
    from scripts import verify_gap_lr_seed_replication_run as run_verifier
except ModuleNotFoundError:
    torch = None
    reconstruction = None
    run_verifier = None


class BlindAdjudicationPolicyTests(unittest.TestCase):
    @staticmethod
    def fixture():
        run_keys = sorted(adjudicator.EXPECTED_RUN_KEYS)
        public = {
            key: {
                "receipt_type": "gap_lr_seed_replication_run_integrity_public",
                "status": "passed",
                "experiment_id": adjudicator.EXPERIMENT_ID,
                "run_id": adjudicator.EXPECTED_RUNS[key][0],
                "seed": adjudicator.EXPECTED_RUNS[key][1],
                "arm": adjudicator.EXPECTED_RUNS[key][2],
                "gap_scale": adjudicator.EXPECTED_RUNS[key][3],
                "learning_rate": adjudicator.EXPECTED_RUNS[key][4],
                "bindings": {
                    "execution_protocol_commit": adjudicator.EXECUTION_PROTOCOL_COMMIT,
                    "internal_receipt_sha256": key * 8,
                },
                "publication": {"sanitized_for_github": True},
                "completion": {"summary": {"amp_contract_passed": True}},
            }
            for key in run_keys
        }
        manifest = {
            key: {
                "file": key + ".json",
                "run_id": adjudicator.EXPECTED_RUNS[key][0],
                "internal_receipt_sha256": key * 8,
            }
            for key in run_keys
        }
        evidence = {
            "receipt_type": "gap_lr_seed_replication_quality_blind_evidence",
            "status": "adjudication_ready",
            "experiment_id": adjudicator.EXPERIMENT_ID,
            "bindings": {
                "execution_protocol_commit": adjudicator.EXECUTION_PROTOCOL_COMMIT
            },
            "quality_blind": {"generation_quality_metrics_accessed": False},
            "per_run_integrity": {
                "passed_runs": 6,
                "required_runs": 6,
                "all_artifact_hashes_recomputed": True,
                "public_receipts": manifest,
            },
            "configuration_contract": {
                "within_seed_passed": {"4": True, "5": True},
                "between_seed_passed": {"A": True, "B": True, "C": True},
            },
            "initialization": {
                "historical_observed_preupdate_parameter_hash": "not_captured",
                "reconstructed_expected_initialization": {
                    "historical_process_attestation": False
                },
                "model_init_previews": {
                    "4": {"max_abs_channel_delta_lsb": 0},
                    "5": {"max_abs_channel_delta_lsb": 1},
                },
            },
            "runtime": {
                "planned": {"execution_mode": "fully_serial"},
                "hardware": {"same_model_driver_and_memory": True},
                "runs": {
                    adjudicator.EXPECTED_RUNS[key][0]: {} for key in run_keys
                },
                "directly_observed_overlaps": [
                    {
                        "runs": [
                            "arm_b_g1_3_lr_fixed_s4",
                            "arm_a_g1_0_lr_fixed_s5",
                        ],
                        "different_logged_gpu_indices": True,
                    },
                    {
                        "runs": [
                            "arm_c_g1_3_lr_matched_s4",
                            "arm_b_g1_3_lr_fixed_s5",
                        ],
                        "different_logged_gpu_indices": True,
                    },
                ],
            },
            "deviations": [
                {"id": item} for item in sorted(adjudicator.EXPECTED_DEVIATIONS)
            ],
            "claim_exclusions": sorted(adjudicator.REQUIRED_EXCLUSIONS),
        }
        init_runs = {
            run_id: {
                "copy_contract": {
                    "all_destination_tensors_covered": True,
                    "missing_destination_names": [],
                    "shape_dtype_mismatches": [],
                },
                "net": {"tensor_count": 424, "sha256": "a" * 64},
                "ema": {"sha256": "a" * 64},
            }
            for run_id in sorted(adjudicator.EXPECTED_RUN_IDS)
        }
        initialization = {
            "status": "passed",
            "quality_blind": {"generation_quality_metrics_accessed": False},
            "cross_run": {
                "all_six_reconstructed_net_hashes_equal": True,
                "all_six_initialization_contract_hashes_equal": True,
            },
            "runs": init_runs,
        }
        return evidence, initialization, public

    def test_documented_deviations_are_accepted_before_quality(self):
        evidence, initialization, public = self.fixture()
        verdict, failures, affected = adjudicator.evaluate(
            evidence, initialization, public
        )
        self.assertEqual(verdict, "machine_recommends_acceptance")
        self.assertEqual(failures, [])
        self.assertEqual(affected, [])

    def test_global_reconstruction_failure_requires_all_six_runs(self):
        evidence, initialization, public = self.fixture()
        initialization["cross_run"][
            "all_six_reconstructed_net_hashes_equal"
        ] = False
        verdict, failures, affected = adjudicator.evaluate(
            evidence, initialization, public
        )
        self.assertEqual(verdict, "rerun_required")
        self.assertTrue(any("initialization" in item for item in failures))
        self.assertEqual(set(affected), adjudicator.EXPECTED_RUN_KEYS)

    def test_quality_access_breaks_blind_gate(self):
        evidence, initialization, public = self.fixture()
        evidence["quality_blind"]["generation_quality_metrics_accessed"] = True
        verdict, failures, affected = adjudicator.evaluate(
            evidence, initialization, public
        )
        self.assertEqual(verdict, "rerun_required")
        self.assertTrue(any("quality-blind" in item for item in failures))
        self.assertEqual(set(affected), adjudicator.EXPECTED_RUN_KEYS)

    def test_public_path_scan_rejects_server_identifiers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text('{"run_dir":"/data/raw/private"}\n')
            self.assertFalse(adjudicator.public_text_is_sanitized(path))
            path.write_text('{"run_id":"arm_a_g1_0_lr_fixed_s4"}\n')
            self.assertTrue(adjudicator.public_text_is_sanitized(path))
            for leaked in (
                "/root/private",
                "user@internal-host",
                "192.168.1.9",
                "GPU-01234567-89ab-cdef-0123-456789abcdef",
                "file:///private/result",
            ):
                path.write_text('{"value":' + repr(leaked).replace("'", '"') + '}\n')
                self.assertFalse(adjudicator.public_text_is_sanitized(path))


@unittest.skipIf(torch is None, "PyTorch runtime is unavailable; run formal suite remotely")
class CanonicalInitializationTests(unittest.TestCase):
    def test_canonical_hash_is_stable_and_tensor_sensitive(self):
        module = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=True))
        first = reconstruction.canonical_module(module)
        second = reconstruction.canonical_module(module)
        self.assertEqual(first, second)
        with torch.no_grad():
            module[0].weight[0, 0] += 1
        changed = reconstruction.canonical_module(module)
        self.assertNotEqual(first["sha256"], changed["sha256"])

    def test_canonical_v1_golden_vector(self):
        class Golden(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "a_buffer", torch.tensor([1, 2, 3], dtype=torch.int16)
                )
                self.z_param = torch.nn.Parameter(
                    torch.tensor([[1.0, -0.0], [3.0, -2.5]], dtype=torch.float32)
                )

        result = reconstruction.canonical_module(Golden())
        self.assertEqual(
            result["sha256"],
            "4de4d769cdd5fcaf3370f9a2f9307394e7f02099c41d41c41898f7b68323e3e9",
        )

    def test_summary_binds_amp_skips_and_successful_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_summary.csv"
            fieldnames = [
                "attempted_iteration",
                "successful_optimizer_steps",
                "processed_kimg",
                "loss",
                "grad_scale",
                "step_skipped",
                "schedule",
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                successful = 0
                for attempted in range(1, 2001):
                    skipped = int(attempted <= 8)
                    successful += 1 - skipped
                    writer.writerow(
                        {
                            "attempted_iteration": attempted,
                            "successful_optimizer_steps": successful,
                            "processed_kimg": attempted * 0.128,
                            "loss": 15.0,
                            "grad_scale": 256,
                            "step_skipped": skipped,
                            "schedule": "global_sigmoid",
                        }
                    )
            result = run_verifier.validate_summary(path)
            self.assertEqual(result["attempted_iterations"], 2000)
            self.assertEqual(result["successful_optimizer_steps"], 1992)
            self.assertEqual(result["amp_skipped_steps"], 8)
            self.assertTrue(result["amp_contract_passed"])


if __name__ == "__main__":
    unittest.main()
