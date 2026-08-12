"""Fail-closed tests for the post-run quality-blind adjudication layer."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts import adjudicate_gap_lr_seed_replication as adjudicator
from scripts import gap_lr_seed_replication_contract as audit_contract


try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None

if torch is not None:
    from scripts import build_gap_lr_seed_replication_blind_evidence as builder
    from scripts import reconstruct_gap_lr_seed_initialization as reconstruction
    from scripts import verify_gap_lr_seed_replication_run as run_verifier
else:
    builder = None
    reconstruction = None
    run_verifier = None


TOOLING_COMMIT = "1" * 40
RECONSTRUCTION_SOURCE_SHA256 = "2" * 64
EVIDENCE_BUILDER_SOURCE_SHA256 = "3" * 64
RUN_VERIFIER_SOURCE_SHA256 = "4" * 64
INITIALIZATION_REPORT_SHA256 = "5" * 64


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class BlindAdjudicationPolicyTests(unittest.TestCase):
    @staticmethod
    def evaluate(evidence, initialization, public):
        return adjudicator.evaluate(
            evidence,
            initialization,
            public,
            tooling_commit=TOOLING_COMMIT,
            reconstruction_source_sha256=RECONSTRUCTION_SOURCE_SHA256,
            evidence_builder_source_sha256=EVIDENCE_BUILDER_SOURCE_SHA256,
            run_verifier_source_sha256=RUN_VERIFIER_SOURCE_SHA256,
        )

    @staticmethod
    def fixture():
        run_keys = sorted(adjudicator.EXPECTED_RUN_KEYS)
        timeline = {
            "arm_a_g1_0_lr_fixed_s4": (
                "2026-08-12T10:00:00+00:00",
                "2026-08-12T10:30:00+00:00",
                "2026-08-12T10:40:00+00:00",
                "2026-08-12T10:50:00+00:00",
            ),
            "arm_b_g1_3_lr_fixed_s4": (
                "2026-08-12T11:00:00+00:00",
                "2026-08-12T11:30:00+00:00",
                "2026-08-12T11:40:00+00:00",
                "2026-08-12T11:50:00+00:00",
            ),
            "arm_c_g1_3_lr_matched_s4": (
                "2026-08-12T12:00:00+00:00",
                "2026-08-12T12:30:00+00:00",
                "2026-08-12T12:40:00+00:00",
                "2026-08-12T12:50:00+00:00",
            ),
            "arm_a_g1_0_lr_fixed_s5": (
                "2026-08-12T11:05:00+00:00",
                "2026-08-12T11:35:00+00:00",
                "2026-08-12T11:45:00+00:00",
                "2026-08-12T11:55:00+00:00",
            ),
            "arm_b_g1_3_lr_fixed_s5": (
                "2026-08-12T12:05:00+00:00",
                "2026-08-12T12:35:00+00:00",
                "2026-08-12T12:45:00+00:00",
                "2026-08-12T12:55:00+00:00",
            ),
            "arm_c_g1_3_lr_matched_s5": (
                "2026-08-12T13:00:00+00:00",
                "2026-08-12T13:30:00+00:00",
                "2026-08-12T13:40:00+00:00",
                "2026-08-12T13:50:00+00:00",
            ),
        }
        strengthened_verified = {
            run_id: f"2026-08-13T00:{index:02d}:00+00:00"
            for index, run_id in enumerate(audit_contract.EXPECTED_RUNTIME)
        }

        public = {}
        manifest = {}
        for key in run_keys:
            run_id, seed, arm, gap, lr = adjudicator.EXPECTED_RUNS[key]
            skips = 9 if key == "seed5_C" else 8
            internal_sha = digest(key + ":internal")
            artifacts = {
                name: {"sha256": digest(key + ":" + name), "size_bytes": 1}
                for name in adjudicator.EXPECTED_ARTIFACT_KEYS
            }
            filename = f"{key}.integrity.public.json"
            public_sha = digest(key + ":public")
            public[key] = {
                "schema_version": 1,
                "receipt_type": "gap_lr_seed_replication_run_integrity_public",
                "status": "passed",
                "experiment_id": adjudicator.EXPERIMENT_ID,
                "run_id": run_id,
                "seed": seed,
                "arm": arm,
                "gap_scale": gap,
                "learning_rate": lr,
                "bindings": {
                    "execution_protocol_commit": adjudicator.EXECUTION_PROTOCOL_COMMIT,
                    "training_code_commit": audit_contract.TRAINING_CODE_COMMIT,
                    "source_audit_receipt_sha256": audit_contract.SOURCE_AUDIT_RECEIPT_SHA256,
                    "matrix_sha256": audit_contract.MATRIX_SHA256,
                    "dataset_sha256": audit_contract.DATA_SHA256,
                    "transfer_checkpoint_sha256": audit_contract.TRANSFER_SHA256,
                    "internal_receipt_sha256": internal_sha,
                    "verifier_source_sha256": RUN_VERIFIER_SOURCE_SHA256,
                },
                "completion": {
                    "budget_kimg": 256,
                    "summary": {
                        "rows": 2000,
                        "final_processed_kimg": 256.0,
                        "attempted_iterations": 2000,
                        "successful_optimizer_steps": 2000 - skips,
                        "amp_skipped_steps": skips,
                        "max_allowed_amp_skips": 16,
                        "final_gradscaler_scale": 128.0 if key == "seed5_C" else 256.0,
                        "amp_contract_passed": True,
                    },
                    "stats": {"records": 9, "final_kimg": 256.0},
                },
                "final_training_state": {
                    "cur_nimg": 256000,
                    "attempted_iteration": 2000,
                    "successful_optimizer_steps": 2000 - skips,
                    "gradscaler_scale": 128.0 if key == "seed5_C" else 256.0,
                    "optimizer_parameter_states": 416,
                    "tensors_checked": 1248,
                },
                "final_ema_snapshot": {
                    "ema_present": True,
                    "ema_finite": True,
                    "ema_tensors_checked": 424,
                },
                "artifact_manifest": artifacts,
                "verified_at_utc": strengthened_verified[run_id],
                "publication": {
                    "sanitized_for_github": True,
                    "absolute_paths_removed": True,
                    "raw_artifacts_retained_external_to_git": True,
                },
            }
            manifest[key] = {
                "run_id": run_id,
                "file": filename,
                "sha256": public_sha,
                "internal_receipt_sha256": internal_sha,
            }

        module_summary = {
            **audit_contract.EXPECTED_MODULE_SUMMARY,
            "sha256": "a" * 64,
        }
        init_runs = {}
        run_hashes = {}
        for key in run_keys:
            run_id, seed, arm, _gap, _lr = adjudicator.EXPECTED_RUNS[key]
            init_runs[run_id] = {
                "seed": seed,
                "arm": arm,
                "training_options_sha256": public[key]["artifact_manifest"][
                    "training_options"
                ]["sha256"],
                "internal_integrity_receipt_sha256": public[key]["bindings"][
                    "internal_receipt_sha256"
                ],
                "interface_kwargs": copy.deepcopy(audit_contract.EXPECTED_INTERFACE),
                "initialization_contract_sha256": "b" * 64,
                "copy_contract": {
                    "source_tensor_count": 425,
                    "destination_tensor_count": 424,
                    "missing_destination_names": [],
                    "source_only_ignored_by_destination_iterating_copy": copy.deepcopy(
                        audit_contract.EXPECTED_SOURCE_ONLY
                    ),
                    "shape_dtype_mismatches": [],
                    "all_destination_tensors_covered": True,
                },
                "ema_copy_contract_equal": True,
                "net": copy.deepcopy(module_summary),
                "ema": copy.deepcopy(module_summary),
            }
            run_hashes[run_id] = "a" * 64

        initialization = {
            "schema_version": 1,
            "receipt_type": "gap_lr_seed_initialization_reconstruction",
            "status": "passed",
            "experiment_id": adjudicator.EXPERIMENT_ID,
            "quality_blind": {
                "generation_quality_metrics_accessed": False,
                "inputs_read": [
                    "receipt-bound training_options.json",
                    "frozen transfer checkpoint",
                    "frozen implementation modules",
                    "frozen dataset metadata",
                ],
                "inputs_explicitly_not_read": [
                    "FID",
                    "KID",
                    "quality-evaluation outputs",
                    "trained network snapshots",
                    "training states",
                ],
            },
            "bindings": {
                "execution_protocol_commit": adjudicator.EXECUTION_PROTOCOL_COMMIT,
                "adjudication_tooling_commit": TOOLING_COMMIT,
                "training_code_commit": audit_contract.TRAINING_CODE_COMMIT,
                "source_audit_receipt_sha256": audit_contract.SOURCE_AUDIT_RECEIPT_SHA256,
                "matrix_sha256": audit_contract.MATRIX_SHA256,
                "dataset_sha256": audit_contract.DATA_SHA256,
                "transfer_checkpoint_sha256": audit_contract.TRANSFER_SHA256,
                "tool_source_sha256": RECONSTRUCTION_SOURCE_SHA256,
            },
            "interpretation": {
                "hash_kind": "reconstructed_expected_initialization_hash",
                "historical_observed_preupdate_hash_captured": False,
                "does_not_attest_historical_process_memory": True,
                "rng_state_reconstructed": False,
                "scope": "transferred tensor state only",
            },
            "canonicalization": {
                "schema": "ECT_CANONICAL_TORCH_MODULE_V1",
                "ordering": "UTF-8 fully-qualified tensor name, then kind",
                "fields": [
                    "kind",
                    "name",
                    "dtype",
                    "rank",
                    "shape",
                    "nbytes",
                    "raw_bytes",
                ],
                "raw_bytes": "detach, CPU, contiguous, row-major, little-endian",
                "metadata_integer_encoding": "unsigned 64-bit big-endian",
                "excluded": [
                    "module mode",
                    "requires_grad",
                    "non-tensor attributes",
                ],
            },
            "runs": init_runs,
            "cross_run": {
                "all_six_reconstructed_net_hashes_equal": True,
                "all_six_initialization_contract_hashes_equal": True,
                "all_six_dataset_interfaces_equal": True,
                "distinct_reconstructed_net_hashes": ["a" * 64],
                "distinct_initialization_contract_hashes": ["b" * 64],
                "run_hashes": run_hashes,
            },
        }

        def preview_pair(seed: int, arm: str, changed: bool):
            ids = {
                4: {
                    "A": "arm_a_g1_0_lr_fixed_s4",
                    "B": "arm_b_g1_3_lr_fixed_s4",
                    "C": "arm_c_g1_3_lr_matched_s4",
                },
                5: {
                    "A": "arm_a_g1_0_lr_fixed_s5",
                    "B": "arm_b_g1_3_lr_fixed_s5",
                    "C": "arm_c_g1_3_lr_matched_s5",
                },
            }
            return {
                "first": ids[seed]["A"],
                "second": ids[seed][arm],
                "shape_hwc": [64, 64, 3],
                "exact_pixel_values_equal": not changed,
                "max_abs_channel_delta_lsb": int(changed),
                "differing_channel_values": 10 if changed else 0,
                "differing_pixels": 8 if changed else 0,
                "positive_one_lsb": 5 if changed else 0,
                "negative_one_lsb": 5 if changed else 0,
                "greater_than_one_lsb": 0,
            }

        runtime_runs = {}
        for run_id, expected in audit_contract.EXPECTED_RUNTIME.items():
            first, last, exit_at, verified = timeline[run_id]
            key = f"seed{expected['seed']}_{expected['arm']}"
            runtime_runs[run_id] = {
                "seed": expected["seed"],
                "arm": expected["arm"],
                "segment_id": expected["segment_id"],
                "port": expected["port"],
                "logged_gpu_index": expected["logged_gpu_index"],
                "device_alias": f"device_{expected['logged_gpu_index']}",
                "gpu_evidence": {
                    "logged_index": "direct launcher assertion",
                    "uuid_mapping": (
                        "inference from one pre-launch hardware sidecar; index stability "
                        "and per-run UUID were not attested"
                    ),
                },
                "first_progress_at_utc": first,
                "last_progress_at_utc": last,
                "exit_marker_at_utc": exit_at,
                "interval_definition": "first stats record through timestamped exit marker",
                "interval_scope": (
                    "observed training-phase lower bound, not exact process lifetime"
                ),
                "exit_timestamp_source": (
                    "destroy_process_group warning immediately after clean Exiting marker"
                ),
                "exit_timestamp_timezone_assumption": "Asia/Shanghai (UTC+08:00)",
                "strengthened_integrity_verified_at_utc": strengthened_verified[
                    run_id
                ],
                "historical_integrity_receipt": {
                    "role": (
                        "posthoc_reverification_after_original_launcher_exit"
                        if run_id == "arm_a_g1_0_lr_fixed_s4"
                        else "inline_before_launcher_done"
                    ),
                    "schema_version": 1,
                    "receipt_sha256": digest(run_id + ":historical-receipt"),
                    "receipt_size_bytes": 1,
                    "verified_at_utc": verified,
                    "artifact_manifest_equal_to_strengthened_receipt": True,
                    "retained_external_to_git": True,
                },
                "public_receipt": manifest[key]["file"],
                "public_receipt_sha256": manifest[key]["sha256"],
                "external_training_log_sha256": digest(run_id + ":log"),
            }

        evidence_manifest = {
            "launch_provenance": {
                "file": "launch_provenance.txt",
                "sha256": digest("launch_provenance"),
                "size_bytes": 1,
                "retained_external_to_git": True,
            },
            "hardware": {
                "file": "hardware.txt",
                "sha256": digest("hardware"),
                "size_bytes": 1,
                "retained_external_to_git": True,
            },
            "software": {
                "file": "software.txt",
                "sha256": digest("software"),
                "size_bytes": 1,
                "retained_external_to_git": True,
            },
            "original_launcher_log": {
                "file": "gap_lr_matched_q128_s45_replication_v1.launcher.log",
                "sha256": digest("original_launcher"),
                "size_bytes": 1,
                "retained_external_to_git": True,
            },
            "seed4_recovery_launcher_log": {
                "file": "seed4_resume.launcher.log",
                "sha256": digest("seed4_launcher"),
                "size_bytes": 1,
                "retained_external_to_git": True,
            },
            "seed5_recovery_launcher_log": {
                "file": "seed5_resume.launcher.log",
                "sha256": digest("seed5_launcher"),
                "size_bytes": 1,
                "retained_external_to_git": True,
            },
            "initialization_reconstruction": {
                "file": "initialization_reconstruction.json",
                "sha256": INITIALIZATION_REPORT_SHA256,
                "size_bytes": 1,
                "retained_external_to_git": False,
            },
        }
        deviations = audit_contract.expected_deviations()

        evidence = {
            "schema_version": 1,
            "receipt_type": "gap_lr_seed_replication_quality_blind_evidence",
            "status": "adjudication_ready",
            "experiment_id": adjudicator.EXPERIMENT_ID,
            "quality_blind": {
                "generation_quality_metrics_accessed": False,
                "decision_frozen_before_quality_evaluation": True,
                "attestation_kind": (
                    "workflow-scope declaration; not cryptographic proof"
                ),
                "excluded_inputs": ["FID", "KID", "quality-evaluation outputs"],
            },
            "bindings": {
                "execution_protocol_commit": adjudicator.EXECUTION_PROTOCOL_COMMIT,
                "adjudication_tooling_commit": TOOLING_COMMIT,
                "training_code_commit": audit_contract.TRAINING_CODE_COMMIT,
                "source_audit_receipt_sha256": audit_contract.SOURCE_AUDIT_RECEIPT_SHA256,
                "matrix_sha256": audit_contract.MATRIX_SHA256,
                "dataset_sha256": audit_contract.DATA_SHA256,
                "transfer_checkpoint_sha256": audit_contract.TRANSFER_SHA256,
                "initialization_reconstruction_sha256": INITIALIZATION_REPORT_SHA256,
                "evidence_builder_source_sha256": EVIDENCE_BUILDER_SOURCE_SHA256,
            },
            "per_run_integrity": {
                "passed_runs": 6,
                "required_runs": 6,
                "all_artifact_hashes_recomputed": True,
                "public_receipts": manifest,
            },
            "configuration_contract": {
                "within_seed_allowed_differences": [
                    "loss_kwargs.global_gap_scale",
                    "optimizer_kwargs.lr",
                    "run_dir",
                ],
                "within_seed_passed": {"4": True, "5": True},
                "between_seed_allowed_differences": ["seed", "run_dir"],
                "between_seed_passed": {"A": True, "B": True, "C": True},
            },
            "initialization": {
                "historical_observed_preupdate_parameter_hash": "not_captured",
                "reconstructed_expected_initialization": {
                    "status": "passed",
                    "report_sha256": INITIALIZATION_REPORT_SHA256,
                    "hash_kind": "reconstructed_expected_initialization_hash",
                    "all_six_equal": True,
                    "distinct_hashes": ["a" * 64],
                    "historical_process_attestation": False,
                },
                "model_init_previews": {
                    "4": {
                        "sha256": {"A": "6" * 64, "B": "6" * 64, "C": "6" * 64},
                        "pairwise_against_A": [
                            preview_pair(4, "B", False),
                            preview_pair(4, "C", False),
                        ],
                        "exact_file_hashes_equal": True,
                        "max_abs_channel_delta_lsb": 0,
                        "role": "generated FP16 diagnostic preview; not a parameter hash",
                    },
                    "5": {
                        "sha256": {"A": "7" * 64, "B": "8" * 64, "C": "8" * 64},
                        "pairwise_against_A": [
                            preview_pair(5, "B", True),
                            preview_pair(5, "C", True),
                        ],
                        "exact_file_hashes_equal": False,
                        "max_abs_channel_delta_lsb": 1,
                        "role": "generated FP16 diagnostic preview; not a parameter hash",
                    },
                },
                "data_images": {
                    "all_six_sha256_equal": True,
                    "sha256": "9" * 64,
                },
            },
            "runtime": {
                "planned": {
                    "seed_order": [4, 5],
                    "arm_order": ["A", "B", "C"],
                    "gpu_index": 1,
                    "execution_mode": "fully_serial",
                    "automatic_retry": False,
                },
                "hardware": {
                    "devices": [
                        {
                            "device_alias": "device_0",
                            "logged_gpu_index": 0,
                            "name": "NVIDIA A100 80GB PCIe",
                            "driver_version": "535.54.03",
                            "memory_total": "81920 MiB",
                        },
                        {
                            "device_alias": "device_1",
                            "logged_gpu_index": 1,
                            "name": "NVIDIA A100 80GB PCIe",
                            "driver_version": "535.54.03",
                            "memory_total": "81920 MiB",
                        },
                    ],
                    "prelaunch_sidecar_entries_equivalent": True,
                    "snapshot_scope": (
                        "single pre-original-launch sidecar; not per-run attestation"
                    ),
                    "per_run_cuda_uuid_attested": False,
                    "full_gpu_uuids_retained_only_in_internal_sidecar": True,
                },
                "software": {
                    "python": "3.12.12",
                    "torch": "2.6.0+cu124",
                    "cuda": "12.4",
                    "cudnn": "90100",
                },
                "launcher_segments": [
                    {
                        "segment_id": "original",
                        "kind": "original",
                        "launcher_log_sha256": digest("original launcher"),
                        "launcher_log_size_bytes": 1,
                        "exact_command_preserved": False,
                        "committed_launcher_reconstructible": True,
                        "events": audit_contract.expected_launcher_events("original"),
                    },
                    {
                        "segment_id": "seed4_recovery",
                        "kind": "manual_recovery",
                        "launcher_log_sha256": digest("seed4 recovery"),
                        "launcher_log_size_bytes": 1,
                        "exact_command_preserved": False,
                        "committed_launcher_reconstructible": False,
                        "events": audit_contract.expected_launcher_events(
                            "seed4_recovery"
                        ),
                        "pid": 100,
                        "pid_file_sha256": digest("seed4 pid"),
                        "pid_file_mtime_filesystem_clock_utc": (
                            "2026-08-12T10:59:00+00:00"
                        ),
                    },
                    {
                        "segment_id": "seed5_recovery",
                        "kind": "manual_recovery",
                        "launcher_log_sha256": digest("seed5 recovery"),
                        "launcher_log_size_bytes": 1,
                        "exact_command_preserved": False,
                        "committed_launcher_reconstructible": False,
                        "events": audit_contract.expected_launcher_events(
                            "seed5_recovery"
                        ),
                        "pid": 101,
                        "pid_file_sha256": digest("seed5 pid"),
                        "pid_file_mtime_filesystem_clock_utc": (
                            "2026-08-12T10:59:01+00:00"
                        ),
                    },
                ],
                "runs": runtime_runs,
                "directly_observed_overlaps": [
                    {
                        "runs": [
                            "arm_b_g1_3_lr_fixed_s4",
                            "arm_a_g1_0_lr_fixed_s5",
                        ],
                        "directly_observed_overlap_start_utc": (
                            "2026-08-12T11:05:00+00:00"
                        ),
                        "directly_observed_overlap_end_utc": (
                            "2026-08-12T11:40:00+00:00"
                        ),
                        "duration_seconds": 2100.0,
                        "logged_gpu_indices": [0, 1],
                        "different_logged_gpu_indices": True,
                    },
                    {
                        "runs": [
                            "arm_c_g1_3_lr_matched_s4",
                            "arm_b_g1_3_lr_fixed_s5",
                        ],
                        "directly_observed_overlap_start_utc": (
                            "2026-08-12T12:05:00+00:00"
                        ),
                        "directly_observed_overlap_end_utc": (
                            "2026-08-12T12:40:00+00:00"
                        ),
                        "duration_seconds": 2100.0,
                        "logged_gpu_indices": [0, 1],
                        "different_logged_gpu_indices": True,
                    },
                ],
                "clock_model": {
                    "application_clock_sources": [
                        "stats.jsonl epoch timestamps",
                        "timestamped process-exit warning",
                        "historical and strengthened receipt verified_at_utc",
                    ],
                    "filesystem_clock_not_used_as_application_time": True,
                    "acceptance_gate": "diagnostic only; no exact process-start claim",
                    "observed_filesystem_minus_application_offset_seconds": {
                        "minimum": 65.6,
                        "maximum": 65.7,
                    },
                },
            },
            "deviations": deviations,
            "missing_evidence": [
                "historical post-transfer/pre-forward parameter hash",
                "per-run CUDA UUID attestation",
                "per-run recovery software snapshot",
                "complete recovery command lines and process environments",
                "original seed4-A verifier failure output and exit status",
            ],
            "claim_exclusions": sorted(adjudicator.REQUIRED_EXCLUSIONS),
            "evidence_manifest": evidence_manifest,
            "publication": {
                "sanitized_for_github": True,
                "absolute_paths_hostnames_accounts_ips_and_full_gpu_uuids_removed": True,
            },
        }
        return evidence, initialization, public

    def assert_rerun(self, evidence, initialization, public, phrase=None):
        verdict, failures, affected = self.evaluate(evidence, initialization, public)
        self.assertEqual(verdict, "rerun_required")
        self.assertTrue(failures)
        self.assertTrue(affected)
        if phrase is not None:
            self.assertTrue(any(phrase in item for item in failures), failures)

    def test_complete_documented_package_is_machine_recommendation_only(self):
        evidence, initialization, public = self.fixture()
        verdict, failures, affected = self.evaluate(evidence, initialization, public)
        self.assertEqual(verdict, "machine_recommends_acceptance")
        self.assertEqual(failures, [])
        self.assertEqual(affected, [])

    def test_init_cross_hashes_are_recomputed(self):
        evidence, initialization, public = self.fixture()
        run_id = "arm_a_g1_0_lr_fixed_s4"
        initialization["runs"][run_id]["net"]["sha256"] = "c" * 64
        initialization["runs"][run_id]["ema"]["sha256"] = "c" * 64
        self.assert_rerun(evidence, initialization, public, "initialization")

    def test_missing_init_hash_and_false_ema_copy_are_rejected(self):
        for mutation in ("missing_hash", "false_copy"):
            evidence, initialization, public = self.fixture()
            row = initialization["runs"]["arm_a_g1_0_lr_fixed_s4"]
            if mutation == "missing_hash":
                row["net"].pop("sha256")
            else:
                row["ema_copy_contract_equal"] = False
            with self.subTest(mutation=mutation):
                self.assert_rerun(evidence, initialization, public, "initialization")

    def test_init_receipt_and_options_cross_bindings_are_enforced(self):
        for field in ("internal", "options"):
            evidence, initialization, public = self.fixture()
            row = initialization["runs"]["arm_a_g1_0_lr_fixed_s4"]
            if field == "internal":
                row["internal_integrity_receipt_sha256"] = "c" * 64
            else:
                row["training_options_sha256"] = "c" * 64
            with self.subTest(field=field):
                self.assert_rerun(evidence, initialization, public, "cross-binding")

    def test_public_receipt_closed_schema_and_exact_types(self):
        mutations = ("missing_artifacts", "extra_key", "boolean_size", "string_lr")
        for mutation in mutations:
            evidence, initialization, public = self.fixture()
            receipt = public["seed4_A"]
            if mutation == "missing_artifacts":
                receipt.pop("artifact_manifest")
            elif mutation == "extra_key":
                receipt["unexpected"] = True
            elif mutation == "boolean_size":
                receipt["artifact_manifest"]["stats"]["size_bytes"] = True
            else:
                receipt["learning_rate"] = "0.0001"
            with self.subTest(mutation=mutation):
                self.assert_rerun(evidence, initialization, public, "public per-run")

    def test_all_fixed_json_types_are_exact(self):
        cases = (
            ("evidence_schema_float", lambda e, i, p: e.__setitem__("schema_version", 1.0)),
            ("evidence_schema_bool", lambda e, i, p: e.__setitem__("schema_version", True)),
            ("public_seed_float", lambda e, i, p: p["seed4_A"].__setitem__("seed", 4.0)),
            (
                "public_budget_float",
                lambda e, i, p: p["seed4_A"]["completion"].__setitem__(
                    "budget_kimg", 256.0
                ),
            ),
            (
                "public_rows_float",
                lambda e, i, p: p["seed4_A"]["completion"]["summary"].__setitem__(
                    "rows", 2000.0
                ),
            ),
            (
                "public_tensors_float",
                lambda e, i, p: p["seed4_A"]["final_training_state"].__setitem__(
                    "tensors_checked", 1248.0
                ),
            ),
            ("init_schema_float", lambda e, i, p: i.__setitem__("schema_version", 1.0)),
            (
                "init_seed_float",
                lambda e, i, p: i["runs"]["arm_a_g1_0_lr_fixed_s4"].__setitem__(
                    "seed", 4.0
                ),
            ),
            (
                "init_tensor_count_float",
                lambda e, i, p: i["runs"]["arm_a_g1_0_lr_fixed_s4"]["net"].__setitem__(
                    "tensor_count", 424.0
                ),
            ),
            (
                "runtime_gpu_bool",
                lambda e, i, p: e["runtime"]["planned"].__setitem__("gpu_index", True),
            ),
            (
                "preview_derived_max_bool",
                lambda e, i, p: e["initialization"]["model_init_previews"]["4"].__setitem__(
                    "max_abs_channel_delta_lsb", False
                ),
            ),
            (
                "preview_derived_max_float",
                lambda e, i, p: e["initialization"]["model_init_previews"]["5"].__setitem__(
                    "max_abs_channel_delta_lsb", 1.0
                ),
            ),
            (
                "init_evidence_all_equal_int",
                lambda e, i, p: e["initialization"][
                    "reconstructed_expected_initialization"
                ].__setitem__("all_six_equal", 1),
            ),
            (
                "init_evidence_attestation_int",
                lambda e, i, p: e["initialization"][
                    "reconstructed_expected_initialization"
                ].__setitem__("historical_process_attestation", 0),
            ),
        )
        for label, mutate in cases:
            evidence, initialization, public = self.fixture()
            mutate(evidence, initialization, public)
            with self.subTest(label=label):
                self.assert_rerun(evidence, initialization, public)

    def test_quality_access_breaks_blind_gate(self):
        evidence, initialization, public = self.fixture()
        evidence["quality_blind"]["generation_quality_metrics_accessed"] = True
        self.assert_rerun(evidence, initialization, public, "quality-blind")

    def test_runtime_records_and_derived_overlap_are_recomputed(self):
        mutations = ("wrong_seed", "same_gpu", "inverted_time", "negative_duration")
        for mutation in mutations:
            evidence, initialization, public = self.fixture()
            runtime = evidence["runtime"]
            if mutation == "wrong_seed":
                runtime["runs"]["arm_b_g1_3_lr_fixed_s4"]["seed"] = 999
            elif mutation == "same_gpu":
                runtime["runs"]["arm_b_g1_3_lr_fixed_s4"]["logged_gpu_index"] = 1
            elif mutation == "inverted_time":
                runtime["runs"]["arm_b_g1_3_lr_fixed_s4"][
                    "exit_marker_at_utc"
                ] = "2026-08-12T10:00:00+00:00"
            else:
                runtime["directly_observed_overlaps"][0]["duration_seconds"] = -1.0
            with self.subTest(mutation=mutation):
                self.assert_rerun(evidence, initialization, public, "runtime")

    def test_id_only_deviation_is_rejected(self):
        evidence, initialization, public = self.fixture()
        evidence["deviations"][0] = {"id": "D1"}
        self.assert_rerun(evidence, initialization, public, "deviation")

    def test_contradictory_deviation_narrative_is_rejected(self):
        evidence, initialization, public = self.fixture()
        evidence["deviations"][1]["observed"] = (
            "no overlap occurred; fully serial and protocol-exact"
        )
        self.assert_rerun(evidence, initialization, public, "deviation")

    def test_runtime_verification_time_is_public_receipt_bound(self):
        evidence, initialization, public = self.fixture()
        evidence["runtime"]["runs"]["arm_b_g1_3_lr_fixed_s4"][
            "strengthened_integrity_verified_at_utc"
        ] = "2026-08-12T12:10:00+00:00"
        self.assert_rerun(evidence, initialization, public, "runtime")

    def test_historical_inline_verification_precedes_next_start(self):
        evidence, initialization, public = self.fixture()
        historical = evidence["runtime"]["runs"]["arm_b_g1_3_lr_fixed_s4"][
            "historical_integrity_receipt"
        ]
        historical["verified_at_utc"] = "2026-08-12T12:10:00+00:00"
        historical["receipt_sha256"] = "c" * 64
        self.assert_rerun(evidence, initialization, public, "runtime")

    def test_historical_verification_cannot_precede_run_exit(self):
        evidence, initialization, public = self.fixture()
        historical = evidence["runtime"]["runs"]["arm_b_g1_3_lr_fixed_s4"][
            "historical_integrity_receipt"
        ]
        historical["verified_at_utc"] = "2026-08-12T11:20:00+00:00"
        historical["receipt_sha256"] = "d" * 64
        self.assert_rerun(evidence, initialization, public, "runtime")

    def test_nonfinite_preview_is_rejected(self):
        evidence, initialization, public = self.fixture()
        evidence["initialization"]["model_init_previews"]["5"][
            "max_abs_channel_delta_lsb"
        ] = float("nan")
        self.assert_rerun(evidence, initialization, public, "initialization")

    def test_public_path_scan_rejects_identifier_carriers(self):
        leaked_values = (
            "/root/private",
            "prefix=/root/private",
            "user@internal-host",
            "192.168.1.9",
            "GPU-01234567-89ab-cdef-0123-456789abcdef",
            "file:///private/result",
            "s3://secret-bucket/internal",
            "\\\\internal-host\\share",
            "fe80::1",
            "../../private/result",
            "ect-internal-01",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text('{"run_id":"arm_a_g1_0_lr_fixed_s4"}\n')
            self.assertTrue(adjudicator.public_text_is_sanitized(path))
            for leaked in leaked_values:
                path.write_text(json.dumps({"value": leaked}) + "\n")
                with self.subTest(leaked=leaked):
                    self.assertFalse(adjudicator.public_text_is_sanitized(path))
            path.write_text(json.dumps({"/root/private": True}) + "\n")
            self.assertFalse(adjudicator.public_text_is_sanitized(path))

    def test_duplicate_key_and_nonfinite_json_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text('{"value":"/root/private","value":"safe"}\n')
            self.assertFalse(adjudicator.public_text_is_sanitized(path))
            path.write_text('{"value":NaN}\n')
            self.assertFalse(adjudicator.public_text_is_sanitized(path))
        with self.assertRaises(ValueError):
            audit_contract.loads_strict('{"a":1,"a":2}')
        with self.assertRaises(ValueError):
            audit_contract.loads_strict('{"a":Infinity}')


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

    def test_fractional_summary_counters_are_rejected(self):
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
                for attempted in range(1, 2001):
                    writer.writerow(
                        {
                            "attempted_iteration": attempted + 0.5,
                            "successful_optimizer_steps": attempted + 0.5,
                            "processed_kimg": attempted * 0.128,
                            "loss": 15.0,
                            "grad_scale": 256,
                            "step_skipped": 0.5,
                            "schedule": "global_sigmoid",
                        }
                    )
            with self.assertRaisesRegex(SystemExit, "exact integers"):
                run_verifier.validate_summary(path)

    def test_training_options_require_exact_json_types(self):
        run_dir = Path(
            "/data/raw/ECT/ect_runs/gap_lr_matched_q128_s45_replication_v1/"
            "arm_a_g1_0_lr_fixed_s4"
        )
        baseline = run_verifier.expected_options(run_dir, "A", 4)
        run_verifier.validate_options(copy.deepcopy(baseline), run_dir, "A", 4)
        for label, path, value in (
            ("enable_amp", ("enable_amp",), 1),
            ("seed", ("seed",), 4.0),
            ("batch_gpu", ("batch_gpu",), 16.0),
            ("q", ("loss_kwargs", "q"), 128),
        ):
            candidate = copy.deepcopy(baseline)
            target = candidate
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(label=label):
                with self.assertRaisesRegex(SystemExit, "exact frozen"):
                    run_verifier.validate_options(candidate, run_dir, "A", 4)

    def test_launcher_segment_parser_is_an_exact_state_machine(self):
        valid = """nohup: ignored
START seed=4 arm=B gap=1.3 lr=0.0001 gpu=0 port=29842
DONE seed=4 arm=B integrity=passed
START seed=4 arm=C gap=1.3 lr=0.00012963523762588692 gpu=0 port=29843
DONE seed=4 arm=C integrity=passed
SEED4_REMAINING_ARMS_COMPLETE
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "launcher.log"
            path.write_text(valid)
            starts, events = builder.parse_launcher_segment("seed4_recovery", path)
            self.assertEqual(len(starts), 2)
            self.assertEqual(
                events, audit_contract.expected_launcher_events("seed4_recovery")
            )
            for tampered in (
                valid.replace("START seed=4 arm=B", "NOTSTART seed=4 arm=B"),
                valid.replace(
                    "START seed=4 arm=B gap=1.3 lr=0.0001 gpu=0 port=29842\n"
                    "DONE seed=4 arm=B integrity=passed",
                    "DONE seed=4 arm=B integrity=passed\n"
                    "START seed=4 arm=B gap=1.3 lr=0.0001 gpu=0 port=29842",
                ),
                valid.replace("DONE seed=4 arm=B", "UNDONE seed=4 arm=B"),
            ):
                path.write_text(tampered)
                with self.assertRaises(SystemExit):
                    builder.parse_launcher_segment("seed4_recovery", path)


class FormalRuntimePresenceTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("ECT_FORMAL_AUDIT") == "1",
        "formal runtime assertion is enabled only in the remote audit suite",
    )
    def test_formal_suite_cannot_skip_pytorch_tests(self):
        self.assertIsNotNone(torch, "formal audit requires PyTorch")


if __name__ == "__main__":
    unittest.main()
