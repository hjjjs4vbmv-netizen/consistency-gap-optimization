import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from analysis.q256_fresh_crossed_switch_n12_matpool_v1 import authorization


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ElevenSeedAuthorizationFailClosedTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.control = self.root / "control"
        self.output = self.root / "output"
        self.protocol_path = self.root / "protocol.json"
        write_json(self.protocol_path, {
            "paths": {
                "control_root": str(self.control),
                "formal_output_root": str(self.output),
                "repository_root": str(self.root),
            }
        })
        self.protocol_sha = sha256(self.protocol_path)

        self.numeric_path = self.control / "numeric_recovery2_authorization.json"
        write_json(self.numeric_path, {
            "schema": authorization.NUMERIC_RECOVERY_SCHEMA,
            "status": "AUTHOR_APPROVED",
            "protocol_sha256": self.protocol_sha,
            "manual_recovery_index": 2,
            "automatic_retry_count": 0,
            "max_recoverable_nonfinite_loss_attempts_per_cell": 1,
            "original_failure_must_be_preserved": True,
            "quality_metrics_observed_before_amendment": False,
        })

        self.failure_path = (
            self.output / "archive" / "manual-recovery-2" / "seed38_AB_failed_attempt2"
            / "compute_failure_receipt.json"
        )
        write_json(self.failure_path, {
            "schema": "ect.q256.fresh-crossed-switch-compute-completion/v1",
            "status": "FAIL", "label": "seed38:AB", "exit_code": 1,
            "hard_timeout": False,
        })

        receipt_hashes = {}
        for seed in authorization.INCLUDED_SEEDS:
            path = self.output / "training" / f"seed{seed}" / "seed_completion_receipt.json"
            write_json(path, {
                "schema": "ect.q256.fresh-crossed-switch-seed-completion/v1",
                "status": "PASS", "seed": seed, "protocol_sha256": self.protocol_sha,
            })
            receipt_hashes[str(seed)] = sha256(path)

        self.authorization_path = self.control / "eleven_seed_authorization.json"
        self.valid = {
            "schema": authorization.SCHEMA,
            "status": "AUTHOR_APPROVED",
            "authorized_at": "2026-09-01T12:00:00Z",
            "protocol_sha256": self.protocol_sha,
            "amendment_commit": "a" * 40,
            "scope": "exclude terminally failed seed38 and continue with the eleven complete seeds",
            "excluded_seed": 38,
            "included_seeds": list(authorization.INCLUDED_SEEDS),
            "expected_evaluation_jobs": 242,
            "original_n12_claim_abandoned": True,
            "analysis_population": "AUTHOR_AMENDED_N11_COMPLETE_CASE",
            "minimum_negative_directions_for_strong_success": 10,
            "decode_forbidden_before_full_amended_seal": True,
            "automatic_retry_count": 0,
            "quality_metrics_observed_before_amendment": False,
            "numeric_recovery2_authorization_sha256": sha256(self.numeric_path),
            "terminal_failed_compute_receipt_sha256": sha256(self.failure_path),
            "seed_completion_receipt_sha256": receipt_hashes,
        }
        write_json(self.authorization_path, self.valid)

    def tearDown(self):
        self.temporary.cleanup()

    def validate(self):
        return authorization.validate_eleven_seed_authorization(
            self.authorization_path, self.protocol_path
        )

    def rewrite_authorization(self, mutation):
        changed = copy.deepcopy(self.valid)
        mutation(changed)
        write_json(self.authorization_path, changed)

    def test_complete_authorization_and_all_sources_pass(self):
        self.assertEqual(self.validate()["included_seeds"], list(authorization.INCLUDED_SEEDS))

    def test_handcrafted_minimal_json_is_rejected(self):
        write_json(self.authorization_path, {
            "status": "AUTHOR_APPROVED",
            "protocol_sha256": self.protocol_sha,
            "excluded_seed": 38,
            "included_seeds": list(authorization.INCLUDED_SEEDS),
            "expected_evaluation_jobs": 242,
            "original_n12_claim_abandoned": True,
            "decode_forbidden_before_full_amended_seal": True,
        })
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_schema_tampering_is_rejected(self):
        self.rewrite_authorization(lambda value: value.__setitem__("schema", "wrong"))
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_every_frozen_scalar_field_is_fail_closed(self):
        mutations = {
            "status": "PASS",
            "authorized_at": "not-a-timestamp",
            "protocol_sha256": "0" * 64,
            "amendment_commit": "not-a-commit",
            "scope": "exclude a failed seed",
            "excluded_seed": 37,
            "included_seeds": list(range(31, 42)),
            "expected_evaluation_jobs": 241,
            "original_n12_claim_abandoned": False,
            "analysis_population": "N12",
            "minimum_negative_directions_for_strong_success": 9,
            "decode_forbidden_before_full_amended_seal": False,
            "automatic_retry_count": 1,
            "quality_metrics_observed_before_amendment": True,
        }
        for key, changed in mutations.items():
            with self.subTest(key=key):
                value = copy.deepcopy(self.valid)
                value[key] = changed
                write_json(self.authorization_path, value)
                with self.assertRaises(RuntimeError):
                    self.validate()
        write_json(self.authorization_path, self.valid)

    def test_quality_observation_tampering_is_rejected(self):
        self.rewrite_authorization(
            lambda value: value.__setitem__("quality_metrics_observed_before_amendment", True)
        )
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_decision_threshold_tampering_is_rejected(self):
        self.rewrite_authorization(
            lambda value: value.__setitem__("minimum_negative_directions_for_strong_success", 9)
        )
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_numeric_authorization_hash_tampering_is_rejected(self):
        self.rewrite_authorization(
            lambda value: value.__setitem__("numeric_recovery2_authorization_sha256", "0" * 64)
        )
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_terminal_failure_hash_tampering_is_rejected(self):
        self.rewrite_authorization(
            lambda value: value.__setitem__("terminal_failed_compute_receipt_sha256", "0" * 64)
        )
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_completion_receipt_inventory_tampering_is_rejected(self):
        def mutate(value):
            value["seed_completion_receipt_sha256"].pop("31")
        self.rewrite_authorization(mutate)
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_completion_receipt_hash_tampering_is_rejected(self):
        def mutate(value):
            value["seed_completion_receipt_sha256"]["31"] = "0" * 64
        self.rewrite_authorization(mutate)
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_numeric_authorization_content_tampering_is_rejected(self):
        value = json.loads(self.numeric_path.read_text(encoding="utf-8"))
        value["quality_metrics_observed_before_amendment"] = True
        write_json(self.numeric_path, value)
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_completion_receipt_content_tampering_is_rejected(self):
        path = self.output / "training" / "seed31" / "seed_completion_receipt.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["status"] = "FAIL"
        write_json(path, value)
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_terminal_failure_content_tampering_is_rejected(self):
        value = json.loads(self.failure_path.read_text(encoding="utf-8"))
        value["label"] = "seed38:AA"
        write_json(self.failure_path, value)
        with self.assertRaises(RuntimeError):
            self.validate()


if __name__ == "__main__":
    unittest.main()
