"""Regression tests for the seed-4/5 Role E evaluation handoff."""

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/validate_gap_lr_seed_replication_eval_handoff.py"
SPEC = importlib.util.spec_from_file_location("validate_role_e_handoff", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RoleEEvaluationHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = json.loads(MODULE.DEFAULT_HANDOFF.read_text(encoding="utf-8"))

    def validate_mutation(self, payload: dict) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "handoff.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return MODULE.validate_handoff(path)

    def test_committed_handoff_passes(self) -> None:
        report = MODULE.validate_handoff()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["endpoint_count"], 6)
        self.assertEqual(report["sample_block_count"], 3)
        self.assertEqual(report["raw_checkpoint_rehash"], "not_run")

    def test_checkpoint_hash_drift_fails_closed(self) -> None:
        payload = copy.deepcopy(self.source)
        payload["checkpoint_contract"]["endpoints"][0]["checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(MODULE.HandoffError, "checkpoint hash drift"):
            self.validate_mutation(payload)

    def test_reusing_original_sampling_block_fails_closed(self) -> None:
        payload = copy.deepcopy(self.source)
        block = payload["evaluation_contract"]["sample_blocks"][0]
        block.update(block_id="block_0_4999", start=0, end=4999)
        with self.assertRaisesRegex(MODULE.HandoffError, "block ID drift"):
            self.validate_mutation(payload)

    def test_latest_alias_fails_closed(self) -> None:
        payload = copy.deepcopy(self.source)
        payload["checkpoint_contract"]["checkpoint_filename"] = (
            "network-snapshot-latest.pkl"
        )
        with self.assertRaisesRegex(MODULE.HandoffError, "final numbered checkpoint"):
            self.validate_mutation(payload)

    def test_claiming_independent_blind_review_fails_closed(self) -> None:
        payload = copy.deepcopy(self.source)
        payload["authorization"]["independent_quality_blind_review_claimed"] = True
        with self.assertRaisesRegex(MODULE.HandoffError, "must not claim"):
            self.validate_mutation(payload)

    def test_fid50k_authorization_fails_closed(self) -> None:
        payload = copy.deepcopy(self.source)
        payload["authorization"]["fid50k_seed4_seed5_authorized"] = True
        with self.assertRaisesRegex(MODULE.HandoffError, "FID-50k must be forbidden"):
            self.validate_mutation(payload)


if __name__ == "__main__":
    unittest.main()
