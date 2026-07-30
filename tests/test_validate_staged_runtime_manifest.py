import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import validate_staged_runtime_manifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ValidateStagedRuntimeManifestTest(unittest.TestCase):
    def make_manifests(self, root: Path):
        checkpoint = root / "network-snapshot-latest.pkl"
        checkpoint.write_bytes(b"checkpoint")
        receipt = root / "seed3_fixed_256k.integrity.json"
        checkpoint_sha = sha256(checkpoint)
        receipt.write_text(json.dumps({
            "schema_version": 1,
            "status": "passed",
            "checkpoint_id": "confirmatory-256k-seed3-fixed",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "training_run_id": "confirmatory-256k-seed3-fixed",
            "method": "fixed",
            "training_seed": 3,
            "budget_kimg": 256,
            "completion_passed": True,
            "logs_state_consistent": True,
            "finite_loss_state_passed": True,
            "checker_version": "1",
            "checker_git_commit": "125ece6d018e10a1c2cf13ea6e3beeda09667d23",
            "checked_at_unix": 1_753_822_000,
        }), encoding="utf-8")
        frozen = {
            "manifest_kind": "frozen-logical-checkpoint-matrix",
            "protocol": "staged-checkpoint-evaluation-v1",
            "runtime_binding": {"versioned_paths": False},
            "cells": [{
                "checkpoint_id": "confirmatory-256k-seed3-fixed",
                "method": "fixed",
                "training_seed": 3,
                "budget_kimg": 256,
                "schedule_q": 256,
                "schedule_identity": "sigmoid",
                "global_gap_scale": 1.0,
                "checkpoint_sha256": checkpoint_sha,
                "executed_training_source_commit": "3a0d603da97dd93ddbb6c7ce49e4a7351d54bb43",
                "training_integrity_receipt": {
                    "receipt_filename": receipt.name,
                    "receipt_sha256": sha256(receipt),
                },
            }],
        }
        runtime = {
            "protocol": "staged-checkpoint-evaluation-v1",
            "cells": [{
                **{key: value for key, value in frozen["cells"][0].items()
                   if key != "training_integrity_receipt"},
                "checkpoint": str(checkpoint),
                "integrity_receipt": str(receipt),
            }],
        }
        frozen_path = root / "frozen.json"
        runtime_path = root / "runtime.json"
        frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
        runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
        return frozen_path, runtime_path

    def test_accepts_exact_runtime_binding(self):
        with TemporaryDirectory() as temp_dir:
            frozen, runtime = self.make_manifests(Path(temp_dir))
            rows = validate_staged_runtime_manifest.validate(frozen, runtime)
            self.assertEqual([row["checkpoint_id"] for row in rows], ["confirmatory-256k-seed3-fixed"])

    def test_rejects_runtime_identity_drift(self):
        with TemporaryDirectory() as temp_dir:
            frozen, runtime = self.make_manifests(Path(temp_dir))
            payload = json.loads(runtime.read_text(encoding="utf-8"))
            payload["cells"][0]["global_gap_scale"] = 1.2
            runtime.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "global_gap_scale"):
                validate_staged_runtime_manifest.validate(frozen, runtime)

    def test_rejects_receipt_checksum_drift(self):
        with TemporaryDirectory() as temp_dir:
            frozen, runtime = self.make_manifests(Path(temp_dir))
            payload = json.loads(runtime.read_text(encoding="utf-8"))
            receipt = Path(payload["cells"][0]["integrity_receipt"])
            receipt.write_text(receipt.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "receipt SHA256"):
                validate_staged_runtime_manifest.validate(frozen, runtime)


if __name__ == "__main__":
    unittest.main()
