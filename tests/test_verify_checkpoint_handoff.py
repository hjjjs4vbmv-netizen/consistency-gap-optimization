import hashlib
import json
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import verify_checkpoint_handoff


class VerifyCheckpointHandoffTest(unittest.TestCase):
    def test_prospective_matrix_requires_main_merge_or_protected_tag(self):
        matrix_path = Path(__file__).resolve().parents[1] / "configs" / "q128_1024k_prospective_matrix.frozen.json"
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        freeze = matrix["freeze_activation"]
        self.assertEqual(freeze["required_location"], "merged_main_or_immutable_protected_tag")
        self.assertTrue(freeze["branch_push_is_insufficient"])
        self.assertIn("merged into main", freeze["training_prohibited_until"])
        contract = matrix["training_contract"]
        self.assertEqual(contract["training_source_status"], "clean_main_commit")
        self.assertEqual(
            contract["executed_training_source_commit"],
            "aae014c3a630a3a86801238cd0a8ff4ecd39c3d8",
        )

    def test_independent_verifier_recomputes_archive_and_checkpoint_hashes(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "checkpoint.pkl"
            checkpoint.write_bytes(b"checkpoint bytes")
            archive = root / "handoff.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                handle.add(checkpoint, arcname=checkpoint.name)
            checksum = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = root / "sender.json"
            manifest.write_text(json.dumps({
                "handoff_id": "unit-test",
                "archive_sha256": checksum(archive),
                "checkpoints": [{
                    "checkpoint_id": "checkpoint-a",
                    "filename": checkpoint.name,
                    "sha256": checksum(checkpoint),
                }],
            }), encoding="utf-8")
            receipt = verify_checkpoint_handoff.verify(manifest, archive, "independent-role-d")
            self.assertEqual(receipt["status"], "passed")
            self.assertTrue(receipt["archive_sha256_matches"])
            self.assertTrue(receipt["checkpoint_sha256_matches"])
