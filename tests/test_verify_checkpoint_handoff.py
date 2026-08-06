import hashlib
import json
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import verify_checkpoint_handoff


class VerifyCheckpointHandoffTest(unittest.TestCase):
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

