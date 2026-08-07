import hashlib
import json
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import verify_handoff


class VerifyHandoffTest(unittest.TestCase):
    def test_recomputes_archive_and_checkpoint_hashes(self):
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
                "handoff_id": "supplementary-unit-test",
                "archive_sha256": checksum(archive),
                "checkpoints": [{
                    "checkpoint_id": "checkpoint-a",
                    "filename": checkpoint.name,
                    "sha256": checksum(checkpoint),
                }],
            }), encoding="utf-8")

            receipt = verify_handoff.verify(manifest, archive, "independent-receiver")

            self.assertEqual(receipt["status"], "passed")
            self.assertEqual(receipt["verification_role"], "independent receiver")
            self.assertTrue(receipt["archive_sha256_matches"])
            self.assertTrue(receipt["checkpoint_sha256_matches"])

    def test_rejects_unsafe_archive_member(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "unsafe.tar"
            source = root / "checkpoint.pkl"
            source.write_bytes(b"checkpoint bytes")
            with tarfile.open(archive, "w") as handle:
                handle.add(source, arcname="../checkpoint.pkl")
            manifest = root / "sender.json"
            manifest.write_text(json.dumps({"archive_sha256": "x", "checkpoints": [{
                "checkpoint_id": "checkpoint-a", "filename": "checkpoint.pkl", "sha256": "x",
            }]}), encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "unsafe path"):
                verify_handoff.verify(manifest, archive, "independent-receiver")
