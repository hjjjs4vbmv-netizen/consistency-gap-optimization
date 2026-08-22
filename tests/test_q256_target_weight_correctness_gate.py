import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from scripts import run_q256_target_weight_correctness_gate as gate


class JUnitGateTest(unittest.TestCase):
    def test_git_status_command_is_compatible_with_server_git_1_8(self):
        self.assertEqual(
            gate.GIT_STATUS_COMMAND,
            ("git", "status", "--porcelain", "--untracked-files=all"),
        )

    def test_visible_gpu_binding_requires_the_exact_full_uuid(self):
        expected = "GPU-11111111-2222-3333-4444-555555555555"
        with mock.patch.dict(
            "os.environ", {"CUDA_VISIBLE_DEVICES": expected}, clear=False
        ):
            self.assertEqual(gate.require_visible_gpu_uuid(expected), expected)
        with mock.patch.dict(
            "os.environ", {"CUDA_VISIBLE_DEVICES": "0"}, clear=False
        ), self.assertRaisesRegex(gate.GateError, "full GPU UUID exactly"):
            gate.require_visible_gpu_uuid(expected)

    def test_source_manifest_must_match_head_without_special_index_flags(self):
        root = Path(self.tempdir.name) / "repo"
        root.mkdir()
        subprocess.check_call(["git", "init", "-q"], cwd=root)
        tracked = root / "tracked.py"
        tracked.write_text("value = 1\n", encoding="utf-8")
        subprocess.check_call(["git", "add", "tracked.py"], cwd=root)
        subprocess.check_call(
            [
                "git", "-c", "user.name=Gate Test", "-c",
                "user.email=gate@example.invalid", "commit", "-q", "-m", "fixture",
            ],
            cwd=root,
        )
        source = {
            "files": [
                {
                    "path": "tracked.py",
                    "sha256": gate.sha256_file(tracked),
                }
            ]
        }
        gate.verify_launcher_source_matches_head(root, source)
        subprocess.check_call(
            ["git", "update-index", "--assume-unchanged", "tracked.py"],
            cwd=root,
        )
        tracked.write_text("value = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(gate.GateError, "unsafe index flag"):
            gate.verify_launcher_source_matches_head(root, source)

    def write_junit(self, body):
        root = Path(self.tempdir.name)
        path = root / "pytest.xml"
        path.write_text(
            '<testsuites><testsuite name="pytest">' + body
            + '</testsuite></testsuites>',
            encoding="utf-8",
        )
        return path

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_required_tests_must_all_pass_without_skip(self):
        body = "".join(
            f'<testcase classname="{classname}" name="{name}" />'
            for classname, name in gate.REQUIRED_TEST_CASES
        )
        parsed = gate.parse_junit(self.write_junit(body))
        self.assertEqual(parsed["failures"], 0)
        self.assertEqual(parsed["errors"], 0)
        self.assertEqual(parsed["skipped"], 0)
        self.assertEqual(parsed["required_test_failures"], [])

    def test_missing_required_test_fails_closed(self):
        path = self.write_junit(
            '<testcase classname="x" name="unrelated" />'
        )
        with self.assertRaisesRegex(gate.GateError, "exactly once"):
            gate.parse_junit(path)

    def test_skipped_required_cuda_test_is_reported(self):
        body = "".join(
            (
                f'<testcase classname="{classname}" name="{name}">'
                + ('<skipped message="no CUDA" />' if (classname, name) == gate.REQUIRED_TEST_CASES[-1] else '')
                + '</testcase>'
            )
            for classname, name in gate.REQUIRED_TEST_CASES
        )
        parsed = gate.parse_junit(self.write_junit(body))
        self.assertEqual(parsed["skipped"], 1)
        classname, name = gate.REQUIRED_TEST_CASES[-1]
        self.assertEqual(
            parsed["required_test_failures"], [f"{classname}::{name}"]
        )

    def test_duplicate_required_identity_fails_closed(self):
        body = "".join(
            f'<testcase classname="{classname}" name="{name}" />'
            for classname, name in gate.REQUIRED_TEST_CASES
        )
        classname, name = gate.REQUIRED_TEST_CASES[0]
        body += f'<testcase classname="{classname}" name="{name}" />'
        with self.assertRaisesRegex(gate.GateError, "exactly once"):
            gate.parse_junit(self.write_junit(body))

    def test_exclusive_writer_never_overwrites_a_published_receipt(self):
        path = Path(self.tempdir.name) / "receipt.json"
        gate.write_exclusive(path, b"first\n")
        self.assertEqual(path.read_bytes(), b"first\n")
        with self.assertRaises(FileExistsError):
            gate.write_exclusive(path, b"second\n")
        self.assertEqual(path.read_bytes(), b"first\n")


if __name__ == "__main__":
    unittest.main()
