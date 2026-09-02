import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_anonymity.py"
SPEC = importlib.util.spec_from_file_location("audit_anonymity", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AnonymityAuditTests(unittest.TestCase):
    def test_detects_private_paths_credentials_and_repository_url(self):
        text = "\n".join(
            [
                "output=/mnt/ect_project/runs/example",
                "cache=/data/raw/ECT/private",
                "local=/Users/researcher/project",
                "account=ECT001",
                "token=ghp_abcdefghijklmnopqrstuvwxyz123456",
                "source=https://github.com/hjjjs4vbmv-netizen/consistency-gap-optimization",
            ]
        )
        findings = list(MODULE.scan_text("README.md", text, sorted(MODULE.PATTERNS)))
        self.assertEqual(
            {item.rule for item in findings},
            {
                "project_mount_path",
                "private_data_path",
                "macos_user_path",
                "server_account",
                "github_token",
                "collaboration_repo_url",
                "collaboration_owner",
            },
        )

    def test_sensitive_excerpts_are_redacted(self):
        secret = "ghp_abcdefghijklmnopqrstuvwxyz123456"
        findings = list(
            MODULE.scan_text(
                "credentials.txt",
                f"token={secret}\n",
                sorted(MODULE.PATTERNS),
            )
        )
        rendered_output = "\n".join(
            f"{item.path}:{item.line}: [{item.rule}] {item.excerpt}"
            for item in findings
        )
        self.assertNotIn(secret, rendered_output)
        self.assertIn("<redacted>", rendered_output)

    def test_clean_relative_paths_pass(self):
        text = "output=${ECT_RUNS_ROOT}/fixed/seed3\nsource=./results/summary.csv\n"
        findings = list(MODULE.scan_text("README.md", text, sorted(MODULE.PATTERNS)))
        self.assertEqual(findings, [])

    def test_recursive_scan_ignores_binary_and_cache_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("path=/root/private/run\n", encoding="utf-8")
            (root / "image.png").write_bytes(b"\x89PNG\x00/root/not-text")
            cache = root / "__pycache__"
            cache.mkdir()
            (cache / "leak.txt").write_text("path=/root/ignored\n", encoding="utf-8")

            findings = MODULE.scan(root, sorted(MODULE.PATTERNS), use_git=False)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].rule, "linux_root_path")
        self.assertEqual(findings[0].path, "README.md")

    def test_recursive_scan_rejects_agents_and_git_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".agents").mkdir()
            (root / ".git").mkdir()

            findings = MODULE.scan(root, sorted(MODULE.PATTERNS), use_git=False)

        self.assertEqual(
            {(item.rule, item.path) for item in findings},
            {("agents_metadata", ".agents"), ("git_metadata", ".git")},
        )


if __name__ == "__main__":
    unittest.main()
