#!/usr/bin/env python3
"""Scan a release tree for likely identity, credential, and private-path leaks."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence


DEFAULT_EXCLUDES = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".vscode",
    "__pycache__",
}

# These files intentionally contain the literal examples used by the scanner.
DEFAULT_FILE_EXCLUDES = {
    "docs/ANONYMIZATION_AUDIT.md",
    "scripts/audit_anonymity.py",
    "tests/test_audit_anonymity.py",
}

TEXT_SUFFIXES = {
    ".cfg", ".csv", ".env", ".ini", ".json", ".jsonl", ".md", ".py",
    ".rst", ".sh", ".toml", ".tsv", ".txt", ".yaml", ".yml",
}

PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "generic_secret": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)\b"
        r"\s*[:=]\s*['\"]?[A-Za-z0-9+/_.-]{8,}"
    ),
    "windows_user_path": re.compile(r"(?i)\b[A-Z]:\\Users\\[^\\\s]+"),
    "linux_root_path": re.compile(r"(?<![A-Za-z0-9_])/root/(?!\.cache/torch/hub/checkpoints/)"),
    "linux_home_path": re.compile(r"(?<![A-Za-z0-9_])/home/[^/\s]+/"),
    "project_mount_path": re.compile(r"(?<![A-Za-z0-9_])/mnt/(?:ect_project|mydata|workspace)(?:/|\b)"),
    "collaboration_repo_url": re.compile(
        r"https?://github\.com/hjjjs4vbmv-netizen/recurrence_of_ect(?:\.git)?(?:[/\s]|$)",
        re.IGNORECASE,
    ),
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
}

SENSITIVE_RULES = {"github_token", "generic_secret", "private_key"}


@dataclass(frozen=True)
class Finding:
    rule: str
    path: str
    line: int
    excerpt: str


def tracked_files(root: Path) -> list[Path] | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return [root / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def iter_files(root: Path, use_git: bool = True) -> Iterator[Path]:
    candidates = tracked_files(root) if use_git else None
    if candidates is None:
        candidates = list(root.rglob("*"))
    for path in candidates:
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        relative_name = relative.as_posix()
        if not path.is_file() or any(part in DEFAULT_EXCLUDES for part in relative.parts):
            continue
        if relative_name in DEFAULT_FILE_EXCLUDES:
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"Dockerfile", "Makefile"}:
            yield path


def scan_text(relative_path: str, text: str, rules: Sequence[str]) -> Iterator[Finding]:
    for line_number, line in enumerate(text.splitlines(), start=1):
        for rule in rules:
            if PATTERNS[rule].search(line):
                if rule in SENSITIVE_RULES:
                    excerpt = "<redacted>"
                else:
                    excerpt = line.strip()
                    if len(excerpt) > 200:
                        excerpt = excerpt[:197] + "..."
                yield Finding(rule=rule, path=relative_path, line=line_number, excerpt=excerpt)


def scan(root: Path, rules: Sequence[str], use_git: bool = True) -> list[Finding]:
    findings: list[Finding] = []
    for path in iter_files(root, use_git=use_git):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        relative = path.relative_to(root).as_posix()
        findings.extend(scan_text(relative, text, rules))
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", help="release tree to scan")
    parser.add_argument("--json", dest="json_path", help="write machine-readable findings")
    parser.add_argument(
        "--rule",
        action="append",
        choices=sorted(PATTERNS),
        help="scan only selected rule; may be repeated",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="scan the directory recursively instead of using git ls-files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    rules = args.rule or sorted(PATTERNS)
    findings = scan(root, rules, use_git=not args.all_files)

    for finding in findings:
        print(f"{finding.path}:{finding.line}: [{finding.rule}] {finding.excerpt}")

    if args.json_path:
        output = Path(args.json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps([asdict(item) for item in findings], indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"anonymity audit: {len(findings)} finding(s), {len(rules)} rule(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
