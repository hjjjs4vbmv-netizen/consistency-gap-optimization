#!/usr/bin/env python3
"""Independently verify a sender checkpoint archive for a Role D handoff.

This tool intentionally has no repository-specific imports.  Role D can copy
it with the sender manifest to an independent checkout and verify the archive
bytes plus every declared checkpoint digest before emitting a portable receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"[verify_checkpoint_handoff] ERROR: cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"[verify_checkpoint_handoff] ERROR: {label} must be a JSON object")
    return value


def safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:*") as handle:
        base = destination.resolve()
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if target != base and base not in target.parents:
                raise SystemExit("[verify_checkpoint_handoff] ERROR: archive contains an unsafe path")
        handle.extractall(destination, filter="data")


def verify(sender_manifest: Path, archive: Path, verifier_identity: str) -> dict[str, Any]:
    sender = load_object(sender_manifest, "sender manifest")
    checkpoints = sender.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise SystemExit("[verify_checkpoint_handoff] ERROR: sender manifest lacks checkpoints")
    if not verifier_identity.strip():
        raise SystemExit("[verify_checkpoint_handoff] ERROR: --verifier-identity must be non-empty")

    observed_archive_sha = sha256_file(archive)
    expected_archive_sha = sender.get("archive_sha256")
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="role-d-handoff-") as temp:
        root = Path(temp)
        safe_extract(archive, root)
        for item in checkpoints:
            if not isinstance(item, dict):
                raise SystemExit("[verify_checkpoint_handoff] ERROR: malformed checkpoint declaration")
            checkpoint_id = item.get("checkpoint_id")
            filename = item.get("filename")
            expected_sha = item.get("sha256")
            if not all(isinstance(value, str) and value for value in (checkpoint_id, filename, expected_sha)):
                raise SystemExit("[verify_checkpoint_handoff] ERROR: incomplete checkpoint declaration")
            matches = list(root.rglob(filename))
            actual_sha = sha256_file(matches[0]) if len(matches) == 1 and matches[0].is_file() else None
            results.append({
                "checkpoint_id": checkpoint_id,
                "filename": filename,
                "expected_sha256": expected_sha,
                "observed_sha256": actual_sha,
                "match": actual_sha == expected_sha,
            })
    archive_match = isinstance(expected_archive_sha, str) and observed_archive_sha == expected_archive_sha
    all_checkpoint_matches = all(row["match"] for row in results)
    return {
        "schema_version": 1,
        "verification_role": "Role D independent receiver",
        "status": "passed" if archive_match and all_checkpoint_matches else "failed",
        "handoff_id": sender.get("handoff_id"),
        "verifier_identity": verifier_identity,
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "verifier_python": platform.python_version(),
        "sender_manifest_filename": sender_manifest.name,
        "sender_manifest_sha256": sha256_file(sender_manifest),
        "archive_filename": archive.name,
        "expected_archive_sha256": expected_archive_sha,
        "observed_archive_sha256": observed_archive_sha,
        "archive_sha256_matches": archive_match,
        "checkpoint_sha256_matches": all_checkpoint_matches,
        "checkpoints": results,
        "claim_effect": (
            "Archive and declared checkpoint identities were independently recomputed."
            if archive_match and all_checkpoint_matches else
            "Checkpoint identity verification failed; this handoff must not support formal evaluation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sender-manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--verifier-identity", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(args.sender_manifest, args.archive, args.verifier_identity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if receipt["status"] != "passed":
        raise SystemExit("[verify_checkpoint_handoff] ERROR: verification failed")


if __name__ == "__main__":
    main()
