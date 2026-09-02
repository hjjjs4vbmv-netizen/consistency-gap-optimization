#!/usr/bin/env python3
"""Dependency-free validation for the author-amended n=11 evidence chain."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path


SCHEMA = "ect.q256.fresh-crossed-switch-eleven-seed-amendment/v1"
NUMERIC_RECOVERY_SCHEMA = (
    "ect.q256.fresh-crossed-switch-numeric-recovery-v2-authorization/v1"
)
SEEDS = tuple(range(31, 43))
EXCLUDED_SEED = 38
INCLUDED_SEEDS = tuple(seed for seed in SEEDS if seed != EXCLUDED_SEED)
EXPECTED_EVALUATION_JOBS = len(INCLUDED_SEEDS) * 22
MINIMUM_NEGATIVE_DIRECTIONS = 10
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"regular JSON artifact required: {path}")
    with path.open("rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _valid_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return True


def _validate_structure(value: dict, protocol_sha256: str) -> None:
    expected = {
        "schema": SCHEMA,
        "status": "AUTHOR_APPROVED",
        "protocol_sha256": protocol_sha256,
        "scope": "exclude terminally failed seed38 and continue with the eleven complete seeds",
        "excluded_seed": EXCLUDED_SEED,
        "included_seeds": list(INCLUDED_SEEDS),
        "expected_evaluation_jobs": EXPECTED_EVALUATION_JOBS,
        "original_n12_claim_abandoned": True,
        "analysis_population": "AUTHOR_AMENDED_N11_COMPLETE_CASE",
        "minimum_negative_directions_for_strong_success": MINIMUM_NEGATIVE_DIRECTIONS,
        "decode_forbidden_before_full_amended_seal": True,
        "automatic_retry_count": 0,
        "quality_metrics_observed_before_amendment": False,
    }
    mismatches = [
        key for key, expected_value in expected.items()
        if value.get(key) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            "eleven-seed authorization binding mismatch: " + ", ".join(mismatches)
        )
    if not _valid_utc_timestamp(value.get("authorized_at")):
        raise RuntimeError("eleven-seed authorization timestamp mismatch")
    if not HEX40.fullmatch(str(value.get("amendment_commit", ""))):
        raise RuntimeError("eleven-seed amendment commit is not a SHA-1")
    for key in (
        "numeric_recovery2_authorization_sha256",
        "terminal_failed_compute_receipt_sha256",
    ):
        if not HEX64.fullmatch(str(value.get(key, ""))):
            raise RuntimeError(f"invalid eleven-seed SHA-256 binding: {key}")
    receipts = value.get("seed_completion_receipt_sha256")
    expected_keys = {str(seed) for seed in INCLUDED_SEEDS}
    if not isinstance(receipts, dict) or set(receipts) != expected_keys:
        raise RuntimeError("eleven-seed completion receipt inventory mismatch")
    if any(not HEX64.fullmatch(str(digest)) for digest in receipts.values()):
        raise RuntimeError("invalid eleven-seed completion receipt SHA-256")


def _validate_numeric_authorization(path: Path, expected_sha: str,
                                    protocol_sha: str) -> None:
    value = load_object(path)
    if sha256_file(path) != expected_sha:
        raise RuntimeError("numeric recovery v2 authorization SHA-256 mismatch")
    expected = {
        "schema": NUMERIC_RECOVERY_SCHEMA,
        "status": "AUTHOR_APPROVED",
        "protocol_sha256": protocol_sha,
        "manual_recovery_index": 2,
        "automatic_retry_count": 0,
        "max_recoverable_nonfinite_loss_attempts_per_cell": 1,
        "original_failure_must_be_preserved": True,
        "quality_metrics_observed_before_amendment": False,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise RuntimeError("numeric recovery v2 authorization content mismatch")


def _validate_seed_receipts(output_root: Path, authorization: dict,
                            protocol_sha: str) -> None:
    expected_hashes = authorization["seed_completion_receipt_sha256"]
    for seed in INCLUDED_SEEDS:
        path = output_root / "training" / f"seed{seed}" / "seed_completion_receipt.json"
        receipt = load_object(path)
        if (receipt.get("schema")
                != "ect.q256.fresh-crossed-switch-seed-completion/v1"
                or receipt.get("status") != "PASS"
                or receipt.get("seed") != seed
                or receipt.get("protocol_sha256") != protocol_sha):
            raise RuntimeError(f"invalid completed seed receipt: seed{seed}")
        if sha256_file(path) != expected_hashes[str(seed)]:
            raise RuntimeError(f"completed seed receipt SHA-256 mismatch: seed{seed}")


def _validate_terminal_failure(output_root: Path, expected_sha: str) -> None:
    path = output_root / "training" / "seed38" / "AB" / "compute_completion_receipt.json"
    value = load_object(path)
    if sha256_file(path) != expected_sha:
        raise RuntimeError("terminal seed38/AB failure receipt identity mismatch")
    if (value.get("schema")
            != "ect.q256.fresh-crossed-switch-compute-completion/v1"
            or value.get("status") != "FAIL"
            or value.get("label") != "seed38:AB"
            or value.get("exit_code") != 1
            or value.get("hard_timeout") is not False):
        raise RuntimeError("terminal seed38/AB failure receipt content mismatch")


def analysis_population_line(seeds: tuple[int, ...], *, amended: bool) -> str:
    rendered = ", ".join(map(str, seeds))
    if amended:
        return (
            f"{len(seeds)} complete seeds ({rendered}); seed38 excluded by explicit "
            "author amendment; the original n=12 claim is abandoned: True"
        )
    return (
        f"{len(seeds)} preregistered seeds ({rendered}); no author amendment or "
        "seed exclusion"
    )


def validate_eleven_seed_authorization(
    path: Path,
    protocol_path: Path,
    *,
    require_commit: bool = False,
    repository_root: Path | None = None,
    verify_source_artifacts: bool = True,
) -> dict:
    """Validate the amendment and, by default, every source hash it asserts.

    ``verify_source_artifacts=False`` is intentionally available only for tools
    that inspect a detached public evidence bundle. Formal pipeline consumers
    use the default and therefore fail closed when any source artifact is
    absent or changed.
    """

    path = path.resolve(strict=True)
    protocol_path = protocol_path.resolve(strict=True)
    authorization = load_object(path)
    protocol = load_object(protocol_path)
    protocol_sha = sha256_file(protocol_path)
    _validate_structure(authorization, protocol_sha)

    if require_commit:
        repo = Path(repository_root or protocol["paths"]["repository_root"])
        head = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if authorization["amendment_commit"] != head:
            raise RuntimeError("eleven-seed authorization commit mismatch")

    if verify_source_artifacts:
        paths = protocol.get("paths")
        if not isinstance(paths, dict):
            raise RuntimeError("protocol paths required for authorization validation")
        control_root = Path(paths["control_root"])
        output_root = Path(paths["formal_output_root"])
        _validate_numeric_authorization(
            control_root / "numeric_recovery2_authorization.json",
            authorization["numeric_recovery2_authorization_sha256"],
            protocol_sha,
        )
        _validate_seed_receipts(output_root, authorization, protocol_sha)
        _validate_terminal_failure(
            output_root,
            authorization["terminal_failed_compute_receipt_sha256"],
        )
    return authorization
