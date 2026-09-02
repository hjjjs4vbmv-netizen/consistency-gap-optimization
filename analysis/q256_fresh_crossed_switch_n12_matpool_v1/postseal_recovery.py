#!/usr/bin/env python3
"""Authorize and preserve one post-seal statistics/report recovery."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.q256_fresh_crossed_switch_n12_matpool_v1 import experiment  # noqa: E402


def validate_authorization(path: Path, protocol_path: Path,
                           *, require_commit: bool = False) -> dict:
    value = experiment.load_json(path.resolve(strict=True))
    index = value.get("manual_postseal_report_recovery_index")
    if index not in {1, 2}:
        raise RuntimeError("invalid postseal report recovery index")
    expected = {
        "status": "AUTHOR_APPROVED",
        "protocol_sha256": experiment.sha256_file(protocol_path),
        "expected_evaluation_jobs": experiment.ELEVEN_JOB_COUNT,
        "evaluation_rerun_forbidden": True,
        "decoded_results_reused_without_redecode": True,
        "automatic_retry_count": 0,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise RuntimeError("postseal report recovery authorization binding mismatch")
    eleven_path = Path(value["eleven_seed_authorization_path"])
    experiment.validate_eleven_seed_authorization(eleven_path, protocol_path)
    evaluation_path = Path(value["evaluation_recovery_authorization_path"])
    experiment.validate_evaluation_recovery1_authorization(evaluation_path, protocol_path)
    if value.get("eleven_seed_authorization_sha256") != experiment.sha256_file(eleven_path):
        raise RuntimeError("postseal recovery eleven-seed authorization mismatch")
    if value.get("evaluation_recovery_authorization_sha256") != experiment.sha256_file(evaluation_path):
        raise RuntimeError("postseal recovery evaluation authorization mismatch")
    if index == 2:
        prior_path = Path(value["prior_postseal_authorization_path"])
        prior = validate_authorization(prior_path, protocol_path)
        if (prior.get("manual_postseal_report_recovery_index") != 1
                or value.get("prior_postseal_authorization_sha256")
                != experiment.sha256_file(prior_path)):
            raise RuntimeError("postseal recovery v2 prior authorization mismatch")
    if require_commit:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        if value.get("amendment_commit") != head:
            raise RuntimeError("postseal report recovery commit mismatch")
    return value


def authorize(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve(strict=True)
    protocol = experiment.load_json(protocol_path)
    experiment.validate_protocol(protocol, protocol_path)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    if not experiment.HEX40.fullmatch(args.amendment_commit) or args.amendment_commit != head:
        raise RuntimeError("postseal report recovery amendment is not current HEAD")
    eleven_path = args.eleven_seed_authorization.resolve(strict=True)
    experiment.validate_eleven_seed_authorization(eleven_path, protocol_path)
    evaluation_path = args.evaluation_recovery_authorization.resolve(strict=True)
    evaluation = experiment.validate_evaluation_recovery1_authorization(
        evaluation_path, protocol_path
    )
    root = Path(protocol["paths"]["formal_output_root"])
    eval_root = root / evaluation["evaluation_dir"]
    seal_path = eval_root / "evaluation_matrix_seal.json"
    seal = experiment.load_json(seal_path.resolve(strict=True))
    if (seal.get("status") != "SEALED_PASS"
            or seal.get("sealed_jobs") != experiment.ELEVEN_JOB_COUNT
            or seal.get("expected_jobs") != experiment.ELEVEN_JOB_COUNT
            or seal.get("automatic_retry_count") != 0
            or len(list((eval_root / "seals").glob("*.json"))) != experiment.ELEVEN_JOB_COUNT
            or list(eval_root.glob("jobs/*/failure_receipt.json"))):
        raise RuntimeError("postseal recovery requires the complete failure-free matrix seal")
    decoded_path = args.decoded_results.resolve(strict=True)
    decoded = experiment.load_json(decoded_path)
    if (decoded.get("status") != "PASS" or not decoded.get("decoded_after_full_seal")
            or decoded.get("job_count") != experiment.ELEVEN_JOB_COUNT
            or decoded.get("matrix_seal_sha256") != experiment.sha256_file(seal_path)
            or decoded.get("evaluation_recovery_authorization_sha256")
            != experiment.sha256_file(evaluation_path)):
        raise RuntimeError("postseal recovery requires the sealed decoded matrix")
    failed_statistics = args.failed_statistics.resolve(strict=True)
    if (failed_statistics / "analysis.json").exists() \
            or (failed_statistics / "primary_decision.json").exists():
        raise RuntimeError("failed statistics directory unexpectedly contains final decisions")
    expected_partial = {"H_C_I_Q_G_per_seed.csv", "decoded_evaluation_results.csv",
                        "AULC_diagnostic.csv"}
    if {path.name for path in failed_statistics.iterdir()} != expected_partial:
        raise RuntimeError("unexpected partial statistics artifact set")
    archive_root = root / "archive" / "manual-postseal-report-recovery-1"
    archive_statistics = archive_root / "statistics_failed_attempt1"
    archive_receipt = archive_root / "postseal_report_recovery1_archive_receipt.json"
    for path in (archive_root, archive_statistics, archive_receipt,
                 root / "REPORT_11SEED.md", root / "final_completion_receipt_11seed.json",
                 root / "SHA256SUMS_11SEED.txt"):
        if path.exists():
            raise RuntimeError(f"postseal report recovery destination already exists: {path}")
    payload = {
        "schema": "ect.q256.postseal-report-recovery/v1",
        "status": "AUTHOR_APPROVED", "authorized_at": experiment.utc_now(),
        "manual_postseal_report_recovery_index": 1,
        "protocol_sha256": experiment.sha256_file(protocol_path),
        "amendment_commit": head, "expected_evaluation_jobs": experiment.ELEVEN_JOB_COUNT,
        "evaluation_rerun_forbidden": True,
        "decoded_results_reused_without_redecode": True,
        "automatic_retry_count": 0,
        "failure_reason": "NumPy boolean was not JSON serializable",
        "eleven_seed_authorization_path": str(eleven_path),
        "eleven_seed_authorization_sha256": experiment.sha256_file(eleven_path),
        "evaluation_recovery_authorization_path": str(evaluation_path),
        "evaluation_recovery_authorization_sha256": experiment.sha256_file(evaluation_path),
        "matrix_seal_path": str(seal_path),
        "matrix_seal_sha256": experiment.sha256_file(seal_path),
        "decoded_results_path": str(decoded_path),
        "decoded_results_sha256": experiment.sha256_file(decoded_path),
        "failed_statistics_path": str(failed_statistics),
        "archive_statistics_path": str(archive_statistics),
        "archive_receipt_path": str(archive_receipt),
    }
    experiment.atomic_json(args.destination.resolve(), payload)
    print(json.dumps({"status": "AUTHOR_APPROVED",
                      "manual_postseal_report_recovery_index": 1}, sort_keys=True))


def authorize_v2(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve(strict=True)
    protocol = experiment.load_json(protocol_path)
    experiment.validate_protocol(protocol, protocol_path)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    if not experiment.HEX40.fullmatch(args.amendment_commit) or args.amendment_commit != head:
        raise RuntimeError("postseal report recovery v2 amendment is not current HEAD")
    prior_path = args.prior_authorization.resolve(strict=True)
    prior = validate_authorization(prior_path, protocol_path)
    if prior.get("manual_postseal_report_recovery_index") != 1:
        raise RuntimeError("postseal recovery v2 requires the exact v1 authorization")
    prior_receipt_path = args.prior_archive_receipt.resolve(strict=True)
    prior_receipt = experiment.load_json(prior_receipt_path)
    if (prior_receipt.get("status") != "PASS"
            or prior_receipt.get("postseal_report_recovery_authorization_sha256")
            != experiment.sha256_file(prior_path)
            or prior_receipt.get("evaluation_rerun_performed") is not False
            or prior_receipt.get("redecode_performed") is not False):
        raise RuntimeError("postseal recovery v2 requires the PASS v1 archive receipt")
    failure_log = args.failure_log.resolve(strict=True)
    if "evaluation recovery1 commit mismatch" not in failure_log.read_text(encoding="utf-8"):
        raise RuntimeError("postseal recovery v2 requires the exact v1 authorization-chain failure")
    root = Path(protocol["paths"]["formal_output_root"])
    statistics_path = root / "analysis_11seed_recovery1" / "statistics"
    if statistics_path.exists():
        raise RuntimeError("postseal recovery v2 requires an absent fresh statistics destination")
    archive_root = root / "archive" / "manual-postseal-report-recovery-2"
    archive_log = archive_root / "postseal_report_recovery1_failed.log"
    archive_receipt = archive_root / "postseal_report_recovery2_preparation_receipt.json"
    for path in (archive_root, archive_log, archive_receipt,
                 root / "REPORT_11SEED.md", root / "final_completion_receipt_11seed.json",
                 root / "SHA256SUMS_11SEED.txt"):
        if path.exists():
            raise RuntimeError(f"postseal report recovery v2 destination already exists: {path}")
    eleven_path = Path(prior["eleven_seed_authorization_path"])
    evaluation_path = Path(prior["evaluation_recovery_authorization_path"])
    payload = {
        "schema": "ect.q256.postseal-report-recovery/v1",
        "status": "AUTHOR_APPROVED", "authorized_at": experiment.utc_now(),
        "manual_postseal_report_recovery_index": 2,
        "protocol_sha256": experiment.sha256_file(protocol_path),
        "amendment_commit": head, "expected_evaluation_jobs": experiment.ELEVEN_JOB_COUNT,
        "evaluation_rerun_forbidden": True,
        "decoded_results_reused_without_redecode": True,
        "automatic_retry_count": 0,
        "failure_reason": "v1 postseal recovery inherited an obsolete current-commit gate",
        "eleven_seed_authorization_path": str(eleven_path),
        "eleven_seed_authorization_sha256": experiment.sha256_file(eleven_path),
        "evaluation_recovery_authorization_path": str(evaluation_path),
        "evaluation_recovery_authorization_sha256": experiment.sha256_file(evaluation_path),
        "matrix_seal_path": prior["matrix_seal_path"],
        "matrix_seal_sha256": prior["matrix_seal_sha256"],
        "decoded_results_path": prior["decoded_results_path"],
        "decoded_results_sha256": prior["decoded_results_sha256"],
        "prior_postseal_authorization_path": str(prior_path),
        "prior_postseal_authorization_sha256": experiment.sha256_file(prior_path),
        "prior_archive_receipt_path": str(prior_receipt_path),
        "prior_archive_receipt_sha256": experiment.sha256_file(prior_receipt_path),
        "failure_log_path": str(failure_log),
        "failure_log_sha256": experiment.sha256_file(failure_log),
        "archive_failure_log_path": str(archive_log),
        "archive_receipt_path": str(archive_receipt),
    }
    experiment.atomic_json(args.destination.resolve(), payload)
    print(json.dumps({"status": "AUTHOR_APPROVED",
                      "manual_postseal_report_recovery_index": 2}, sort_keys=True))


def prepare(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve(strict=True)
    authorization_path = args.authorization.resolve(strict=True)
    authorization = validate_authorization(authorization_path, protocol_path,
                                           require_commit=True)
    decoded_path = Path(authorization["decoded_results_path"])
    seal_path = Path(authorization["matrix_seal_path"])
    if (experiment.sha256_file(decoded_path) != authorization["decoded_results_sha256"]
            or experiment.sha256_file(seal_path) != authorization["matrix_seal_sha256"]):
        raise RuntimeError("sealed postseal inputs changed after authorization")
    index = authorization["manual_postseal_report_recovery_index"]
    if index == 1:
        source = Path(authorization["failed_statistics_path"])
        archive = Path(authorization["archive_statistics_path"])
        before = experiment.directory_manifest(source)
        archive.parent.mkdir(parents=True, exist_ok=False)
        os.rename(source, archive)
        after = experiment.directory_manifest(archive)
        if before != after:
            raise RuntimeError("failed statistics archive identity mismatch")
        file_count = len(after)
        manifest_sha = experiment.canonical_sha256(after)
    else:
        source = Path(authorization["failure_log_path"])
        archive = Path(authorization["archive_failure_log_path"])
        if experiment.sha256_file(source) != authorization["failure_log_sha256"]:
            raise RuntimeError("v1 postseal failure log changed after authorization")
        archive.parent.mkdir(parents=True, exist_ok=False)
        experiment.copy_exclusive(source, archive)
        file_count = 1
        manifest_sha = experiment.sha256_file(archive)
    receipt = {
        "schema": "ect.q256.postseal-report-recovery-archive/v1",
        "status": "PASS", "archived_at": experiment.utc_now(),
        "postseal_report_recovery_authorization_sha256":
            experiment.sha256_file(authorization_path),
        "source_path": str(source), "archive_path": str(archive),
        "file_count": file_count,
        "archive_identity_sha256": manifest_sha,
        "matrix_seal_sha256": authorization["matrix_seal_sha256"],
        "decoded_results_sha256": authorization["decoded_results_sha256"],
        "evaluation_rerun_performed": False, "redecode_performed": False,
    }
    experiment.atomic_json(Path(authorization["archive_receipt_path"]), receipt)
    print(json.dumps({"status": "PASS", "archived_files": file_count}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    auth = sub.add_parser("authorize")
    auth.add_argument("--protocol", type=Path, required=True)
    auth.add_argument("--eleven-seed-authorization", type=Path, required=True)
    auth.add_argument("--evaluation-recovery-authorization", type=Path, required=True)
    auth.add_argument("--decoded-results", type=Path, required=True)
    auth.add_argument("--failed-statistics", type=Path, required=True)
    auth.add_argument("--amendment-commit", required=True)
    auth.add_argument("--destination", type=Path, required=True)
    auth.set_defaults(func=authorize)
    auth_v2 = sub.add_parser("authorize-v2")
    auth_v2.add_argument("--protocol", type=Path, required=True)
    auth_v2.add_argument("--prior-authorization", type=Path, required=True)
    auth_v2.add_argument("--prior-archive-receipt", type=Path, required=True)
    auth_v2.add_argument("--failure-log", type=Path, required=True)
    auth_v2.add_argument("--amendment-commit", required=True)
    auth_v2.add_argument("--destination", type=Path, required=True)
    auth_v2.set_defaults(func=authorize_v2)
    prep = sub.add_parser("prepare")
    prep.add_argument("--protocol", type=Path, required=True)
    prep.add_argument("--authorization", type=Path, required=True)
    prep.set_defaults(func=prepare)
    return root


def main() -> int:
    args = parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
