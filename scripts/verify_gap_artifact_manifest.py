#!/usr/bin/env python3
"""Fail-closed consistency checks for the canonical Gap artifact manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STATE_K = {32.128: "032128", 64.128: "064128", 128.128: "128128", 256.0: "256000"}
REQUIRED_CHECKPOINT_FIELDS = {
    "id",
    "experiment_id",
    "trajectory_id",
    "single_uninterrupted_trajectory",
    "run_id",
    "training_seed",
    "arm",
    "gap",
    "lr",
    "kimg",
    "state_id",
    "optimizer_steps",
    "training_code_commit",
    "provenance_commit",
    "pr",
    "training_state",
    "optimizer_state",
    "network_snapshot",
}


class AuditFailure(RuntimeError):
    pass


def git_bytes(repo: Path, commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise AuditFailure(
            f"cannot read git:{commit}:{path}: {result.stderr.decode().strip()}"
        )
    return result.stdout


def git_paths(repo: Path, commit: str, root: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", commit, root],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise AuditFailure(f"cannot list git:{commit}:{root}")
    return [line for line in result.stdout.splitlines() if line]


def ensure_commit(repo: Path, commit: str) -> None:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode:
        raise AuditFailure(f"missing Git commit: {commit}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_sha(value: Any, label: str, allow_null: bool = False) -> None:
    if allow_null and value is None:
        return
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise AuditFailure(f"invalid SHA256 for {label}: {value!r}")


def check_ref(repo: Path, ref: dict[str, Any], label: str) -> bytes:
    data = git_bytes(repo, ref["commit"], ref["path"])
    observed = sha256(data)
    if observed != ref["sha256"]:
        raise AuditFailure(
            f"SHA256 mismatch for {label}: expected {ref['sha256']}, observed {observed}"
        )
    return data


def near(left: float, right: float, label: str, atol: float = 1e-12) -> None:
    if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol):
        raise AuditFailure(f"numeric mismatch for {label}: {left} != {right}")


def checkpoint_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = manifest.get("checkpoint_records", [])
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        missing = REQUIRED_CHECKPOINT_FIELDS - set(record)
        if missing:
            raise AuditFailure(f"checkpoint {record.get('id')} missing {sorted(missing)}")
        if record["id"] in index:
            raise AuditFailure(f"duplicate checkpoint id: {record['id']}")
        if "latest" in record["training_state"]["path"]:
            raise AuditFailure(f"canonical training state uses latest alias: {record['id']}")
        if "latest" in record["network_snapshot"]["path"]:
            raise AuditFailure(f"canonical network snapshot uses latest alias: {record['id']}")
        require_sha(record["training_state"]["sha256"], f"{record['id']} training state")
        require_sha(
            record["network_snapshot"]["sha256"],
            f"{record['id']} network snapshot",
            allow_null=True,
        )
        if record["optimizer_state"]["container"] != record["training_state"]["path"]:
            raise AuditFailure(f"optimizer state container mismatch: {record['id']}")
        index[record["id"]] = record
    return index


def check_longitudinal(
    repo: Path, manifest: dict[str, Any], checkpoints: dict[str, dict[str, Any]]
) -> int:
    bundle = manifest["evidence_bundles"]["same_trajectory_longitudinal"]
    commit = bundle["artifact_commit"]
    handoff = json.loads(check_ref(repo, bundle["handoff_receipt"], "Role D handoff"))
    summary_data = check_ref(repo, bundle["raw_summary"], "longitudinal summary")
    check_ref(repo, bundle["plotting_script"], "longitudinal plotting script")
    check_ref(repo, bundle["figure"], "longitudinal figure")

    ids = bundle["checkpoint_record_ids"]
    points = [checkpoints[item] for item in ids]
    identity_fields = ("experiment_id", "trajectory_id", "run_id", "training_seed", "arm", "gap", "lr")
    for field in identity_fields:
        values = {json.dumps(point[field], sort_keys=True) for point in points}
        if len(values) != 1:
            raise AuditFailure(f"longitudinal identity differs at field {field}: {values}")
    if not all(point["single_uninterrupted_trajectory"] for point in points):
        raise AuditFailure("longitudinal bundle includes a non-uninterrupted record")
    if [point["kimg"] for point in points] != sorted(point["kimg"] for point in points):
        raise AuditFailure("longitudinal kimg values are not increasing")
    if [point["optimizer_steps"] for point in points] != sorted(
        point["optimizer_steps"] for point in points
    ):
        raise AuditFailure("longitudinal optimizer steps are not increasing")

    trajectory = handoff["trajectory"]
    if trajectory["run_id"] != points[0]["run_id"] or trajectory["training_seed"] != 3:
        raise AuditFailure("handoff trajectory identity differs from manifest")
    if trajectory.get("single_uninterrupted_run") is not True:
        raise AuditFailure("handoff does not certify one uninterrupted run")
    handoff_artifacts = {item["state_id"]: item for item in handoff["artifacts"]}

    receipts: dict[float, dict[str, Any]] = {}
    for point in points:
        artifact = handoff_artifacts[point["state_id"]]
        if artifact["training_state"]["sha256"] != point["training_state"]["sha256"]:
            raise AuditFailure(f"handoff state hash mismatch: {point['id']}")
        if artifact["network_snapshot"]["sha256"] != point["network_snapshot"]["sha256"]:
            raise AuditFailure(f"handoff checkpoint hash mismatch: {point['id']}")
        suffix = STATE_K[point["kimg"]]
        path = f"analysis/same_trajectory_longitudinal/radam_update_audit_stateful_k{suffix}.json"
        receipt = json.loads(git_bytes(repo, commit, path))
        receipts[point["kimg"]] = receipt
        provenance = receipt["provenance"]
        if provenance["training_state_sha256"] != point["training_state"]["sha256"]:
            raise AuditFailure(f"audit state hash mismatch: {point['id']}")
        if provenance["checkpoint_sha256"] != point["network_snapshot"]["sha256"]:
            raise AuditFailure(f"audit checkpoint hash mismatch: {point['id']}")
        if provenance["training_state_meta"]["successful_optimizer_steps"] != point["optimizer_steps"]:
            raise AuditFailure(f"optimizer-step mismatch: {point['id']}")
        if receipt["source_state_non_committing"]["preserved"] is not True:
            raise AuditFailure(f"source state was not preserved: {point['id']}")

    rows = list(csv.DictReader(io.StringIO(summary_data.decode())))
    if len(rows) != 4:
        raise AuditFailure(f"expected four longitudinal summary rows, found {len(rows)}")
    for row in rows:
        kimg = float(row["K_kimg"])
        receipt = receipts[kimg]
        near(row["R_grad"], receipt["whole_model"]["R_grad"], f"R_grad@{kimg}")
        near(row["R_opt"], receipt["whole_model"]["R_opt"], f"R_opt@{kimg}")
        near(row["c_K_star"], receipt["whole_model"]["c_K_star"], f"c_K_star@{kimg}")

    artifact_manifest = json.loads(
        git_bytes(repo, commit, "analysis/same_trajectory_longitudinal/artifact_sha256.json")
    )
    base = "analysis/same_trajectory_longitudinal"
    for path, expected in artifact_manifest.items():
        observed = sha256(git_bytes(repo, commit, f"{base}/{path}"))
        if observed != expected:
            raise AuditFailure(f"PR #49 bundle manifest mismatch: {path}")
    return len(artifact_manifest)


def check_mechanism(
    repo: Path, manifest: dict[str, Any], checkpoints: dict[str, dict[str, Any]]
) -> int:
    bundle = manifest["evidence_bundles"]["prospective_scalar_history"]
    commit = bundle["artifact_commit"]
    paths = bundle["canonical_result_paths"]
    hashes = bundle["result_sha256"]
    ids = bundle["checkpoint_record_ids"]
    if not (len(paths) == len(hashes) == len(ids) == 4):
        raise AuditFailure("mechanism bundle must contain exactly four aligned points")
    for path, expected, checkpoint_id in zip(paths, hashes, ids):
        require_sha(expected, f"mechanism result {path}")
        raw = git_bytes(repo, commit, path)
        if sha256(raw) != expected:
            raise AuditFailure(f"mechanism result hash mismatch: {path}")
        result = json.loads(raw)
        checkpoint = checkpoints[checkpoint_id]
        if result["source_state_sha256"] != checkpoint["training_state"]["sha256"]:
            raise AuditFailure(f"mechanism source-state mismatch: {path}")
        near(result["source_nimg"] / 1000.0, checkpoint["kimg"], f"source nimg {path}")
        if result["source_optimizer_steps"] != checkpoint["optimizer_steps"]:
            raise AuditFailure(f"mechanism optimizer-step mismatch: {path}")
        if checkpoint["training_state"]["path"] not in result["execution_command"]:
            raise AuditFailure(f"mechanism command does not bind numbered state: {path}")
        for field in bundle["raw_arrays"]["hash_fields_in_result"]:
            require_sha(result.get(field), f"{path}:{field}")

    script = check_ref(repo, bundle["plotting_script"], "mechanism plotting script").decode()
    if 'analysis/real_history/k{k}/scalar_prediction.json' not in script:
        raise AuditFailure("mechanism plotter no longer reads canonical per-K results")
    if 'Path("analysis/real_history/scalar_prediction.json")' in script:
        raise AuditFailure("mechanism plotter reads forbidden mutable-latest result")
    check_ref(repo, bundle["figure"], "mechanism figure")

    forbidden = manifest["noncanonical_artifacts"][0]
    forbidden_raw = git_bytes(repo, forbidden["commit"], forbidden["path"])
    if sha256(forbidden_raw) != forbidden["sha256"]:
        raise AuditFailure("noncanonical duplicate hash changed")
    duplicate = json.loads(forbidden_raw)
    canonical_k256 = json.loads(git_bytes(repo, commit, paths[-1]))
    if duplicate["source_state_sha256"] == canonical_k256["source_state_sha256"]:
        raise AuditFailure("expected mutable-latest duplicate to have a distinct state hash")
    if "training-state-latest.pt" not in duplicate["execution_command"]:
        raise AuditFailure("noncanonical duplicate no longer documents its latest alias")
    resolution = forbidden["resolved_by"]
    tombstone = json.loads(
        git_bytes(repo, resolution["commit"], forbidden["path"])
    )
    if tombstone.get("status") != "NON_CANONICAL_TOMBSTONE":
        raise AuditFailure("mutable-latest duplicate was not replaced by a tombstone")
    if tombstone.get("canonical_result") != paths[-1]:
        raise AuditFailure("mechanism tombstone points to the wrong canonical result")
    if tombstone.get("superseded_payload_sha256") != forbidden["sha256"]:
        raise AuditFailure("mechanism tombstone does not bind the superseded payload")
    return len(paths)


def check_crossk_h20(
    repo: Path, manifest: dict[str, Any], checkpoints: dict[str, dict[str, Any]]
) -> tuple[int, int]:
    bundle = manifest["evidence_bundles"]["crossk_h20_scalar_history"]
    commit = bundle["artifact_commit"]
    ensure_commit(repo, commit)

    raw_hashes = bundle["raw_prediction_sha256"]
    if len(raw_hashes) != 16:
        raise AuditFailure("PR #58 h20 bundle must contain exactly 16 raw NPY arrays")
    expected_paths = {
        f"analysis/crossk_scalar_history/{label}/raw_predictions/{name}.npy"
        for label in ("k32", "k64", "k128", "k256")
        for name in (
            "a_star_series",
            "h_actual_h20",
            "h_pred_scalar_h20",
            "weights_h20",
        )
    }
    if set(raw_hashes) != expected_paths:
        raise AuditFailure("PR #58 h20 raw-array path set is incomplete or unexpected")
    for path, expected in raw_hashes.items():
        require_sha(expected, f"PR #58 raw array {path}")
        observed = sha256(git_bytes(repo, commit, path))
        if observed != expected:
            raise AuditFailure(f"PR #58 raw-array hash mismatch: {path}")

    summary_ref = {"commit": commit, **bundle["summary"]}
    summary = json.loads(check_ref(repo, summary_ref, "PR #58 cross-K summary"))
    for label, expected in bundle["headline_weighted_r2"].items():
        rows = [
            row
            for row in summary[label]["horizons"]
            if row["horizon_steps"] == 20
        ]
        if len(rows) != 1:
            raise AuditFailure(f"PR #58 summary has no unique h20 row for {label}")
        near(rows[0]["weighted_R2"], expected, f"PR #58 h20 R2 {label}")

    for field, label in (
        ("recompute_test", "PR #58 h20 recompute test"),
        ("plotting_script", "PR #58 cross-K plotting script"),
    ):
        check_ref(repo, {"commit": commit, **bundle[field]}, label)
    for index, figure in enumerate(bundle["h20_figures"]):
        check_ref(repo, {"commit": commit, **figure}, f"PR #58 h20 figure {index}")

    external = bundle["full_matrix_external_raw"]
    external_manifest = json.loads(
        check_ref(
            repo,
            {"commit": commit, **external["manifest"]},
            "PR #58 full-matrix external manifest",
        )
    )
    label_to_record = dict(
        zip(
            ("k32", "k64", "k128", "k256"),
            (checkpoints[item] for item in bundle["checkpoint_record_ids"]),
        )
    )
    external_records = 0
    for label, record in label_to_record.items():
        checkpoint = external_manifest["checkpoints"][label]
        if checkpoint["sha256"] != record["training_state"]["sha256"]:
            raise AuditFailure(f"PR #58 checkpoint hash mismatch: {label}")
        for field in ("path", "sha256", "size_bytes"):
            if field not in checkpoint:
                raise AuditFailure(f"PR #58 checkpoint locator missing {field}: {label}")
        require_sha(checkpoint["sha256"], f"PR #58 checkpoint {label}")
        if not checkpoint["path"].startswith("/") or checkpoint["size_bytes"] <= 0:
            raise AuditFailure(f"PR #58 checkpoint locator malformed: {label}")
        external_records += 1
        for name, item in external_manifest["raw_inputs"][label].items():
            if not isinstance(item, dict) or "kind" not in item:
                continue
            for field in ("path", "sha256", "size_bytes"):
                if field not in item:
                    raise AuditFailure(f"PR #58 raw locator missing {field}: {label}/{name}")
            require_sha(item["sha256"], f"PR #58 external raw {label}/{name}")
            if not item["path"].startswith("/") or item["size_bytes"] <= 0:
                raise AuditFailure(f"PR #58 raw locator malformed: {label}/{name}")
            external_records += 1
    if external_records != 30:
        raise AuditFailure(
            f"PR #58 external manifest expected 30 hash-bound records, found {external_records}"
        )
    if external.get("durable_external_locator") is not None:
        raise AuditFailure("B002 scope changed; re-audit the claimed durable locator")
    return len(raw_hashes), external_records


def parse_checksum_manifest(data: bytes) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(data.decode().splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise AuditFailure(f"invalid checksum line {line_number}")
        digest, path = match.groups()
        if path in checksums:
            raise AuditFailure(f"duplicate checksum path: {path}")
        checksums[path] = digest
    return checksums


def check_disjoint(
    repo: Path, manifest: dict[str, Any], checkpoints: dict[str, dict[str, Any]]
) -> tuple[int, int]:
    bundle = manifest["evidence_bundles"]["disjoint_fid_kid_5k"]
    commit = bundle["artifact_commit"]
    root = bundle["raw_metric_root"]
    manifest_bytes = git_bytes(repo, commit, bundle["raw_metric_manifest"]["path"])
    if sha256(manifest_bytes) != bundle["raw_metric_manifest"]["sha256"]:
        raise AuditFailure("PR #53 checksum manifest hash mismatch")
    checksums = parse_checksum_manifest(manifest_bytes)
    for relative_path, expected in checksums.items():
        observed = sha256(git_bytes(repo, commit, f"{root}/{relative_path}"))
        if observed != expected:
            raise AuditFailure(f"PR #53 raw artifact hash mismatch: {relative_path}")

    metric_paths = [path for path in checksums if path.startswith("blocks/")]
    if len(metric_paths) != bundle["raw_metric_files"]:
        raise AuditFailure("PR #53 metric-file count mismatch")
    record_by_seed_arm = {
        (record["training_seed"], record["arm"].lower()): record
        for record in (checkpoints[item] for item in bundle["checkpoint_record_ids"])
    }
    path_re = re.compile(
        r"blocks/block_(\d+)_(\d+)/seed([345])/arm_([abc])/"
        r"metric-(fid5k_full|kid5k_full)\.jsonl"
    )
    cell_keys: set[tuple[int, int, int, str]] = set()
    for relative_path in metric_paths:
        match = path_re.fullmatch(relative_path)
        if match is None:
            raise AuditFailure(f"unexpected PR #53 metric path: {relative_path}")
        start, end, seed, arm, metric = match.groups()
        seed_i = int(seed)
        raw = json.loads(git_bytes(repo, commit, f"{root}/{relative_path}"))
        if raw["metric"] != metric:
            raise AuditFailure(f"metric path/content mismatch: {relative_path}")
        checkpoint = record_by_seed_arm[(seed_i, arm)]
        snapshot_path = raw.get("snapshot_pkl", "")
        if checkpoint["run_id"] not in snapshot_path:
            raise AuditFailure(f"metric run/checkpoint mismatch: {relative_path}")
        if not snapshot_path.endswith("/network-snapshot-000008.pkl"):
            raise AuditFailure(f"metric used non-numbered checkpoint: {relative_path}")
        cell_keys.add((int(start), int(end), seed_i, arm))
    if len(cell_keys) != bundle["evaluation_cells"]:
        raise AuditFailure("PR #53 evaluation-cell count mismatch")

    for field in ("blockwise_csv", "summary_csv", "decision"):
        ref = {"commit": commit, **bundle[field]}
        check_ref(repo, ref, f"PR #53 {field}")

    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from rebuild_disjoint_5k_summary import verify_committed_tables

    verify_committed_tables(repo, commit)
    aggregation = bundle["aggregation_script"]
    ensure_commit(repo, aggregation["commit"])
    git_bytes(repo, aggregation["commit"], aggregation["path"])
    return len(checksums), len(cell_keys)


def run_audit(repo: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise AuditFailure("unsupported manifest schema")
    if manifest["policy"].get("new_hypotheses_allowed") is not False:
        raise AuditFailure("manifest does not freeze new hypothesis exploration")
    for number, pr in manifest["pull_requests"].items():
        ensure_commit(repo, pr["head_commit"])
        if "merge_commit" in pr:
            ensure_commit(repo, pr["merge_commit"])
        if "integration_commit" in pr:
            ensure_commit(repo, pr["integration_commit"])
        if int(number) not in (47, 49, 50, 51, 52, 53, 58):
            raise AuditFailure(f"unexpected PR in target set: {number}")
    checkpoints = checkpoint_index(manifest)
    longitudinal_artifacts = check_longitudinal(repo, manifest, checkpoints)
    mechanism_points = check_mechanism(repo, manifest, checkpoints)
    crossk_h20_arrays, crossk_external_records = check_crossk_h20(
        repo, manifest, checkpoints
    )
    disjoint_artifacts, disjoint_cells = check_disjoint(repo, manifest, checkpoints)
    blockers = manifest.get("blocking_findings", [])
    return {
        "manifest_id": manifest["manifest_id"],
        "verification_status": "ANALYZED",
        "structural_checks": "PASS",
        "pull_requests_checked": len(manifest["pull_requests"]),
        "checkpoint_records_checked": len(checkpoints),
        "longitudinal_bundle_artifacts_checked": longitudinal_artifacts,
        "mechanism_points_checked": mechanism_points,
        "crossk_h20_raw_arrays_checked": crossk_h20_arrays,
        "crossk_external_records_checked": crossk_external_records,
        "disjoint_bundle_artifacts_checked": disjoint_artifacts,
        "disjoint_evaluation_cells_checked": disjoint_cells,
        "blocking_findings": blockers,
        "publication_ready": not blockers,
    }


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo / "evidence" / "gap_artifact_manifest_v1.json",
    )
    parser.add_argument("--require-publication-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_audit(args.repo.resolve(), args.manifest.resolve())
    except (AuditFailure, KeyError, ValueError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("PASS: structural lineage and committed hashes are internally consistent")
        print(
            f"checked {report['checkpoint_records_checked']} checkpoint records, "
            f"{report['crossk_h20_raw_arrays_checked']} PR #58 h20 arrays, "
            f"{report['disjoint_bundle_artifacts_checked']} PR #53 artifacts, and "
            f"{report['disjoint_evaluation_cells_checked']} evaluation cells"
        )
        print(f"publication_ready={str(report['publication_ready']).lower()}")
        for finding in report["blocking_findings"]:
            print(f"{finding['id']} [{finding['severity']}] {finding['detail']}")
    if args.require_publication_ready and not report["publication_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
