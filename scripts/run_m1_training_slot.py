#!/usr/bin/env python3
"""Run one four-branch M1 roster slot in its frozen rotation order."""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from analysis.q256_optimizer_restart_ema_rebuild_v1 import verify_crn
from scripts import build_m1_evaluation_slots as evaluation_slots
from scripts import seal_m1_g4_canaries as g4_sealer
from scripts.run_m1_training_gates import (
    REPO_ROOT, acquire_gpu_lock, command, environment, load_training_manifest,
    probe_gpu_runtime,
    prepare_resume_history, source_for, validate_branch_manifest,
    validate_baseline_repo, validate_gate_seed_artifacts,
    validate_recorded_runtime_probe, write_branch_manifest,
)
from training import m1, reproducibility, schedule_switch
from training import ct_training_loop


ATTEMPT_SCHEMA = "ect.m1.training-attempt/v1"
ATTEMPT_LAUNCH_SCHEMA = "ect.m1.training-attempt-launch/v1"
ATTEMPT_LAUNCH_PREFIX = "M1_ATTEMPT_LAUNCH "
SCIENTIFIC_FLOATING_POINT_MESSAGES = {
    "factorial source times must be finite and positive",
    "factorial base times must be finite and satisfy 0 <= r <= t",
    "non-finite realized factorial time or gap",
    "target realized times must satisfy 0 <= r <= t",
    "denominator realized times must satisfy 0 <= r <= t",
    "target realized gaps must be strictly positive",
    "denominator realized gaps must be strictly positive",
    "non-finite RAdam moment state",
    "non-finite online-EMA distance",
}


def scientific_floating_point(lines) -> bool:
    prefix = "FloatingPointError: "
    for raw in lines:
        line = re.sub(r"^\[rank0\]:\s*", "", raw)
        if not line.startswith(prefix):
            continue
        message = line[len(prefix):]
        if (
            message in SCIENTIFIC_FLOATING_POINT_MESSAGES
            or message.startswith("strict factorial training invariant failure:")
        ):
            return True
    return False


def load_attempt_receipts(run_dir: Path, branch: str, seed: int) -> list[dict]:
    logs = sorted(run_dir.glob("formal-attempt-*.log"))
    paths = sorted(run_dir.glob("formal-attempt-*.json"))
    if len(logs) != len(paths):
        raise RuntimeError(f"M1 attempt log/receipt count mismatch: {run_dir}")
    receipts = []
    for index, path in enumerate(paths, start=1):
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("schema") != ATTEMPT_SCHEMA
            or value.get("branch") != branch
            or value.get("seed") != seed
            or value.get("attempt_index") != index
            or value.get("log_path") != str(logs[index - 1].resolve())
            or not Path(value.get("resume_path", "")).is_absolute()
            or re.fullmatch(
                r"[0-9a-f]{64}", str(value.get("resume_sha256", ""))
            ) is None
            or not isinstance(value.get("launch_command"), list)
            or not value["launch_command"]
            or value.get("status") not in {
                "PASS", "SCIENTIFIC_FAILURE", "INCOMPLETE_TECHNICAL"
            }
        ):
            raise RuntimeError(f"invalid M1 attempt receipt: {path}")
        receipts.append(value)
    return receipts


def write_attempt_launch_header(
    handle, branch: str, seed: int, attempt_index: int, resume: Path,
    command_value: list[str],
) -> dict:
    resume = resume.resolve(strict=True)
    if (
        command_value != ["TERMINAL_REVALIDATION"]
        and f"--resume={resume}" not in command_value
    ):
        raise RuntimeError("M1 launch command does not bind its resume checkpoint")
    value = {
        "schema": ATTEMPT_LAUNCH_SCHEMA,
        "seed": seed,
        "branch": branch,
        "attempt_index": attempt_index,
        "resume_path": str(resume),
        "resume_sha256": schedule_switch.sha256_file(str(resume)),
        "command": command_value,
    }
    handle.write(
        (ATTEMPT_LAUNCH_PREFIX + json.dumps(value, sort_keys=True) + "\n").encode()
    )
    handle.flush()
    os.fsync(handle.fileno())
    return value


def read_attempt_launch_header(
    log: Path, branch: str, seed: int, attempt_index: int,
) -> dict:
    with log.open("rb") as handle:
        line = handle.readline().decode("utf-8")
    if not line.startswith(ATTEMPT_LAUNCH_PREFIX):
        raise RuntimeError(f"M1 attempt log lacks its launch provenance: {log}")
    value = json.loads(line[len(ATTEMPT_LAUNCH_PREFIX):])
    if (
        value.get("schema") != ATTEMPT_LAUNCH_SCHEMA
        or value.get("seed") != seed
        or value.get("branch") != branch
        or value.get("attempt_index") != attempt_index
        or not Path(value.get("resume_path", "")).is_absolute()
        or re.fullmatch(r"[0-9a-f]{64}", str(value.get("resume_sha256", ""))) is None
        or not isinstance(value.get("command"), list)
        or not value["command"]
    ):
        raise RuntimeError(f"invalid M1 attempt launch provenance: {log}")
    return value


def write_attempt_receipt(
    run_dir: Path, branch: str, seed: int, attempt_index: int, log: Path,
    resume: Path, status: str, exit_code, reason: str,
    rollback_evidence: Path = None,
) -> None:
    branch_manifest_path = run_dir / "formal_run_manifest.json"
    branch_manifest = json.loads(branch_manifest_path.read_text(encoding="utf-8"))
    launch = read_attempt_launch_header(log, branch, seed, attempt_index)
    if resume is not None and launch["resume_path"] != str(resume.resolve()):
        raise RuntimeError("M1 receipt resume differs from its launch provenance")
    reproducibility.atomic_json_dump(
        {
            "schema": ATTEMPT_SCHEMA,
            "status": status,
            "reason": reason,
            "seed": seed,
            "branch": branch,
            "attempt_index": attempt_index,
            "resume_path": launch["resume_path"],
            "resume_sha256": launch["resume_sha256"],
            "launch_command": launch["command"],
            "log_path": str(log.resolve()),
            "log_sha256": schedule_switch.sha256_file(str(log.resolve())),
            "branch_manifest_path": str(branch_manifest_path.resolve()),
            "branch_manifest_sha256": schedule_switch.sha256_file(
                str(branch_manifest_path.resolve())
            ),
            "training_manifest_sha256": branch_manifest[
                "training_manifest_sha256"
            ],
            "frozen_source_state_sha256": branch_manifest["source_state"][
                "sha256"
            ],
            "exit_code": exit_code,
            "rollback_evidence": (
                None if rollback_evidence is None
                else str(rollback_evidence.resolve())
            ),
        },
        run_dir / f"formal-attempt-{attempt_index}.json",
        overwrite=False,
    )


def recover_interrupted_attempt(
    run_dir: Path, branch: str, seed: int, *, authorized: bool
) -> None:
    logs = sorted(run_dir.glob("formal-attempt-*.log"))
    receipts = sorted(run_dir.glob("formal-attempt-*.json"))
    if len(logs) == len(receipts):
        return
    if not authorized or len(logs) != len(receipts) + 1:
        raise RuntimeError(f"M1 attempt log/receipt count mismatch: {run_dir}")
    index = len(logs)
    lines = logs[-1].read_text(encoding="utf-8", errors="replace").splitlines()
    scientific = scientific_floating_point(lines)
    write_attempt_receipt(
        run_dir, branch, seed, index, logs[-1], None,
        "SCIENTIFIC_FAILURE" if scientific else "INCOMPLETE_TECHNICAL",
        None,
        "NUMERIC_FLOATING_POINT" if scientific else "LAUNCHER_INTERRUPTED",
    )


def _rewrite_csv_to_attempt(
    path: Path, expected_fields, last_attempt: int, first_attempt: int
) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise RuntimeError(f"retry CSV schema mismatch: {path}")
        rows = [
            row for row in reader
            if int(row["attempted_iteration"]) <= last_attempt
        ]
    attempts = [int(row["attempted_iteration"]) for row in rows]
    if attempts != list(range(first_attempt, last_attempt + 1)):
        raise RuntimeError(f"retry CSV does not cover checkpoint boundary: {path}")
    temporary = path.with_name(f".{path.name}.retry.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=expected_fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def rollback_working_history(
    run_dir: Path, boundary: int, retry_index: int, rejected_states=()
) -> Path:
    evidence = run_dir / f"retry-evidence-{retry_index}"
    evidence.mkdir(exist_ok=False)
    names = (
        "train_summary.csv", "schedule_switch_training_telemetry_v1.csv",
        "training_options.json", "log.txt",
    )
    for name in names:
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, evidence / name)
    rejected_paths = {Path(row["path"]).resolve() for row in rejected_states}
    moved_states = []
    for state_path in run_dir.glob("training-state-*.pt"):
        match = re.fullmatch(r"training-state-kimg(\d{6})\.pt", state_path.name)
        future = match is not None and int(match.group(1)) * 1000 // 128 > boundary
        if state_path.resolve() in rejected_paths or future:
            destination = evidence / state_path.name
            moved_states.append({
                "path": str(state_path.resolve()),
                "sha256": schedule_switch.sha256_file(str(state_path)),
                "destination": str(destination.resolve()),
                "reason": "REJECTED" if state_path.resolve() in rejected_paths else "FUTURE",
            })
            shutil.move(str(state_path), destination)
    if moved_states or rejected_states:
        reproducibility.atomic_json_dump(
            {"rejected": list(rejected_states), "moved": moved_states},
            evidence / "resume_state_rollback.json", overwrite=False,
        )
    summary = run_dir / "train_summary.csv"
    _rewrite_csv_to_attempt(
        summary, ct_training_loop._TRAIN_SUMMARY_FIELDS, boundary, 1
    )
    telemetry = run_dir / "schedule_switch_training_telemetry_v1.csv"
    if boundary == schedule_switch.SWITCH_ATTEMPT:
        if telemetry.exists():
            telemetry.unlink()
    else:
        _rewrite_csv_to_attempt(
            telemetry, ct_training_loop._M1_SCHEDULE_SWITCH_TELEMETRY_FIELDS,
            boundary, schedule_switch.SWITCH_ATTEMPT + 1,
        )
    return evidence


def load_and_validate(run_dir: Path, manifest_path: Path) -> dict:
    manifest = schedule_switch.load_run_manifest(manifest_path)
    milestones = {}
    for kimg in (640, 768, 896, 1024):
        path = run_dir / f"training-state-kimg{kimg:06d}.pt"
        state = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
        schedule_switch.verify_switched_state(state, manifest)
        m1.validate_resumed_state(state, manifest)
        expected_nimg = kimg * 1000
        if (
            int(state.get("cur_nimg", -1)) != expected_nimg
            or int(state.get("attempted_iteration", -1)) != expected_nimg // 128
        ):
            raise RuntimeError(f"M1 milestone progress mismatch: {path}")
        if kimg == 1024:
            m1.validate_terminal_state(state, manifest)
        milestones[str(kimg)] = {
            "state_path": str(path.resolve()),
            "state_sha256": schedule_switch.sha256_file(str(path)),
            "attempted_iteration": int(state["attempted_iteration"]),
            "cur_nimg": int(state["cur_nimg"]),
        }
    return {"status": "PASS", "milestones": milestones}


def select_resume_state(candidates, manifest: dict):
    rejected = []
    selected = None
    for path, named_attempt in candidates:
        if not path.is_file():
            continue
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
            schedule_switch.verify_switched_state(state, manifest)
            m1.validate_resumed_state(state, manifest)
            attempt = int(state.get("attempted_iteration", -1))
            if (
                attempt < schedule_switch.SWITCH_ATTEMPT
                or attempt > 8000
                or int(state.get("cur_nimg", -1)) != attempt * 128
                or (named_attempt is not None and attempt != named_attempt)
            ):
                raise RuntimeError("resume progress does not match checkpoint identity")
            candidate = (attempt, named_attempt is not None, path, state)
            if selected is None or candidate[:2] > selected[:2]:
                selected = candidate
        except Exception as exc:
            rejected.append({"path": str(path.resolve()), "reason": str(exc)})
    if selected is None:
        return None, None, rejected
    return selected[2], selected[3], rejected


def validate_training_gate_receipt(path: Path, training: dict) -> dict:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected_seeds = [row["seed"] for row in training["roster"][:2]]
    checks = receipt.get("seeds", [])
    hex64 = re.compile(r"^[0-9a-f]{64}$")
    normalized_keys = {
        *(f"{branch}_continuous32" for branch in ("K_A", "K_B", "R_A", "R_B")),
        "K_A_shadow_on_off", "K_B_shadow_on_off",
        "K_A_legacy_resume", "K_B_legacy_resume",
    }
    details_valid = all(
        hex64.fullmatch(str(row.get("crn_sha256", ""))) is not None
        and set(row.get("manifest_sha256_by_branch", {}))
        == {"K_A", "K_B", "R_A", "R_B"}
        and set(row.get("baseline_manifest_sha256_by_branch", {})) == {"K_A", "K_B"}
        and set(row.get("normalized_state_sha256", {})) == normalized_keys
        and set(row.get("fresh_radam_step_sha256", {})) == {"R_A", "R_B"}
        and all(
            hex64.fullmatch(str(digest)) is not None
            for field in (
                "manifest_sha256_by_branch", "baseline_manifest_sha256_by_branch",
                "normalized_state_sha256", "fresh_radam_step_sha256",
            )
            for digest in row[field].values()
        )
        for row in checks
    )
    if (
        receipt.get("schema") != "ect.m1.training-gates/v1"
        or receipt.get("status") != "PASS"
        or receipt.get("training_manifest_sha256")
        != training["_training_manifest_sha256"]
        or receipt.get("baseline", {}).get("head")
        != "890a85a8ef4d9effb48f653111a70b5f15b249de"
        or [row.get("seed") for row in checks] != expected_seeds
        or any(row.get("status") != "PASS" for row in checks)
        or not details_valid
    ):
        raise RuntimeError("M1 G1-G3 receipt does not bind this frozen manifest")
    baseline = receipt["baseline"]
    validate_baseline_repo(Path(baseline.get("repo", "")))
    pause_patch = Path(baseline.get("pause_patch_entry", ""))
    if (
        pause_patch.is_symlink()
        or not pause_patch.is_file()
        or schedule_switch.sha256_file(str(pause_patch.resolve()))
        != baseline.get("pause_patch_sha256")
    ):
        raise RuntimeError("M1 G1-G3 baseline pause artifact mismatch")
    for check, row in zip(checks, training["roster"][:2]):
        validate_gate_seed_artifacts(check, training, row)
    validate_recorded_runtime_probe(training, receipt.get("hardware_probe"))
    return receipt


def validate_g4_receipt(
    path: Path, training: dict, evaluation_manifest: Path,
    training_gate_receipt: Path,
) -> dict:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != "ect.m1.g4-canary-seal/v1"
        or receipt.get("status") != "PASS"
        or receipt.get("protocol_id") != m1.PROTOCOL_ID
        or receipt.get("quality_eligible") is not False
        or receipt.get("quality_generation") is not False
        or receipt.get("quality_metrics_executed") is not False
        or receipt.get("training_manifest_sha256")
        != training["_training_manifest_sha256"]
        or receipt.get("implementation_commit") != training["implementation_commit"]
        or receipt.get("evaluation_manifest_sha256")
        != schedule_switch.sha256_file(str(evaluation_manifest))
        or receipt.get("training_gates_receipt_sha256")
        != schedule_switch.sha256_file(str(training_gate_receipt))
        or receipt.get("evaluator_commit")
        != "d6aba02fb88e9db0993623895eb2228ed717d810"
        or receipt.get("canary_count") != 5
    ):
        raise RuntimeError("M1 G4 receipt does not bind this frozen launch")
    child_paths = [
        Path(row.get("path", "")) for row in receipt.get("canary_receipts", [])
        if isinstance(row, dict)
    ]
    parents = {path.parent.resolve() for path in child_paths}
    if len(child_paths) != 5 or len(parents) != 1:
        raise RuntimeError("M1 G4 seal does not identify five co-located canaries")
    training_identity = evaluation_slots.load_training_identity(
        Path(training["_training_manifest_path"])
    )
    rebuilt = g4_sealer.seal(
        training_identity, training_gate_receipt,
        schedule_switch.sha256_file(str(evaluation_manifest)), parents.pop()
    )
    if rebuilt != receipt:
        raise RuntimeError("M1 G4 seal differs from its five validated canaries")
    return receipt


def run_branch(
    training: dict, row: dict, branch: str, gpu: int, *, technical_retry: bool
) -> dict:
    slot_root = Path(training["output_root"]) / row["roster_slot"]
    run_dir = slot_root / branch
    manifest_path = run_dir / "formal_run_manifest.json"
    latest = run_dir / "training-state-latest.pt"
    branch_init = run_dir / "training-state-kimg000512.pt"
    source = source_for(row, branch)
    rollback_evidence = None
    if run_dir.exists():
        resume_candidates = [
            (latest, None),
            *((run_dir / f"training-state-kimg{kimg:06d}.pt", kimg * 1000 // 128)
              for kimg in (1024, 896, 768, 640)),
            (branch_init, schedule_switch.SWITCH_ATTEMPT),
        ]
        if not manifest_path.is_file():
            raise RuntimeError(f"incomplete existing M1 run directory: {run_dir}")
        manifest = validate_branch_manifest(
            manifest_path, training, row, branch, run_dir, shadow=True
        )
        m1.verify_source_artifacts(manifest)
        resume, state, resume_rejections = select_resume_state(
            resume_candidates, manifest
        )
        recover_interrupted_attempt(
            run_dir, branch, row["seed"], authorized=technical_retry,
        )
        receipts = load_attempt_receipts(run_dir, branch, row["seed"])
        if receipts and receipts[-1]["status"] == "SCIENTIFIC_FAILURE":
            return {
                "status": "SCIENTIFIC_FAILURE",
                "reason": receipts[-1]["reason"],
                "attempt_receipt": str(
                    run_dir / f"formal-attempt-{len(receipts)}.json"
                ),
            }
        if resume is None:
            if not (
                technical_retry and receipts
                and receipts[-1]["status"] == "INCOMPLETE_TECHNICAL"
            ):
                raise RuntimeError(
                    f"existing M1 run has no resumable full state: {run_dir}"
                )
            resume = Path(source["source_state_path"])
        if state is None:
            state = torch.load(resume, map_location="cpu", weights_only=False)
        if int(state.get("attempted_iteration", -1)) == 8000:
            if receipts and receipts[-1]["status"] == "PASS":
                return load_and_validate(run_dir, manifest_path)
            if not (
                technical_retry and receipts
                and receipts[-1]["status"] == "INCOMPLETE_TECHNICAL"
            ):
                raise RuntimeError("terminal M1 state lacks a successful attempt receipt")
            result_record = load_and_validate(run_dir, manifest_path)
            log_index = len(receipts) + 1
            log = run_dir / f"formal-attempt-{log_index}.log"
            with log.open("xb") as handle:
                write_attempt_launch_header(
                    handle, branch, row["seed"], log_index, resume,
                    ["TERMINAL_REVALIDATION"],
                )
                handle.write(b"terminal state revalidated after technical interruption\n")
            write_attempt_receipt(
                run_dir, branch, row["seed"], log_index, log, resume,
                "PASS", 0, "TERMINAL_REVALIDATED",
            )
            return result_record
        if receipts and receipts[-1]["status"] == "INCOMPLETE_TECHNICAL":
            if not technical_retry:
                raise RuntimeError(
                    f"M1 technical retry requires --technical-retry: {run_dir}"
                )
        elif receipts:
            raise RuntimeError(f"nonterminal M1 run has invalid prior status: {run_dir}")
        boundary = int(state.get("attempted_iteration", -1))
        if resume.resolve() == Path(manifest["source_state"]["path"]).resolve():
            schedule_switch.verify_source_state(state, manifest)
        else:
            m1.validate_resumed_state(state, manifest)
            if boundary == schedule_switch.SWITCH_ATTEMPT:
                source_state = torch.load(
                    manifest["source_state"]["path"],
                    map_location="cpu", weights_only=False,
                )
                m1.validate_branch_init_against_source(
                    state, source_state, manifest
                )
        if receipts and receipts[-1]["status"] == "INCOMPLETE_TECHNICAL":
            rollback_evidence = rollback_working_history(
                run_dir, boundary, len(receipts) + 1, resume_rejections
            )
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        prepare_resume_history(Path(source["source_state_path"]).parent, run_dir)
        manifest_path = write_branch_manifest(
            training, row, branch, run_dir, shadow=True
        )
        m1.verify_source_artifacts(
            schedule_switch.load_run_manifest(manifest_path)
        )
        resume = Path(source["source_state_path"])

    prior_receipts = load_attempt_receipts(run_dir, branch, row["seed"])
    log_index = len(prior_receipts) + 1
    if log_index > 3:
        raise RuntimeError(f"M1 run exhausted its initial attempt plus two retries: {run_dir}")
    log = run_dir / f"formal-attempt-{log_index}.log"
    launch_command = command(training, row, manifest_path, resume)
    with log.open("xb") as handle:
        write_attempt_launch_header(
            handle, branch, row["seed"], log_index, resume, launch_command
        )
        result = subprocess.run(
            launch_command,
            cwd=REPO_ROOT,
            env=environment(gpu, runtime_python=Path(training["runtime_python"])),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        scientific = scientific_floating_point(lines)
        status = "SCIENTIFIC_FAILURE" if scientific else "INCOMPLETE_TECHNICAL"
        reason = (
            "NUMERIC_FLOATING_POINT"
            if scientific else "NONZERO_EXIT"
        )
        write_attempt_receipt(
            run_dir, branch, row["seed"], log_index, log, resume,
            status, result.returncode, reason, rollback_evidence,
        )
        if scientific:
            return {
                "status": status, "reason": reason,
                "attempt_receipt": str(
                    run_dir / f"formal-attempt-{log_index}.json"
                ),
            }
        raise RuntimeError(f"M1 formal training failed: {run_dir}")
    try:
        result_record = load_and_validate(run_dir, manifest_path)
    except Exception:
        write_attempt_receipt(
            run_dir, branch, row["seed"], log_index, log, resume,
            "INCOMPLETE_TECHNICAL", result.returncode,
            "POST_RUN_VALIDATION_FAILED", rollback_evidence,
        )
        raise
    write_attempt_receipt(
        run_dir, branch, row["seed"], log_index, log, resume,
        "PASS", result.returncode, "COMPLETED", rollback_evidence,
    )
    return result_record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--slot", required=True, help="S01 through S16")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--technical-retry", action="store_true")
    parser.add_argument("--training-gates-receipt", type=Path, required=True)
    parser.add_argument("--g4-receipt", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.training_manifest.resolve(strict=True)
    training = load_training_manifest(manifest_path)
    gate_receipt_path = args.training_gates_receipt.resolve(strict=True)
    validate_training_gate_receipt(gate_receipt_path, training)
    g4_receipt_path = args.g4_receipt.resolve(strict=True)
    evaluation_manifest = args.evaluation_manifest.resolve(strict=True)
    validate_g4_receipt(
        g4_receipt_path, training, evaluation_manifest, gate_receipt_path
    )
    matches = [row for row in training["roster"] if row["roster_slot"] == args.slot]
    if len(matches) != 1:
        raise RuntimeError("slot must identify exactly one frozen roster row")
    row = matches[0]
    slot_root = Path(training["output_root"]) / row["roster_slot"]
    slot_root.mkdir(parents=True, exist_ok=True)
    receipt_path = slot_root / "training_receipt.json"
    if receipt_path.exists():
        raise RuntimeError("slot already has a training receipt")
    gpu_lock = acquire_gpu_lock(training, args.gpu)
    hardware = probe_gpu_runtime(training, args.gpu)
    launch_index = len(list(slot_root.glob("launch-*.json"))) + 1
    reproducibility.atomic_json_dump(
        {
            "schema": "ect.m1.training-launch/v1",
            "training_manifest_path": str(manifest_path),
            "training_manifest_sha256": training["_training_manifest_sha256"],
            "implementation_commit": training["implementation_commit"],
            "training_gates_receipt_path": str(gate_receipt_path),
            "training_gates_receipt_sha256": schedule_switch.sha256_file(
                str(gate_receipt_path)
            ),
            "g4_receipt_path": str(g4_receipt_path),
            "g4_receipt_sha256": schedule_switch.sha256_file(
                str(g4_receipt_path)
            ),
            "evaluation_manifest_path": str(evaluation_manifest),
            "evaluation_manifest_sha256": schedule_switch.sha256_file(
                str(evaluation_manifest)
            ),
            "roster_slot": row["roster_slot"], "seed": row["seed"],
            "gpu": args.gpu, "technical_retry": args.technical_retry,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "hardware_probe": hardware,
        },
        slot_root / f"launch-{launch_index:03d}.json", overwrite=False,
    )
    branches = {}
    for branch in row["order"]:
        branches[branch] = run_branch(
            training, row, branch, args.gpu,
            technical_retry=args.technical_retry,
        )
    crn = {}
    for label, pair in (("R", ("R_A", "R_B")), ("K", ("K_A", "K_B"))):
        if all(branches[name]["status"] == "PASS" for name in pair):
            paths = {
                name: slot_root / name / "schedule_switch_training_telemetry_v1.csv"
                for name in pair
            }
            crn[label] = {
                "status": "PASS",
                "branches": list(pair),
                "series_sha256": verify_crn.verify_pair(
                    paths, row["seed"], "formal", pair
                ),
            }
        else:
            crn[label] = {
                "status": "NOT_AVAILABLE_SCIENTIFIC_FAILURE",
                "branches": list(pair),
            }
    if crn["R"]["status"] == crn["K"]["status"] == "PASS":
        if crn["R"]["series_sha256"] != crn["K"]["series_sha256"]:
            raise RuntimeError("complete M1 branches do not share one four-way CRN stream")
    status = (
        "PASS"
        if all(value["status"] == "PASS" for value in branches.values())
        else "COMPLETE_WITH_SCIENTIFIC_FAILURES"
    )
    reproducibility.atomic_json_dump(
        {
            "schema": "ect.m1.training-slot/v1",
            "status": status,
            "training_manifest_path": str(manifest_path),
            "training_manifest_sha256": schedule_switch.sha256_file(
                str(manifest_path)
            ),
            "roster_slot": row["roster_slot"],
            "seed": row["seed"],
            "order": row["order"],
            "branches": branches,
            "crn": crn,
        },
        receipt_path,
        overwrite=False,
    )
    gpu_lock.close()
    print(f"M1_TRAINING_SLOT_{status} slot={row['roster_slot']} seed={row['seed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
