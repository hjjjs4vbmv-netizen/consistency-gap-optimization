#!/usr/bin/env python3
"""Copy compact PASS evidence to the durable cloud mount without raw arrays."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path


ROOT = Path("/root/q256_eval")
DURABLE = Path("/mnt/ect_project/q256_seed14_18_streaming_eval_20260822")
ALL_ARMS = ("A", "B", "C", "D")
SMALL_JOB_FILES = (
    "metric-kid50k_full.jsonl",
    "metric-fid50k_full.jsonl",
    "training_options.json",
    "log.txt",
    "sample.png",
    "data.png",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    DURABLE.mkdir(mode=0o750, parents=True, exist_ok=True)
    for name in ("receipts", "jobs", "logs", "integrity", "deploy"):
        (DURABLE / name).mkdir(mode=0o750, exist_ok=True)
    selected_arms = tuple(
        arm.strip() for arm in os.environ.get("Q256_EVAL_ARMS", "A,B,C,D").split(",")
    )
    if not selected_arms or len(set(selected_arms)) != len(selected_arms) or any(
        arm not in ALL_ARMS for arm in selected_arms
    ):
        raise RuntimeError(f"invalid Q256_EVAL_ARMS: {selected_arms}")
    expected_jobs = 5 * len(selected_arms) * 6 * 2
    while True:
        copied = 0
        receipts = [
            receipt
            for seed in range(14, 19)
            for receipt in sorted((ROOT / "receipts").glob(f"seed{seed}-*.json"))
        ]
        for receipt in receipts:
            target_receipt = DURABLE / "receipts" / receipt.name
            job_id = receipt.stem
            durable_job = DURABLE / "jobs" / job_id
            durable_record = durable_job / "durable_copy_receipt.json"
            if durable_record.is_file():
                copied += 1
                continue
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            if payload.get("status") != "PASS" or payload.get("seed") not in range(14, 19):
                raise RuntimeError(f"refuse non-PASS receipt: {receipt}")
            if payload.get("arm") not in selected_arms:
                continue
            source_job = Path(payload["job_dir"]).resolve(strict=True)
            temporary = durable_job.parent / f".{job_id}.tmp-{os.getpid()}"
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(mode=0o750)
            files = {}
            for filename in SMALL_JOB_FILES:
                source = source_job / filename
                if source.is_file() and not source.is_symlink():
                    destination = temporary / filename
                    shutil.copy2(source, destination)
                    files[filename] = {
                        "bytes": destination.stat().st_size,
                        "sha256": sha256_file(destination),
                    }
            process_log = ROOT / "logs" / f"{job_id}.process.log"
            if process_log.is_file():
                shutil.copy2(process_log, temporary / "process.log")
                files["process.log"] = {
                    "bytes": (temporary / "process.log").stat().st_size,
                    "sha256": sha256_file(temporary / "process.log"),
                }
            shutil.copy2(receipt, temporary / "job_receipt.json")
            receipt_sha = sha256_file(temporary / "job_receipt.json")
            record = {
                "schema": "ect.q256.seed14-18.durable-evaluation-copy/v1",
                "status": "PASS",
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "job_id": job_id,
                "source_receipt_sha256": receipt_sha,
                "raw_generated_artifacts": "retained_on_ephemeral_evaluation_node",
                "regenerable_raw_artifact_hashes": payload["artifacts"],
                "compact_files": files,
            }
            (temporary / "durable_copy_receipt.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.rename(temporary, durable_job)
            shutil.copy2(receipt, target_receipt)
            copied += 1
            print(f"[q256-durable] PASS {job_id}", flush=True)
        state = {
            "schema": "ect.q256.seed14-18.durable-copy-state/v1",
            "status": "PASS" if copied == expected_jobs else "RUNNING",
            "copied_job_count": copied,
            "expected_job_count": expected_jobs,
            "arms": list(selected_arms),
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        temporary_state = DURABLE / f".durable_copy_state.json.tmp-{os.getpid()}"
        temporary_state.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_state, DURABLE / "durable_copy_state.json")
        if copied == expected_jobs:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
