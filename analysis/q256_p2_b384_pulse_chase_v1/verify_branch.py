#!/usr/bin/env python3
"""Seal one completed 384->512->640 P2 branch without evaluating quality."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training import pulse_chase, reproducibility


def read_rows(path: Path, first: int, last: int, arm: str) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    attempts = [int(row["attempted_iteration"]) for row in rows]
    failures = []
    if attempts != list(range(first, last + 1)):
        failures.append(f"attempt tape is not exact {first}..{last}")
    if any(row["arm"] != arm for row in rows):
        failures.append(f"telemetry arm is not uniformly {arm}")
    for row in rows:
        if not math.isfinite(float(row["loss"])):
            failures.append("non-finite loss")
            break
        for key in (
            "loss_nonfinite_count", "sanitized_grad_nonfinite_count",
            "update_nonfinite_count", "model_nonfinite_count",
            "ema_nonfinite_count", "factor_nonfinite_count",
            "nonpositive_denominator_count",
        ):
            if int(row[key]):
                failures.append(f"nonzero {key}")
                break
    return {
        "path": str(path.resolve()),
        "sha256": pulse_chase.sha256_file(path),
        "rows": len(rows),
        "fieldnames": list(fields),
        "amp_skips": sum(int(row["step_skipped"]) for row in rows),
        "failures": failures,
    }


def endpoint(run_dir: Path, manifest: dict, kimg: int) -> dict:
    state_path = run_dir / f"training-state-kimg{kimg:06d}.pt"
    snapshot_path = run_dir / f"network-snapshot-kimg{kimg:06d}.pkl"
    if state_path.is_symlink() or snapshot_path.is_symlink():
        raise RuntimeError("endpoint artifacts must not be symlinks")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    expected_attempt = kimg * 1000 // pulse_chase.BATCH_SIZE
    if int(state.get("cur_nimg", -1)) != kimg * 1000:
        raise RuntimeError(f"cur_nimg mismatch at {kimg}")
    if int(state.get("attempted_iteration", -1)) != expected_attempt:
        raise RuntimeError(f"attempted iteration mismatch at {kimg}")
    phase = "pulse" if kimg == 512 else "chase"
    if state.get("pulse_chase") != pulse_chase.state_metadata(
        manifest, phase=phase
    ):
        raise RuntimeError(f"P2 state metadata mismatch at {kimg}")
    arm = manifest["pulse_arm"] if kimg == 512 else "A"
    if state.get("factorial") != pulse_chase.factorial_for_arm(arm):
        raise RuntimeError(f"factorial identity mismatch at {kimg}")
    internal = pulse_chase.internal_state_hashes(state)
    with snapshot_path.open("rb") as handle:
        snapshot = pickle.load(handle)
    if reproducibility.module_state_sha256(snapshot["ema"]) != internal["ema"]:
        raise RuntimeError(f"EMA snapshot mismatch at {kimg}")
    return {
        "kimg": kimg,
        "attempted_iteration": expected_attempt,
        "successful_optimizer_steps": int(state["successful_optimizer_steps"]),
        "training_state": {
            "path": str(state_path.resolve()),
            "bytes": state_path.stat().st_size,
            "sha256": pulse_chase.sha256_file(state_path),
            "internal_state_sha256": internal,
        },
        "ema_snapshot": {
            "path": str(snapshot_path.resolve()),
            "bytes": snapshot_path.stat().st_size,
            "sha256": pulse_chase.sha256_file(snapshot_path),
            "ema_internal_sha256": internal["ema"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    manifest = pulse_chase.load_run_manifest(manifest_path)
    if run_dir != Path(manifest["immutable_output_root"]).resolve():
        raise RuntimeError("run directory/manifest mismatch")
    source = manifest["source_state"]
    source_path = Path(source["path"]).resolve(strict=True)
    if source_path.stat().st_size != source["bytes"]:
        raise RuntimeError("source bytes changed")
    if pulse_chase.sha256_file(source_path) != source["sha256"]:
        raise RuntimeError("source SHA256 changed")
    pulse = endpoint(run_dir, manifest, 512)
    chase = endpoint(run_dir, manifest, 640)
    pulse_tape = read_rows(
        run_dir / "p2_pulse_training_telemetry_v1.csv",
        pulse_chase.SOURCE_ATTEMPT + 1,
        pulse_chase.PULSE_END_ATTEMPT,
        manifest["pulse_arm"],
    )
    chase_tape = read_rows(
        run_dir / "p2_chase_training_telemetry_v1.csv",
        pulse_chase.PULSE_END_ATTEMPT + 1,
        pulse_chase.CHASE_END_ATTEMPT,
        "A",
    )
    failures = pulse_tape["failures"] + chase_tape["failures"]
    payload = {
        "schema": "ect.q256.p2-branch-completion/v1",
        "status": "PASS" if not failures else "FAIL_CLOSED",
        "seed": manifest["seed"],
        "branch": manifest["branch"],
        "pulse_arm": manifest["pulse_arm"],
        "chase_arm": "A",
        "protocol_sha256": manifest["protocol_sha256"],
        "implementation_commit": manifest["implementation_commit"],
        "gpu_index": manifest["gpu_index"],
        "gpu_uuid": manifest["gpu_uuid"],
        "source_state_sha256": source["sha256"],
        "source_internal_state_sha256": source["internal_state_sha256"],
        "endpoints": [pulse, chase],
        "telemetry": {"pulse": pulse_tape, "chase": chase_tape},
        "manifest_path": str(manifest_path),
        "manifest_sha256": pulse_chase.sha256_file(manifest_path),
        "failures": failures,
    }
    reproducibility.atomic_json_dump(payload, args.output, overwrite=False)
    print(json.dumps({"status": payload["status"], "seed": manifest["seed"],
                      "branch": manifest["branch"]}))
    return 0 if not failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
