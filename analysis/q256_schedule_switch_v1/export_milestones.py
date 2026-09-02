#!/usr/bin/env python3
"""Export immutable schedule-switch milestone states, EMA snapshots, and receipts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import platform
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training import reproducibility, schedule_switch


FORMAL_MILESTONES = (640, 768, 896, 1024)
PARITY_MILESTONES = (640,)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_receipt() -> dict:
    cuda_visible = torch.cuda.is_available()
    gpu_uuid = None
    if cuda_visible:
        raw_uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
        gpu_uuid = None if raw_uuid is None else str(raw_uuid)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": cuda_visible,
        "gpu_uuid": gpu_uuid,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_visible else None,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--recover-partial-export", action="store_true",
        help="resume only a strictly validated partial milestone export",
    )
    args = parser.parse_args()
    # Export runs in a fresh process; re-assert the frozen deterministic runtime
    # policy before recording its receipt instead of inheriting PyTorch defaults.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run_dir = args.run_dir.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    manifest = schedule_switch.load_run_manifest(manifest_path)
    if run_dir != Path(manifest["immutable_output_root"]).resolve():
        raise RuntimeError("run directory does not match frozen output root")
    milestones = (
        PARITY_MILESTONES
        if manifest["run_kind"] == "parity"
        else FORMAL_MILESTONES
    )
    manifest_sha = sha256_file(manifest_path)
    runtime = runtime_receipt()
    records = []
    for kimg in milestones:
        source_path = run_dir / f"training-state-kimg{kimg:06d}.pt"
        if not source_path.is_file() or source_path.is_symlink():
            raise RuntimeError(f"missing immutable training milestone: {source_path}")
        milestone_dir = run_dir / f"kimg{kimg:04d}"
        partial = milestone_dir.exists()
        if partial and not args.recover_partial_export:
            raise RuntimeError(f"milestone directory already exists: {milestone_dir}")
        milestone_dir.mkdir(exist_ok=args.recover_partial_export)
        state_path = milestone_dir / "training-state.pt"
        snapshot_path = milestone_dir / "network-snapshot.pkl"
        receipt_path = milestone_dir / "milestone_receipt.json"
        if state_path.exists():
            if (not args.recover_partial_export or state_path.is_symlink()
                    or not state_path.is_file()
                    or os.stat(source_path).st_ino != os.stat(state_path).st_ino
                    or os.stat(source_path).st_dev != os.stat(state_path).st_dev):
                raise RuntimeError(f"invalid pre-existing milestone state: {state_path}")
        else:
            if partial:
                raise RuntimeError(f"partial milestone lacks state hardlink: {milestone_dir}")
            os.link(source_path, state_path)
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if int(state.get("cur_nimg", -1)) != kimg * 1000:
            raise RuntimeError(f"state processed-image mismatch at {kimg} kimg")
        if int(state.get("attempted_iteration", -1)) != kimg * 1000 // 128:
            raise RuntimeError(f"state attempted-iteration mismatch at {kimg} kimg")
        schedule_switch.verify_switched_state(state, manifest)
        rank_states = state["rank_states"]
        if len(rank_states) != 1 or int(
            rank_states[0]["sampler_state"].get("consumed_samples", -1)
        ) != kimg * 1000:
            raise RuntimeError(f"sampler-state mismatch at {kimg} kimg")
        internal = schedule_switch.internal_state_hashes(state)
        ema = copy.deepcopy(state["ema"]).eval().requires_grad_(False)
        snapshot = {
            "ema": ema,
            "loss_fn": None,
            "augment_pipe": None,
            "dataset_kwargs": dict(state["trajectory_config"]["dataset_kwargs"]),
        }
        if snapshot_path.exists():
            if not args.recover_partial_export or snapshot_path.is_symlink():
                raise RuntimeError(f"invalid pre-existing snapshot: {snapshot_path}")
        else:
            reproducibility.atomic_pickle_dump(snapshot, snapshot_path, overwrite=False)
        with snapshot_path.open("rb") as handle:
            snapshot_check = pickle.load(handle)
        if reproducibility.module_state_sha256(snapshot_check["ema"]) != internal["ema"]:
            raise RuntimeError(f"snapshot EMA mismatch at {kimg} kimg")
        receipt = {
            "schema": "ect.q256.schedule-switch-milestone/v1",
            "status": "PASS",
            "seed": manifest["seed"],
            "branch": manifest["branch"],
            "origin_arm": manifest["origin_arm"],
            "continuation_arm": manifest["continuation_arm"],
            "milestone_kimg": kimg,
            "attempted_iteration": int(state["attempted_iteration"]),
            "successful_optimizer_steps": int(state["successful_optimizer_steps"]),
            "cur_nimg": int(state["cur_nimg"]),
            "sampler_consumed_samples": int(
                rank_states[0]["sampler_state"]["consumed_samples"]
            ),
            "training_state": {
                "path": str(state_path),
                "bytes": state_path.stat().st_size,
                "sha256": sha256_file(state_path),
                "internal_state_sha256": internal,
            },
            "network_snapshot": {
                "path": str(snapshot_path),
                "bytes": snapshot_path.stat().st_size,
                "sha256": sha256_file(snapshot_path),
                "ema_internal_sha256": internal["ema"],
            },
            "protocol_sha256": manifest["protocol_sha256"],
            "formal_run_manifest_sha256": manifest_sha,
            "implementation_commit": manifest["implementation_commit"],
            "runtime": runtime,
        }
        if receipt_path.exists():
            if not args.recover_partial_export or receipt_path.is_symlink():
                raise RuntimeError(f"invalid pre-existing receipt: {receipt_path}")
            prior = json.loads(receipt_path.read_text(encoding="utf-8"))
            for key in ("schema", "status", "seed", "branch", "milestone_kimg",
                        "protocol_sha256", "formal_run_manifest_sha256"):
                if prior.get(key) != receipt.get(key):
                    raise RuntimeError(f"pre-existing receipt mismatch ({key}): {receipt_path}")
            for artifact in ("training_state", "network_snapshot"):
                if prior.get(artifact, {}).get("sha256") != receipt[artifact]["sha256"]:
                    raise RuntimeError(f"pre-existing receipt artifact mismatch: {receipt_path}")
            receipt = prior
        else:
            reproducibility.atomic_json_dump(receipt, receipt_path, overwrite=False)
        records.append({
            "kimg": kimg,
            "directory": str(milestone_dir),
            "training_state_sha256": receipt["training_state"]["sha256"],
            "network_snapshot_sha256": receipt["network_snapshot"]["sha256"],
            "milestone_receipt_sha256": sha256_file(receipt_path),
            "internal_state_sha256": internal,
        })
        del state, ema, snapshot, snapshot_check
    expected_final_attempt = manifest["final_kimg"] * 1000 // 128
    completion = {
        "schema": "ect.q256.schedule-switch-training-completion/v1",
        "status": "PASS",
        "seed": manifest["seed"],
        "branch": manifest["branch"],
        "run_kind": manifest["run_kind"],
        "source_attempted_iteration": schedule_switch.SWITCH_ATTEMPT,
        "final_attempted_iteration": expected_final_attempt,
        "additional_attempted_iterations": (
            expected_final_attempt - schedule_switch.SWITCH_ATTEMPT
        ),
        "milestone_count": len(records),
        "milestones": records,
        "protocol_sha256": manifest["protocol_sha256"],
        "formal_run_manifest_sha256": manifest_sha,
        "runtime": runtime,
    }
    if (run_dir / "checkpoint_manifest.json").exists():
        raise RuntimeError("checkpoint manifest already exists before completion write")
    reproducibility.atomic_json_dump(
        completion, run_dir / "checkpoint_manifest.json", overwrite=False
    )
    reproducibility.atomic_json_dump(
        {
            "schema": "ect.q256.schedule-switch-trajectory-completion/v1",
            "status": "PASS",
            "checkpoint_manifest_sha256": sha256_file(
                run_dir / "checkpoint_manifest.json"
            ),
            "protocol_sha256": manifest["protocol_sha256"],
        },
        run_dir / "trajectory_completion_receipt.json",
        overwrite=False,
    )
    print(json.dumps({"status": "PASS", "milestones": len(records),
                      "run_dir": str(run_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
