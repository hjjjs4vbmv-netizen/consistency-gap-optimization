"""Independent D-to-D continuation; PR108 D-restore behavior is unchanged."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import torch

from training import m1, reproducibility, schedule_switch

PROTOCOL_ID = schedule_switch.D_RESTORE_PROTOCOL
ENGINEERING_PROTOCOL_ID = schedule_switch.D_RESTORE_ENGINEERING_PROTOCOL
METADATA_KEY = "d_restore"
READOUTS = m1.READOUTS
initialize_ema_512 = m1.initialize_ema_512
update_ema_512 = m1.update_ema_512
checkpoint_metadata = m1.checkpoint_metadata
evaluator_snapshot = m1.evaluator_snapshot
readout_module = m1.readout_module


def _equal_state(left, right):
    """Read-only byte comparison, including live CUDA state versus archived CPU state."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.dtype != right.dtype or left.shape != right.shape:
            return False
        a = left.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        b = right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        return torch.equal(a, b)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_equal_state(a, b) for a, b in zip(left, right))
    return m1._equal_state(left, right)


def is_manifest(manifest):
    return manifest is not None and manifest.get("experiment_protocol") in {
        PROTOCOL_ID, ENGINEERING_PROTOCOL_ID,
    }


def apply_optimizer_intervention(optimizer, branch):
    if branch != "DD" or optimizer.__class__.__name__ != "RAdam":
        raise RuntimeError("D-restore requires a keep-RAdam branch")
    return 0


def initial_metadata(manifest, reset_count, successful_steps_at_init=0):
    if not is_manifest(manifest) or reset_count != 0:
        raise RuntimeError("invalid D-restore initialization")
    origin = manifest["origin_arm"]
    if manifest["branch"] != "DD" or origin != "D" or manifest["continuation_arm"] != "D":
        raise RuntimeError("D-restore branch/history mismatch")
    if type(successful_steps_at_init) is not int or successful_steps_at_init < 0:
        raise RuntimeError("invalid source successful-step count")
    return {
        "protocol_id": manifest["experiment_protocol"],
        "branch": manifest["branch"], "seed": manifest["seed"],
        "source_path": manifest["source_state"]["path"],
        "source_binding": copy.deepcopy(manifest["source_binding"]),
        "source_history": origin, "current_arm": "D",
        "current_target_gap_scale": 1.0, "current_denominator_gap_scale": 1.1,
        "optimizer_operation": "keep", "optimizer_reset_count": 0,
        "ema_512_init_count": 1, "initialized_at_nimg": 512000,
        "initialized_emas": ["E_KEEP", "E_512"],
        "successful_steps_at_init": successful_steps_at_init,
        "successful_steps_since_init": 0,
    }


def validate_resumed_state(state, manifest):
    metadata = state.get(METADATA_KEY)
    if not isinstance(metadata, dict) or "ema_512" not in state or "m1" in state or "history_component" in state:
        raise RuntimeError("resume requires independent D-restore metadata and E_512")
    expected = initial_metadata(manifest, 0)
    for key, value in expected.items():
        if key.startswith("successful_steps_"):
            continue
        if metadata.get(key) != value:
            raise RuntimeError(f"D-restore resume mismatch: {key}")
    initial = metadata.get("successful_steps_at_init")
    steps = metadata.get("successful_steps_since_init")
    if any(type(v) is not int or v < 0 for v in (initial, steps)):
        raise RuntimeError("invalid D-restore successful-step counts")
    attempt = state.get("attempted_iteration", -1)
    if not 4000 <= attempt <= 8000 or state.get("cur_nimg") != attempt * 128:
        raise RuntimeError("D-restore progress mismatch")
    if steps > attempt - 4000 or state.get("successful_optimizer_steps") != initial + steps:
        raise RuntimeError("D-restore successful-step counts do not reconcile")
    schedule_switch.verify_switched_state(state, manifest)
    if state.get("factorial", {}).get("arm") != metadata["source_history"]:
        raise RuntimeError("source history was overwritten")
    return copy.deepcopy(metadata)


def validate_branch_init_against_source(state, source, manifest):
    schedule_switch.verify_source_state(source, manifest)
    validate_resumed_state(state, manifest)
    if state["attempted_iteration"] != 4000:
        raise RuntimeError("branch-init must precede attempt 4001")
    for target, original in (("net", "net"), ("ema", "ema"), ("ema_512", "net")):
        if not _equal_state(state[target].state_dict(), source[original].state_dict()):
            raise RuntimeError(f"branch-init changed {target}")
    for key in ("optimizer_state", "gradscaler_state", "rank_states", "loss_fn_state",
                "successful_optimizer_steps", "cur_tick", "tick_start_nimg",
                "snapshot_grid_z", "snapshot_grid_c", "snapshot_grid_size", "factorial"):
        if not _equal_state(state[key], source[key]):
            raise RuntimeError(f"branch-init changed {key}")
    return True


def validate_terminal_state(state, manifest):
    metadata = validate_resumed_state(state, manifest)
    if state["attempted_iteration"] != 8000:
        raise RuntimeError("terminal requires attempt 8000")
    required = {"net", "ema", "ema_512", "optimizer_state", "gradscaler_state",
                "rank_states", "loss_fn_state", "cur_tick", "tick_start_nimg",
                "trajectory_config", "trajectory_config_sha256", "reproducibility_schema"}
    if required - state.keys():
        raise RuntimeError(f"incomplete terminal: {sorted(required - state.keys())}")
    if state["reproducibility_schema"] != reproducibility.TRAINING_STATE_SCHEMA:
        raise RuntimeError("terminal schema mismatch")
    if state["trajectory_config"]["seed"] != manifest["seed"] or state["trajectory_config"]["total_kimg"] != 1024:
        raise RuntimeError("terminal trajectory mismatch")
    if len(state["rank_states"]) != 1 or state["rank_states"][0]["sampler_state"]["consumed_samples"] != 1024000:
        raise RuntimeError("terminal sampler mismatch")
    for name in READOUTS:
        readout_module(state, name)
    return metadata


def save_branch_init_state(state, run_dir):
    metadata = state[METADATA_KEY]
    manifest = {
        "experiment_protocol": metadata["protocol_id"], "branch": metadata["branch"],
        "seed": metadata["seed"], "origin_arm": metadata["source_history"],
        "continuation_arm": "D",
        "source_binding": copy.deepcopy(metadata["source_binding"]), "run_kind": "formal",
        "source_state": {"path": metadata["source_path"]},
    }
    validate_resumed_state(state, manifest)
    if state["attempted_iteration"] != 4000:
        raise RuntimeError("branch-init must precede attempt 4001")
    required = {"net", "ema", "optimizer_state", "gradscaler_state", "rank_states",
                "loss_fn_state", "trajectory_config", "trajectory_config_sha256",
                "reproducibility_schema", "snapshot_grid_z", "snapshot_grid_c", "snapshot_grid_size"}
    if required - state.keys():
        raise RuntimeError("incomplete branch-init")
    path = os.path.join(run_dir, "training-state-kimg000512.pt")
    reproducibility.atomic_torch_save(state, path, overwrite=False)
    return path


def validate_against_da_init(state, da_init, source, manifest):
    """Compare computation state directly; policy/identity metadata may differ."""
    validate_branch_init_against_source(state, source, manifest)
    old = da_init.get("history_component", {})
    if (old.get("branch") != "DA" or old.get("seed") != manifest["seed"]
            or old.get("ema_512_init_count") != 1):
        raise RuntimeError("reference is not the corresponding original DA branch-init")
    for key in ("net", "ema", "ema_512"):
        if not _equal_state(state[key].state_dict(), da_init[key].state_dict()):
            raise RuntimeError("DD/DA boundary module mismatch: " + key)
    for key in ("optimizer_state", "gradscaler_state", "rank_states", "loss_fn_state",
                "attempted_iteration", "successful_optimizer_steps", "cur_nimg",
                "cur_tick", "tick_start_nimg", "snapshot_grid_z", "snapshot_grid_c",
                "snapshot_grid_size", "factorial"):
        if key in state or key in da_init:
            if not _equal_state(state.get(key), da_init.get(key)):
                raise RuntimeError("DD/DA boundary state mismatch: " + key)
    return True
