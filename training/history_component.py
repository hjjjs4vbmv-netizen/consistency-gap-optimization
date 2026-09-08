"""C/D history to common A, with one readout EMA and no optimizer intervention."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from training import m1, reproducibility, schedule_switch

PROTOCOL_ID = schedule_switch.HISTORY_COMPONENT_PROTOCOL
ENGINEERING_PROTOCOL_ID = schedule_switch.HISTORY_COMPONENT_ENGINEERING_PROTOCOL
METADATA_KEY = "history_component"
READOUTS = m1.READOUTS
initialize_ema_512 = m1.initialize_ema_512
update_ema_512 = m1.update_ema_512
checkpoint_metadata = m1.checkpoint_metadata
evaluator_snapshot = m1.evaluator_snapshot
readout_module = m1.readout_module


def validate_initial_receipt(current, reference):
    """Compare inherited initialization and scientific configuration, allowing paths/factors."""
    if current["seed"] != reference["seed"] or current["hashes"] != reference["hashes"]:
        raise RuntimeError("new prefix initialization differs from archived A/B")
    source = copy.deepcopy(reference["trajectory_config"])
    target = copy.deepcopy(current["trajectory_config"])
    manifest = {"final_kimg": 1024}
    if not schedule_switch.trajectory_configs_compatible(source, target, manifest):
        raise RuntimeError("new prefix scientific configuration differs from archived A/B")
    return {"status": "PASS", "seed": current["seed"], "scope": "initial hashes and scientific trajectory config"}


def check_archived_initial(run_dir, seed):
    root = Path(os.environ["ECT_HISTORY_OLD_PREFIX_ROOT"])
    current = json.loads((Path(run_dir)/"initial_state_receipt_v1.json").read_text())
    reports = []
    for arm in ("A", "B"):
        path = root/f"seed{seed}"/f"prefix_{arm}"/"initial_state_receipt_v1.json"
        reports.append({"old_arm": arm, **validate_initial_receipt(current, json.loads(path.read_text()))})
    reproducibility.atomic_json_dump({"checks":reports}, os.path.join(run_dir, "initial_pairing_check.json"), overwrite=False)


def observe_first_training_forward(net, run_dir):
    """One scalar dtype receipt from the actual denoiser call; no tensor or RNG operation."""
    path = os.path.join(run_dir, "first_actual_forward.json")
    if os.path.exists(path):
        return
    if not hasattr(net, "model"):
        raise RuntimeError("cannot locate the ECT denoiser for input-dtype observation")
    def observe(module, args):
        reproducibility.atomic_json_dump({"input_dtype":str(args[0].dtype),
            "module_class":module.__class__.__name__, "training":module.training,
            "scope":"first actual inner-denoiser forward in this trajectory"}, path, overwrite=False)
        handle.remove()
    handle = net.model.register_forward_pre_hook(observe)


def is_manifest(manifest):
    return manifest is not None and manifest.get("experiment_protocol") in {
        PROTOCOL_ID, ENGINEERING_PROTOCOL_ID,
    }


def apply_optimizer_intervention(optimizer, branch):
    if branch not in {"CA", "DA", "AA"} or optimizer.__class__.__name__ != "RAdam":
        raise RuntimeError("history component requires a keep-RAdam branch")
    return 0


def initial_metadata(manifest, reset_count, successful_steps_at_init=0):
    if not is_manifest(manifest) or reset_count != 0:
        raise RuntimeError("invalid history-component initialization")
    origin = manifest["origin_arm"]
    if manifest["branch"] != origin + "A" or origin not in {"A", "C", "D"}:
        raise RuntimeError("history-component branch/history mismatch")
    if origin == "A" and manifest["experiment_protocol"] != ENGINEERING_PROTOCOL_ID:
        raise RuntimeError("A control is engineering-only")
    if type(successful_steps_at_init) is not int or successful_steps_at_init < 0:
        raise RuntimeError("invalid source successful-step count")
    return {
        "protocol_id": manifest["experiment_protocol"],
        "branch": manifest["branch"], "seed": manifest["seed"],
        "source_path": manifest["source_state"]["path"],
        "source_history": origin, "current_arm": "A",
        "current_target_gap_scale": 1.0, "current_denominator_gap_scale": 1.0,
        "optimizer_operation": "keep", "optimizer_reset_count": 0,
        "ema_512_init_count": 1, "initialized_at_nimg": 512000,
        "initialized_emas": ["E_KEEP", "E_512"],
        "successful_steps_at_init": successful_steps_at_init,
        "successful_steps_since_init": 0,
    }


def validate_resumed_state(state, manifest):
    metadata = state.get(METADATA_KEY)
    if not isinstance(metadata, dict) or "ema_512" not in state or "m1" in state:
        raise RuntimeError("resume requires independent history-component metadata and E_512")
    expected = initial_metadata(manifest, 0)
    for key, value in expected.items():
        if key.startswith("successful_steps_"):
            continue
        if metadata.get(key) != value:
            raise RuntimeError(f"history-component resume mismatch: {key}")
    initial = metadata.get("successful_steps_at_init")
    steps = metadata.get("successful_steps_since_init")
    if any(type(v) is not int or v < 0 for v in (initial, steps)):
        raise RuntimeError("invalid history-component successful-step counts")
    attempt = state.get("attempted_iteration", -1)
    if not 4000 <= attempt <= 8000 or state.get("cur_nimg") != attempt * 128:
        raise RuntimeError("history-component progress mismatch")
    if steps > attempt - 4000 or state.get("successful_optimizer_steps") != initial + steps:
        raise RuntimeError("history-component successful-step counts do not reconcile")
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
        if not m1._equal_state(state[target].state_dict(), source[original].state_dict()):
            raise RuntimeError(f"branch-init changed {target}")
    for key in ("optimizer_state", "gradscaler_state", "rank_states", "loss_fn_state",
                "successful_optimizer_steps", "cur_tick", "tick_start_nimg",
                "snapshot_grid_z", "snapshot_grid_c", "snapshot_grid_size", "factorial"):
        if not m1._equal_state(state[key], source[key]):
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
        "continuation_arm": "A", "run_kind": "formal",
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
