"""Training-state operations for the M1 optimizer/EMA intervention."""

from __future__ import annotations

import copy
import os

import torch

from training import reproducibility, schedule_switch


PROTOCOL_ID = schedule_switch.M1_HISTORY_PERSISTENCE_PROTOCOL
READOUTS = ("ONLINE", "E_KEEP", "E_512")
GATE_START_ATTEMPTS = {4016: (4000,), 4032: (4000, 4016)}


def is_m1_manifest(manifest: dict | None) -> bool:
    return (
        manifest is not None
        and manifest.get("experiment_protocol") == PROTOCOL_ID
    )


def optimizer_intervention(branch: str) -> str:
    if branch in {"K_A", "K_B"}:
        return "keep"
    if branch in {"R_A", "R_B"}:
        return "reset"
    raise RuntimeError(f"invalid M1 branch: {branch}")


def validate_gate_resume_boundary(target_attempt: int, restored_attempt: int) -> None:
    allowed = GATE_START_ATTEMPTS.get(target_attempt)
    if allowed is None or restored_attempt not in allowed:
        raise RuntimeError(
            f"M1 gate target {target_attempt} requires restored attempt in {allowed}"
        )


def verify_source_artifacts(manifest: dict) -> None:
    source = manifest["source_state"]
    provenance = source.get("provenance_receipt", {})
    artifacts = {"provenance_receipt": provenance}
    artifacts.update(source.get("support_files", {}))
    if set(artifacts) != {
        "provenance_receipt", "train_summary.csv",
        "factorial_training_telemetry_v1.csv", "training_options.json",
    }:
        raise RuntimeError("M1 source support-file binding is incomplete")
    for name, identity in artifacts.items():
        path = identity.get("path")
        if not isinstance(path, str) or not os.path.isabs(path):
            raise RuntimeError(f"M1 {name} path is invalid")
        if not os.path.isfile(path) or os.path.islink(path):
            raise RuntimeError(f"M1 {name} is not a regular file")
        expected_bytes = identity.get("bytes")
        if expected_bytes is not None and os.path.getsize(path) != expected_bytes:
            raise RuntimeError(f"M1 {name} byte count mismatch")
        if schedule_switch.sha256_file(path) != identity.get("sha256"):
            raise RuntimeError(f"M1 {name} SHA256 mismatch")


def apply_optimizer_intervention(optimizer, branch: str) -> int:
    """Apply the one-time K/R operation after the full source restore."""
    if optimizer.__class__.__name__ != "RAdam":
        raise RuntimeError("M1 requires torch.optim.RAdam")
    param_groups_before = reproducibility.state_sha256(
        optimizer.state_dict()["param_groups"]
    )
    intervention = optimizer_intervention(branch)
    if intervention == "reset":
        optimizer.state.clear()
    if reproducibility.state_sha256(
        optimizer.state_dict()["param_groups"]
    ) != param_groups_before:
        raise RuntimeError("M1 optimizer intervention changed parameter groups")
    if intervention == "reset" and optimizer.state:
        raise RuntimeError("M1 R branch did not clear RAdam per-parameter state")
    return int(intervention == "reset")


def initialize_ema_512(net):
    """Create the independent 512-kimg EMA readout without consuming RNG."""
    return copy.deepcopy(net).eval().requires_grad_(False)


@torch.no_grad()
def update_ema_512(ema_512, net, beta: float) -> None:
    """Follow the baseline attempted-update EMA clock (parameters only)."""
    for p_ema, p_net in zip(ema_512.parameters(), net.parameters()):
        p_ema.copy_(p_net.detach().lerp(p_ema, beta))


def initial_metadata(
    manifest: dict, reset_count: int, successful_steps_at_init: int = 0
) -> dict:
    if not is_m1_manifest(manifest):
        raise RuntimeError("not an M1 manifest")
    expected_reset = int(optimizer_intervention(manifest["branch"]) == "reset")
    if reset_count != expected_reset:
        raise RuntimeError("M1 reset count does not match branch")
    if (
        isinstance(successful_steps_at_init, bool)
        or not isinstance(successful_steps_at_init, int)
        or successful_steps_at_init < 0
    ):
        raise RuntimeError("invalid M1 source successful-step counter")
    return {
        "protocol_id": PROTOCOL_ID,
        "branch": manifest["branch"],
        "seed": manifest["seed"],
        "source_sha256": manifest["source_state"]["sha256"],
        "initialized_at_nimg": schedule_switch.SWITCH_NIMG,
        "reset_count": reset_count,
        "initialized_emas": ["E_KEEP", "E_512"],
        "shadow_update_enabled": bool(manifest.get("m1_shadow_update", True)),
        "successful_steps_at_init": successful_steps_at_init,
        "successful_steps_since_init": 0,
    }


def validate_branch_init_against_source(
    branch_state: dict, source_state: dict, manifest: dict
) -> str:
    schedule_switch.verify_source_state(source_state, manifest)
    validate_resumed_state(branch_state, manifest)
    if (
        int(branch_state.get("attempted_iteration", -1))
        != schedule_switch.SWITCH_ATTEMPT
        or int(branch_state.get("cur_nimg", -1))
        != schedule_switch.SWITCH_NIMG
    ):
        raise RuntimeError("M1 branch-init progress differs from source")
    module_pairs = (
        ("net", "net"), ("ema", "ema"), ("ema_512", "net")
    )
    for branch_key, source_key in module_pairs:
        if reproducibility.module_state_sha256(branch_state[branch_key]) != (
            reproducibility.module_state_sha256(source_state[source_key])
        ):
            raise RuntimeError(f"M1 branch-init {branch_key} differs from source")
    exact_keys = (
        "gradscaler_state", "rank_states", "loss_fn_state",
        "successful_optimizer_steps", "cur_tick", "tick_start_nimg",
        "snapshot_grid_z", "snapshot_grid_c", "snapshot_grid_size", "factorial",
    )
    for key in exact_keys:
        if reproducibility.state_sha256(branch_state[key]) != (
            reproducibility.state_sha256(source_state[key])
        ):
            raise RuntimeError(f"M1 branch-init changed restored {key}")
    branch_optimizer = branch_state["optimizer_state"]
    source_optimizer = source_state["optimizer_state"]
    if optimizer_intervention(manifest["branch"]) == "keep":
        if reproducibility.state_sha256(branch_optimizer) != (
            reproducibility.state_sha256(source_optimizer)
        ):
            raise RuntimeError("M1 K branch changed restored optimizer")
    elif (
        branch_optimizer.get("state")
        or branch_optimizer.get("param_groups") != source_optimizer.get("param_groups")
    ):
        raise RuntimeError("M1 R branch reset more than optimizer state")
    expected_reset = int(optimizer_intervention(manifest["branch"]) == "reset")
    if branch_state["m1"].get("reset_count") != expected_reset:
        raise RuntimeError("M1 branch-init reset count mismatch")
    return reproducibility.state_sha256({
        "source": schedule_switch.internal_state_hashes(source_state),
        "branch": schedule_switch.internal_state_hashes(branch_state),
        "ema_512": reproducibility.module_state_sha256(branch_state["ema_512"]),
        "m1": branch_state["m1"],
    })


def validate_resumed_state(state: dict, manifest: dict) -> dict:
    metadata = state.get("m1")
    if not isinstance(metadata, dict) or "ema_512" not in state:
        raise RuntimeError("M1 resume requires m1 metadata and E_512")
    expected = initial_metadata(
        manifest, int(optimizer_intervention(manifest["branch"]) == "reset")
    )
    for key, value in expected.items():
        if key in {"successful_steps_at_init", "successful_steps_since_init"}:
            continue
        if metadata.get(key) != value:
            raise RuntimeError(f"M1 resume metadata mismatch: {key}")
    initial = metadata.get("successful_steps_at_init")
    steps = metadata.get("successful_steps_since_init")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (initial, steps)
    ):
        raise RuntimeError("invalid M1 successful-step counters")
    total = state.get("successful_optimizer_steps")
    if total is not None and total != initial + steps:
        raise RuntimeError("M1 successful-step counters do not reconcile")
    return copy.deepcopy(metadata)


def checkpoint_metadata(metadata: dict, successful_steps: int) -> dict:
    result = copy.deepcopy(metadata)
    result["successful_steps_since_init"] = successful_steps
    return result


def validate_terminal_state(state: dict, manifest: dict) -> dict:
    metadata = validate_resumed_state(state, manifest)
    required = {
        "net", "ema", "ema_512", "optimizer_state", "gradscaler_state",
        "rank_states", "loss_fn_state", "attempted_iteration",
        "successful_optimizer_steps", "cur_nimg", "cur_tick",
        "tick_start_nimg", "trajectory_config", "trajectory_config_sha256",
        "reproducibility_schema", "factorial", "schedule_switch",
    }
    missing = sorted(required - set(state))
    if missing:
        raise RuntimeError("M1 terminal state is incomplete: " + ", ".join(missing))
    if int(state["attempted_iteration"]) != 8000:
        raise RuntimeError("M1 terminal state must be attempt 8000")
    if int(state["cur_nimg"]) != 1024 * 1000:
        raise RuntimeError("M1 terminal state must be 1024 kimg")
    if state["reproducibility_schema"] != reproducibility.TRAINING_STATE_SCHEMA:
        raise RuntimeError("M1 terminal state schema mismatch")
    modules = [state[name] for name in ("net", "ema", "ema_512")]
    if any(not isinstance(module, torch.nn.Module) for module in modules):
        raise RuntimeError("M1 terminal readouts must be torch modules")
    signatures = [
        [(key, tuple(value.shape), value.dtype) for key, value in module.state_dict().items()]
        for module in modules
    ]
    if signatures[1:] != signatures[:1] * 2:
        raise RuntimeError("M1 terminal readout structures differ")
    trajectory = state["trajectory_config"]
    if reproducibility.state_sha256(trajectory) != state["trajectory_config_sha256"]:
        raise RuntimeError("M1 terminal trajectory hash is invalid")
    if (
        trajectory.get("seed") != manifest["seed"]
        or trajectory.get("total_kimg") != 1024
    ):
        raise RuntimeError("M1 terminal trajectory identity mismatch")
    ranks = state["rank_states"]
    if (
        not isinstance(ranks, list)
        or len(ranks) != 1
        or int(ranks[0].get("sampler_state", {}).get("consumed_samples", -1))
        != 1024 * 1000
    ):
        raise RuntimeError("M1 terminal sampler progress mismatch")
    if metadata["successful_steps_since_init"] > 4000:
        raise RuntimeError("M1 terminal successful-step counter exceeds attempts")
    return metadata


def branch_init_path(run_dir: str) -> str:
    return os.path.join(run_dir, "training-state-kimg000512.pt")


def save_branch_init_state(state: dict, run_dir: str) -> str:
    required = {
        "net", "ema", "ema_512", "optimizer_state", "gradscaler_state",
        "rank_states", "loss_fn_state", "attempted_iteration",
        "successful_optimizer_steps", "cur_nimg", "cur_tick",
        "tick_start_nimg", "trajectory_config", "trajectory_config_sha256",
        "reproducibility_schema", "factorial", "schedule_switch",
        "snapshot_grid_z", "snapshot_grid_c", "snapshot_grid_size", "m1",
    }
    missing = sorted(required - set(state))
    if missing:
        raise RuntimeError(
            "M1 branch-init is not a complete training state: "
            + ", ".join(missing)
        )
    if int(state.get("cur_nimg", -1)) != schedule_switch.SWITCH_NIMG:
        raise RuntimeError("M1 branch-init must be written at 512 kimg")
    if int(state.get("attempted_iteration", -1)) != schedule_switch.SWITCH_ATTEMPT:
        raise RuntimeError("M1 branch-init must precede attempt 4001")
    validate_resumed_state(state, {
        "experiment_protocol": PROTOCOL_ID,
        "branch": state["m1"]["branch"],
        "seed": state["m1"]["seed"],
        "source_state": {"sha256": state["m1"]["source_sha256"]},
        "m1_shadow_update": state["m1"]["shadow_update_enabled"],
    })
    path = branch_init_path(run_dir)
    reproducibility.atomic_torch_save(state, path, overwrite=False)
    return path


def readout_module(state: dict, readout: str):
    if readout not in READOUTS:
        raise ValueError(f"unsupported M1 readout: {readout}")
    key = {"ONLINE": "net", "E_KEEP": "ema", "E_512": "ema_512"}[readout]
    module = state.get(key)
    if not isinstance(module, torch.nn.Module):
        raise RuntimeError(f"M1 state is missing {readout}")
    return module


def evaluator_snapshot(state: dict, readout: str) -> dict:
    module = copy.deepcopy(readout_module(state, readout))
    module.eval().requires_grad_(False).cpu()
    return {
        "ema": module,
        "loss_fn": None,
        "augment_pipe": None,
        "dataset_kwargs": dict(state["trajectory_config"]["dataset_kwargs"]),
    }
