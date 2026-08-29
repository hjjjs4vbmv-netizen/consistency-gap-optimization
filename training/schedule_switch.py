"""Fail-closed q256 A/B schedule-switch manifest and state validation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re

from training import reproducibility


PROTOCOL = "q256_ab_crossed_switch_v1"
SEED3_7_PROTOCOL = "q256_ab_crossed_switch_seed3_7_v1"
SEED3_7_PROTOCOL_V2 = "q256_ab_crossed_switch_seed3_7_v2"
SUPPORTED_PROTOCOL_SEEDS = {
    PROTOCOL: tuple(range(14, 19)),
    SEED3_7_PROTOCOL: tuple(range(3, 8)),
    SEED3_7_PROTOCOL_V2: tuple(range(3, 8)),
}
RUN_MANIFEST_SCHEMA = "ect.q256.schedule-switch-run-manifest/v1"
STATE_SCHEMA = "ect.q256.schedule-switch-state/v1"
SWITCH_KIMG = 512
SWITCH_NIMG = SWITCH_KIMG * 1000
SWITCH_ATTEMPT = SWITCH_NIMG // 128
ARM_FACTORS = {
    "A": (1.0, 1.0),
    "B": (1.1, 1.1),
}
FORMAL_BRANCHES = {
    "A_to_B": ("A", "B"),
    "B_to_A": ("B", "A"),
}
PARITY_BRANCHES = {
    "A_to_A": ("A", "A"),
    "B_to_B": ("B", "B"),
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def internal_state_hashes(state: dict) -> dict:
    rank_states = state["rank_states"]
    return {
        "net": reproducibility.module_state_sha256(state["net"]),
        "ema": reproducibility.module_state_sha256(state["ema"]),
        "optimizer": reproducibility.state_sha256(state["optimizer_state"]),
        "gradscaler": reproducibility.state_sha256(state["gradscaler_state"]),
        "rank_rng": [
            reproducibility.state_sha256(item["rng_state"])
            for item in rank_states
        ],
        "rank_sampler": [
            reproducibility.state_sha256(item["sampler_state"])
            for item in rank_states
        ],
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_run_manifest(path: str) -> dict:
    path = os.path.abspath(path)
    _require(os.path.isfile(path) and not os.path.islink(path),
             "schedule-switch manifest must be a regular file")
    with open(path, "rt", encoding="utf-8") as handle:
        manifest = json.load(handle)
    _require(manifest.get("schema") == RUN_MANIFEST_SCHEMA,
             "unsupported schedule-switch manifest schema")
    experiment_protocol = manifest.get("experiment_protocol")
    _require(experiment_protocol in SUPPORTED_PROTOCOL_SEEDS,
             "schedule-switch protocol identity mismatch")
    run_kind = manifest.get("run_kind")
    _require(run_kind in {"parity", "formal"}, "invalid schedule-switch run kind")
    branches = PARITY_BRANCHES if run_kind == "parity" else FORMAL_BRANCHES
    branch = manifest.get("branch")
    _require(branch in branches, "invalid schedule-switch branch")
    origin, continuation = branches[branch]
    _require(manifest.get("origin_arm") == origin,
             "schedule-switch origin arm mismatch")
    _require(manifest.get("continuation_arm") == continuation,
             "schedule-switch continuation arm mismatch")
    _require(manifest.get("seed") in SUPPORTED_PROTOCOL_SEEDS[experiment_protocol],
             "schedule-switch seed is outside the frozen protocol cohort")
    _require(manifest.get("switch_kimg") == SWITCH_KIMG,
             "schedule-switch point must be 512 kimg")
    expected_final = 640 if run_kind == "parity" else 1024
    _require(manifest.get("final_kimg") == expected_final,
             "schedule-switch final budget mismatch")
    _require(_HEX64.fullmatch(str(manifest.get("protocol_sha256", ""))) is not None,
             "invalid protocol SHA256")
    _require(_HEX40.fullmatch(str(manifest.get("implementation_commit", ""))) is not None,
             "invalid implementation commit")
    _require(_HEX64.fullmatch(str(manifest.get(
        "source_checkpoint_manifest_sha256", ""))) is not None,
        "invalid source checkpoint-manifest SHA256")
    source = manifest.get("source_state")
    _require(isinstance(source, dict), "missing source-state record")
    _require(os.path.isabs(str(source.get("path", ""))),
             "source-state path must be absolute")
    _require(isinstance(source.get("bytes"), int) and source["bytes"] > 0,
             "invalid source-state byte count")
    _require(_HEX64.fullmatch(str(source.get("sha256", ""))) is not None,
             "invalid source-state SHA256")
    expected_hashes = source.get("internal_state_sha256")
    required_hashes = {
        "net", "ema", "optimizer", "gradscaler", "rank_rng", "rank_sampler"
    }
    _require(isinstance(expected_hashes, dict)
             and set(expected_hashes) == required_hashes,
             "incomplete source internal-state hashes")
    for name in ("net", "ema", "optimizer", "gradscaler"):
        _require(_HEX64.fullmatch(str(expected_hashes[name])) is not None,
                 f"invalid source {name} hash")
    for name in ("rank_rng", "rank_sampler"):
        _require(isinstance(expected_hashes[name], list)
                 and len(expected_hashes[name]) == 1
                 and _HEX64.fullmatch(str(expected_hashes[name][0])) is not None,
                 f"invalid source {name} hash list")
    return manifest


def continuation_factorial(manifest: dict) -> dict:
    arm = manifest["continuation_arm"]
    target, denominator = ARM_FACTORS[arm]
    return {
        "enabled": True,
        "protocol": "q256_target_weight_v1",
        "arm": arm,
        "target_gap_scale": target,
        "denominator_gap_scale": denominator,
    }


def state_metadata(manifest: dict) -> dict:
    source = manifest["source_state"]
    return {
        "schema": STATE_SCHEMA,
        "experiment_protocol": manifest["experiment_protocol"],
        "run_kind": manifest["run_kind"],
        "branch": manifest["branch"],
        "origin_arm": manifest["origin_arm"],
        "continuation_arm": manifest["continuation_arm"],
        "switch_kimg": SWITCH_KIMG,
        "source_state_path": source["path"],
        "source_state_sha256": source["sha256"],
        "source_internal_state_hashes": copy.deepcopy(
            source["internal_state_sha256"]
        ),
        "source_checkpoint_manifest_sha256": manifest[
            "source_checkpoint_manifest_sha256"
        ],
        "protocol_sha256": manifest["protocol_sha256"],
        "implementation_commit": manifest["implementation_commit"],
    }


def verify_resume_state_file(path: str, manifest: dict) -> None:
    source = manifest["source_state"]
    _require(os.path.realpath(path) == os.path.realpath(source["path"]),
             "resume path does not match frozen source-state path")
    _require(os.path.getsize(path) == source["bytes"],
             "source-state byte count mismatch")
    _require(sha256_file(path) == source["sha256"],
             "source-state file SHA256 mismatch")


def verify_source_state(state: dict, manifest: dict) -> dict:
    required = (
        "net", "ema", "optimizer_state", "gradscaler_state",
        "attempted_iteration", "successful_optimizer_steps", "cur_nimg",
        "rank_states", "factorial", "trajectory_config",
        "trajectory_config_sha256",
    )
    missing = [name for name in required if name not in state]
    _require(not missing, "source training-state missing fields: " + ", ".join(missing))
    _require(int(state["cur_nimg"]) == SWITCH_NIMG,
             "source training-state is not exactly 512 kimg")
    _require(int(state["attempted_iteration"]) == SWITCH_ATTEMPT,
             "source attempted iteration is not exactly 4000")
    origin = manifest["origin_arm"]
    target, denominator = ARM_FACTORS[origin]
    factorial = state["factorial"]
    _require(factorial.get("protocol") == "q256_target_weight_v1"
             and factorial.get("arm") == origin
             and float(factorial.get("target_gap_scale")) == target
             and float(factorial.get("denominator_gap_scale")) == denominator,
             "source factorial identity mismatch")
    trajectory = state["trajectory_config"]
    _require(reproducibility.state_sha256(trajectory)
             == state["trajectory_config_sha256"],
             "source trajectory-config SHA256 mismatch")
    _require(int(trajectory.get("seed", -1)) == manifest["seed"],
             "source trajectory seed mismatch")
    ranks = state["rank_states"]
    _require(len(ranks) == 1, "schedule-switch requires WORLD_SIZE=1 source")
    _require(int(ranks[0]["sampler_state"].get("consumed_samples", -1))
             == SWITCH_NIMG, "source sampler cursor is not exactly 512000")
    actual_hashes = internal_state_hashes(state)
    _require(actual_hashes == manifest["source_state"]["internal_state_sha256"],
             "source internal-state hash mismatch")
    return actual_hashes


def verify_switched_state(state: dict, manifest: dict) -> None:
    _require(state.get("schedule_switch") == state_metadata(manifest),
             "resumed schedule-switch metadata mismatch")
    _require(state.get("factorial", {}).get("arm") == manifest["origin_arm"],
             "switched state lost its origin factorial identity")


def trajectory_configs_compatible(source: dict, current: dict, manifest: dict) -> bool:
    source = reproducibility.canonical_json_data(copy.deepcopy(source))
    current = reproducibility.canonical_json_data(copy.deepcopy(current))
    source_total = source.pop("total_kimg", None)
    current_total = current.pop("total_kimg", None)
    if source_total != 1024 or current_total != manifest["final_kimg"]:
        return False
    source_loss = source.get("loss_kwargs")
    current_loss = current.get("loss_kwargs")
    if not isinstance(source_loss, dict) or not isinstance(current_loss, dict):
        return False
    for key in ("arm", "target_gap_scale", "denominator_gap_scale"):
        source_loss.pop(key, None)
        current_loss.pop(key, None)
    return source == current
