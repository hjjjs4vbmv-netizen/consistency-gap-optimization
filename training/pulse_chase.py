"""Fail-closed state contract for q256 B@384 pulse/chase P2."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path

from training import reproducibility


PROTOCOL = "q256_b384_pulse_chase_v1"
RUN_MANIFEST_SCHEMA = "ect.q256.p2-pulse-chase-run-manifest/v1"
STATE_SCHEMA = "ect.q256.p2-pulse-chase-state/v1"
SOURCE_KIMG = 384
PULSE_END_KIMG = 512
CHASE_END_KIMG = 640
BATCH_SIZE = 128
SOURCE_ATTEMPT = SOURCE_KIMG * 1000 // BATCH_SIZE
PULSE_END_ATTEMPT = PULSE_END_KIMG * 1000 // BATCH_SIZE
CHASE_END_ATTEMPT = CHASE_END_KIMG * 1000 // BATCH_SIZE
SEEDS = tuple(range(19, 29))
BRANCHES = {
    "Early-switch": {"pulse_arm": "A", "chase_arm": "A"},
    "Late-switch": {"pulse_arm": "B", "chase_arm": "A"},
}
ARM_FACTORS = {"A": (1.0, 1.0), "B": (1.1, 1.1)}
ASSET_SHA256 = {
    "dataset": "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372",
    "transfer": "4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da",
    "runtime_sif": "9d5f2c9e68f1f7dcaa20457bf6e0b6fa46f74a8605edaf5d49fdccf9f6bb62ea",
}
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def internal_state_hashes(state: dict) -> dict:
    """Hash every scientific state component named by the P2 protocol."""
    ranks = state["rank_states"]
    return {
        "online_parameters": reproducibility.module_state_sha256(state["net"]),
        "ema": reproducibility.module_state_sha256(state["ema"]),
        "radam": reproducibility.state_sha256(state["optimizer_state"]),
        "gradscaler": reproducibility.state_sha256(state["gradscaler_state"]),
        "rank_rng": [
            reproducibility.state_sha256(item["rng_state"]) for item in ranks
        ],
        "sampler": [
            reproducibility.state_sha256(item["sampler_state"]) for item in ranks
        ],
        "loss_control": reproducibility.state_sha256(state["loss_fn_state"]),
        "trajectory_config": reproducibility.state_sha256(
            state["trajectory_config"]
        ),
        "data_cursor": [
            int(item["sampler_state"]["consumed_samples"]) for item in ranks
        ],
        "attempted_iteration": int(state["attempted_iteration"]),
        "successful_optimizer_steps": int(state["successful_optimizer_steps"]),
        "cur_nimg": int(state["cur_nimg"]),
    }


def state_record(path: os.PathLike[str] | str, state: dict) -> dict:
    path = Path(path).resolve(strict=True)
    _require(path.is_file() and not path.is_symlink(), "state must be regular")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "internal_state_sha256": internal_state_hashes(state),
    }


def load_run_manifest(path: os.PathLike[str] | str) -> dict:
    manifest_path = Path(path).resolve(strict=True)
    _require(
        manifest_path.is_file() and not manifest_path.is_symlink(),
        "P2 manifest must be a regular file",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == RUN_MANIFEST_SCHEMA, "manifest schema")
    _require(manifest.get("experiment_protocol") == PROTOCOL, "protocol id")
    run_kind = manifest.get("run_kind")
    _require(run_kind in {"formal", "smoke"}, "invalid P2 run kind")
    expected_seeds = SEEDS if run_kind == "formal" else (18,)
    _require(manifest.get("seed") in expected_seeds, "invalid P2 seed")
    branch = manifest.get("branch")
    _require(branch in BRANCHES, "invalid P2 branch")
    expected = BRANCHES[branch]
    _require(manifest.get("pulse_arm") == expected["pulse_arm"], "pulse arm")
    _require(manifest.get("chase_arm") == "A", "chase arm must be A")
    _require(manifest.get("source_kimg") == SOURCE_KIMG, "source boundary")
    _require(manifest.get("pulse_end_kimg") == PULSE_END_KIMG, "pulse boundary")
    _require(manifest.get("chase_end_kimg") == CHASE_END_KIMG, "chase boundary")
    _require(
        _HEX64.fullmatch(str(manifest.get("protocol_sha256", ""))) is not None,
        "protocol SHA256",
    )
    _require(
        _HEX40.fullmatch(str(manifest.get("implementation_commit", "")))
        is not None,
        "implementation commit",
    )
    assets = manifest.get("asset_sha256")
    _require(assets == ASSET_SHA256, "asset hashes differ from frozen P2 assets")
    source = manifest.get("source_state")
    _require(isinstance(source, dict), "missing source-state record")
    _require(os.path.isabs(str(source.get("path", ""))), "absolute source path")
    _require(isinstance(source.get("bytes"), int) and source["bytes"] > 0,
             "source byte count")
    _require(_HEX64.fullmatch(str(source.get("sha256", ""))) is not None,
             "source SHA256")
    required_internal = {
        "online_parameters", "ema", "radam", "gradscaler", "rank_rng",
        "sampler", "loss_control", "trajectory_config", "data_cursor",
        "attempted_iteration", "successful_optimizer_steps", "cur_nimg",
    }
    internal = source.get("internal_state_sha256")
    _require(isinstance(internal, dict) and set(internal) == required_internal,
             "incomplete source internal-state record")
    for name in (
        "online_parameters", "ema", "radam", "gradscaler",
        "loss_control", "trajectory_config",
    ):
        _require(_HEX64.fullmatch(str(internal[name])) is not None,
                 f"invalid source internal hash: {name}")
    for name in ("rank_rng", "sampler"):
        _require(
            isinstance(internal[name], list) and len(internal[name]) == 1
            and _HEX64.fullmatch(str(internal[name][0])) is not None,
            f"invalid source rank hash: {name}",
        )
    _require(internal["attempted_iteration"] == SOURCE_ATTEMPT,
             "source attempted iteration")
    _require(internal["cur_nimg"] == SOURCE_KIMG * 1000, "source cur_nimg")
    _require(internal["data_cursor"] == [SOURCE_KIMG * 1000], "source cursor")
    output = manifest.get("immutable_output_root")
    _require(os.path.isabs(str(output or "")), "absolute immutable output root")
    if run_kind == "formal":
        expected_gpu = 0 if manifest["seed"] <= 23 else 1
        _require(manifest.get("gpu_index") == expected_gpu,
                 "frozen GPU assignment")
    _require(_HEX64.fullmatch(str(manifest.get("source_inventory_sha256", "")))
             is not None, "source inventory SHA256")
    return manifest


def factorial_for_arm(arm: str) -> dict:
    target, denominator = ARM_FACTORS[arm]
    return {
        "enabled": True,
        "protocol": "q256_target_weight_v1",
        "arm": arm,
        "target_gap_scale": target,
        "denominator_gap_scale": denominator,
    }


def phase_for_resume_state(state: dict, manifest: dict) -> dict:
    cur_nimg = int(state.get("cur_nimg", -1))
    attempted = int(state.get("attempted_iteration", -1))
    if cur_nimg == SOURCE_KIMG * 1000 and attempted == SOURCE_ATTEMPT:
        _require("pulse_chase" not in state, "source already carries P2 branch state")
        return {
            "name": "pulse",
            "start_kimg": SOURCE_KIMG,
            "end_kimg": PULSE_END_KIMG,
            "arm": manifest["pulse_arm"],
        }
    if cur_nimg == PULSE_END_KIMG * 1000 and attempted == PULSE_END_ATTEMPT:
        expected = state_metadata(manifest, phase="pulse")
        _require(state.get("pulse_chase") == expected,
                 "512 state P2 metadata mismatch")
        return {
            "name": "chase",
            "start_kimg": PULSE_END_KIMG,
            "end_kimg": CHASE_END_KIMG,
            "arm": "A",
        }
    raise RuntimeError("P2 resume must be exactly B@384 or branch@512")


def state_metadata(manifest: dict, *, phase: str) -> dict:
    _require(phase in {"pulse", "chase"}, "invalid P2 phase")
    source = manifest["source_state"]
    return {
        "schema": STATE_SCHEMA,
        "experiment_protocol": PROTOCOL,
        "run_kind": manifest["run_kind"],
        "seed": manifest["seed"],
        "branch": manifest["branch"],
        "phase_completed": phase,
        "pulse_arm": manifest["pulse_arm"],
        "chase_arm": "A",
        "source_state_path": source["path"],
        "source_state_sha256": source["sha256"],
        "source_internal_state_sha256": copy.deepcopy(
            source["internal_state_sha256"]
        ),
        "protocol_sha256": manifest["protocol_sha256"],
        "implementation_commit": manifest["implementation_commit"],
        "asset_sha256": copy.deepcopy(ASSET_SHA256),
        "gpu_index": manifest["gpu_index"],
        "gpu_uuid": manifest["gpu_uuid"],
    }


def verify_resume_state_file(path: os.PathLike[str] | str, manifest: dict,
                             phase: dict) -> None:
    path = Path(path).resolve(strict=True)
    if phase["name"] == "pulse":
        source = manifest["source_state"]
        _require(path == Path(source["path"]).resolve(), "source path mismatch")
        _require(path.stat().st_size == source["bytes"], "source bytes mismatch")
        _require(sha256_file(path) == source["sha256"], "source SHA mismatch")
    else:
        expected = (
            Path(manifest["immutable_output_root"])
            / f"training-state-kimg{PULSE_END_KIMG:06d}.pt"
        ).resolve()
        _require(path == expected, "chase must resume the immutable 512 state")


def verify_resume_state(state: dict, manifest: dict, phase: dict) -> None:
    required = {
        "net", "ema", "optimizer_state", "gradscaler_state", "loss_fn_state",
        "rank_states", "factorial", "trajectory_config",
        "trajectory_config_sha256", "attempted_iteration",
        "successful_optimizer_steps", "cur_nimg",
    }
    _require(required.issubset(state), "resume training-state is incomplete")
    _require(
        reproducibility.state_sha256(state["trajectory_config"])
        == state["trajectory_config_sha256"],
        "resume trajectory-config hash mismatch",
    )
    if phase["name"] == "pulse":
        _require(state["factorial"] == factorial_for_arm("B"),
                 "P2 source is not B@384")
        _require(internal_state_hashes(state)
                 == manifest["source_state"]["internal_state_sha256"],
                 "source internal state hash mismatch")
    else:
        _require(state["factorial"] == factorial_for_arm(manifest["pulse_arm"]),
                 "512 pulse-arm identity mismatch")
    ranks = state["rank_states"]
    _require(len(ranks) == 1, "P2 requires one rank/GPU per cell")
    _require(int(ranks[0]["sampler_state"]["consumed_samples"])
             == phase["start_kimg"] * 1000, "resume data cursor mismatch")


def trajectory_configs_compatible(source: dict, current: dict,
                                  manifest: dict, phase: dict) -> bool:
    source = reproducibility.canonical_json_data(copy.deepcopy(source))
    current = reproducibility.canonical_json_data(copy.deepcopy(current))
    if source.pop("total_kimg", None) != phase["start_kimg"]:
        return False
    if current.pop("total_kimg", None) != phase["end_kimg"]:
        return False
    source.pop("matched_randomness_audit", None)
    current.pop("matched_randomness_audit", None)
    source_loss = source.get("loss_kwargs")
    current_loss = current.get("loss_kwargs")
    if not isinstance(source_loss, dict) or not isinstance(current_loss, dict):
        return False
    for key in ("arm", "target_gap_scale", "denominator_gap_scale"):
        source_loss.pop(key, None)
        current_loss.pop(key, None)
    return source == current
