#!/usr/bin/env python3
"""Fail-closed cross-arm and exact-resume verifier for the q256 smoke gate.

The four input directories must already have immutable PASS receipts from
``verify_q256_target_weight_arm.py``.  The artifacts loaded here are trusted
training outputs from this repository: PyTorch state files are pickle-backed
and must never be supplied by an untrusted party.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_q256_target_weight_matrix as launcher
from scripts import verify_q256_target_weight_arm as arm_verifier
from training import reproducibility


VERIFIER_VERSION = "2"
VALIDATION_SCHEMA = "ect.q256.target-weight-smoke-matrix-validation/v2"
AMP_SKIP_POLICY = {
    "schema": "ect.q256.target-weight-amp-skip-policy/v2",
    "kind": "observe_then_require_cross_arm_count_equivalence_within_seed",
    "allowed_region": "tick_0_amp_warmup_only",
    "warmup_processed_nimg_exclusive_upper_bound": 10_000,
    "require_finite_loss": True,
    "require_raw_nonfinite_exactly_on_skipped_attempts": True,
    "require_cross_arm_equal_skip_count_within_seed": True,
    "require_cross_arm_equal_successful_update_count_within_seed": True,
    "allow_objective_dependent_skip_locations": True,
}
VALIDATION_FILENAME = "q256_target_weight_smoke_matrix_validation_v2.json"

ARMS = tuple(arm_verifier.ARMS)
COMMON_TRAJECTORY_FIELDS = (
    "batch_sha256",
    "t_sha256",
    "base_r_sha256",
)
TARGET_TRAJECTORY_FIELDS = (
    "target_r_sha256",
    "target_delta_sha256",
    "target_r_zero_count",
    "target_r_equal_t_count",
    "target_scaled_to_zero_count",
    "target_delta_min",
    "target_delta_max",
    "target_delta_mean",
)
DENOMINATOR_TRAJECTORY_FIELDS = (
    "denominator_r_sha256",
    "denominator_delta_sha256",
    "denominator_r_zero_count",
    "denominator_r_equal_t_count",
    "denominator_scaled_to_zero_count",
    "denominator_delta_min",
    "denominator_delta_max",
    "denominator_delta_mean",
)
NATIVE_FIELD_SUFFIXES = (
    "r_sha256",
    "delta_sha256",
    "r_zero_count",
    "r_equal_t_count",
    "scaled_to_zero_count",
    "delta_min",
    "delta_max",
    "delta_mean",
)
NONCOMPUTATIONAL_TELEMETRY_FIELDS = frozenset(
    ("elapsed_sec", "gpu_hours_cumulative")
)


class MatrixVerificationError(RuntimeError):
    """The smoke matrix or exact-resume comparison is not admissible."""


def fail(message: str) -> None:
    raise MatrixVerificationError(message)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot load {label} {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must contain one JSON object: {path}")
    return value


def _bound_artifact_path(run_dir: Path, raw_relative: Any, label: str) -> Path:
    if not isinstance(raw_relative, str) or not raw_relative:
        fail(f"{label} has an invalid path")
    relative = Path(raw_relative)
    if (
        relative.is_absolute()
        or relative == Path(".")
        or ".." in relative.parts
        or relative.as_posix() != raw_relative
    ):
        fail(f"{label} has a non-canonical or escaping path: {raw_relative!r}")
    path = run_dir / relative
    current = run_dir
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            fail(f"{label} traverses a symlink: {current}")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(run_dir)
    except (OSError, ValueError) as exc:
        fail(f"{label} is missing or escapes its run directory: {exc}")
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        fail(f"{label} is missing or empty: {resolved}")
    return resolved


def _validate_immutable_arm_pass(
    run_dir: Path, *, arm: str, seed: int
) -> dict[str, Any]:
    """Revalidate one arm and its pre-existing immutable artifact bindings."""

    validation_path = run_dir / arm_verifier.VALIDATION_FILENAME
    hashes_path = run_dir / arm_verifier.HASH_RECEIPT_FILENAME
    if not validation_path.is_file() or not hashes_path.is_file():
        fail(
            f"arm {arm} lacks the complete immutable single-arm PASS receipt set: "
            f"{validation_path}, {hashes_path}"
        )
    if validation_path.is_symlink() or hashes_path.is_symlink():
        fail(f"arm {arm} verifier receipts must not be symlinks")

    validation = _load_json(validation_path, f"arm {arm} validation receipt")
    hashes = _load_json(hashes_path, f"arm {arm} artifact-hash receipt")
    identity = {"status": "passed", "mode": "smoke", "arm": arm, "seed": seed}
    if validation.get("schema") != arm_verifier.VALIDATION_SCHEMA:
        fail(f"arm {arm} validation receipt has an unexpected schema")
    if hashes.get("schema") != arm_verifier.HASH_RECEIPT_SCHEMA:
        fail(f"arm {arm} artifact-hash receipt has an unexpected schema")
    for key, expected in identity.items():
        if validation.get(key) != expected or hashes.get(key) != expected:
            fail(f"arm {arm} immutable PASS identity mismatch for {key!r}")
    if validation.get("run_dir") != str(run_dir):
        fail(f"arm {arm} validation receipt is bound to another run directory")
    if hashes.get("run_dir") != str(run_dir):
        fail(f"arm {arm} artifact-hash receipt is bound to another run directory")
    if validation.get("amp_skip_policy") != AMP_SKIP_POLICY:
        fail(f"arm {arm} immutable PASS uses another AMP skip policy")
    enforcement = validation.get("amp_skip_signature_expected_value_enforced")
    if not isinstance(enforcement, bool):
        fail(f"arm {arm} immutable PASS has an invalid AMP enforcement mode")
    observed_skip_attempts = validation.get("amp_skip_attempts")
    if (
        not isinstance(observed_skip_attempts, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in observed_skip_attempts
        )
        or observed_skip_attempts != sorted(set(observed_skip_attempts))
    ):
        fail(f"arm {arm} immutable PASS has a malformed AMP skip signature")

    artifacts = hashes.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        fail(f"arm {arm} artifact-hash receipt contains no bindings")
    required = set(arm_verifier.CORE_ARTIFACTS) | {
        arm_verifier.VALIDATION_FILENAME
    }
    missing = sorted(required - set(artifacts))
    if missing:
        fail(f"arm {arm} artifact-hash receipt lacks required bindings: {missing}")
    resolved_paths: set[Path] = set()
    for raw_relative, binding in artifacts.items():
        path = _bound_artifact_path(
            run_dir, raw_relative, f"arm {arm} PASS-bound artifact"
        )
        if path in resolved_paths:
            fail(f"arm {arm} artifact-hash receipt aliases one artifact twice")
        resolved_paths.add(path)
        if not isinstance(binding, dict) or set(binding) != {"bytes", "sha256"}:
            fail(f"arm {arm} has a malformed binding for {raw_relative!r}")
        size = binding["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            fail(f"arm {arm} has an invalid byte count for {raw_relative!r}")
        try:
            digest = arm_verifier.require_sha256(
                binding["sha256"], f"arm {arm} artifact {raw_relative!r} SHA256"
            )
        except arm_verifier.VerificationError as exc:
            fail(str(exc))
        if path.stat().st_size != size or arm_verifier.sha256_file(path) != digest:
            fail(f"arm {arm} PASS-bound artifact changed: {path}")

    try:
        current = arm_verifier.verify_run(
            run_dir,
            arm=arm,
            seed=seed,
            mode="smoke",
            expected_skip_attempts=(
                observed_skip_attempts if enforcement else None
            ),
            write_receipts=False,
        )
    except arm_verifier.VerificationError as exc:
        fail(f"arm {arm} failed fresh single-arm revalidation: {exc}")
    if current != validation:
        fail(f"arm {arm} immutable validation receipt is not the current report")
    return {
        "report": current,
        "validation_receipt_sha256": arm_verifier.sha256_file(validation_path),
        "artifact_hash_receipt_sha256": arm_verifier.sha256_file(hashes_path),
    }


def _load_state(run_dir: Path) -> dict[str, Any]:
    value = arm_verifier.torch_load_trusted(run_dir / "training-state-latest.pt")
    if not isinstance(value, dict):
        fail(f"training state must be a dict: {run_dir}")
    return value


def _load_arm_inputs(run_dir: Path, *, arm: str, seed: int) -> dict[str, Any]:
    immutable = _validate_immutable_arm_pass(run_dir, arm=arm, seed=seed)
    try:
        runner_completion = launcher.validate_existing_runner_completion(run_dir)
    except launcher.LaunchError as exc:
        fail(f"arm {arm} lacks a monitor-bound runner PASS: {exc}")
    initial = _load_json(
        run_dir / "initial_state_receipt_v1.json", f"arm {arm} initial receipt"
    )
    rows = arm_verifier.read_telemetry(
        run_dir / "factorial_training_telemetry_v1.csv"
    )
    state = _load_state(run_dir)
    rank_states = state.get("rank_states")
    if not isinstance(rank_states, list) or len(rank_states) != 1:
        fail(f"arm {arm} final state does not contain exactly one rank")
    rank = rank_states[0]
    if not isinstance(rank, dict):
        fail(f"arm {arm} final rank state is malformed")
    return {
        "run_dir": run_dir,
        "immutable": immutable,
        "runner_completion": runner_completion,
        "initial": initial,
        "rows": rows,
        "state": state,
        "final_rng_sha256": reproducibility.state_sha256(rank["rng_state"]),
        "final_sampler_sha256": reproducibility.state_sha256(
            rank["sampler_state"]
        ),
    }


def _require_same(
    values: Sequence[tuple[str, Any]], label: str
) -> Any:
    reference_arm, reference = values[0]
    mismatches = [(arm, value) for arm, value in values[1:] if value != reference]
    if mismatches:
        fail(
            f"{label} mismatch: reference {reference_arm}={reference!r}, "
            f"others={mismatches!r}"
        )
    return reference


def _field_trajectory(rows: Sequence[Mapping[str, str]], field: str) -> list[str]:
    return [row[field] for row in rows]


def _compare_field_group(
    arms: Mapping[str, dict[str, Any]], group: Sequence[str], fields: Sequence[str], label: str
) -> dict[str, str]:
    result: dict[str, str] = {}
    for field in fields:
        trajectory = _require_same(
            [(arm, _field_trajectory(arms[arm]["rows"], field)) for arm in group],
            f"{label}.{field}",
        )
        result[field] = reproducibility.state_sha256(trajectory)
    return result


def _compare_native_target_denominator(
    arms: Mapping[str, dict[str, Any]], arm: str
) -> None:
    for attempt, row in enumerate(arms[arm]["rows"], start=1):
        for suffix in NATIVE_FIELD_SUFFIXES:
            target = row[f"target_{suffix}"]
            denominator = row[f"denominator_{suffix}"]
            if target != denominator:
                fail(
                    f"native arm {arm} target/denominator {suffix} mismatch "
                    f"at attempt {attempt}: {target!r} != {denominator!r}"
                )


def _state_component_hashes(state: Mapping[str, Any]) -> dict[str, str]:
    ranks = state["rank_states"]
    rank = ranks[0]
    control_state = {
        field: state[field]
        for field in (
            "reproducibility_schema",
            "attempted_iteration",
            "successful_optimizer_steps",
            "cur_nimg",
            "cur_tick",
            "tick_start_nimg",
            "factorial",
            "trajectory_config",
            "trajectory_config_sha256",
        )
    }
    return {
        "net": reproducibility.module_state_sha256(state["net"]),
        "ema": reproducibility.module_state_sha256(state["ema"]),
        "optimizer": reproducibility.state_sha256(state["optimizer_state"]),
        "gradscaler": reproducibility.state_sha256(state["gradscaler_state"]),
        "rank_rng": reproducibility.state_sha256(rank["rng_state"]),
        "rank_sampler": reproducibility.state_sha256(rank["sampler_state"]),
        "loss": reproducibility.state_sha256(state["loss_fn_state"]),
        "control_state": reproducibility.state_sha256(control_state),
        "snapshot_grid_z": reproducibility.state_sha256(
            state["snapshot_grid_z"]
        ),
        "snapshot_grid_c": reproducibility.state_sha256(
            state["snapshot_grid_c"]
        ),
        "snapshot_grid_size": reproducibility.state_sha256(
            state["snapshot_grid_size"]
        ),
    }


def _computational_telemetry(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    return [
        {
            field: row[field]
            for field in arm_verifier.TELEMETRY_FIELDS
            if field not in NONCOMPUTATIONAL_TELEMETRY_FIELDS
        }
        for row in rows
    ]


def _verify_resume_pair(
    uninterrupted_dir: Path,
    resumed_dir: Path,
    *,
    arm: str,
    seed: int,
) -> dict[str, Any]:
    uninterrupted_dir = uninterrupted_dir.expanduser().resolve()
    resumed_dir = resumed_dir.expanduser().resolve()
    if uninterrupted_dir == resumed_dir:
        fail("uninterrupted and resumed run directories must be distinct")
    try:
        provenance = launcher.validate_exact_resume_provenance(
            uninterrupted_dir,
            resumed_dir,
            arm=arm,
            seed=seed,
            runtime_command=[sys.executable],
            process_env=os.environ,
        )
    except launcher.LaunchError as exc:
        fail(f"exact-resume provenance failed: {exc}")
    uninterrupted = _load_arm_inputs(
        uninterrupted_dir, arm=arm, seed=seed
    )
    resumed = _load_arm_inputs(resumed_dir, arm=arm, seed=seed)

    _require_same(
        [
            (
                "uninterrupted",
                (
                    uninterrupted["immutable"]["report"]["source_git_head"],
                    uninterrupted["immutable"]["report"][
                        "source_content_sha256"
                    ],
                ),
            ),
            (
                "resumed",
                (
                    resumed["immutable"]["report"]["source_git_head"],
                    resumed["immutable"]["report"]["source_content_sha256"],
                ),
            ),
        ],
        "exact-resume source identity",
    )
    _require_same(
        [
            ("uninterrupted", uninterrupted["initial"]["hashes"]),
            ("resumed", resumed["initial"]["hashes"]),
        ],
        "exact-resume initial state hashes",
    )
    uninterrupted_hashes = _state_component_hashes(uninterrupted["state"])
    resumed_hashes = _state_component_hashes(resumed["state"])
    for component in uninterrupted_hashes:
        if uninterrupted_hashes[component] != resumed_hashes[component]:
            fail(
                f"exact-resume final {component} mismatch: "
                f"uninterrupted={uninterrupted_hashes[component]}, "
                f"resumed={resumed_hashes[component]}"
            )

    uninterrupted_telemetry = _computational_telemetry(uninterrupted["rows"])
    resumed_telemetry = _computational_telemetry(resumed["rows"])
    if uninterrupted_telemetry != resumed_telemetry:
        for index, (left, right) in enumerate(
            zip(uninterrupted_telemetry, resumed_telemetry), start=1
        ):
            differing = [field for field in left if left[field] != right[field]]
            if differing:
                fail(
                    "exact-resume computational telemetry mismatch at attempt "
                    f"{index}, fields={differing}"
                )
        fail(
            "exact-resume computational telemetry length mismatch: "
            f"{len(uninterrupted_telemetry)} != {len(resumed_telemetry)}"
        )
    return {
        "status": "passed",
        "arm": arm,
        "uninterrupted_run_dir": str(uninterrupted_dir),
        "resumed_run_dir": str(resumed_dir),
        "final_component_sha256": uninterrupted_hashes,
        "computational_telemetry_sha256": reproducibility.state_sha256(
            uninterrupted_telemetry
        ),
        "excluded_noncomputational_fields": sorted(
            NONCOMPUTATIONAL_TELEMETRY_FIELDS
        ),
        "uninterrupted_validation_receipt_sha256": uninterrupted["immutable"][
            "validation_receipt_sha256"
        ],
        "resumed_validation_receipt_sha256": resumed["immutable"][
            "validation_receipt_sha256"
        ],
        "uninterrupted_artifact_hash_receipt_sha256": uninterrupted[
            "immutable"
        ]["artifact_hash_receipt_sha256"],
        "resumed_artifact_hash_receipt_sha256": resumed["immutable"][
            "artifact_hash_receipt_sha256"
        ],
        "uninterrupted_runner_completion_sha256": uninterrupted[
            "runner_completion"
        ]["sha256"],
        "resumed_runner_completion_sha256": resumed["runner_completion"][
            "sha256"
        ],
        "provenance": provenance,
        "provenance_sha256": provenance["provenance_sha256"],
    }


def _default_receipt_path(run_dirs: Mapping[str, Path]) -> Path:
    common = Path(os.path.commonpath([str(path) for path in run_dirs.values()]))
    if common in run_dirs.values():
        common = common.parent
    return common / VALIDATION_FILENAME


def verify_smoke_matrix(
    run_dirs: Mapping[str, Path],
    *,
    seed: int = 3,
    receipt_path: Path | None = None,
    resume_pair: tuple[Path, Path, str] | None = None,
    write_receipt: bool = True,
) -> dict[str, Any]:
    """Verify one complete seed-3 A/B/C/D smoke matrix."""

    if seed != 3:
        fail("the frozen q256 smoke matrix uses seed 3")
    if set(run_dirs) != set(ARMS):
        fail(
            f"smoke matrix arm keys must be exactly {list(ARMS)}, "
            f"got {sorted(run_dirs)}"
        )
    resolved = {
        arm: Path(run_dirs[arm]).expanduser().resolve() for arm in ARMS
    }
    if len(set(resolved.values())) != len(ARMS):
        fail("each smoke arm must use a distinct run directory")
    arms = {
        arm: _load_arm_inputs(resolved[arm], arm=arm, seed=seed) for arm in ARMS
    }

    initial_hashes = _require_same(
        [(arm, arms[arm]["initial"]["hashes"]) for arm in ARMS],
        "common initial model/EMA/optimizer/GradScaler/RNG/sampler hashes",
    )
    common_initial = _require_same(
        [
            (arm, arms[arm]["initial"]["common_initial_state_sha256"])
            for arm in ARMS
        ],
        "common initial-state digest",
    )
    source_git_head = _require_same(
        [(arm, arms[arm]["immutable"]["report"]["source_git_head"]) for arm in ARMS],
        "source Git head",
    )
    source_content = _require_same(
        [
            (arm, arms[arm]["immutable"]["report"]["source_content_sha256"])
            for arm in ARMS
        ],
        "source content digest",
    )
    skip_attempts_by_arm = {
        arm: arms[arm]["immutable"]["report"]["amp_skip_attempts"]
        for arm in ARMS
    }
    skip_count = _require_same(
        [(arm, len(skip_attempts_by_arm[arm])) for arm in ARMS],
        "AMP skip count",
    )
    successful_optimizer_steps = _require_same(
        [
            (
                arm,
                arms[arm]["immutable"]["report"][
                    "successful_optimizer_steps"
                ],
            )
            for arm in ARMS
        ],
        "successful optimizer-step count",
    )
    skip_enforcement = _require_same(
        [
            (
                arm,
                arms[arm]["immutable"]["report"][
                    "amp_skip_signature_expected_value_enforced"
                ],
            )
            for arm in ARMS
        ],
        "AMP skip-signature enforcement mode",
    )
    final_rng = _require_same(
        [(arm, arms[arm]["final_rng_sha256"]) for arm in ARMS],
        "final rank RNG state",
    )
    final_sampler = _require_same(
        [(arm, arms[arm]["final_sampler_sha256"]) for arm in ARMS],
        "final sampler state",
    )

    common_trajectories = _compare_field_group(
        arms, ARMS, COMMON_TRAJECTORY_FIELDS, "all-arms common trajectory"
    )
    target_scale_1 = _compare_field_group(
        arms, ("A", "D"), TARGET_TRAJECTORY_FIELDS, "target scale 1.0"
    )
    target_scale_1p1 = _compare_field_group(
        arms, ("B", "C"), TARGET_TRAJECTORY_FIELDS, "target scale 1.1"
    )
    denominator_scale_1 = _compare_field_group(
        arms,
        ("A", "C"),
        DENOMINATOR_TRAJECTORY_FIELDS,
        "denominator scale 1.0",
    )
    denominator_scale_1p1 = _compare_field_group(
        arms,
        ("B", "D"),
        DENOMINATOR_TRAJECTORY_FIELDS,
        "denominator scale 1.1",
    )
    _compare_native_target_denominator(arms, "A")
    _compare_native_target_denominator(arms, "B")

    resume_result = None
    if resume_pair is not None:
        uninterrupted_dir, resumed_dir, resume_arm = resume_pair
        if resume_arm not in ARMS:
            fail(f"unknown exact-resume arm {resume_arm!r}")
        resume_result = _verify_resume_pair(
            uninterrupted_dir, resumed_dir, arm=resume_arm, seed=seed
        )

    report = {
        "schema": VALIDATION_SCHEMA,
        "status": "passed",
        "verifier_version": VERIFIER_VERSION,
        "mode": "smoke",
        "seed": seed,
        "arms": {
            arm: {
                "run_dir": str(resolved[arm]),
                "validation_receipt_sha256": arms[arm]["immutable"][
                    "validation_receipt_sha256"
                ],
                "artifact_hash_receipt_sha256": arms[arm]["immutable"][
                    "artifact_hash_receipt_sha256"
                ],
                "runner_completion_path": arms[arm]["runner_completion"]["path"],
                "runner_completion_sha256": arms[arm]["runner_completion"][
                    "sha256"
                ],
            }
            for arm in ARMS
        },
        "source_git_head": source_git_head,
        "source_content_sha256": source_content,
        "initial_common_state_sha256": common_initial,
        "initial_component_hashes": initial_hashes,
        "final_rank_rng_sha256": final_rng,
        "final_sampler_sha256": final_sampler,
        "amp_skip_attempts_by_arm": skip_attempts_by_arm,
        "amp_skip_count": skip_count,
        "successful_optimizer_steps": successful_optimizer_steps,
        "amp_skip_signature_expected_value_enforced": skip_enforcement,
        "amp_skip_policy": AMP_SKIP_POLICY,
        "trajectory_checks": {
            "all_arms_common": common_trajectories,
            "target_A_equals_D": target_scale_1,
            "target_B_equals_C": target_scale_1p1,
            "denominator_A_equals_C": denominator_scale_1,
            "denominator_B_equals_D": denominator_scale_1p1,
            "native_A_target_equals_denominator": True,
            "native_B_target_equals_denominator": True,
        },
        "exact_resume": resume_result,
    }

    if write_receipt:
        target = (
            Path(receipt_path).expanduser().resolve()
            if receipt_path is not None
            else _default_receipt_path(resolved).resolve()
        )
        if target.exists():
            fail(f"immutable matrix PASS receipt already exists: {target}")
        try:
            reproducibility.atomic_json_dump(report, target, overwrite=False)
        except FileExistsError:
            fail(f"immutable matrix PASS receipt already exists: {target}")
        report["validation_receipt"] = str(target)
        report["validation_receipt_sha256"] = arm_verifier.sha256_file(target)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument(
            f"--arm-{arm.lower()}-run-dir", type=Path, required=True
        )
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--receipt-path", type=Path)
    parser.add_argument("--uninterrupted-run-dir", type=Path)
    parser.add_argument("--resumed-run-dir", type=Path)
    parser.add_argument("--resume-arm", choices=ARMS)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="re-run all cross-arm/exact-resume checks without writing a receipt",
    )
    args = parser.parse_args()
    pair_values = (
        args.uninterrupted_run_dir,
        args.resumed_run_dir,
        args.resume_arm,
    )
    if any(value is not None for value in pair_values) and not all(
        value is not None for value in pair_values
    ):
        parser.error(
            "--uninterrupted-run-dir, --resumed-run-dir, and --resume-arm "
            "must be supplied together"
        )
    resume_pair = pair_values if all(value is not None for value in pair_values) else None
    try:
        report = verify_smoke_matrix(
            {
                arm: getattr(args, f"arm_{arm.lower()}_run_dir") for arm in ARMS
            },
            seed=args.seed,
            receipt_path=args.receipt_path,
            resume_pair=resume_pair,
            write_receipt=not args.check_only,
        )
    except MatrixVerificationError as exc:
        raise SystemExit(
            f"[verify_q256_target_weight_smoke_matrix] ERROR: {exc}"
        ) from exc
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
