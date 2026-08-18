#!/usr/bin/env python3
"""Run the hash-bound cross-seed optimizer-geometry operation.

The operation has two deliberately different layers at one frozen state per
training seed:

* Layer A is a same-state, one-step virtual fork.  It delegates to
  ``analysis/radam_stateful_update_audit.py`` and restores the complete RAdam
  state plus GradScaler before comparing g=1.0 with g=1.3.
* Layer B is the canonical #47/#58 20-step prospective scalar-history replay.
  It first creates the paired raw history, then runs
  ``analysis/scalar_history_predictor.py`` at the same endpoint (step 19).

``seed3`` is an already-existing, immutable reference artifact.  Only seed4
and seed5 are executed here, and each must be bound to a distinct exact Arm-A
training trajectory.  The runner rejects mutable ``latest`` aliases, missing
hashes, a partially populated output root, and every protocol drift that would
turn batch replication into an alleged training-seed replication.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "cross-seed-gap-induced-optimizer-divergence-v1"
MANIFEST_KIND = "cross-seed-optimizer-geometry-runtime-binding"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
NEW_SEEDS = (4, 5)
EXPECTED_SEED45_CHECKPOINTS = {
    4: "ac94e7b07e5b7628e6b14b26155fb3de09e42373497183d39aba4fe9863663c9",
    5: "21fab0e501bb27032c0e49a553b05a2800ea0fbe20a2a1d94a6bbf5276f2b72a",
}


def fail(message: str) -> None:
    raise SystemExit(f"[cross-seed optimizer geometry] ERROR: {message}")


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {label} {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must be a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically write a receipt; never leave a valid-looking partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        fail(f"{label} must be a concrete lowercase SHA-256, not a placeholder")
    return value


def require_number(value: Any, label: str) -> float:
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(float(value)):
        fail(f"{label} must be a finite number")
    return float(value)


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    return value


def reject_latest_alias(path_text: Any, label: str) -> str:
    if not isinstance(path_text, str) or not path_text.strip():
        fail(f"{label} must be a non-empty explicit path")
    if "latest" in Path(path_text).name.lower():
        fail(f"{label} must name an exact archived/numbered file, not a latest alias: {path_text}")
    return path_text


def _seed_rows(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = manifest.get("seed_rows")
    if not isinstance(rows, list) or len(rows) != 3:
        fail("seed_rows must contain exactly the seed3, seed4, and seed5 rows")
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        row = require_mapping(row, "seed row")
        seed = row.get("training_seed")
        if not isinstance(seed, int) or seed in indexed:
            fail("each seed row needs a unique integer training_seed")
        indexed[seed] = row
    if tuple(sorted(indexed)) != (3, 4, 5):
        fail("seed_rows must be exactly training seeds [3, 4, 5]")
    return indexed


def validate_manifest(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Validate logical binding only; file hashes are checked immediately before use."""
    if manifest.get("schema_version") != 1:
        fail("schema_version must be 1")
    if manifest.get("manifest_kind") != MANIFEST_KIND:
        fail(f"manifest_kind must be {MANIFEST_KIND!r}")
    if manifest.get("protocol") != PROTOCOL_ID:
        fail(f"protocol must be {PROTOCOL_ID!r}")

    canonical = require_mapping(manifest.get("canonical"), "canonical")
    if canonical.get("budget_kimg") != 256:
        fail("this operation is frozen at K=256 kimg")
    arm = require_mapping(canonical.get("arm"), "canonical.arm")
    if arm.get("name") != "arm_a" or arm.get("method") != "fixed" or arm.get("gap_scale") != 1.0:
        fail("canonical.arm must bind the fixed Arm-A g=1.0 state")

    layer_a = require_mapping(canonical.get("layer_a"), "canonical.layer_a")
    layer_b = require_mapping(canonical.get("layer_b"), "canonical.layer_b")
    for layer, label in ((layer_a, "layer_a"), (layer_b, "layer_b")):
        if layer.get("reference_gain") != 1.0 or layer.get("candidate_gain") != 1.3:
            fail(f"canonical.{label} must compare g=1.0 against g=1.3")
        if layer.get("batch_size") != 128:
            fail(f"canonical.{label}.batch_size must be the frozen 128")
    if layer_a.get("batch_gpu") != 16 or layer_a.get("probe_rng_seed") != 20260810:
        fail("Layer A must use canonical batch_gpu=16 and probe_rng_seed=20260810")
    if layer_a.get("lr") != 1e-4:
        fail("Layer A must use the canonical RAdam lr=1e-4")
    if layer_a.get("support_atol") != 0.0 or layer_a.get("amp") is not True:
        fail("Layer A must retain exact support and restored AMP/GradScaler")
    if (layer_b.get("n_steps"), layer_b.get("eval_step"), layer_b.get("probe_rng_seed")) != (20, 19, 20260809):
        fail("Layer B must be the canonical 20-step #47/#58 endpoint replay")
    if layer_b.get("lr") != 1e-4:
        fail("Layer B must use the canonical RAdam lr=1e-4")

    dataset = require_mapping(manifest.get("dataset"), "dataset")
    reject_latest_alias(dataset.get("path"), "dataset.path")
    require_sha(dataset.get("sha256"), "dataset.sha256")

    rows = _seed_rows(manifest)
    trajectory_ids: set[str] = set()
    for seed, row in rows.items():
        if row.get("state_kimg") != 256:
            fail(f"seed{seed} must bind K=256, not another training stage")
        trajectory_id = row.get("training_trajectory_id")
        if not isinstance(trajectory_id, str) or not trajectory_id or trajectory_id in trajectory_ids:
            fail("every row must name a unique non-empty training_trajectory_id")
        trajectory_ids.add(trajectory_id)
        require_sha(require_mapping(row.get("training_state"), f"seed{seed}.training_state").get("sha256"),
                    f"seed{seed}.training_state.sha256")
        checkpoint = require_mapping(row.get("checkpoint"), f"seed{seed}.checkpoint")
        require_sha(checkpoint.get("sha256"), f"seed{seed}.checkpoint.sha256")
        if not isinstance(row.get("schedule_q"), int):
            fail(f"seed{seed}.schedule_q must be recorded as an integer")

    if rows[3].get("row_kind") != "existing_artifact":
        fail("seed3 must be an existing-artifact reference row; do not rerun it")
    existing = require_mapping(rows[3].get("existing"), "seed3.existing")
    for key in ("layer_a_receipt", "layer_b_receipt"):
        artifact = require_mapping(existing.get(key), f"seed3.existing.{key}")
        if not isinstance(artifact.get("path"), str) or not artifact["path"]:
            fail(f"seed3.existing.{key}.path must be repository-relative")
        require_sha(artifact.get("sha256"), f"seed3.existing.{key}.sha256")
    raw_predictions = require_mapping(
        existing.get("layer_b_h20_raw_predictions"), "seed3.existing.layer_b_h20_raw_predictions",
    )
    for key in ("h_pred_scalar", "h_actual", "weights", "a_star_series"):
        artifact = require_mapping(raw_predictions.get(key), f"seed3 existing raw {key}")
        if not isinstance(artifact.get("path"), str) or not artifact["path"]:
            fail(f"seed3 existing raw {key}.path must be repository-relative")
        require_sha(artifact.get("sha256"), f"seed3 existing raw {key}.sha256")

    for seed in NEW_SEEDS:
        row = rows[seed]
        if row.get("row_kind") != "new_independent_training_trajectory":
            fail(f"seed{seed} must be marked new_independent_training_trajectory")
        state = require_mapping(row["training_state"], f"seed{seed}.training_state")
        checkpoint = require_mapping(row["checkpoint"], f"seed{seed}.checkpoint")
        reject_latest_alias(state.get("path"), f"seed{seed}.training_state.path")
        reject_latest_alias(checkpoint.get("path"), f"seed{seed}.checkpoint.path")
        if checkpoint["sha256"] != EXPECTED_SEED45_CHECKPOINTS[seed]:
            fail(f"seed{seed}.checkpoint.sha256 disagrees with the Role-C→D handoff")
        commit = row.get("executed_training_source_commit")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            fail(f"seed{seed}.executed_training_source_commit must be a 40-hex commit")
    return rows


def check_bound_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        fail(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        fail(f"{label} SHA-256 mismatch: {actual} != {expected_sha256}")


def read_existing_seed3(row: dict[str, Any], canonical: dict[str, Any]) -> dict[str, Any]:
    """Validate existing immutable seed3 receipts without pretending to rerun them."""
    existing = row["existing"]
    artifacts: dict[str, tuple[Path, dict[str, Any]]] = {}
    for key in ("layer_a_receipt", "layer_b_receipt"):
        artifact = existing[key]
        path = REPO_ROOT / artifact["path"]
        check_bound_file(path, artifact["sha256"], f"seed3 {key}")
        artifacts[key] = (path, load_json(path, f"seed3 {key}"))

    layer_a = artifacts["layer_a_receipt"][1]
    provenance = require_mapping(layer_a.get("provenance"), "seed3 Layer A provenance")
    if provenance.get("training_state_sha256") != row["training_state"]["sha256"]:
        fail("seed3 Layer A receipt is bound to a different training state")
    if provenance.get("checkpoint_sha256") != row["checkpoint"]["sha256"]:
        fail("seed3 Layer A receipt is bound to a different checkpoint")
    verify_layer_a_receipt(layer_a, canonical, label="seed3 existing Layer A")

    layer_b = artifacts["layer_b_receipt"][1]
    if layer_b.get("source_state_sha256") != row["training_state"]["sha256"]:
        fail("seed3 Layer B receipt is bound to a different training state")
    verify_layer_b_receipt(layer_b, canonical, label="seed3 existing Layer B")
    raw_predictions = existing["layer_b_h20_raw_predictions"]
    for key, artifact in raw_predictions.items():
        check_bound_file(REPO_ROOT / artifact["path"], artifact["sha256"], f"seed3 raw {key}")
    return {
        "training_seed": 3,
        "row_kind": row["row_kind"],
        "training_trajectory_id": row["training_trajectory_id"],
        "state_kimg": row["state_kimg"],
        "schedule_q": row["schedule_q"],
        "training_state": row["training_state"],
        "checkpoint": row["checkpoint"],
        "layer_a": {"storage": "repository", "receipt_path": existing["layer_a_receipt"]["path"],
                    "receipt_sha256": existing["layer_a_receipt"]["sha256"], "executed": False},
        "layer_b": {"storage": "repository", "receipt_path": existing["layer_b_receipt"]["path"],
                    "receipt_sha256": existing["layer_b_receipt"]["sha256"], "executed": False,
                    "raw_predictions": existing["layer_b_h20_raw_predictions"]},
    }


def verify_layer_a_receipt(receipt: dict[str, Any], canonical: dict[str, Any], *, label: str) -> None:
    if receipt.get("gains") != [1.0, 1.3]:
        fail(f"{label} gains are not [1.0, 1.3]")
    random_contract = require_mapping(receipt.get("randomness_contract"), f"{label}.randomness_contract")
    for key in ("same_minibatch", "same_t", "same_noise", "same_dropout_rng_state"):
        if random_contract.get(key) is not True:
            fail(f"{label} does not establish {key}")
    source = require_mapping(receipt.get("source_state_non_committing"), f"{label}.source_state_non_committing")
    if source.get("preserved") is not True:
        fail(f"{label} committed or failed to preserve its source state")
    branches = receipt.get("branches")
    if not isinstance(branches, list) or len(branches) != 2 or any(branch.get("step_skipped") for branch in branches):
        fail(f"{label} has an AMP-skipped or incomplete virtual branch")
    whole = require_mapping(receipt.get("whole_model"), f"{label}.whole_model")
    if whole.get("gauge_defined") is not True:
        fail(f"{label} has an undefined optimizer gauge")
    for key in ("a_K_star", "R_grad", "s_K_star", "c_K_star", "R_opt",
                "on_support_gauge_dispersion_energy"):
        require_number(whole.get(key), f"{label}.whole_model.{key}")
    stateful = require_mapping(receipt.get("stateful_radam"), f"{label}.stateful_radam")
    if stateful.get("gradscaler_restored") is not True or stateful.get("moments_nontrivial") is not True:
        fail(f"{label} did not restore non-trivial moments and GradScaler")
    if stateful.get("support_atol") != canonical["layer_a"]["support_atol"]:
        fail(f"{label} uses a different support threshold")
    provenance = require_mapping(receipt.get("provenance"), f"{label}.provenance")
    source_meta = require_mapping(provenance.get("training_state_meta"), f"{label}.training_state_meta")
    if require_number(source_meta.get("cur_nimg"), f"{label}.cur_nimg") != 256000.0:
        fail(f"{label} is not bound to an exact 256-kimg training state")


def verify_layer_b_receipt(receipt: dict[str, Any], canonical: dict[str, Any], *, label: str) -> None:
    layer_b = canonical["layer_b"]
    if receipt.get("T_steps") != layer_b["n_steps"] or receipt.get("n_steps") != layer_b["n_steps"]:
        fail(f"{label} is not a 20-step replay")
    if receipt.get("eval_step") != layer_b["eval_step"]:
        fail(f"{label} does not evaluate the canonical endpoint step 19")
    for key in ("a_star_mean", "a_star_std", "weighted_RMSE_scalar_vs_actual",
                "corr_scalar_vs_actual", "weighted_R2_scalar_vs_actual", "R_opt"):
        require_number(receipt.get(key), f"{label}.{key}")
    if int(receipt.get("effective_coords", 0)) <= 0:
        fail(f"{label} has no effective coordinate support")
    if require_number(receipt.get("source_nimg"), f"{label}.source_nimg") != 256000.0:
        fail(f"{label} is not bound to an exact 256-kimg training state")


def run_command(command: list[str], *, label: str) -> None:
    print("[cross-seed optimizer geometry]", label)
    print("  $", " ".join(command))
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        fail(f"{label} failed with exit code {completed.returncode}")


def artifact_record(root: Path, path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"expected artifact was not created: {path}")
    return {
        "storage": "operation",
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def execute_new_seed(row: dict[str, Any], canonical: dict[str, Any], dataset: dict[str, Any],
                     root: Path, python: str, device: str) -> dict[str, Any]:
    seed = row["training_seed"]
    state = Path(row["training_state"]["path"])
    checkpoint = Path(row["checkpoint"]["path"])
    data = Path(dataset["path"])
    check_bound_file(state, row["training_state"]["sha256"], f"seed{seed} training state")
    check_bound_file(checkpoint, row["checkpoint"]["sha256"], f"seed{seed} checkpoint")
    check_bound_file(data, dataset["sha256"], "dataset")

    seed_root = root / f"seed{seed}"
    layer_a_dir = seed_root / "layer_a"
    layer_b_raw = seed_root / "layer_b" / "raw"
    layer_b_dir = seed_root / "layer_b"
    layer_a = canonical["layer_a"]
    layer_b = canonical["layer_b"]
    state_text, checkpoint_text, data_text = str(state), str(checkpoint), str(data)

    run_command([
        python, str(REPO_ROOT / "analysis" / "radam_stateful_update_audit.py"),
        "--training-state", state_text, "--checkpoint", checkpoint_text, "--data", data_text,
        "--state-kimg", "256", "--batch-size", str(layer_a["batch_size"]),
        "--batch-gpu", str(layer_a["batch_gpu"]), "--seed", str(layer_a["probe_rng_seed"]),
        "--support-atol", str(layer_a["support_atol"]), "--lr", str(layer_a["lr"]), "--amp",
        "--device", device, "--out", str(layer_a_dir),
    ], label=f"seed{seed} Layer A same-state virtual fork")
    layer_a_receipt = layer_a_dir / "radam_update_audit_stateful.json"
    layer_a_payload = load_json(layer_a_receipt, f"seed{seed} Layer A receipt")
    verify_layer_a_receipt(layer_a_payload, canonical, label=f"seed{seed} Layer A")
    provenance = require_mapping(layer_a_payload.get("provenance"), f"seed{seed} Layer A provenance")
    for key, expected in (("training_state_sha256", row["training_state"]["sha256"]),
                          ("checkpoint_sha256", row["checkpoint"]["sha256"]),
                          ("dataset_sha256", dataset["sha256"])):
        if provenance.get(key) != expected:
            fail(f"seed{seed} Layer A {key} does not match the bound manifest")

    request = {
        "protocol": PROTOCOL_ID,
        "seed": seed,
        "training_trajectory_id": row["training_trajectory_id"],
        "training_state": row["training_state"],
        "checkpoint": row["checkpoint"],
        "dataset": dataset,
        "n_steps": layer_b["n_steps"], "eval_step": layer_b["eval_step"],
        "reference_gain": layer_b["reference_gain"], "candidate_gain": layer_b["candidate_gain"],
        "batch_size": layer_b["batch_size"], "lr": layer_b["lr"],
        "probe_rng_seed": layer_b["probe_rng_seed"],
        "replay_note": "Canonical #47/#58 prospective replay from the real fixed Arm-A RAdam state.",
    }
    write_json(layer_b_dir / "request.json", request)
    run_command([
        python, str(REPO_ROOT / "analysis" / "real_history_sweep.py"),
        "--training-state", state_text, "--checkpoint", checkpoint_text, "--data", data_text,
        "--batch-size", str(layer_b["batch_size"]), "--n-steps", str(layer_b["n_steps"]),
        "--seed", str(layer_b["probe_rng_seed"]), "--g-candidate", str(layer_b["candidate_gain"]),
        "--lr", str(layer_b["lr"]), "--device", device, "--out", str(layer_b_raw),
    ], label=f"seed{seed} Layer B paired raw-history fork")
    scalar_receipt = layer_b_dir / "scalar_history_prediction.json"
    run_command([
        python, str(REPO_ROOT / "analysis" / "scalar_history_predictor.py"),
        "--training-state", state_text,
        "--grad-history-1", str(layer_b_raw / "grad_history_1.npy"),
        "--grad-history-g", str(layer_b_raw / "grad_history_g.npy"),
        "--u1", str(layer_b_raw / "u1.npy"), "--ug", str(layer_b_raw / "ug.npy"),
        "--u1-history", str(layer_b_raw / "u1_history.npy"),
        "--ug-history", str(layer_b_raw / "ug_history.npy"),
        "--eval-step", str(layer_b["eval_step"]), "--lr", str(layer_b["lr"]),
        "--seed", str(layer_b["probe_rng_seed"]),
        "--out", str(scalar_receipt),
    ], label=f"seed{seed} Layer B scalar-history predictor")
    scalar_payload = load_json(scalar_receipt, f"seed{seed} Layer B receipt")
    verify_layer_b_receipt(scalar_payload, canonical, label=f"seed{seed} Layer B")
    if scalar_payload.get("source_state_sha256") != row["training_state"]["sha256"]:
        fail(f"seed{seed} Layer B receipt is bound to a different training state")
    if scalar_payload.get("update_source") != "history":
        fail(f"seed{seed} Layer B must use full per-step update histories")

    raw_names = (
        "grad_history_1.npy", "grad_history_g.npy", "u1.npy", "ug.npy",
        "u1_history.npy", "ug_history.npy", "sweep_meta.json",
    )
    raw_artifacts = {name: artifact_record(root, layer_b_raw / name) for name in raw_names}
    for name in ("h_pred_scalar.npy", "h_actual.npy", "weights.npy"):
        raw_artifacts[name] = artifact_record(root, layer_b_dir / name)
    write_json(layer_b_dir / "raw_artifact_manifest.json", {
        "protocol": PROTOCOL_ID, "request_sha256": sha256_file(layer_b_dir / "request.json"),
        "raw_artifacts": raw_artifacts,
    })
    return {
        "training_seed": seed,
        "row_kind": row["row_kind"],
        "training_trajectory_id": row["training_trajectory_id"],
        "state_kimg": row["state_kimg"], "schedule_q": row["schedule_q"],
        "training_state": row["training_state"], "checkpoint": row["checkpoint"],
        "executed_training_source_commit": row["executed_training_source_commit"],
        "layer_a": {**artifact_record(root, layer_a_receipt), "receipt_path": "seed%d/layer_a/radam_update_audit_stateful.json" % seed,
                    "executed": True},
        "layer_b": {**artifact_record(root, scalar_receipt), "receipt_path": "seed%d/layer_b/scalar_history_prediction.json" % seed,
                    "executed": True, "raw_artifact_manifest": artifact_record(root, layer_b_dir / "raw_artifact_manifest.json")},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="filled, hash-bound runtime binding JSON")
    parser.add_argument("--out", type=Path, required=True, help="new server operation root; must not exist")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for child analysis commands")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main(argv: list[str] | None = None) -> int:
    args = parse_args() if argv is None else parse_args_from(argv)
    if args.out.exists():
        fail(f"refusing to reuse existing operation root: {args.out}")
    manifest = load_json(args.manifest, "bound manifest")
    rows = validate_manifest(manifest)
    canonical = manifest["canonical"]
    dataset = manifest["dataset"]
    # Validate the immutable anchor before creating any server output.
    seed_results = [read_existing_seed3(rows[3], canonical)]

    args.out.mkdir(parents=True, exist_ok=False)
    try:
        for seed in NEW_SEEDS:
            seed_results.append(execute_new_seed(rows[seed], canonical, dataset, args.out, args.python, args.device))
        schedule_q_values = [result["schedule_q"] for result in seed_results]
        audit = {
            "schema_version": 1, "protocol": PROTOCOL_ID, "status": "passed",
            "created_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
            "source_commit": source_commit(), "runner_script_sha256": sha256_file(Path(__file__)),
            "bound_manifest": {"path": str(args.manifest), "sha256": sha256_file(args.manifest)},
            "canonical": canonical, "dataset": dataset,
            "seed_results": seed_results,
            "replication_accounting": {
                "training_seeds": [3, 4, 5],
                "new_execution_training_seeds": list(NEW_SEEDS),
                "existing_reference_training_seed": 3,
                "all_kimg_equal": all(result["state_kimg"] == 256 for result in seed_results),
                "all_training_trajectory_ids_distinct": len({result["training_trajectory_id"] for result in seed_results}) == 3,
                "all_schedule_q_equal": len(set(schedule_q_values)) == 1,
                "schedule_q_by_seed": {str(result["training_seed"]): result["schedule_q"] for result in seed_results},
                "batch_replication_is_not_training_seed_replication": True,
            },
        }
        write_json(args.out / "audit_manifest.json", audit)
    except BaseException:
        # The partially written operation root intentionally remains as a forensic record.
        # A retry requires a new root, preventing accidental mixing of raw artifacts.
        raise
    print(f"[cross-seed optimizer geometry] passed; wrote {args.out / 'audit_manifest.json'}")
    return 0


def parse_args_from(argv: list[str]) -> argparse.Namespace:
    previous = sys.argv
    try:
        sys.argv = [previous[0], *argv]
        return parse_args()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())
