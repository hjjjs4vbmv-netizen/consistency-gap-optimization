#!/usr/bin/env python3
"""Fail-closed inventory for the frozen q256 B@384 source cohort."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training import reproducibility


SEEDS = (3, 4, 5)
EXPECTED_SOURCE_SHA = {
    3: "5173a6b1532c3589c8dd1e6095ab3fca4fffd77331c08932688d11df5e7cf7b8",
    4: "724d47531a8ded39af61cd98265efa8dc1dc6ed03e2e080886a243ad9650d210",
    5: "23805fe2eceefed7ed58006f96253d5f5fcfa32887e0833b7af7a5750a2fcb17",
}
DATASET_SHA256 = "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372"
REPLAY_COMMIT = "c8721a05227f3ff171f8dc1f559a64d58281c0ae"
SELECTION_REASON = (
    "正式 seeds3–5 中，所有 seed 都存在 exact full state 的最早 "
    "post-256 common milestone"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def floating_nonfinite(value: Any) -> tuple[int, int]:
    tensors = 0
    nonfinite = 0
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            tensors += 1
            nonfinite += int((~torch.isfinite(value)).sum())
    elif isinstance(value, dict):
        for item in value.values():
            child_tensors, child_nonfinite = floating_nonfinite(item)
            tensors += child_tensors
            nonfinite += child_nonfinite
    elif isinstance(value, (tuple, list)):
        for item in value:
            child_tensors, child_nonfinite = floating_nonfinite(item)
            tensors += child_tensors
            nonfinite += child_nonfinite
    return tensors, nonfinite


def inventory_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["arm"] == "B" and row["budget_kimg"] == "384"
            and int(row["seed"]) in SEEDS
        ]
    indexed = {int(row["seed"]): row for row in rows}
    if set(indexed) != set(SEEDS) or len(rows) != len(SEEDS):
        raise RuntimeError("inventory does not contain exactly one B@384 row per seed")
    return indexed


def audit_one(seed: int, row: dict[str, str], source_root: Path) -> dict[str, Any]:
    path = source_root / f"seed{seed}" / "armB" / "training-state-kimg000384.pt"
    failures: list[str] = []
    if not path.is_file():
        return {"seed": seed, "path": str(path), "status": "FAIL_CLOSED",
                "failures": ["source state missing"]}
    before = sha256_file(path)
    if before != EXPECTED_SOURCE_SHA[seed] or before != row["replay_state_sha256"]:
        failures.append("whole-file SHA256 mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "net", "ema", "optimizer_state", "gradscaler_state", "loss_fn_state",
        "rank_states", "factorial", "trajectory_config", "trajectory_config_sha256",
        "cur_nimg", "attempted_iteration", "successful_optimizer_steps",
    }
    missing = sorted(required - set(state))
    if missing:
        failures.append("missing fields: " + ", ".join(missing))
    fingerprint = {
        "online_model": reproducibility.module_state_sha256(state["net"]),
        "ema_model": reproducibility.module_state_sha256(state["ema"]),
        "optimizer": reproducibility.state_sha256(state["optimizer_state"]),
        "gradscaler": reproducibility.state_sha256(state["gradscaler_state"]),
        "loss_control": reproducibility.state_sha256(state["loss_fn_state"]),
        "trajectory_config": reproducibility.state_sha256(state["trajectory_config"]),
    }
    for key, column in (
        ("online_model", "online_model_canonical_sha256"),
        ("ema_model", "ema_model_canonical_sha256"),
        ("optimizer", "optimizer_canonical_sha256"),
    ):
        if fingerprint[key] != row[column]:
            failures.append(f"{key} internal SHA256 mismatch")
    factorial = state["factorial"]
    trajectory = state["trajectory_config"]
    rank_states = state["rank_states"]
    if int(state["cur_nimg"]) != 384000:
        failures.append("cur_nimg != 384000")
    if int(state["attempted_iteration"]) != 3000:
        failures.append("attempted_iteration != 3000")
    if int(trajectory.get("seed", -1)) != seed:
        failures.append("seed identity mismatch")
    if factorial.get("arm") != "B":
        failures.append("origin arm != B")
    if float(factorial.get("target_gap_scale", -1)) != 1.1:
        failures.append("target_gap_scale != 1.1")
    if float(factorial.get("denominator_gap_scale", -1)) != 1.1:
        failures.append("denominator_gap_scale != 1.1")
    if fingerprint["trajectory_config"] != state["trajectory_config_sha256"]:
        failures.append("trajectory config SHA256 is internally invalid")
    if row["dataset_sha256"] != DATASET_SHA256:
        failures.append("dataset SHA256 mismatch")
    if row["git_commit"] != REPLAY_COMMIT:
        failures.append("PR #79 replay commit mismatch")
    if not state["gradscaler_state"]:
        failures.append("GradScaler state is empty")
    if not isinstance(rank_states, list) or len(rank_states) != 1:
        failures.append("rank state count != 1")
        rank_receipt = {}
    else:
        rank = rank_states[0]
        sampler = rank.get("sampler_state", {})
        if int(sampler.get("consumed_samples", -1)) != 384000:
            failures.append("sampler consumed_samples != 384000")
        rank_receipt = {
            "rng_sha256": reproducibility.state_sha256(rank.get("rng_state")),
            "sampler_sha256": reproducibility.state_sha256(sampler),
            "sampler_consumed_samples": sampler.get("consumed_samples"),
        }
    steps = []
    for item in state["optimizer_state"].get("state", {}).values():
        if "step" not in item:
            failures.append("RAdam state missing parameter-group step")
            continue
        value = item["step"]
        steps.append(int(value.item()) if isinstance(value, torch.Tensor) else int(value))
    if not steps:
        failures.append("RAdam step counters missing")
    tensor_count, nonfinite_count = floating_nonfinite({
        "net": state["net"].state_dict(),
        "ema": state["ema"].state_dict(),
        "optimizer": state["optimizer_state"],
        "gradscaler": state["gradscaler_state"],
    })
    if nonfinite_count:
        failures.append(f"{nonfinite_count} non-finite floating values")
    after = sha256_file(path)
    if after != before:
        failures.append("source file changed during inventory")
    return {
        "seed": seed,
        "path": str(path.resolve()),
        "inventory_path": row["replay_state_path"],
        "file_sha256_before": before,
        "file_sha256_after": after,
        "fingerprint": fingerprint,
        "gradscaler_state": state["gradscaler_state"],
        "radam_step_summary": {
            "parameter_state_count": len(steps),
            "min": min(steps) if steps else None,
            "max": max(steps) if steps else None,
            "unique": sorted(set(steps)),
        },
        "rank_state": rank_receipt,
        "cur_nimg": int(state["cur_nimg"]),
        "attempted_iteration": int(state["attempted_iteration"]),
        "successful_optimizer_steps": int(state["successful_optimizer_steps"]),
        "factorial": factorial,
        "trajectory_config_sha256": state["trajectory_config_sha256"],
        "floating_tensor_count": tensor_count,
        "floating_nonfinite_count": nonfinite_count,
        "status": "PASS" if not failures else "FAIL_CLOSED",
        "failures": failures,
    }


def write_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--inventory", type=Path,
        default=Path("results/q256_target_weight_replay_curve_seed3_5/replay_checkpoint_inventory.csv"),
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()
    rows = inventory_rows(args.inventory)
    sources = [audit_one(seed, rows[seed], args.source_root) for seed in SEEDS]
    passed = all(item["status"] == "PASS" for item in sources)
    payload = {
        "schema": "ect.q256.b384-same-state-source-inventory/v1",
        "status": "PASS" if passed else "FAIL_CLOSED",
        "classification": "post-hoc mechanism replication, not a new confirmatory quality experiment",
        "formal_cohort": list(SEEDS),
        "origin_history": "B",
        "source_kimg": 384,
        "selection_reason": SELECTION_REASON,
        "no_result_dependent_selection": True,
        "dataset_sha256": DATASET_SHA256,
        "pr79_replay_execution_commit": REPLAY_COMMIT,
        "sources": sources,
    }
    write_exclusive(args.out_json, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# q256 B@384 same-state source inventory", "",
        f"- Status: **{payload['status']}**",
        f"- Formal cohort: `{list(SEEDS)}`", "- Origin history: `B`",
        "- Source budget: `384 kimg`", f"- Selection rule: {SELECTION_REASON}", "",
        "| seed | file SHA256 | online | EMA | optimizer | RNG/sampler | finite | status |",
        "|---:|---|---|---|---|---|---|---|",
    ]
    for item in sources:
        fp = item.get("fingerprint", {})
        lines.append(
            f"| {item['seed']} | `{item.get('file_sha256_before', '')}` | "
            f"`{fp.get('online_model', '')}` | `{fp.get('ema_model', '')}` | "
            f"`{fp.get('optimizer', '')}` | "
            f"`{item.get('rank_state', {}).get('rng_sha256', '')}` / "
            f"`{item.get('rank_state', {}).get('sampler_sha256', '')}` | "
            f"{item.get('floating_nonfinite_count', 'NA') == 0} | {item['status']} |"
        )
    write_exclusive(args.out_md, "\n".join(lines) + "\n")
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
