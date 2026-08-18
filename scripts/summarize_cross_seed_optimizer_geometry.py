#!/usr/bin/env python3
"""Summarize a completed cross-seed optimizer-geometry operation.

The summary is intentionally descriptive.  It reports the raw-gradient and
optimizer-update residuals, the support-aware h_i dispersion, and Layer B's
prospective scalar-history explanatory metrics without turning any of them
into a causal claim about endpoint quality.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "cross-seed-gap-induced-optimizer-divergence-v1"


def fail(message: str) -> None:
    raise SystemExit(f"[cross-seed optimizer geometry summary] ERROR: {message}")


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {label} {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must be a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any, label: str) -> float:
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(float(value)):
        fail(f"{label} must be finite")
    return float(value)


def resolve_receipt(root: Path, record: dict[str, Any], label: str) -> Path:
    storage = record.get("storage")
    path_text = record.get("receipt_path")
    # Early operation roots recorded the generic artifact field ``sha256``;
    # completed roots remain valid when that value matches the receipt bytes.
    expected = record.get("receipt_sha256", record.get("sha256"))
    if storage == "repository":
        path = REPO_ROOT / str(path_text)
    elif storage == "operation":
        path = root / str(path_text)
    else:
        fail(f"{label} has invalid receipt storage {storage!r}")
    if not path.is_file():
        fail(f"{label} receipt is missing: {path}")
    if not isinstance(expected, str) or sha256_file(path) != expected:
        fail(f"{label} receipt SHA-256 does not match audit_manifest")
    return path


def receipt_row(root: Path, seed_result: dict[str, Any]) -> dict[str, Any]:
    seed = seed_result.get("training_seed")
    if not isinstance(seed, int):
        fail("seed result has no integer training_seed")
    layer_a_path = resolve_receipt(root, seed_result.get("layer_a", {}), f"seed{seed} Layer A")
    layer_b_path = resolve_receipt(root, seed_result.get("layer_b", {}), f"seed{seed} Layer B")
    layer_a = load_json(layer_a_path, f"seed{seed} Layer A receipt")
    layer_b = load_json(layer_b_path, f"seed{seed} Layer B receipt")
    whole = layer_a.get("whole_model")
    if not isinstance(whole, dict) or whole.get("gauge_defined") is not True:
        fail(f"seed{seed} Layer A receipt lacks a defined gauge")
    if layer_b.get("T_steps") != 20 or layer_b.get("eval_step") != 19:
        fail(f"seed{seed} Layer B receipt is not the required 20-step endpoint replay")

    on_support_energy = finite(whole.get("on_support_gauge_dispersion_energy"),
                               f"seed{seed}.on_support_gauge_dispersion_energy")
    if on_support_energy < 0:
        fail(f"seed{seed} has negative h_i dispersion energy")
    row = {
        "seed": seed,
        "row_kind": seed_result.get("row_kind"),
        "training_trajectory_id": seed_result.get("training_trajectory_id"),
        "state_kimg": seed_result.get("state_kimg"),
        "schedule_q": seed_result.get("schedule_q"),
        "a_K_star": finite(whole.get("a_K_star"), f"seed{seed}.a_K_star"),
        "R_grad": finite(whole.get("R_grad"), f"seed{seed}.R_grad"),
        "s_K_star": finite(whole.get("s_K_star"), f"seed{seed}.s_K_star"),
        "c_K_star": finite(whole.get("c_K_star"), f"seed{seed}.c_K_star"),
        "R_opt": finite(whole.get("R_opt"), f"seed{seed}.R_opt"),
        "h_i_dispersion_on_support": math.sqrt(on_support_energy),
        "h_i_off_support_candidate_energy": finite(whole.get("off_support_candidate_energy_exact"),
                                                      f"seed{seed}.off_support_candidate_energy_exact"),
        "H_K_identity_check": finite(whole.get("H_K"), f"seed{seed}.H_K"),
        "scalar_history_a_star_mean": finite(layer_b.get("a_star_mean"), f"seed{seed}.a_star_mean"),
        "scalar_history_a_star_std": finite(layer_b.get("a_star_std"), f"seed{seed}.a_star_std"),
        "scalar_history_R2": finite(layer_b.get("weighted_R2_scalar_vs_actual"), f"seed{seed}.scalar R2"),
        "scalar_history_corr": finite(layer_b.get("corr_scalar_vs_actual"), f"seed{seed}.scalar corr"),
        "scalar_history_wRMSE": finite(layer_b.get("weighted_RMSE_scalar_vs_actual"), f"seed{seed}.scalar wRMSE"),
        "layer_a_receipt": str(layer_a_path),
        "layer_b_receipt": str(layer_b_path),
    }
    return row


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def render_markdown(rows: list[dict[str, Any]], accounting: dict[str, Any]) -> str:
    lines = [
        "# Cross-seed replication of gap-induced optimizer divergence",
        "",
        "The table contains one same-state Layer A receipt and one 20-step Layer B receipt per training seed. A row is a training-trajectory observation, not a repeated-minibatch estimate.",
        "",
        "| seed | row | K | schedule q | a* | R_grad | R_opt | h_i disp. (on support) | scalar-history R² | Corr | wRMSE |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {seed} | {row_kind} | {state_kimg} | {schedule_q} | {a_K_star:.6f} | "
            "{R_grad:.6f} | {R_opt:.6f} | {h_i_dispersion_on_support:.6f} | "
            "{scalar_history_R2:.6f} | {scalar_history_corr:.6f} | {scalar_history_wRMSE:.6f} |".format(**row)
        )
    lines.extend([
        "",
        "`h_i` dispersion is the square root of the exact on-support dispersion energy. `H_K` is retained in the CSV as an algebraic identity check (`H_K = R_opt` after off-support energy is included), not as a second mechanism measurement.",
        "",
        "## Accounting and claim boundary",
        "",
        f"- K=256 for every row: `{accounting.get('all_kimg_equal')}`.",
        f"- Distinct named training trajectories: `{accounting.get('all_training_trajectory_ids_distinct')}`.",
        f"- Layer A uses shared minibatch/t/noise/dropout **within** each seed; this pairing does not substitute for training-seed replication.",
        f"- schedule q by seed: `{accounting.get('schedule_q_by_seed')}`; all schedules equal: `{accounting.get('all_schedule_q_equal')}`.",
        "",
        "The historical seed-3 anchor is explicitly retained with its recorded schedule q. If the schedule-q accounting is false, do not pool all three rows as a pure same-configuration seed effect; report the q256 seed4/5 trajectories as independent replications and the seed-3 row as a hash-bound mechanism anchor. In every case, the scalar-history values quantify prospective update-ratio explanatory power, not endpoint-quality causality.",
        "",
    ])
    return "\n".join(lines)


def summarize(audit_root: Path, out: Path) -> list[dict[str, Any]]:
    if not audit_root.is_dir():
        fail(f"audit root does not exist: {audit_root}")
    if out.exists():
        fail(f"refusing to overwrite existing summary directory: {out}")
    manifest = load_json(audit_root / "audit_manifest.json", "audit manifest")
    if manifest.get("protocol") != PROTOCOL_ID or manifest.get("status") != "passed":
        fail("audit_manifest is not a passed cross-seed optimizer-geometry operation")
    results = manifest.get("seed_results")
    if not isinstance(results, list) or {row.get("training_seed") for row in results if isinstance(row, dict)} != {3, 4, 5}:
        fail("audit_manifest must contain exactly seeds 3, 4, and 5")
    rows = sorted((receipt_row(audit_root, result) for result in results), key=lambda row: row["seed"])
    accounting = manifest.get("replication_accounting")
    if not isinstance(accounting, dict) or accounting.get("batch_replication_is_not_training_seed_replication") is not True:
        fail("audit_manifest is missing training-seed replication accounting")
    out.mkdir(parents=True, exist_ok=False)
    fields = list(rows[0])
    with (out / "optimizer_geometry_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1, "protocol": PROTOCOL_ID,
        "audit_manifest_sha256": sha256_file(audit_root / "audit_manifest.json"),
        "replication_accounting": accounting, "rows": rows,
    }
    write_json(out / "optimizer_geometry_summary.json", summary)
    (out / "OPTIMIZER_GEOMETRY_TABLE.md").write_text(render_markdown(rows, accounting), encoding="utf-8")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = summarize(args.audit_root, args.out)
    print(f"wrote {len(rows)} seed rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
