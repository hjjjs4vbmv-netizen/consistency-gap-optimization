"""Deterministically rebuild cross-seed q256 moment-reset summaries."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


REQUIRED_METRICS = (
    "a_star", "grad_cosine", "R_grad", "s_star", "c_star", "update_cosine",
    "R_opt", "update_norm_ratio", "h_i_weighted_mean", "h_i_weighted_std",
    "h_i_p05", "h_i_p50", "h_i_p95", "effective_support_coordinate_coverage",
    "effective_support_energy_coverage", "off_support_candidate_energy_exact",
    "H_K", "H_equals_R_opt_identity_residual",
    "H_K_squared_minus_R_opt_squared_energy_gap",
)


def _read(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _finite_metrics(receipt: dict) -> bool:
    whole = receipt["whole_model"]
    return all(isinstance(whole.get(key), (int, float))
               and math.isfinite(float(whole[key])) for key in REQUIRED_METRICS)


def _pair_gate(real: dict, reset: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    identity = (real["training_seed"], real["audit_seed"])
    if identity != (reset["training_seed"], reset["audit_seed"]):
        errors.append("receipt identity mismatch")
    if real["condition"] != "real" or reset["condition"] != "reset_moments":
        errors.append("condition mismatch")
    for receipt in (real, reset):
        if receipt.get("reference_gap_scale") != 1.0 or receipt.get("probe_gap_scale") != 1.1:
            errors.append(f"{receipt.get('condition')}: gap-scale metadata mismatch")
        if not _finite_metrics(receipt):
            errors.append(f"{receipt.get('condition')}: required metric non-finite")
        if not receipt["source_state_non_committing"].get("preserved"):
            errors.append(f"{receipt.get('condition')}: source state not preserved")
        if not receipt["whole_model"].get("H_K_equals_R_opt_identity"):
            errors.append(f"{receipt.get('condition')}: H=R_opt identity failed")
        if receipt["whole_model"].get("step_skipped"):
            errors.append(f"{receipt.get('condition')}: virtual step skipped")
        for label, branch in receipt["branches"].items():
            if branch.get("step_skipped"):
                errors.append(f"{receipt.get('condition')}/{label}: virtual step skipped")
            if not branch.get("gradient_injection_identical"):
                errors.append(f"{receipt.get('condition')}/{label}: cached gradient changed")
            if not branch.get("gradscaler_preserved"):
                errors.append(f"{receipt.get('condition')}/{label}: GradScaler changed")
    for label in ("reference", "probe"):
        real_hash = real["gradient_contract"][label]["gradient_sha256"]
        reset_hash = reset["gradient_contract"][label]["gradient_sha256"]
        if real_hash != reset_hash:
            errors.append(f"{label}: real/reset gradient hash mismatch")
    if real["whole_model"]["R_grad"] != reset["whole_model"]["R_grad"]:
        errors.append("real/reset R_grad mismatch")
    reset_contracts = [reset["branches"][label]["reset_contract"]
                       for label in ("reference", "probe")]
    for label, contract in zip(("reference", "probe"), reset_contracts):
        if not (contract and contract["exp_avg_all_zero"]
                and contract["exp_avg_sq_all_zero"]
                and contract["per_parameter_step_preserved"]
                and contract["param_groups_preserved"]):
            errors.append(f"reset_moments/{label}: reset contract failed")
    return not errors, errors


def build(receipts_root: Path) -> tuple[dict, list[dict], str, dict]:
    real_paths = sorted(receipts_root.glob("seed*/audit*/real.json"))
    pairs = []
    all_errors = []
    for real_path in real_paths:
        reset_path = real_path.with_name("reset_moments.json")
        if not reset_path.is_file():
            all_errors.append(f"missing {reset_path}")
            continue
        real, reset = _read(real_path), _read(reset_path)
        passed, errors = _pair_gate(real, reset)
        suppression = 1.0 - reset["whole_model"]["R_opt"] / real["whole_model"]["R_opt"]
        pairs.append({
            "training_seed": real["training_seed"],
            "audit_seed": real["audit_seed"],
            "R_opt_real": real["whole_model"]["R_opt"],
            "R_opt_reset": reset["whole_model"]["R_opt"],
            "R_grad": real["whole_model"]["R_grad"],
            "suppression": suppression,
            "pair_gate_pass": passed,
        })
        all_errors.extend(f"seed{real['training_seed']}/audit{real['audit_seed']}: {error}"
                          for error in errors)
    by_seed: dict[int, list[dict]] = {}
    for pair in pairs:
        by_seed.setdefault(pair["training_seed"], []).append(pair)
    rows = []
    for seed in sorted(by_seed):
        values = [item["suppression"] for item in by_seed[seed]]
        rows.append({
            "training_seed": seed,
            "audit_count": len(values),
            "median_suppression": statistics.median(values),
            "min_suppression": min(values),
            "max_suppression": max(values),
            "median_R_opt_real": statistics.median(item["R_opt_real"] for item in by_seed[seed]),
            "median_R_opt_reset": statistics.median(item["R_opt_reset"] for item in by_seed[seed]),
            "all_pair_gates_pass": all(item["pair_gate_pass"] for item in by_seed[seed]),
        })
    complete_seeds = {row["training_seed"] for row in rows if row["audit_count"] == 8}
    seed_medians = {row["training_seed"]: row["median_suppression"] for row in rows}
    go_count = sum(value >= 0.25 for value in seed_medians.values())
    gate = {
        "required_training_seeds": [3, 4, 5],
        "required_audits_per_seed": 8,
        "complete_cross_seed_receipts": complete_seeds == {3, 4, 5},
        "seeds_with_median_suppression_ge_0p25": go_count,
        "at_least_two_of_three_medians_ge_0p25": go_count >= 2,
        "no_seed_median_below_minus_0p05": (
            len(seed_medians) == 3 and all(value >= -0.05 for value in seed_medians.values())),
        "all_pair_gates_pass": bool(pairs) and all(item["pair_gate_pass"] for item in pairs),
        "all_required_metrics_finite": not any("non-finite" in error for error in all_errors),
    }
    gate["go"] = all((gate["complete_cross_seed_receipts"],
                      gate["at_least_two_of_three_medians_ge_0p25"],
                      gate["no_seed_median_below_minus_0p05"],
                      gate["all_pair_gates_pass"], gate["all_required_metrics_finite"]))
    provenance = {}
    for real_path in real_paths:
        receipt = _read(real_path)
        seed = str(receipt["training_seed"])
        provenance[seed] = {
            "source_state_sha256": receipt["provenance"]["source_state_sha256"],
            "checkpoint_sha256": receipt["provenance"]["checkpoint_sha256"],
            "code_commit": receipt["provenance"]["code_commit"],
            "runner_sha256": receipt["provenance"]["runner_sha256"],
            "audit_library_sha256": receipt["provenance"]["audit_library_sha256"],
        }
    summary = {
        "schema_version": 1,
        "reference_gap_scale": 1.0,
        "probe_gap_scale": 1.1,
        "suppression_definition": "1 - R_opt_reset / R_opt_real",
        "audit_seeds": sorted({item["audit_seed"] for item in pairs}),
        "per_audit": sorted(pairs, key=lambda item: (item["training_seed"], item["audit_seed"])),
        "per_training_seed": rows,
        "preregistered_go_gate": gate,
        "errors": sorted(all_errors),
    }
    verdict = "GO" if gate["go"] else "NO-GO"
    report_lines = [
        "# q256 g=1.00/1.10 RAdam moment-reset manipulation check",
        "",
        f"**Pre-registered verdict: {verdict}.**",
        "",
        "## Seed-level results",
        "",
        "| Training seed | Audit batches | Median suppression | Min | Max |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['training_seed']} | {row['audit_count']} | "
            f"{row['median_suppression']:.6f} | {row['min_suppression']:.6f} | "
            f"{row['max_suppression']:.6f} |")
    report_lines.extend([
        "",
        "Suppression is defined as `1 - R_opt_reset / R_opt_real`.",
        "",
        "## Interpretation boundary",
        "",
        "This audit can determine whether clearing accumulated RAdam moments lowers optimizer-update divergence for the formal g=1.10 treatment at the frozen q256 source states. It does not establish that optimizer memory caused an FID improvement. Audit minibatches are not independent training replicates, and this is not a full-training intervention.",
        "",
        "No training, sample generation, FID, or KID computation was performed.",
    ])
    return summary, rows, "\n".join(report_lines) + "\n", provenance


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    summary, rows, report, provenance = build(args.receipts_root)
    _write_json(args.out / "summary.json", summary)
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("training_seed", "audit_count", "median_suppression", "min_suppression",
                  "max_suppression", "median_R_opt_real", "median_R_opt_reset",
                  "all_pair_gates_pass")
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "REPORT.md").write_text(report, encoding="utf-8")
    _write_json(args.out / "provenance_manifest.json", {
        "code_and_source_states": provenance,
        "summary_inputs": [str(path.relative_to(args.receipts_root))
                           for path in sorted(args.receipts_root.glob("seed*/audit*/*.json"))],
    })
    print(json.dumps(summary["preregistered_go_gate"], indent=2, sort_keys=True))
    return 0 if summary["preregistered_go_gate"]["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
