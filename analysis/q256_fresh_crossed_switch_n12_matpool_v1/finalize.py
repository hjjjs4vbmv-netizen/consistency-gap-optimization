#!/usr/bin/env python3
"""Create final execution/statistics report, compute ledger, and checksums."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.q256_fresh_crossed_switch_n12_matpool_v1 import experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    protocol = experiment.load_json(protocol_path)
    experiment.validate_protocol(protocol, protocol_path)
    root = args.output_root.resolve(strict=True)
    if root != Path(protocol["paths"]["formal_output_root"]).resolve():
        raise RuntimeError("final output root differs from protocol")
    training = experiment.load_json(root / "training_matrix_completion_receipt.json")
    integrity = experiment.load_json(root / "training_integrity_report.json")
    seal = experiment.load_json(root / "evaluation" / "evaluation_matrix_seal.json")
    analysis = experiment.load_json(root / "analysis" / "statistics" / "analysis.json")
    if [training.get("status"), integrity.get("status"), seal.get("status"), analysis.get("status")] != ["PASS", "PASS", "SEALED_PASS", "PASS"]:
        raise RuntimeError("finalization requires complete execution, seal, and analysis PASS")
    ledger = []
    for path in sorted((root / "training").glob("seed*/**/compute_completion_receipt.json")):
        row = experiment.load_json(path)
        ledger.append({"phase": "training", "label": row["label"],
                       "gpu_index": "", "elapsed_seconds": row["elapsed_seconds"],
                       "gpu_hours": row["elapsed_seconds"] / 3600})
    for path in sorted((root / "evaluation" / "receipts").glob("*.json")):
        row = experiment.load_json(path)
        ledger.append({"phase": "evaluation", "label": row["opaque_id"],
                       "gpu_index": row["gpu_index"], "elapsed_seconds": row["elapsed_seconds"],
                       "gpu_hours": row["elapsed_seconds"] / 3600})
    cost_path = root / "compute_cost.csv"
    with cost_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger[0]))
        writer.writeheader(); writer.writerows(ledger)
    head = subprocess.check_output(["git", "-C", protocol["paths"]["repository_root"], "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", protocol["paths"]["repository_root"], "status", "--porcelain"], text=True).strip()
    h = analysis["summaries"]["H"]
    report = f"""# Fresh q256 crossed-switch replication report

## Execution and integrity status

- Training matrix: PASS ({integrity['counts']['prefixes']}/24 prefixes; {integrity['counts']['suffixes']}/48 suffixes).
- Blind evaluation: SEALED_PASS ({seal['sealed_jobs']}/264 jobs), decoded only after the full matrix seal.
- Protocol SHA256: `{experiment.sha256_file(protocol_path)}`.
- Implementation commit: `{protocol['implementation_commit']}`; final HEAD `{head}`; clean worktree: `{not bool(status)}`.
- Host: `{protocol['hostname']}`; GPU UUIDs: {', '.join(gpu['uuid'] for gpu in protocol['gpus'])}.
- Dataset SHA256: `{protocol['assets']['dataset']['sha256']}`.
- Transfer SHA256: `{protocol['assets']['transfer']['sha256']}`.
- Rebuilt runtime manifest SHA256: `{protocol['assets']['runtime_manifest']['sha256']}`.

## Statistical primary verdict

- Decision: **{analysis['primary_verdict']}**.
- H mean: {h['mean']:.9g}; median: {h['median']:.9g}; sample SD: {h['sample_sd']:.9g}.
- Two-sided 95% CI: [{h['ci95_two_sided'][0]:.9g}, {h['ci95_two_sided'][1]:.9g}].
- TOST two-sided 90% CI: [{h['ci90_two_sided'][0]:.9g}, {h['ci90_two_sided'][1]:.9g}].
- Exact two-sided sign-flip p: {h['exact_two_sided_sign_flip_p']:.9g}; negative directions: {h['negative_count']}/12.

## Claim boundary

The primary classification is determined only by seed-level H from 1024-kimg NFE1 FID-50k under the frozen rules. NFE2, KID, intermediate milestones, AULC, single-cell BA, interaction, and checkpoint-quality diagnostics are descriptive and cannot alter or rescue the primary verdict. Execution PASS establishes protocol and data integrity; it does not itself establish the scientific hypothesis.
"""
    report_path = root / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    deliverables = [protocol_path, args.preflight.resolve(strict=True), args.public_manifest.resolve(strict=True),
                    root / "training_integrity_report.json", root / "evaluation" / "evaluation_matrix_seal.json",
                    root / "analysis" / "decoded_results.json", root / "analysis" / "statistics" / "analysis.json",
                    root / "analysis" / "statistics" / "primary_decision.json",
                    root / "analysis" / "statistics" / "H_C_I_Q_G_per_seed.csv",
                    root / "analysis" / "statistics" / "decoded_evaluation_results.csv",
                    root / "analysis" / "statistics" / "AULC_diagnostic.csv", cost_path, report_path]
    sums = root / "SHA256SUMS.txt"
    with sums.open("x", encoding="utf-8") as handle:
        for path in deliverables:
            handle.write(f"{experiment.sha256_file(path)}  {path}\n")
    experiment.atomic_json(root / "final_completion_receipt.json", {
        "schema": "ect.q256.fresh-crossed-switch-final-completion/v1", "status": "PASS",
        "execution_integrity": "PASS", "primary_verdict": analysis["primary_verdict"],
        "protocol_sha256": experiment.sha256_file(protocol_path), "implementation_commit": protocol["implementation_commit"],
        "final_git_head": head, "final_git_clean": not bool(status), "sha256sums": str(sums),
        "compute_ledger_rows": len(ledger),
    })
    print(json.dumps({"status": "PASS", "primary_verdict": analysis["primary_verdict"],
                      "report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
