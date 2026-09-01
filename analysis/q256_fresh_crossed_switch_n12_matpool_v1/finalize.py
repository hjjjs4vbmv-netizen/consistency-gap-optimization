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
    parser.add_argument("--eleven-seed-authorization", type=Path)
    parser.add_argument("--evaluation-recovery-authorization", type=Path)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    protocol = experiment.load_json(protocol_path)
    experiment.validate_protocol(protocol, protocol_path)
    root = args.output_root.resolve(strict=True)
    if root != Path(protocol["paths"]["formal_output_root"]).resolve():
        raise RuntimeError("final output root differs from protocol")
    seeds = experiment.SEEDS
    expected_jobs = 264
    eval_root = root / "evaluation"
    analysis_root = root / "analysis"
    training_path = root / "training_matrix_completion_receipt.json"
    integrity_path = root / "training_integrity_report.json"
    authorization_sha = None
    recovery_authorization_sha = None
    recovery_preparation_receipt = None
    if args.eleven_seed_authorization is not None:
        authorization_path = args.eleven_seed_authorization.resolve(strict=True)
        experiment.validate_eleven_seed_authorization(
            authorization_path, protocol_path,
            require_commit=args.evaluation_recovery_authorization is None
        )
        authorization_sha = experiment.sha256_file(authorization_path)
        seeds = experiment.ELEVEN_SEEDS
        expected_jobs = experiment.ELEVEN_JOB_COUNT
        eval_root = root / "evaluation_11seed"
        analysis_root = root / "analysis_11seed"
        training_path = root / "training_matrix_11seed_completion_receipt.json"
        integrity_path = root / "training_integrity_11seed_report.json"
    if args.evaluation_recovery_authorization is not None:
        if authorization_sha is None:
            raise RuntimeError("evaluation recovery finalization requires eleven-seed authorization")
        recovery_path = args.evaluation_recovery_authorization.resolve(strict=True)
        recovery = experiment.validate_evaluation_recovery1_authorization(
            recovery_path, protocol_path, require_commit=True
        )
        if recovery.get("eleven_seed_authorization_sha256") != authorization_sha:
            raise RuntimeError("evaluation recovery finalization amendment binding mismatch")
        recovery_authorization_sha = experiment.sha256_file(recovery_path)
        eval_root = root / recovery["evaluation_dir"]
        analysis_root = root / recovery["analysis_dir"]
        recovery_preparation_receipt = (
            root / "archive" / "manual-evaluation-recovery-1"
            / "evaluation_recovery1_preparation_receipt.json"
        )
    training = experiment.load_json(training_path)
    integrity = experiment.load_json(integrity_path)
    seal = experiment.load_json(eval_root / "evaluation_matrix_seal.json")
    analysis = experiment.load_json(analysis_root / "statistics" / "analysis.json")
    if [training.get("status"), integrity.get("status"), seal.get("status"), analysis.get("status")] != ["PASS", "PASS", "SEALED_PASS", "PASS"]:
        raise RuntimeError("finalization requires complete execution, seal, and analysis PASS")
    if authorization_sha is not None:
        bound = [training.get("eleven_seed_authorization_sha256"),
                 integrity.get("eleven_seed_authorization_sha256"),
                 seal.get("eleven_seed_authorization_sha256"),
                 analysis.get("eleven_seed_authorization_sha256")]
        if bound != [authorization_sha] * 4:
            raise RuntimeError("final eleven-seed artifacts are not amendment-bound")
    if recovery_authorization_sha is not None:
        bound = [seal.get("evaluation_recovery_authorization_sha256"),
                 analysis.get("evaluation_recovery_authorization_sha256")]
        if bound != [recovery_authorization_sha] * 2:
            raise RuntimeError("final artifacts are not evaluation-recovery-bound")
        preparation = experiment.load_json(recovery_preparation_receipt)
        if (preparation.get("status") != "PASS"
                or preparation.get("evaluation_recovery_authorization_sha256")
                != recovery_authorization_sha
                or preparation.get("metrics_executed") is not False):
            raise RuntimeError("evaluation recovery preparation gate is not PASS")
    ledger = []
    for path in sorted((root / "training").glob("seed*/**/compute_completion_receipt.json")):
        if int(path.parts[-3].removeprefix("seed")) not in seeds:
            continue
        row = experiment.load_json(path)
        ledger.append({"phase": "training", "label": row["label"],
                       "gpu_index": "", "elapsed_seconds": row["elapsed_seconds"],
                       "gpu_hours": row["elapsed_seconds"] / 3600})
    for path in sorted((eval_root / "receipts").glob("*.json")):
        row = experiment.load_json(path)
        ledger.append({"phase": "evaluation", "label": row["opaque_id"],
                       "gpu_index": row["gpu_index"], "elapsed_seconds": row["elapsed_seconds"],
                       "gpu_hours": row["elapsed_seconds"] / 3600})
    suffix = "_11SEED" if authorization_sha is not None else ""
    cost_path = root / f"compute_cost{suffix.lower()}.csv"
    with cost_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger[0]))
        writer.writeheader(); writer.writerows(ledger)
    head = subprocess.check_output(["git", "-C", protocol["paths"]["repository_root"], "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", protocol["paths"]["repository_root"], "status", "--porcelain"], text=True).strip()
    h = analysis["summaries"]["H"]
    report = f"""# Fresh q256 crossed-switch replication report

## Execution and integrity status

- Training matrix: PASS ({integrity['counts']['prefixes']}/{2 * len(seeds)} prefixes; {integrity['counts']['suffixes']}/{4 * len(seeds)} suffixes).
- Blind evaluation: SEALED_PASS ({seal['sealed_jobs']}/{expected_jobs} jobs), decoded only after the full amended matrix seal.
- Manual evaluation recovery: {1 if recovery_authorization_sha else 0}; the failed attempt is preserved and the replacement cache passed a non-metric storage gate.
- Analysis population: {len(seeds)} complete seeds ({', '.join(map(str, seeds))}); seed38 excluded by explicit author amendment; the original n=12 claim is abandoned: {authorization_sha is not None}.
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
- Exact two-sided sign-flip p: {h['exact_two_sided_sign_flip_p']:.9g}; negative directions: {h['negative_count']}/{len(seeds)}.

## Claim boundary

The primary classification is determined only by seed-level H from 1024-kimg NFE1 FID-50k under the frozen rules. NFE2, KID, intermediate milestones, AULC, single-cell BA, interaction, and checkpoint-quality diagnostics are descriptive and cannot alter or rescue the primary verdict. Execution PASS establishes protocol and data integrity; it does not itself establish the scientific hypothesis.
"""
    report_path = root / f"REPORT{suffix}.md"
    report_path.write_text(report, encoding="utf-8")
    deliverables = [protocol_path, args.preflight.resolve(strict=True), args.public_manifest.resolve(strict=True),
                    integrity_path, eval_root / "evaluation_matrix_seal.json",
                    analysis_root / "decoded_results.json", analysis_root / "statistics" / "analysis.json",
                    analysis_root / "statistics" / "primary_decision.json",
                    analysis_root / "statistics" / "H_C_I_Q_G_per_seed.csv",
                    analysis_root / "statistics" / "decoded_evaluation_results.csv",
                    analysis_root / "statistics" / "AULC_diagnostic.csv", cost_path, report_path]
    if args.eleven_seed_authorization is not None:
        deliverables.insert(1, args.eleven_seed_authorization.resolve(strict=True))
    if args.evaluation_recovery_authorization is not None:
        deliverables.insert(2, args.evaluation_recovery_authorization.resolve(strict=True))
        deliverables.insert(3, recovery_preparation_receipt.resolve(strict=True))
    sums = root / f"SHA256SUMS{suffix}.txt"
    with sums.open("x", encoding="utf-8") as handle:
        for path in deliverables:
            handle.write(f"{experiment.sha256_file(path)}  {path}\n")
    experiment.atomic_json(root / f"final_completion_receipt{suffix.lower()}.json", {
        "schema": "ect.q256.fresh-crossed-switch-final-completion/v1", "status": "PASS",
        "execution_integrity": "PASS", "primary_verdict": analysis["primary_verdict"],
        "protocol_sha256": experiment.sha256_file(protocol_path), "implementation_commit": protocol["implementation_commit"],
        "final_git_head": head, "final_git_clean": not bool(status), "sha256sums": str(sums),
        "eleven_seed_authorization_sha256": authorization_sha,
        "evaluation_recovery_authorization_sha256": recovery_authorization_sha,
        "manual_evaluation_recovery_count": 1 if recovery_authorization_sha else 0,
        "included_seeds": list(seeds), "original_n12_claim_abandoned": authorization_sha is not None,
        "compute_ledger_rows": len(ledger),
    })
    print(json.dumps({"status": "PASS", "primary_verdict": analysis["primary_verdict"],
                      "report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
