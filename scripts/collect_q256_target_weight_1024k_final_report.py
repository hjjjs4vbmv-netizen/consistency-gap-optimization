#!/usr/bin/env python3
"""Verify completed training/evaluation artifacts and emit a compact report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


ARMS = ("A", "B", "C", "D")
METRICS = ("kid50k_full", "fid50k_full")
EXPECTED_FEATURE_BYTES = 409_600_128
EXPECTED_SAMPLE_BYTES = 153_600_128


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def read_one_jsonl(path: Path) -> dict:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1, (path, len(lines))
    value = json.loads(lines[0])
    assert isinstance(value, dict)
    return value


def continuation_elapsed(run_dir: Path) -> str:
    text = (run_dir / "log.txt").read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"^tick\s+102\s+kimg\s+1024\.0\s+.*?\btime\s+(.+?)\s+sec/tick",
        text,
        flags=re.MULTILINE,
    )
    assert matches, run_dir
    return matches[-1].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    training_rows = []
    evaluation_rows = []
    audit_by_seed = {}
    for path in args.audit:
        audit = load_json(path)
        assert audit["all_pass"] is True and audit["arm_count"] == 4
        audit_by_seed[int(audit["seed"])] = audit

    for seed in sorted(audit_by_seed):
        audit = audit_by_seed[seed]
        arm_records = {row["arm"]: row for row in audit["arms"]}
        assert tuple(arm_records) == ARMS
        for arm in ARMS:
            record = arm_records[arm]
            run_dir = Path(record["final_state_path"]).parent
            checkpoint = run_dir / "network-snapshot-latest.pkl"
            assert sha256_file(checkpoint) == record["network_snapshot_sha256"]
            training_rows.append(
                {
                    **record,
                    "continuation_elapsed": continuation_elapsed(run_dir),
                }
            )

            for nfe in (1, 2):
                job_id = f"seed{seed}-arm{arm}-nfe{nfe}"
                target = args.eval_root / "jobs" / job_id
                assert target.is_dir() and not target.is_symlink(), target
                log_text = (target / "log.txt").read_text(
                    encoding="utf-8", errors="replace"
                )
                assert "Exiting..." in log_text
                assert "Traceback (most recent call last)" not in log_text

                options = load_json(target / "training_options.json")
                assert options["batch_size"] == 512
                assert options["metrics"] == list(METRICS)
                assert options["metric_repeats"] == 1
                assert options["retain_generated_artifacts"] is True
                assert options["metric_generator_batch"] == 128
                assert options["seed"] == 20_260_730
                assert options["network_kwargs"]["use_fp16"] is False
                assert options["resume_pkl"] == str(checkpoint)
                assert options["mid_t"] == ([] if nfe == 1 else [0.821])
                sample_seeds = options["sample_seeds"]
                assert len(sample_seeds) == 50_000
                assert sample_seeds[0] == 0 and sample_seeds[-1] == 49_999
                assert len(set(sample_seeds)) == 50_000

                samples = target / "generated-samples.npy"
                kid_features = target / "generated-features-kid50k_full-repeat00.npy"
                fid_features = target / "generated-features-fid50k_full-repeat00.npy"
                assert samples.stat().st_size == EXPECTED_SAMPLE_BYTES
                assert kid_features.stat().st_size == EXPECTED_FEATURE_BYTES
                assert fid_features.stat().st_size == EXPECTED_FEATURE_BYTES
                feature_sha = sha256_file(kid_features)
                assert sha256_file(fid_features) == feature_sha

                values = {}
                for metric in METRICS:
                    metric_record = read_one_jsonl(target / f"metric-{metric}.jsonl")
                    assert metric_record["metric"] == metric
                    assert metric_record["num_gpus"] == 1
                    value = float(metric_record["results"][metric])
                    assert math.isfinite(value)
                    values[metric] = value
                evaluation_rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "nfe": nfe,
                        "mid_t": None if nfe == 1 else 0.821,
                        "sample_count": 50_000,
                        "kid50k_full": values["kid50k_full"],
                        "fid50k_full": values["fid50k_full"],
                        "generated_feature_sha256": feature_sha,
                        "job_dir": str(target),
                        "status": "PASS",
                    }
                )

    expected_jobs = 2 * 4 * len(audit_by_seed)
    assert len(training_rows) == 4 * len(audit_by_seed)
    assert len(evaluation_rows) == expected_jobs
    payload = {
        "schema": "ect.q256.target-weight-1024k-final-report/v1",
        "seeds": sorted(audit_by_seed),
        "training_arm_count": len(training_rows),
        "evaluation_job_count": len(evaluation_rows),
        "all_pass": True,
        "training": training_rows,
        "evaluation": evaluation_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
