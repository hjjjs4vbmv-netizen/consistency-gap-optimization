#!/usr/bin/env python3
"""Collect the sealed q128 matched-spacing result matrix without quality selection.

The server archive is used as the complete 210-job base.  Results from the
preassigned server and data-card partitions replace the corresponding base
rows, even when a redundant attempt exists.  This preserves the frozen
partition ownership rule and never chooses a result based on metric quality.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess


JOB_RE = re.compile(r"^seed(\d+)-arm([A-Za-z]+)-kimg(\d+)-nfe([12])$")
ARMS = ["A", "Bsame", "Bmatch", "Cmatch", "Dmatch"]
SEEDS = [3, 4, 5]
BUDGETS = [256, 384, 512, 640, 768, 896, 1024]
NFES = [1, 2]
DEPLOYED_EVALUATOR_SOURCE_SHA256 = {
    "ct_eval.py": "8e17e4cd4e12097e12659a9c8849d42554f24efb25e5255261383d952d878c95",
    "metrics/frechet_inception_distance.py": "efab16b0e42ea551fd9141dd80750075425429771ed85b4200fc1da97b018ecc",
    "metrics/kernel_inception_distance.py": "cb9456b183bfa40b098dfc13720d27f95d29a2ebfddc9fc537165b5a02655248",
    "metrics/metric_main.py": "c312c1d4217036e36738692a6efe88bd6cdc70f7bccc797b1faf68b795f1069c",
    "metrics/metric_utils.py": "8ae7cf341288a79a77aec845487ff9a7742fe91ee5aa98c52b5af5e0a52d5a20",
    "scripts/run_q128_stream_eval_worker.sh": "a357a0ab2bd1f30bd5466add8b1ec1a31a9e9cc28890cc736339399e22fa765a",
}


def read_jsonl_last(path):
    with open(path, "r") as handle:
        lines = [line for line in handle.read().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("empty metric file: {}".format(path))
    return json.loads(lines[-1])


def read_local(root, label):
    rows = {}
    done_root = os.path.join(root, "done")
    jobs_root = os.path.join(root, "jobs")
    for name in sorted(os.listdir(done_root)):
        if not name.endswith(".SEALED_PASS"):
            continue
        job_id = name[: -len(".SEALED_PASS")]
        if JOB_RE.match(job_id) is None:
            continue
        job_root = os.path.join(jobs_root, job_id)
        fid_path = os.path.join(job_root, "metric-fid50k_full.jsonl")
        kid_path = os.path.join(job_root, "metric-kid50k_full.jsonl")
        if not (os.path.isfile(fid_path) and os.path.isfile(kid_path)):
            continue
        fid = read_jsonl_last(fid_path)
        kid = read_jsonl_last(kid_path)
        rows[job_id] = {
            "fid": float(fid["results"]["fid50k_full"]),
            "kid": float(kid["results"]["kid50k_full"]),
            "fid_timestamp": fid.get("timestamp"),
            "kid_timestamp": kid.get("timestamp"),
            "source": label,
        }
    return rows


def read_remote(socket_path, port, host, root, label):
    script = r'''for r in ROOT/done/*.SEALED_PASS; do
  [ -f "$r" ] || continue
  j=${r##*/}; j=${j%.SEALED_PASS}
  f=ROOT/jobs/"$j"/metric-fid50k_full.jsonl
  k=ROOT/jobs/"$j"/metric-kid50k_full.jsonl
  [ -f "$f" ] && [ -f "$k" ] || continue
  printf 'J\t%s\t' "$j"
  tail -n 1 "$f" | tr -d '\n'
  printf '\t'
  tail -n 1 "$k"
done'''.replace("ROOT", root)
    command = [
        "ssh", "-n", "-S", socket_path, "-p", str(port), host, script,
    ]
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=180,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "remote source failed ({}): {}".format(label, process.stderr[-500:])
        )
    rows = {}
    for line in process.stdout.splitlines():
        if not line.startswith("J\t"):
            continue
        _, job_id, fid_text, kid_text = line.split("\t", 3)
        fid = json.loads(fid_text)
        kid = json.loads(kid_text)
        rows[job_id] = {
            "fid": float(fid["results"]["fid50k_full"]),
            "kid": float(kid["results"]["kid50k_full"]),
            "fid_timestamp": fid.get("timestamp"),
            "kid_timestamp": kid.get("timestamp"),
            "source": label,
        }
    return rows


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-root",
        default="/data/raw/ECT/q128_matched_spacing_v1/evaluation",
    )
    parser.add_argument(
        "--server-partition-root",
        default="/data/raw/ECT/q128_server_partition/evaluation",
    )
    parser.add_argument("--data-card-socket", default="/tmp/q128-cloud-29199.sock")
    parser.add_argument("--data-card-port", type=int, default=29199)
    parser.add_argument("--data-card-host", default="root@px-cloud2.matpool.com")
    parser.add_argument(
        "--data-card-root", default="/root/q128_data_eval/evaluation"
    )
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if os.path.exists(args.outdir):
        raise RuntimeError("refusing to overwrite {}".format(args.outdir))
    os.makedirs(args.outdir)

    rows = read_local(args.server_root, "server-main")
    server_partition = read_local(
        args.server_partition_root, "server-partition"
    )
    data_partition = read_remote(
        args.data_card_socket,
        args.data_card_port,
        args.data_card_host,
        args.data_card_root,
        "data-card-partition",
    )

    duplicates = []
    differing_metric_values = 0
    max_fid_difference = 0.0
    max_kid_difference = 0.0
    for label, partition in (
        ("server-partition", server_partition),
        ("data-card-partition", data_partition),
    ):
        for job_id, row in partition.items():
            if job_id in rows:
                fid_difference = abs(rows[job_id]["fid"] - row["fid"])
                kid_difference = abs(rows[job_id]["kid"] - row["kid"])
                duplicates.append(
                    {
                        "job_id": job_id,
                        "authoritative_source": label,
                        "fid_abs_difference": fid_difference,
                        "kid_abs_difference": kid_difference,
                    }
                )
                differing_metric_values += int(fid_difference > 1e-12)
                differing_metric_values += int(kid_difference > 1e-12)
                max_fid_difference = max(max_fid_difference, fid_difference)
                max_kid_difference = max(max_kid_difference, kid_difference)
            rows[job_id] = row

    expected = [
        "seed{}-arm{}-kimg{:06d}-nfe{}".format(seed, arm, budget, nfe)
        for seed in SEEDS
        for arm in ARMS
        for budget in BUDGETS
        for nfe in NFES
    ]
    expected_set = set(expected)
    missing = sorted(expected_set - set(rows))
    extra = sorted(set(rows) - expected_set)
    if missing or extra or len(rows) != 210:
        raise RuntimeError(
            "matrix mismatch: missing={} extra={} count={}".format(
                missing, extra, len(rows)
            )
        )

    results_path = os.path.join(args.outdir, "evaluation_results.csv")
    with open(results_path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "job_id", "seed", "arm", "kimg", "nfe", "mid_t",
                "fid50k_full", "kid50k_full", "samples",
                "sample_seed_start", "sample_seed_end", "metric_seed",
                "precision", "source", "fid_timestamp", "kid_timestamp",
                "status",
            ]
        )
        for job_id in expected:
            match = JOB_RE.match(job_id)
            seed, arm, budget, nfe = match.groups()
            row = rows[job_id]
            writer.writerow(
                [
                    job_id, int(seed), arm, int(budget), int(nfe),
                    "" if int(nfe) == 1 else "0.821",
                    format(row["fid"], ".17g"),
                    format(row["kid"], ".17g"),
                    50000, 0, 49999, 20260730, "fp32", row["source"],
                    row["fid_timestamp"], row["kid_timestamp"], "SEALED_PASS",
                ]
            )

    audit = {
        "schema": "ect.q128-matched-spacing-results-audit/v1",
        "status": "PASS",
        "jobs": 210,
        "metric_values": 420,
        "seeds": SEEDS,
        "arms": ARMS,
        "budgets_kimg": BUDGETS,
        "nfes": NFES,
        "protocol": {
            "precision": "fp32",
            "samples": 50000,
            "sample_seeds": "0-49999",
            "metric_seed": 20260730,
            "nfe2_mid_t": 0.821,
            "feature_reuse_scope": "kid50k_full -> fid50k_full",
            "deployed_evaluator_source_sha256": DEPLOYED_EVALUATOR_SOURCE_SHA256,
        },
        "authoritative_job_counts": {
            "server-main": 210 - len(server_partition) - len(data_partition),
            "server-partition": len(server_partition),
            "data-card-partition": len(data_partition),
        },
        "overlap_jobs": len(duplicates),
        "differing_metric_values": differing_metric_values,
        "max_duplicate_fid_abs_difference": max_fid_difference,
        "max_duplicate_kid_abs_difference": max_kid_difference,
        "selection_rule": (
            "Use preassigned data/server partitions for their disjoint jobs; "
            "never choose by metric quality."
        ),
        "invalidated_directories_included": False,
        "matrix_missing_jobs": 0,
        "matrix_extra_jobs": 0,
    }
    audit_path = os.path.join(args.outdir, "audit.json")
    with open(audit_path, "w") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")

    duplicates_path = os.path.join(args.outdir, "duplicate_attempts.json")
    with open(duplicates_path, "w") as handle:
        json.dump(
            sorted(duplicates, key=lambda item: item["job_id"]),
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    checksums_path = os.path.join(args.outdir, "SHA256SUMS.txt")
    with open(checksums_path, "w") as handle:
        for name in (
            "evaluation_results.csv", "audit.json", "duplicate_attempts.json"
        ):
            handle.write("{}  {}\n".format(sha256(os.path.join(args.outdir, name)), name))

    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
