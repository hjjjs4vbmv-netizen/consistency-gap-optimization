"""Frozen roster, commands and old-control import; no training on import."""
from __future__ import annotations
import csv
import json
from pathlib import Path

EXPERIMENT_ID = "q256_history_component_chase_v1"
ENGINEERING_ID = "q256_history_component_chase_engineering_v1"
SEEDS = tuple(range(50, 66))
FACTORS = {"A": (1.0, 1.0), "B": (1.1, 1.1), "C": (1.1, 1.0), "D": (1.0, 1.1)}
READOUT_BLOCKS = {"E_512": ("B0", "B1", "B2"), "E_KEEP": ("B0",), "ONLINE": ("B0",)}
EVALUATOR_COMMIT = "d6aba02fb88e9db0993623895eb2228ed717d810"
BASE_COMMIT = "7af37a6460bd423ad22e0189bfddc9b713ba0c79"
ROOT = Path(__file__).resolve().parents[2]
EXPECTED_RUNTIME = {"python": "3.11.13", "torch": "2.6.0+cu124", "torch_cuda": "12.4",
                    "numpy": "2.1.2", "scipy": "1.16.1", "cudnn": 90100}


def training_queue():
    return [{"slot": f"S{seed-49:02d}", "seed": seed, "path": arm+"A", "order": order,
             "history": arm, "target": FACTORS[arm][0], "denominator": FACTORS[arm][1],
             "prefix_attempts": 4000, "suffix_attempts": 4000, "status": "PENDING"}
            for seed in SEEDS
            for order, arm in enumerate(("C", "D") if seed % 2 == 0 else ("D", "C"), 1)]


def evaluation_slots():
    return [{"job_id": f"{row['slot']}-{row['path']}-{readout}-{block}",
             "seed": row["seed"], "path": row["path"], "source": "new",
             "readout": readout, "block": block, "status": "PENDING",
             "sample_seed_start": int(block[1])*50000, "sample_seed_end": (int(block[1])+1)*50000-1,
             "metric_seed": 20260730, "nfe": 1, "precision": "fp32", "FID": None, "KID": None,
             "receipt": None}
            for row in training_queue() for readout, blocks in READOUT_BLOCKS.items() for block in blocks]


def import_old_controls(path=None):
    path = path or ROOT / "analysis/q256_optimizer_restart_ema_rebuild_v1/results/endpoint_metrics.json"
    rows = []
    for source, job, seed, branch, readout, block, status, metrics, receipt in json.loads(Path(path).read_text()):
        if source != "original" or branch not in {"K_A", "K_B"}:
            continue
        if seed not in SEEDS or readout not in READOUT_BLOCKS or block not in READOUT_BLOCKS[readout]:
            raise ValueError("old K slot is outside the fixed matrix")
        if status not in {"PASS", "NO_ENDPOINT"}:
            raise ValueError("unexpected old K status")
        rows.append({"job_id": job, "seed": seed, "path": "AA" if branch == "K_A" else "BA",
                     "source": "M1_original", "original_branch": branch,
                     "readout": readout, "block": block, "status": status,
                     "FID": metrics.get("fid50k_full"), "KID": metrics.get("kid50k_full"),
                     "receipt": receipt, "source_commit": BASE_COMMIT})
    keys = {(r["seed"], r["path"], r["readout"], r["block"]) for r in rows}
    missing = {(r["seed"], r["path"]) for r in rows if r["status"] == "NO_ENDPOINT"}
    if len(rows) != 160 or len(keys) != 160 or sum(r["status"] == "PASS" for r in rows) != 145:
        raise ValueError("old controls must contain exactly 145 PASS and 15 NO_ENDPOINT slots")
    if missing != {(58, "AA"), (58, "BA"), (65, "AA")}:
        raise ValueError("old K scientific missingness changed")
    return sorted(rows, key=lambda r: (r["seed"], r["path"], r["readout"], r["block"]))


def manifest(seed, path, source, output, engineering=False):
    if seed not in SEEDS or path not in {"CA", "DA"}:
        if not (engineering and seed == 50 and path == "AA"):
            raise ValueError("outside fixed history-component matrix")
    return {"schema": "ect.q256.schedule-switch-run-manifest/v1",
            "experiment_protocol": ENGINEERING_ID if engineering else EXPERIMENT_ID,
            "run_kind": "formal", "branch": path, "seed": seed,
            "origin_arm": path[0], "continuation_arm": "A", "switch_kimg": 512, "final_kimg": 1024,
            "source_state": {"path": str(Path(source).resolve())},
            "immutable_output_root": str(Path(output).resolve()), "history_component_shadow_update": True}


def command(*, python, dataset, transfer, output, seed, history, phase, resume=None,
            switch_manifest=None, engineering_stop=None):
    if seed not in SEEDS or history not in {"C", "D"}:
        if not (engineering_stop is not None and seed == 50 and history == "A"):
            raise ValueError("outside fixed training matrix")
    if phase not in {"prefix", "suffix"}:
        raise ValueError("invalid phase")
    target, denominator = FACTORS[history if phase == "prefix" else "A"]
    cmd = [str(python), "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
           str(ROOT / "ct_train.py"), f"--data={dataset}", f"--outdir={output}", "--nosubdir",
           "--cond=False", "--arch=ddpmpp", "--precond=ect", "--batch=128", "--batch-gpu=16",
           "--optim=RAdam", "--lr=0.0001", "--dropout=0.2", "--augment=0", "--xflip=False",
           "--mean=-1.1", "--std=2.0", "--mapping=sigmoid", "--global-gap-scale=1.0",
           "--factorial-protocol=q256_target_weight_v1", f"--target-gap-scale={target}",
           f"--denominator-gap-scale={denominator}", "-q", "256", "-k", "8", "-b", "1", "-c", "0",
           "--double=10000", "--ema_beta=0.9993", f"--seed={seed}", "--fp16=True", "--tf32=False",
           "--ls=1.0", "--enable_amp=True", "--bench=False", "--cache=True", "--workers=1",
           "--metrics=none", "--duration=1.024", "--tick=10", "--snap=0", "--dump=0", "--ckpt=10",
           "--sample_every=26", "--eval_every=50", "--mid_t=0.821", "--adaptive-update-kimg=0.5"]
    if phase == "prefix":
        cmd += ["--immutable-checkpoint-kimg=512", f"--stop-after-attempts={engineering_stop or 4000}",
                f"--planned-pause-protocol={ENGINEERING_ID if engineering_stop else EXPERIMENT_ID}"]
        cmd += [f"--resume={resume}"] if resume else [f"--transfer={transfer}"]
    else:
        if not resume or not switch_manifest:
            raise ValueError("suffix requires its own source/resume and manifest")
        cmd += ["--immutable-checkpoint-kimg=640,768,896,1024",
                f"--schedule-switch-manifest={switch_manifest}", f"--resume={resume}"]
        if engineering_stop:
            cmd += [f"--stop-after-attempts={engineering_stop}", f"--planned-pause-protocol={ENGINEERING_ID}"]
    return cmd


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    old, new = import_old_controls(), evaluation_slots()
    write_csv(output / "training_queue.csv", training_queue())
    write_csv(output / "evaluation_slots.csv", new)
    write_csv(output / "old_K_sources.csv", old)
    write_csv(output / "quality_slots_320.csv", old + new)
    (output / "quality_slots_320.json").write_text(json.dumps(old + new, indent=2)+"\n")
    workers = [{"slot": f"S{s-49:02d}", "seed": s,
                "order": [r["path"] for r in training_queue() if r["seed"] == s],
                "node": None, "gpu_index": None, "gpu_uuid": None, "status": "AWAITING_GPU_ASSIGNMENT"}
               for s in SEEDS]
    (output / "worker_manifest.json").write_text(json.dumps(workers, indent=2)+"\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
