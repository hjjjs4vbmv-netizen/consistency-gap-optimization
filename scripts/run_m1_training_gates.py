#!/usr/bin/env python3
"""Run the non-quality M1 G1-G3 training gates on the first two roster seeds."""

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import torch

from analysis.q256_optimizer_restart_ema_rebuild_v1 import verify_crn
from analysis.q256_terminal_history_n30_matpool_v1.run_node import prepare_resume_history
from scripts.build_m1_training_manifest import (
    BRANCHES, GPU_CONTRACT, ORDERS, SOURCE_FIELDS, TRAINING,
    TRAINING_DATASET_SHA256, selected_roster, validate_runtime_receipt,
)
from training import m1, reproducibility, schedule_switch


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_HEAD = "890a85a8ef4d9effb48f653111a70b5f15b249de"
BASELINE_GATE_ENTRY = (
    REPO_ROOT / "analysis" / "q256_optimizer_restart_ema_rebuild_v1"
    / "baseline_gate_entry.py"
)


def canonical_pip_freeze(raw: bytes) -> bytes:
    return b"\n".join(sorted(raw.splitlines())) + b"\n"


def atomic_json(path: Path, value: dict) -> None:
    reproducibility.atomic_json_dump(value, path, overwrite=False)


def load_training_manifest(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != "ect.m1.training-run-manifest/v1"
        or value.get("experiment_protocol") != m1.PROTOCOL_ID
        or len(value.get("roster", [])) != 16
        or value.get("branches") != BRANCHES
        or value.get("training") != TRAINING
        or value.get("gpu_contract") != GPU_CONTRACT
        or value.get("dataset", {}).get("sha256") != TRAINING_DATASET_SHA256
    ):
        raise RuntimeError("invalid M1 training manifest")
    seeds = []
    for index, row in enumerate(value["roster"]):
        if (
            row.get("roster_slot") != f"S{index + 1:02d}"
            or row.get("order") != list(ORDERS[index % len(ORDERS)])
            or not isinstance(row.get("seed"), int)
            or set(row.get("sources", {})) != {"A", "B"}
            or any(
                set(source) != SOURCE_FIELDS
                for source in row["sources"].values()
            )
        ):
            raise RuntimeError("invalid M1 roster parameterization")
        seeds.append(row["seed"])
    if seeds != sorted(seeds) or len(set(seeds)) != 16:
        raise RuntimeError("M1 roster seeds must be unique and ordered")
    for label in (
        "protocol", "baseline_protocol", "source_inventory", "dataset",
        "runtime_receipt",
    ):
        artifact = value.get(label, {})
        artifact_path = Path(artifact.get("path", ""))
        if (
            not artifact_path.is_absolute() or not artifact_path.is_file()
            or schedule_switch.sha256_file(str(artifact_path))
            != artifact.get("sha256")
        ):
            raise RuntimeError(f"M1 {label} identity mismatch")
    inventory = json.loads(
        Path(value["source_inventory"]["path"]).read_text(encoding="utf-8")
    )
    if selected_roster(inventory) != value["roster"]:
        raise RuntimeError("M1 manifest roster differs from bound source inventory")
    if validate_runtime_receipt(Path(value["runtime_receipt"]["path"])) != value.get(
        "runtime_contract"
    ):
        raise RuntimeError("M1 runtime receipt differs from manifest contract")
    runtime = Path(value.get("runtime_python", ""))
    if not runtime.is_absolute() or not runtime.is_file() or not os.access(runtime, os.X_OK):
        raise RuntimeError("M1 runtime Python is unavailable")
    if not Path(value.get("output_root", "")).is_absolute():
        raise RuntimeError("M1 output root must be absolute")
    implementation = value.get("implementation_commit")
    if re.fullmatch(r"[0-9a-f]{40}", str(implementation)) is None:
        raise RuntimeError("invalid M1 implementation commit")
    current = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    if current != implementation:
        raise RuntimeError("checked-out code does not match M1 implementation commit")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT, text=True,
    )
    if dirty:
        raise RuntimeError("M1 implementation repo must be clean before launch")
    value["_training_manifest_path"] = str(path)
    value["_training_manifest_sha256"] = schedule_switch.sha256_file(str(path))
    return value


def acquire_gpu_lock(training: dict, gpu: int):
    uuid = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-gpu=uuid",
        "--format=csv,noheader",
    ], text=True).strip()
    if not uuid.startswith("GPU-"):
        raise RuntimeError("M1 GPU UUID probe failed")
    handle = Path(f"/tmp/m1-{uuid}.lock").open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"M1 GPU {gpu} is already assigned") from exc
    return handle


def probe_gpu_runtime(training: dict, gpu: int) -> dict:
    script = (
        "import json,platform,torch,numpy,scipy; "
        "free,total=torch.cuda.mem_get_info(0); "
        "print(json.dumps({'python':platform.python_version(),"
        "'torch':torch.__version__,'cuda':torch.version.cuda,"
        "'cudnn':torch.backends.cudnn.version(),"
        "'numpy':numpy.__version__,'scipy':scipy.__version__,"
        "'device':torch.cuda.get_device_name(0),'free_bytes':free,"
        "'total_bytes':total}))"
    )
    runtime_env = environment(gpu, runtime_python=Path(training["runtime_python"]))
    runtime = json.loads(subprocess.check_output(
        [training["runtime_python"], "-c", script],
        env=runtime_env, text=True,
    ))
    freeze_output = subprocess.check_output(
        [training["runtime_python"], "-m", "pip", "freeze"], env=runtime_env
    )
    canonical_freeze = canonical_pip_freeze(freeze_output)
    runtime_receipt_path = Path(training["runtime_receipt"]["path"])
    runtime_receipt = json.loads(runtime_receipt_path.read_text(encoding="utf-8"))
    freeze_path = Path(runtime_receipt["pip_freeze"]["path"])
    if canonical_freeze != freeze_path.read_bytes():
        raise RuntimeError("M1 live pip freeze differs from the bound canonical freeze")
    query = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}",
        "--query-gpu=name,uuid,driver_version,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip().split(",")
    if len(query) != 6:
        raise RuntimeError("unexpected nvidia-smi probe output")
    smi = {
        "name": query[0].strip(), "uuid": query[1].strip(),
        "driver_version": query[2].strip(), "free_mib": int(query[3]),
        "total_mib": int(query[4]), "utilization_percent": int(query[5]),
    }
    contract = training["runtime_contract"]
    if any(runtime.get(key) != value for key, value in contract.items()):
        raise RuntimeError("M1 Python/PyTorch/CUDA runtime mismatch")
    gpu_contract = training["gpu_contract"]
    if (
        gpu_contract["name_contains"] not in runtime["device"]
        or gpu_contract["name_contains"] not in smi["name"]
        or not isinstance(runtime.get("cudnn"), int)
        or not smi["uuid"].startswith("GPU-")
        or not smi["driver_version"]
        or smi["free_mib"] < gpu_contract["minimum_free_mib"]
        or smi["utilization_percent"] > gpu_contract["maximum_utilization_percent"]
    ):
        raise RuntimeError("M1 GPU model/free-memory/idle probe failed")
    runtime_python = Path(training["runtime_python"]).resolve(strict=True)
    return {
        "runtime": runtime,
        "runtime_python": str(runtime_python),
        "runtime_prefix": str(runtime_python.parent.parent),
        "runtime_library_paths": [
            str(path) for path in runtime_library_paths(runtime_python)
        ],
        "runtime_receipt_sha256": schedule_switch.sha256_file(
            str(runtime_receipt_path)
        ),
        "canonical_pip_freeze_path": str(freeze_path.resolve(strict=True)),
        "canonical_pip_freeze_sha256": hashlib.sha256(
            canonical_freeze
        ).hexdigest(),
        "nvidia_smi": smi,
        "gpu_index": gpu,
    }


def validate_recorded_runtime_probe(training: dict, hardware: dict) -> None:
    runtime_python = Path(training["runtime_python"]).resolve(strict=True)
    runtime_receipt_path = Path(training["runtime_receipt"]["path"])
    runtime_receipt = json.loads(runtime_receipt_path.read_text(encoding="utf-8"))
    expected = {
        "runtime_python": str(runtime_python),
        "runtime_prefix": str(runtime_python.parent.parent),
        "runtime_library_paths": [
            str(path) for path in runtime_library_paths(runtime_python)
        ],
        "runtime_receipt_sha256": schedule_switch.sha256_file(
            str(runtime_receipt_path)
        ),
        "canonical_pip_freeze_path": str(
            Path(runtime_receipt["pip_freeze"]["path"]).resolve(strict=True)
        ),
        "canonical_pip_freeze_sha256": runtime_receipt["pip_freeze"]["sha256"],
    }
    if (
        not isinstance(hardware, dict)
        or any(hardware.get(key) != value for key, value in expected.items())
        or any(
            hardware.get("runtime", {}).get(key) != value
            for key, value in training["runtime_contract"].items()
        )
    ):
        raise RuntimeError("M1 recorded gate runtime probe differs from its manifest")


def source_for(row: dict, branch: str) -> dict:
    return row["sources"]["B" if branch.endswith("_B") else "A"]


def branch_manifest_value(
    training: dict, row: dict, branch: str, run_dir: Path, *, shadow: bool,
    legacy: bool = False,
) -> dict:
    source = source_for(row, branch)
    identity = training["branches"][branch]
    manifest_branch = {"K_A": "AA", "K_B": "BA"}.get(branch) if legacy else branch
    if legacy and manifest_branch is None:
        raise RuntimeError("legacy comparator is defined only for K branches")
    return {
        "schema": schedule_switch.RUN_MANIFEST_SCHEMA,
        "experiment_protocol": (
            schedule_switch.TERMINAL_HISTORY_N30_PROTOCOL
            if legacy else m1.PROTOCOL_ID
        ),
        "run_kind": "formal",
        "branch": manifest_branch,
        "seed": row["seed"],
        "origin_arm": identity["origin_arm"],
        "continuation_arm": "A",
        "switch_kimg": 512,
        "final_kimg": 1024,
        "protocol_sha256": training[
            "baseline_protocol" if legacy else "protocol"
        ]["sha256"],
        "implementation_commit": (
            BASELINE_HEAD if legacy else training["implementation_commit"]
        ),
        "source_checkpoint_manifest_sha256": training["source_inventory"]["sha256"],
        "training_manifest_sha256": training["_training_manifest_sha256"],
        "source_state": {
            "path": source["source_state_path"],
            "bytes": source["source_state_bytes"],
            "sha256": source["source_state_sha256"],
            "internal_state_sha256": source["internal_state_sha256"],
            "provenance_receipt": {
                "path": source["provenance_receipt_path"],
                "sha256": source["provenance_receipt_sha256"],
            },
            "support_files": source["support_files"],
            "common_initial_state_sha256": source[
                "common_initial_state_sha256"
            ],
        },
        "immutable_output_root": str(run_dir.resolve()),
        **({} if legacy else {"m1_shadow_update": shadow}),
    }


def write_branch_manifest(
    training: dict, row: dict, branch: str, run_dir: Path, *, shadow: bool,
    legacy: bool = False,
) -> Path:
    value = branch_manifest_value(
        training, row, branch, run_dir, shadow=shadow, legacy=legacy
    )
    path = run_dir / "formal_run_manifest.json"
    atomic_json(path, value)
    schedule_switch.load_run_manifest(path)
    return path


def validate_branch_manifest(
    path: Path, training: dict, row: dict, branch: str, run_dir: Path, *,
    shadow: bool, legacy: bool = False,
) -> dict:
    actual = schedule_switch.load_run_manifest(path)
    expected = branch_manifest_value(
        training, row, branch, run_dir, shadow=shadow, legacy=legacy
    )
    if actual != expected:
        raise RuntimeError(f"M1 branch manifest differs from global manifest: {path}")
    return actual


def command(
    training: dict, row: dict, manifest: Path, resume: Path, target=None,
    entry: Path = None,
) -> list[str]:
    cfg = training["training"]
    entry = REPO_ROOT / "ct_train.py" if entry is None else entry
    values = [
        training["runtime_python"], "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node=1", str(entry),
        f"--data={training['dataset']['path']}",
        f"--outdir={manifest.parent}", "--nosubdir", "--cond=False",
        f"--arch={cfg['arch']}", f"--precond={cfg['precond']}",
        f"--batch={cfg['batch']}", f"--batch-gpu={cfg['batch_gpu']}",
        f"--optim={cfg['optimizer']}", f"--lr={cfg['lr']}",
        f"--dropout={cfg['dropout']}", f"--augment={cfg['augment']}",
        f"--xflip={cfg['xflip']}", f"--mean={cfg['mean']}",
        f"--std={cfg['std']}", f"--mapping={cfg['mapping']}",
        f"--global-gap-scale={cfg['global_gap_scale']}",
        "--factorial-protocol=q256_target_weight_v1",
        f"--target-gap-scale={cfg['target_gap_scale']}",
        f"--denominator-gap-scale={cfg['denominator_gap_scale']}",
        "-q", str(cfg["q"]), "-k", str(cfg["k"]),
        "-b", str(cfg["b"]), "-c", str(cfg["c"]),
        f"--double={cfg['double_ticks']}", f"--ema_beta={cfg['ema_beta']}",
        f"--seed={row['seed']}", f"--fp16={cfg['fp16']}",
        f"--tf32={cfg['tf32']}", f"--ls={cfg['loss_scaling']}",
        f"--enable_amp={cfg['enable_amp']}",
        f"--bench={cfg['cudnn_benchmark']}", f"--cache={cfg['cache']}",
        f"--workers={cfg['workers']}", f"--metrics={cfg['metrics']}",
        f"--duration={cfg['final_kimg'] / 1000}", f"--tick={cfg['tick']}",
        f"--snap={cfg['snapshot_tick']}", f"--dump={cfg['state_dump_tick']}",
        f"--ckpt={cfg['checkpoint_tick']}",
        f"--sample_every={cfg['sample_every']}",
        f"--eval_every={cfg['eval_every']}", f"--mid_t={cfg['mid_t']}",
        f"--adaptive-update-kimg={cfg['adaptive_update_kimg']}",
        f"--schedule-switch-manifest={manifest}", f"--resume={resume}",
    ]
    if target is None:
        milestones = ",".join(str(value) for value in cfg["milestone_kimg"])
        values.append(f"--immutable-checkpoint-kimg={milestones}")
    else:
        values.extend([
            f"--stop-after-attempts={target}",
            f"--planned-pause-protocol={m1.PROTOCOL_ID}",
        ])
    return values


def runtime_library_paths(runtime_python: Path) -> list[Path]:
    runtime_python = runtime_python.resolve(strict=True)
    prefix = runtime_python.parent.parent
    site_packages = prefix / "lib/python3.11/site-packages"
    paths = [prefix / "lib", site_packages / "torch/lib"]
    nvidia = site_packages / "nvidia"
    if nvidia.is_dir():
        paths.extend(sorted(path for path in nvidia.glob("*/lib") if path.is_dir()))
    return paths


def environment(
    gpu: int, baseline_repo: Path = None, runtime_python: Path = None,
) -> dict:
    value = os.environ.copy()
    value.pop("PYTHONPATH", None)
    value.pop("PYTHONHOME", None)
    value.update({
        "CUDA_VISIBLE_DEVICES": str(gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "CUDA_CACHE_DISABLE": "1",
        "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
    })
    if baseline_repo is not None:
        value["M1_BASELINE_REPO"] = str(baseline_repo)
    if runtime_python is not None:
        libraries = runtime_library_paths(runtime_python)
        value["LD_LIBRARY_PATH"] = ":".join(str(path) for path in libraries)
        prefix = runtime_python.resolve(strict=True).parent.parent
        value["PATH"] = f"{prefix / 'bin'}:/usr/bin:/bin"
    return value


def validate_baseline_repo(path: Path) -> Path:
    path = path.resolve(strict=True)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=path, text=True,
    )
    if head != BASELINE_HEAD or dirty:
        raise RuntimeError("baseline repo must be clean at exact PR101 head 890a85")
    return path


def run_segment(
    training: dict, row: dict, branch: str, run_dir: Path, *,
    shadow: bool, target: int, gpu: int, resume_existing: bool = False,
    legacy: bool = False, baseline_repo: Path = None,
) -> Path:
    if legacy and baseline_repo is None:
        raise RuntimeError("legacy comparator requires the exact baseline repo")
    if resume_existing:
        manifest = run_dir / "formal_run_manifest.json"
        resume = run_dir / "training-state-latest.pt"
        validate_branch_manifest(
            manifest, training, row, branch, run_dir,
            shadow=shadow, legacy=legacy,
        )
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        source = source_for(row, branch)
        prepare_resume_history(Path(source["source_state_path"]).parent, run_dir)
        manifest = write_branch_manifest(
            training, row, branch, run_dir, shadow=shadow, legacy=legacy
        )
        resume = Path(source["source_state_path"])
    log = run_dir / f"gate-to-{target}.log"
    with log.open("xb") as handle:
        result = subprocess.run(
            command(
                training, row, manifest, resume, target,
                entry=BASELINE_GATE_ENTRY if legacy else None,
            ),
            cwd=baseline_repo if legacy else REPO_ROOT,
            env=environment(
                gpu,
                baseline_repo if legacy else None,
                Path(training["runtime_python"]),
            ), stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        raise RuntimeError(f"M1 gate training failed: {run_dir} -> {target}")
    state = torch.load(
        run_dir / "training-state-latest.pt", map_location="cpu", weights_only=False
    )
    if int(state.get("attempted_iteration", -1)) != target:
        raise RuntimeError("gate checkpoint attempt mismatch")
    expected_nimg = target * training["training"]["batch"]
    if int(state.get("cur_nimg", -1)) != expected_nimg:
        raise RuntimeError("gate checkpoint image progress mismatch")
    ranks = state.get("rank_states")
    if (
        not isinstance(ranks, list) or len(ranks) != 1
        or int(ranks[0].get("sampler_state", {}).get("consumed_samples", -1))
        != expected_nimg
    ):
        raise RuntimeError("gate checkpoint sampler progress mismatch")
    loaded_manifest = schedule_switch.load_run_manifest(manifest)
    schedule_switch.verify_switched_state(state, loaded_manifest)
    if legacy:
        if "m1" in state or "ema_512" in state:
            raise RuntimeError("legacy comparator unexpectedly contains M1 state")
    else:
        m1.validate_resumed_state(state, loaded_manifest)
        branch_init_path = run_dir / "training-state-kimg000512.pt"
        if not branch_init_path.is_file():
            raise RuntimeError("M1 gate did not preserve branch-init@512")
        branch_init = torch.load(
            branch_init_path, map_location="cpu", weights_only=False
        )
        source_state = torch.load(
            loaded_manifest["source_state"]["path"],
            map_location="cpu", weights_only=False,
        )
        m1.validate_branch_init_against_source(
            branch_init, source_state, loaded_manifest
        )
    return run_dir / "training-state-latest.pt"


def normalized_online_state(path: Path, *, include_m1: bool = False) -> str:
    state = torch.load(path, map_location="cpu", weights_only=False)
    value = {
        key: state[key] for key in (
            "optimizer_state", "gradscaler_state", "rank_states",
            "loss_fn_state", "attempted_iteration", "successful_optimizer_steps",
            "cur_nimg", "cur_tick", "tick_start_nimg",
        )
    }
    for key in ("adaptive_signal_window_state", "managed_loss_overflow_count"):
        if key in state:
            value[key] = state[key]
    value["net"] = state["net"].state_dict()
    value["ema"] = state["ema"].state_dict()
    if include_m1:
        value["ema_512"] = state["ema_512"].state_dict()
        value["m1"] = state["m1"]
    return reproducibility.state_sha256(value)


def verify_fresh_radam(source_path: Path, branch: str) -> str:
    state = torch.load(source_path, map_location="cpu", weights_only=False)
    left = copy.deepcopy(state["net"])
    right = copy.deepcopy(state["net"])
    left_opt = torch.optim.RAdam(left.parameters(), lr=1e-4)
    left_opt.load_state_dict(state["optimizer_state"])
    m1.apply_optimizer_intervention(left_opt, branch)
    if len(left_opt.param_groups) != 1:
        raise RuntimeError("M1 fresh-RAdam gate expects one optimizer param group")
    group = {key: copy.deepcopy(value) for key, value in left_opt.param_groups[0].items() if key != "params"}
    right_opt = torch.optim.RAdam(right.parameters(), lr=group["lr"])
    right_opt.param_groups[0].update(group)
    for left_param, right_param in zip(left.parameters(), right.parameters()):
        left_param.grad = torch.ones_like(left_param)
        right_param.grad = torch.ones_like(right_param)
    left_opt.step()
    right_opt.step()
    result = {
        "left_model": reproducibility.module_state_sha256(left),
        "right_model": reproducibility.module_state_sha256(right),
        "left_optimizer": reproducibility.state_sha256(left_opt.state_dict()),
        "right_optimizer": reproducibility.state_sha256(right_opt.state_dict()),
    }
    if result["left_model"] != result["right_model"] or result["left_optimizer"] != result["right_optimizer"]:
        raise RuntimeError("reset RAdam next step is not fresh-RAdam equivalent")
    return reproducibility.state_sha256(result)


def artifact_identity(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"M1 gate artifact is not a regular file: {path}")
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": schedule_switch.sha256_file(str(path))}


def collect_gate_artifacts(seed_root: Path) -> dict:
    paths = {}
    for branch in ("K_A", "K_B", "R_A", "R_B"):
        continuous = seed_root / f"{branch}-continuous32"
        staged = seed_root / f"{branch}-staged"
        for label, path in {
            f"{branch}_continuous_state": continuous / "training-state-latest.pt",
            f"{branch}_continuous_manifest": continuous / "formal_run_manifest.json",
            f"{branch}_continuous_telemetry": continuous / "schedule_switch_training_telemetry_v1.csv",
            f"{branch}_continuous_log": continuous / "gate-to-4032.log",
            f"{branch}_staged_state": staged / "training-state-latest.pt",
            f"{branch}_staged_manifest": staged / "formal_run_manifest.json",
            f"{branch}_staged_log_4016": staged / "gate-to-4016.log",
            f"{branch}_staged_log_4032": staged / "gate-to-4032.log",
        }.items():
            paths[label] = artifact_identity(path)
    for branch in ("K_A", "K_B"):
        shadow = seed_root / f"{branch}-shadow-off"
        legacy = seed_root / f"{branch}-legacy-continuous32"
        for label, path in {
            f"{branch}_shadow_state": shadow / "training-state-latest.pt",
            f"{branch}_shadow_manifest": shadow / "formal_run_manifest.json",
            f"{branch}_shadow_log": shadow / "gate-to-4032.log",
            f"{branch}_legacy_state": legacy / "training-state-latest.pt",
            f"{branch}_legacy_manifest": legacy / "formal_run_manifest.json",
            f"{branch}_legacy_log": legacy / "gate-to-4032.log",
        }.items():
            paths[label] = artifact_identity(path)
    return paths


def validate_gate_seed_artifacts(check: dict, training: dict, row: dict) -> None:
    artifacts = check.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("M1 gate receipt lacks artifact bindings")
    required = set()
    for branch in ("K_A", "K_B", "R_A", "R_B"):
        required.update({
            f"{branch}_continuous_state", f"{branch}_continuous_manifest",
            f"{branch}_continuous_telemetry", f"{branch}_continuous_log",
            f"{branch}_staged_state", f"{branch}_staged_manifest",
            f"{branch}_staged_log_4016", f"{branch}_staged_log_4032",
        })
    for branch in ("K_A", "K_B"):
        required.update({
            f"{branch}_shadow_state", f"{branch}_shadow_manifest",
            f"{branch}_shadow_log", f"{branch}_legacy_state",
            f"{branch}_legacy_manifest", f"{branch}_legacy_log",
        })
    if set(artifacts) != required:
        raise RuntimeError("M1 gate artifact label set mismatch")
    paths = {}
    for label, identity in artifacts.items():
        if not isinstance(identity, dict) or set(identity) != {"path", "sha256"}:
            raise RuntimeError(f"M1 gate artifact identity mismatch: {label}")
        path = Path(identity["path"])
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"M1 gate artifact unavailable: {label}")
        path = path.resolve(strict=True)
        if schedule_switch.sha256_file(str(path)) != identity["sha256"]:
            raise RuntimeError(f"M1 gate artifact SHA256 mismatch: {label}")
        paths[label] = path

    manifests, telemetry = {}, {}
    comparisons = {}
    for branch in ("K_A", "K_B", "R_A", "R_B"):
        continuous_manifest = paths[f"{branch}_continuous_manifest"]
        staged_manifest = paths[f"{branch}_staged_manifest"]
        validate_branch_manifest(
            continuous_manifest, training, row, branch,
            continuous_manifest.parent, shadow=True,
        )
        validate_branch_manifest(
            staged_manifest, training, row, branch,
            staged_manifest.parent, shadow=True,
        )
        continuous_hash = normalized_online_state(
            paths[f"{branch}_continuous_state"], include_m1=True
        )
        staged_hash = normalized_online_state(
            paths[f"{branch}_staged_state"], include_m1=True
        )
        if continuous_hash != staged_hash:
            raise RuntimeError("M1 gate continuous32 differs from staged replay")
        comparisons[f"{branch}_continuous32"] = continuous_hash
        manifests[branch] = continuous_manifest
        telemetry[branch] = paths[f"{branch}_continuous_telemetry"]
    baseline_hashes = {}
    for branch in ("K_A", "K_B"):
        shadow_manifest = paths[f"{branch}_shadow_manifest"]
        legacy_manifest = paths[f"{branch}_legacy_manifest"]
        validate_branch_manifest(
            shadow_manifest, training, row, branch,
            shadow_manifest.parent, shadow=False,
        )
        validate_branch_manifest(
            legacy_manifest, training, row, branch,
            legacy_manifest.parent, shadow=False, legacy=True,
        )
        continuous_hash = normalized_online_state(
            paths[f"{branch}_continuous_state"]
        )
        if continuous_hash != normalized_online_state(paths[f"{branch}_shadow_state"]):
            raise RuntimeError("M1 gate shadow-off trajectory mismatch")
        comparisons[f"{branch}_shadow_on_off"] = continuous_hash
        if continuous_hash != normalized_online_state(paths[f"{branch}_legacy_state"]):
            raise RuntimeError("M1 gate baseline trajectory mismatch")
        comparisons[f"{branch}_legacy_resume"] = continuous_hash
        baseline_hashes[branch] = schedule_switch.sha256_file(str(legacy_manifest))
    bound_seed, manifest_hashes = verify_crn.load_manifest_bindings(manifests)
    if bound_seed != row["seed"]:
        raise RuntimeError("M1 gate artifact manifests bind the wrong seed")
    series = verify_crn.verify(telemetry, row["seed"], "gate32")
    radam = {
        branch: verify_fresh_radam(
            Path(source_for(row, branch)["source_state_path"]), branch
        )
        for branch in ("R_A", "R_B")
    }
    if (
        series != check.get("crn_sha256")
        or manifest_hashes != check.get("manifest_sha256_by_branch")
        or baseline_hashes != check.get("baseline_manifest_sha256_by_branch")
        or comparisons != check.get("normalized_state_sha256")
        or radam != check.get("fresh_radam_step_sha256")
    ):
        raise RuntimeError("M1 G1-G3 artifact-derived evidence mismatch")


def run_gates(
    training: dict, output: Path, gpu: int, baseline_repo: Path
) -> dict:
    rows = training["roster"][:2]
    checks = []
    for row in rows:
        seed_root = output / row["roster_slot"]
        seed_root.mkdir(parents=True, exist_ok=False)
        branch_dirs = {}
        comparison_hashes = {}
        baseline_manifest_hashes = {}
        for branch in ("K_A", "K_B", "R_A", "R_B"):
            continuous = seed_root / f"{branch}-continuous32"
            continuous_state = run_segment(
                training, row, branch, continuous, shadow=True,
                target=4032, gpu=gpu,
            )
            staged = seed_root / f"{branch}-staged"
            run_segment(training, row, branch, staged, shadow=True, target=4016, gpu=gpu)
            staged_state = run_segment(
                training, row, branch, staged, shadow=True,
                target=4032, gpu=gpu, resume_existing=True,
            )
            continuous_hash = normalized_online_state(
                continuous_state, include_m1=True
            )
            staged_hash = normalized_online_state(staged_state, include_m1=True)
            if continuous_hash != staged_hash:
                raise RuntimeError("continuous32 differs from 16+resume+16")
            comparison_hashes[f"{branch}_continuous32"] = continuous_hash
            branch_dirs[branch] = continuous
        for branch in ("K_A", "K_B"):
            shadow_off = seed_root / f"{branch}-shadow-off"
            off_state = run_segment(
                training, row, branch, shadow_off, shadow=False,
                target=4032, gpu=gpu,
            )
            on_state = branch_dirs[branch] / "training-state-latest.pt"
            on_hash = normalized_online_state(on_state)
            off_hash = normalized_online_state(off_state)
            if on_hash != off_hash:
                raise RuntimeError("E_512 shadow changed online/E_KEEP trajectory")
            comparison_hashes[f"{branch}_shadow_on_off"] = on_hash
            legacy_dir = seed_root / f"{branch}-legacy-continuous32"
            legacy_state = run_segment(
                training, row, branch, legacy_dir, shadow=False,
                target=4032, gpu=gpu, legacy=True,
                baseline_repo=baseline_repo,
            )
            legacy_hash = normalized_online_state(legacy_state)
            if legacy_hash != on_hash:
                raise RuntimeError("M1 K branch differs from legal PR101 resume")
            comparison_hashes[f"{branch}_legacy_resume"] = legacy_hash
            baseline_manifest_hashes[branch] = schedule_switch.sha256_file(
                str(legacy_dir / "formal_run_manifest.json")
            )
        radam_hashes = {}
        for branch in ("R_A", "R_B"):
            radam_hashes[branch] = verify_fresh_radam(
                Path(source_for(row, branch)["source_state_path"]), branch
            )
        telemetry = {
            branch: branch_dirs[branch] / "schedule_switch_training_telemetry_v1.csv"
            for branch in branch_dirs
        }
        manifests = {
            branch: branch_dirs[branch] / "formal_run_manifest.json"
            for branch in branch_dirs
        }
        bound_seed, manifest_hashes = verify_crn.load_manifest_bindings(manifests)
        if bound_seed != row["seed"]:
            raise RuntimeError("gate manifests do not bind the roster seed")
        series = verify_crn.verify(telemetry, row["seed"], "gate32")
        checks.append({
            "seed": row["seed"], "status": "PASS", "crn_sha256": series,
            "manifest_sha256_by_branch": manifest_hashes,
            "baseline_manifest_sha256_by_branch": baseline_manifest_hashes,
            "normalized_state_sha256": comparison_hashes,
            "fresh_radam_step_sha256": radam_hashes,
            "artifacts": collect_gate_artifacts(seed_root),
        })
    return {"schema": "ect.m1.training-gates/v1", "status": "PASS", "seeds": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--baseline-repo", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.training_manifest.resolve(strict=True)
    training = load_training_manifest(manifest_path)
    baseline_repo = validate_baseline_repo(args.baseline_repo)
    gpu_lock = acquire_gpu_lock(training, args.gpu)
    hardware = probe_gpu_runtime(training, args.gpu)
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = run_gates(training, args.output, args.gpu, baseline_repo)
    receipt["training_manifest_path"] = str(manifest_path)
    receipt["training_manifest_sha256"] = schedule_switch.sha256_file(
        str(manifest_path)
    )
    receipt["baseline"] = {
        "repo": str(baseline_repo),
        "head": BASELINE_HEAD,
        "pause_patch_entry": str(BASELINE_GATE_ENTRY),
        "pause_patch_sha256": schedule_switch.sha256_file(
            str(BASELINE_GATE_ENTRY)
        ),
    }
    receipt["hardware_probe"] = hardware
    atomic_json(args.output / "g1_g2_g3_receipt.json", receipt)
    gpu_lock.close()
    print("M1_G1_G2_G3_PASS seeds=2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
