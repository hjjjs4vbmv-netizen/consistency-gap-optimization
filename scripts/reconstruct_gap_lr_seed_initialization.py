#!/usr/bin/env python3
"""Quality-blind reconstruction of the six pre-update transferred states.

This is deliberately a reconstruction, not a historical process attestation.
It reads only receipt-bound training options, the frozen dataset metadata, the
frozen transfer checkpoint, and the frozen implementation.  It never opens a
trained snapshot/state or a quality-evaluation artifact.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import pickle
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

import dnnlib
from torch_utils import misc


EXPERIMENT_ID = "gap_lr_matched_q128_s45_replication_v1"
EXECUTION_PROTOCOL_COMMIT = "583c2fe0f914fc1191903d747737fd54b4ba1eef"
TRAINING_CODE_COMMIT = "2357bb1d2531a343bdb4397f5a08f4d42a2d135b"
DATA_SHA256 = "a469a9f1b89d43a4a5a0fea42a351b6f107800fc32712881ea3d0ee8cc3a88c1"
TRANSFER_SHA256 = "4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da"
MAGIC = b"ECT_CANONICAL_TORCH_MODULE_V1\x00"
RUNS = (
    (4, "A", "arm_a_g1_0_lr_fixed_s4"),
    (4, "B", "arm_b_g1_3_lr_fixed_s4"),
    (4, "C", "arm_c_g1_3_lr_matched_s4"),
    (5, "A", "arm_a_g1_0_lr_fixed_s5"),
    (5, "B", "arm_b_g1_3_lr_fixed_s5"),
    (5, "C", "arm_c_g1_3_lr_matched_s5"),
)


def fail(message: str) -> None:
    raise SystemExit("INITIALIZATION RECONSTRUCTION REJECTED: " + message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        fail(f"expected JSON object: {path}")
    return value


def _field(digest: Any, value: bytes) -> None:
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


def tensor_entries(module: torch.nn.Module) -> list[tuple[str, str, torch.Tensor]]:
    entries = [
        *( (name, "parameter", tensor) for name, tensor in module.named_parameters() ),
        *( (name, "buffer", tensor) for name, tensor in module.named_buffers() ),
    ]
    entries.sort(key=lambda item: (item[0].encode("utf-8"), item[1]))
    names = [name for name, _kind, _tensor in entries]
    if len(names) != len(set(names)):
        fail("duplicate parameter/buffer name in canonical stream")
    return entries


def raw_little_endian_bytes(tensor: torch.Tensor) -> bytes:
    if sys.byteorder != "little":
        fail("canonical v1 requires a little-endian host")
    if tensor.layout != torch.strided or tensor.is_quantized:
        fail(
            f"unsupported tensor layout/type: {tensor.layout}, "
            f"quantized={tensor.is_quantized}"
        )
    value = tensor.detach().cpu().contiguous()
    if value.numel() == 0:
        return b""
    return value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")


def canonical_module(
    module: torch.nn.Module, *, include_per_tensor: bool = False
) -> dict[str, Any]:
    entries = tensor_entries(module)
    digest = hashlib.sha256()
    digest.update(MAGIC)
    digest.update(struct.pack(">Q", len(entries)))
    per_tensor: dict[str, str] = {}
    kinds: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    total_raw_bytes = 0
    for name, kind, tensor in entries:
        raw = raw_little_endian_bytes(tensor)
        dtype = str(tensor.dtype).removeprefix("torch.")
        _field(digest, kind.encode("ascii"))
        _field(digest, name.encode("utf-8"))
        _field(digest, dtype.encode("ascii"))
        digest.update(struct.pack(">Q", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack(">Q", int(dimension)))
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)

        leaf = hashlib.sha256()
        leaf.update(MAGIC)
        _field(leaf, kind.encode("ascii"))
        _field(leaf, name.encode("utf-8"))
        _field(leaf, dtype.encode("ascii"))
        leaf.update(struct.pack(">Q", tensor.ndim))
        for dimension in tensor.shape:
            leaf.update(struct.pack(">Q", int(dimension)))
        leaf.update(struct.pack(">Q", len(raw)))
        leaf.update(raw)
        per_tensor[name] = leaf.hexdigest()
        kinds[kind] += 1
        dtypes[dtype] += 1
        total_raw_bytes += len(raw)

    result: dict[str, Any] = {
        "schema": "ECT_CANONICAL_TORCH_MODULE_V1",
        "sha256": digest.hexdigest(),
        "tensor_count": len(entries),
        "kind_counts": dict(sorted(kinds.items())),
        "dtype_counts": dict(sorted(dtypes.items())),
        "total_raw_bytes": total_raw_bytes,
    }
    if include_per_tensor:
        result["per_tensor_sha256"] = per_tensor
    return result


def copy_source_subset(
    source: torch.nn.Module, destination: torch.nn.Module
) -> dict[str, Any]:
    source_tensors = dict(misc.named_params_and_buffers(source))
    destination_tensors = dict(misc.named_params_and_buffers(destination))
    missing = sorted(set(destination_tensors) - set(source_tensors))
    extra = sorted(set(source_tensors) - set(destination_tensors))
    mismatches = []
    for name in sorted(set(source_tensors) & set(destination_tensors)):
        src = source_tensors[name]
        dst = destination_tensors[name]
        if src.shape != dst.shape or src.dtype != dst.dtype:
            mismatches.append(
                {
                    "name": name,
                    "source_shape": list(src.shape),
                    "destination_shape": list(dst.shape),
                    "source_dtype": str(src.dtype),
                    "destination_dtype": str(dst.dtype),
                }
            )
    if missing or mismatches:
        fail(f"incompatible transfer: missing={missing}, mismatches={mismatches}")
    with torch.no_grad():
        for name, dst in destination_tensors.items():
            dst.copy_(source_tensors[name])
    source_only = []
    for name in extra:
        tensor = source_tensors[name]
        source_only.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "raw_bytes": tensor.numel() * tensor.element_size(),
            }
        )
    return {
        "source_tensor_count": len(source_tensors),
        "destination_tensor_count": len(destination_tensors),
        "missing_destination_names": missing,
        "source_only_ignored_by_destination_iterating_copy": source_only,
        "shape_dtype_mismatches": mismatches,
        "all_destination_tensors_covered": not missing and not mismatches,
    }


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_target(options: dict[str, Any]) -> tuple[torch.nn.Module, dict[str, int]]:
    dataset = dnnlib.util.construct_class_by_name(**options["dataset_kwargs"])
    interface = {
        "img_resolution": int(dataset.resolution),
        "img_channels": int(dataset.num_channels),
        "label_dim": int(dataset.label_dim),
    }
    if interface != {"img_resolution": 32, "img_channels": 3, "label_dim": 0}:
        fail(f"unexpected dataset-derived network interface: {interface}")
    del dataset
    net = dnnlib.util.construct_class_by_name(
        **options["network_kwargs"], **interface
    )
    return net, interface


def validate_repo(repo: Path, adjudication_tooling_commit: str) -> None:
    expected_script = (repo / "scripts/reconstruct_gap_lr_seed_initialization.py").resolve()
    if Path(__file__).resolve() != expected_script:
        fail("executed reconstruction script is not the --repo committed path")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True
    ).strip()
    if head != adjudication_tooling_commit:
        fail(
            f"repository HEAD {head} does not equal adjudication tooling commit "
            f"{adjudication_tooling_commit}"
        )
    result = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            TRAINING_CODE_COMMIT,
            "--",
            "ct_train.py",
            "training",
            "dnnlib",
            "torch_utils",
        ],
        cwd=str(repo),
    )
    if result.returncode != 0:
        fail("reconstruction implementation differs from frozen training code")
    status = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "ct_train.py",
            "training",
            "dnnlib",
            "torch_utils",
        ],
        cwd=str(repo),
        text=True,
    ).strip()
    if status:
        fail("tracked or untracked reconstruction modules are dirty")
    protected_scripts = (
        "scripts/reconstruct_gap_lr_seed_initialization.py",
        "scripts/verify_gap_lr_seed_replication_run.py",
        "scripts/build_gap_lr_seed_replication_blind_evidence.py",
        "scripts/adjudicate_gap_lr_seed_replication.py",
    )
    for relative in protected_scripts:
        working = repo / relative
        committed = subprocess.check_output(
            ["git", "show", f"{adjudication_tooling_commit}:{relative}"],
            cwd=str(repo),
        )
        if not working.is_file() or working.read_bytes() != committed:
            fail(f"working adjudication tool differs from committed blob: {relative}")
    repo_root = repo.resolve()
    for origin in (Path(dnnlib.__file__).resolve(), Path(misc.__file__).resolve()):
        if repo_root not in origin.parents:
            fail("imported reconstruction modules did not originate in --repo")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--integrity-receipt-dir", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--transfer", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--adjudication-tooling-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-per-tensor", action="store_true")
    args = parser.parse_args()

    validate_repo(args.repo, args.adjudication_tooling_commit)
    if file_sha256(args.data) != DATA_SHA256:
        fail("dataset hash mismatch")
    transfer_sha = file_sha256(args.transfer)
    if transfer_sha != TRANSFER_SHA256:
        fail(f"transfer hash mismatch: {transfer_sha}")
    with args.transfer.open("rb") as handle:
        checkpoint = pickle.load(handle)
    source = checkpoint.get("ema") if isinstance(checkpoint, dict) else None
    if not isinstance(source, torch.nn.Module):
        fail("transfer does not contain a torch.nn.Module at ['ema']")
    source_interface = {
        "img_resolution": int(source.img_resolution),
        "img_channels": int(source.img_channels),
        "label_dim": int(source.label_dim),
    }
    if source_interface != {"img_resolution": 32, "img_channels": 3, "label_dim": 0}:
        fail(f"unexpected transfer network interface: {source_interface}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "gap_lr_seed_initialization_reconstruction",
        "status": "passed",
        "experiment_id": EXPERIMENT_ID,
        "quality_blind": {
            "generation_quality_metrics_accessed": False,
            "inputs_read": [
                "receipt-bound training_options.json",
                "frozen transfer checkpoint",
                "frozen implementation modules",
                "frozen dataset metadata",
            ],
            "inputs_explicitly_not_read": [
                "FID",
                "KID",
                "quality-evaluation outputs",
                "trained network snapshots",
                "training states",
            ],
        },
        "bindings": {
            "execution_protocol_commit": EXECUTION_PROTOCOL_COMMIT,
            "adjudication_tooling_commit": args.adjudication_tooling_commit,
            "training_code_commit": TRAINING_CODE_COMMIT,
            "dataset_sha256": DATA_SHA256,
            "transfer_checkpoint_sha256": transfer_sha,
            "tool_source_sha256": file_sha256(Path(__file__)),
        },
        "interpretation": {
            "hash_kind": "reconstructed_expected_initialization_hash",
            "historical_observed_preupdate_hash_captured": False,
            "does_not_attest_historical_process_memory": True,
            "rng_state_reconstructed": False,
            "scope": "transferred tensor state only",
        },
        "canonicalization": {
            "schema": "ECT_CANONICAL_TORCH_MODULE_V1",
            "ordering": "UTF-8 fully-qualified tensor name, then kind",
            "fields": ["kind", "name", "dtype", "rank", "shape", "nbytes", "raw_bytes"],
            "raw_bytes": "detach, CPU, contiguous, row-major, little-endian",
            "metadata_integer_encoding": "unsigned 64-bit big-endian",
            "excluded": ["module mode", "requires_grad", "non-tensor attributes"],
        },
        "runs": {},
    }

    interfaces = []
    contracts = []
    expected_run_verifier_sha256 = file_sha256(
        args.repo / "scripts/verify_gap_lr_seed_replication_run.py"
    )
    for seed, arm, run_id in RUNS:
        run_dir = args.experiment_root / run_id
        options_path = run_dir / "training_options.json"
        receipt_path = args.integrity_receipt_dir / f"seed{seed}_{arm}.integrity.json"
        if not options_path.is_file() or not receipt_path.is_file():
            fail(f"missing options or receipt for {run_id}")
        options = load_json(options_path)
        receipt = load_json(receipt_path)
        if (
            receipt.get("schema_version") != 2
            or receipt.get("receipt_type")
            != "gap_lr_seed_replication_run_integrity"
            or receipt.get("experiment_id") != EXPERIMENT_ID
            or receipt.get("status") != "passed"
            or receipt.get("seed") != seed
            or receipt.get("arm") != arm
            or options.get("seed") != seed
            or receipt.get("execution_protocol_commit") != EXECUTION_PROTOCOL_COMMIT
            or receipt.get("training_code_commit") != TRAINING_CODE_COMMIT
            or receipt.get("verifier", {}).get("source_sha256")
            != expected_run_verifier_sha256
            or Path(receipt.get("run_dir", "")).resolve() != run_dir.resolve()
        ):
            fail(f"receipt identity mismatch for {run_id}")
        options_sha = file_sha256(options_path)
        if receipt.get("artifact_sha256", {}).get("training_options") != options_sha:
            fail(f"training_options hash is not receipt-bound for {run_id}")
        if Path(options.get("resume_pkl", "")).resolve() != args.transfer.resolve():
            fail(f"{run_id}: resume_pkl does not resolve to frozen transfer")
        if Path(options.get("dataset_kwargs", {}).get("path", "")).resolve() != args.data.resolve():
            fail(f"{run_id}: dataset path does not resolve to frozen dataset")

        # Reconstruct the transferred tensor state. Construction RNG is not an
        # authoritative replay of the historical process (DataLoader/DDP/CUDA
        # setup is intentionally not recreated), and full transfer coverage
        # makes the pre-copy random tensor values irrelevant to this hash.
        np.random.seed(seed % (1 << 31))
        torch.manual_seed(int(np.random.randint(1 << 31)))
        net, interface = build_target(options)
        import training.dataset as training_dataset
        import training.networks as training_networks

        repo_root = args.repo.resolve()
        for origin in (
            Path(training_dataset.__file__).resolve(),
            Path(training_networks.__file__).resolve(),
        ):
            if repo_root not in origin.parents:
                fail("imported training modules did not originate in --repo")
        if interface != source_interface:
            fail(f"dataset/transfer interface mismatch for {run_id}")
        net = net.train().requires_grad_(True)
        ema = copy.deepcopy(net).eval().requires_grad_(False)
        net_copy = copy_source_subset(source, net)
        ema_copy = copy_source_subset(source, ema)
        net_hash = canonical_module(net, include_per_tensor=args.include_per_tensor)
        ema_hash = canonical_module(ema, include_per_tensor=args.include_per_tensor)
        if net_hash["sha256"] != ema_hash["sha256"]:
            fail(f"{run_id}: reconstructed net/ema hashes differ")

        contract = {
            "training_code_commit": TRAINING_CODE_COMMIT,
            "transfer_checkpoint_sha256": TRANSFER_SHA256,
            "network_kwargs": options["network_kwargs"],
            "interface_kwargs": interface,
            "copy_semantics": "destination-iterating name-matched tensor copy",
        }
        contract_sha = canonical_json_sha256(contract)
        interfaces.append(interface)
        contracts.append(contract_sha)
        report["runs"][run_id] = {
            "seed": seed,
            "arm": arm,
            "training_options_sha256": options_sha,
            "internal_integrity_receipt_sha256": file_sha256(receipt_path),
            "interface_kwargs": interface,
            "initialization_contract_sha256": contract_sha,
            "copy_contract": net_copy,
            "ema_copy_contract_equal": ema_copy == net_copy,
            "net": net_hash,
            "ema": ema_hash,
        }
        del net, ema
        gc.collect()

    hashes = {
        run_id: item["net"]["sha256"]
        for run_id, item in report["runs"].items()
    }
    report["cross_run"] = {
        "all_six_reconstructed_net_hashes_equal": len(set(hashes.values())) == 1,
        "all_six_initialization_contract_hashes_equal": len(set(contracts)) == 1,
        "all_six_dataset_interfaces_equal": len(
            {json.dumps(item, sort_keys=True) for item in interfaces}
        )
        == 1,
        "distinct_reconstructed_net_hashes": sorted(set(hashes.values())),
        "distinct_initialization_contract_hashes": sorted(set(contracts)),
        "run_hashes": hashes,
    }
    if not all(
        (
            report["cross_run"]["all_six_reconstructed_net_hashes_equal"],
            report["cross_run"]["all_six_initialization_contract_hashes_equal"],
            report["cross_run"]["all_six_dataset_interfaces_equal"],
        )
    ):
        fail("six-run reconstruction contract is not identical")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        fail(str(exc))
