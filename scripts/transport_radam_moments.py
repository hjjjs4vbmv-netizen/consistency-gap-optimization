#!/usr/bin/env python3
"""Apply a one-time, whole-model scale transport to saved RAdam moments.

The checkpoint-level entry point is deliberately fail-closed.  It writes a new
checkpoint and JSON sidecar without replacing an existing path, preserves the
source file, and embeds a provenance marker that prevents the output from being
used as the source of another checkpoint-level transport.

Only trusted checkpoints produced by this repository should be loaded.  PyTorch
checkpoints are pickle based and are not safe inputs from untrusted sources.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _datetime
import hashlib
import json
import math
import os
import platform
import random
import sys
import threading
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402


TOOL_VERSION = "1"
MARKER_KEY = "_q256_radam_moment_transport"
MARKER_SCHEMA = "ect.q256.radam-moment-transport-marker/v1"
MANIFEST_SCHEMA = "ect.q256.radam-moment-transport-manifest/v1"

MOMENT_FACTORS = {
    "exp_avg": 1,
    "exp_avg_sq": 2,
    "max_exp_avg_sq": 2,
}

# These RAdam fields may legitimately be tensors (or containers holding
# tensors), but they are state, not moments, and therefore must remain intact.
# Unknown tensor-valued fields are rejected below rather than guessed at.
KNOWN_PRESERVED_TENSOR_STATE_FIELDS = frozenset(
    {
        "step",
        "buffer",
        "radam_buffer",
        "rectification_buffer",
        "radam_rectification_buffer",
    }
)


class TransportError(RuntimeError):
    """Raised when the requested transform cannot be proven safe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameter_id(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    return repr(value)


def _contains_tensor(value: Any, seen: Optional[set] = None) -> bool:
    if torch.is_tensor(value):
        return True
    if seen is None:
        seen = set()
    object_id = id(value)
    if object_id in seen:
        return False
    seen.add(object_id)
    if isinstance(value, Mapping):
        return any(
            _contains_tensor(key, seen) or _contains_tensor(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_tensor(item, seen) for item in value)
    return False


def _tensor_payloads(tensor: torch.Tensor) -> Iterable[Tuple[str, torch.Tensor]]:
    """Yield the logical tensor payloads needed for hashing and norm checks."""
    layout = tensor.layout
    if layout == torch.strided:
        yield "values", tensor
        return
    if layout == torch.sparse_coo:
        yield "indices", tensor._indices()
        yield "values", tensor._values()
        return

    sparse_layouts = {
        getattr(torch, name)
        for name in ("sparse_csr", "sparse_csc", "sparse_bsr", "sparse_bsc")
        if hasattr(torch, name)
    }
    if layout not in sparse_layouts:
        raise TransportError(f"unsupported moment tensor layout: {layout}")
    if layout in {
        getattr(torch, "sparse_csr", None),
        getattr(torch, "sparse_bsr", None),
    }:
        yield "compressed_indices", tensor.crow_indices()
        yield "plain_indices", tensor.col_indices()
    else:
        yield "compressed_indices", tensor.ccol_indices()
        yield "plain_indices", tensor.row_indices()
    yield "values", tensor.values()


def _tensor_values(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.layout == torch.strided:
        return tensor
    if tensor.layout == torch.sparse_coo:
        return tensor._values()
    sparse_layouts = {
        getattr(torch, name)
        for name in ("sparse_csr", "sparse_csc", "sparse_bsr", "sparse_bsc")
        if hasattr(torch, name)
    }
    if tensor.layout in sparse_layouts:
        return tensor.values()
    raise TransportError(f"unsupported moment tensor layout: {tensor.layout}")


def _raw_tensor_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous()
    if cpu.numel() == 0:
        return b""
    # The contiguous tensor owns exactly numel * element_size bytes, so the
    # storage conversion does not include unrelated view/storage contents.
    return bytes(cpu.untyped_storage())


def tensor_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(tensor.layout).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(tensor.device).encode("utf-8"))
    digest.update(b"\0")
    for name, payload in _tensor_payloads(tensor):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(payload.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(payload.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_raw_tensor_bytes(payload))
        digest.update(b"\0")
    return digest.hexdigest()


def tensor_l2_norm(tensor: torch.Tensor) -> float:
    values = _tensor_values(tensor).detach()
    if values.numel() == 0:
        return 0.0
    magnitudes = values.abs().to(dtype=torch.float64)
    return float(torch.linalg.vector_norm(magnitudes.reshape(-1)).item())


def _tensor_metadata(tensor: torch.Tensor) -> Dict[str, Any]:
    return {
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "shape": list(tensor.shape),
        "layout": str(tensor.layout),
        "is_sparse": bool(tensor.layout != torch.strided),
        "requires_grad": bool(tensor.requires_grad),
    }


def _clone_and_scale(tensor: torch.Tensor, factor: float) -> torch.Tensor:
    if tensor.is_quantized:
        raise TransportError("quantized optimizer moments are unsupported")
    if not (tensor.is_floating_point() or tensor.is_complex()):
        raise TransportError(
            f"optimizer moment must be floating point, got {tensor.dtype}"
        )
    source_values = _tensor_values(tensor)
    if not bool(torch.isfinite(source_values).all().item()):
        raise TransportError("optimizer moment contains NaN or Inf before transport")
    if factor == 1.0:
        # This branch makes a=1 a bitwise no-op for supported moment tensors.
        return tensor
    if tensor.layout == torch.strided:
        scaled = tensor.clone(memory_format=torch.preserve_format)
    else:
        scaled = tensor.clone()
    scaled.mul_(factor)
    if not bool(torch.isfinite(_tensor_values(scaled)).all().item()):
        raise TransportError("optimizer moment contains NaN or Inf after transport")
    return scaled


def _norm_ratio_tolerance(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16):
        return 1.0e-2
    if dtype in (torch.float32, torch.complex64):
        return 2.0e-5
    return 1.0e-10


def _validate_optimizer_structure(
    optimizer_state: MutableMapping[str, Any],
) -> Tuple[MutableMapping[Any, Any], Sequence[Mapping[str, Any]], List[Any], List[Any]]:
    if not isinstance(optimizer_state, MutableMapping):
        raise TransportError("optimizer_state must be a mutable mapping")
    states = optimizer_state.get("state")
    groups = optimizer_state.get("param_groups")
    if not isinstance(states, MutableMapping):
        raise TransportError("optimizer_state['state'] must be a mutable mapping")
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
        raise TransportError("optimizer_state['param_groups'] must be a sequence")

    associated: List[Any] = []
    for group_index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise TransportError(
                f"optimizer parameter group {group_index} is not a mapping"
            )
        params = group.get("params")
        if not isinstance(params, Sequence) or isinstance(params, (str, bytes)):
            raise TransportError(
                f"optimizer parameter group {group_index} has no parameter sequence"
            )
        for param_id in params:
            if any(param_id == previous for previous in associated):
                raise TransportError(
                    f"duplicate optimizer parameter id: {_parameter_id(param_id)}"
                )
            associated.append(param_id)

    extras = [
        state_id
        for state_id in states
        if not any(state_id == param_id for param_id in associated)
    ]
    if extras:
        formatted = ", ".join(_parameter_id(item) for item in extras)
        raise TransportError(
            f"optimizer states are not associated with parameter groups: {formatted}"
        )
    missing = [
        param_id
        for param_id in associated
        if param_id not in states or not states[param_id]
    ]
    return states, groups, associated, missing


def _association_hash(groups: Sequence[Mapping[str, Any]]) -> str:
    canonical = []
    for group_index, group in enumerate(groups):
        canonical.append(
            {
                "group_index": group_index,
                "params": [_parameter_id(item) for item in group["params"]],
            }
        )
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def transform_optimizer_state(
    optimizer_state: MutableMapping[str, Any],
    coefficient: float,
    *,
    allow_missing_state: bool = False,
) -> Dict[str, Any]:
    """Transform supported RAdam moments in place and return an audit report.

    This lower-level function intentionally has no one-time marker so its
    algebra can be tested with T_a followed by T_(1/a).  Use
    :func:`transform_checkpoint` for any persisted formal artifact.
    """
    try:
        coefficient = float(coefficient)
    except (TypeError, ValueError) as exc:
        raise TransportError("transport coefficient must be numeric") from exc
    if not math.isfinite(coefficient) or coefficient <= 0.0:
        raise TransportError(
            "transport coefficient must be finite and greater than zero"
        )
    second_factor = coefficient * coefficient
    if not math.isfinite(second_factor):
        raise TransportError("squared transport coefficient is not finite")

    states, groups, associated, missing = _validate_optimizer_structure(optimizer_state)
    if missing and not allow_missing_state:
        formatted = ", ".join(_parameter_id(item) for item in missing)
        raise TransportError(
            f"optimizer state is missing for parameter ids: {formatted}"
        )

    association_before = _association_hash(groups)
    records: List[Dict[str, Any]] = []
    transformed_state_count = 0
    aggregate_before = {name: 0.0 for name in MOMENT_FACTORS}
    aggregate_after = {name: 0.0 for name in MOMENT_FACTORS}

    for param_id in associated:
        if param_id in missing:
            continue
        param_state = states[param_id]
        if not isinstance(param_state, MutableMapping):
            raise TransportError(
                f"optimizer state for parameter {_parameter_id(param_id)} is not a mapping"
            )

        for field, value in param_state.items():
            if field in MOMENT_FACTORS:
                if not torch.is_tensor(value):
                    raise TransportError(
                        f"optimizer field {field!r} for parameter "
                        f"{_parameter_id(param_id)} must be a tensor"
                    )
            elif (
                _contains_tensor(value)
                and field not in KNOWN_PRESERVED_TENSOR_STATE_FIELDS
            ):
                raise TransportError(
                    f"unsupported tensor-valued optimizer-state field {field!r} "
                    f"for parameter {_parameter_id(param_id)}"
                )

        missing_required = [
            name for name in ("exp_avg", "exp_avg_sq") if name not in param_state
        ]
        if missing_required:
            raise TransportError(
                f"optimizer state for parameter {_parameter_id(param_id)} is missing required "
                f"RAdam fields: {', '.join(missing_required)}"
            )

        transformed_state_count += 1
        for field, power in MOMENT_FACTORS.items():
            if field not in param_state:
                continue
            original = param_state[field]
            factor = coefficient if power == 1 else second_factor
            metadata_before = _tensor_metadata(original)
            hash_before = tensor_sha256(original)
            norm_before = tensor_l2_norm(original)
            scaled = _clone_and_scale(original, factor)
            metadata_after = _tensor_metadata(scaled)
            if metadata_after != metadata_before:
                raise TransportError(
                    f"tensor metadata changed for parameter {_parameter_id(param_id)} field {field}"
                )
            hash_after = tensor_sha256(scaled)
            norm_after = tensor_l2_norm(scaled)
            if not math.isfinite(norm_before) or not math.isfinite(norm_after):
                raise TransportError(
                    f"non-finite moment norm for parameter {_parameter_id(param_id)} field {field}"
                )
            observed_ratio: Optional[float]
            if norm_before == 0.0:
                if norm_after != 0.0:
                    raise TransportError(
                        f"zero moment became nonzero for parameter {_parameter_id(param_id)} field {field}"
                    )
                observed_ratio = None
                ratio_verified = True
            else:
                observed_ratio = norm_after / norm_before
                ratio_verified = math.isclose(
                    observed_ratio,
                    factor,
                    rel_tol=_norm_ratio_tolerance(original.dtype),
                    abs_tol=0.0,
                )
                if not ratio_verified:
                    raise TransportError(
                        f"moment norm ratio verification failed for parameter "
                        f"{_parameter_id(param_id)} field {field}: "
                        f"observed {observed_ratio}, expected {factor}"
                    )
            if factor == 1.0 and hash_after != hash_before:
                raise TransportError(
                    f"a=1 changed parameter {_parameter_id(param_id)} field {field}"
                )
            param_state[field] = scaled
            # hypot avoids overflow that can occur when finite norms are
            # explicitly squared before being accumulated.
            aggregate_before[field] = math.hypot(aggregate_before[field], norm_before)
            aggregate_after[field] = math.hypot(aggregate_after[field], norm_after)
            records.append(
                {
                    "parameter_id": _parameter_id(param_id),
                    "field": field,
                    "power": power,
                    "expected_factor": factor,
                    "metadata": metadata_before,
                    "before_sha256": hash_before,
                    "after_sha256": hash_after,
                    "before_l2_norm": norm_before,
                    "after_l2_norm": norm_after,
                    "observed_norm_ratio": observed_ratio,
                    "norm_ratio_verified": ratio_verified,
                }
            )

    association_after = _association_hash(groups)
    if association_after != association_before:
        raise TransportError("optimizer parameter association changed during transport")

    aggregate: Dict[str, Any] = {}
    for field in MOMENT_FACTORS:
        if not any(record["field"] == field for record in records):
            continue
        before = aggregate_before[field]
        after = aggregate_after[field]
        aggregate[field] = {
            "before_l2_norm": before,
            "after_l2_norm": after,
            "observed_norm_ratio": None if before == 0.0 else after / before,
        }

    return {
        "coefficient": coefficient,
        "squared_coefficient": second_factor,
        "parameter_count": len(associated),
        "transformed_state_count": transformed_state_count,
        "missing_state_parameter_ids": [_parameter_id(item) for item in missing],
        "moment_tensor_count": len(records),
        "parameter_association_sha256": association_after,
        "moment_tensors": records,
        "aggregate_moment_norms": aggregate,
    }


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": copy.deepcopy(random.getstate()),
        "torch_cpu": torch.random.get_rng_state().clone(),
        "torch_cuda": None,
        "numpy": None,
    }
    if torch.cuda.is_initialized():
        state["torch_cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    numpy = sys.modules.get("numpy")
    if numpy is not None and hasattr(numpy, "random"):
        state["numpy"] = copy.deepcopy(numpy.random.get_state())
    return state


def _numpy_rng_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if len(left) != len(right):
        return False
    for left_item, right_item in zip(left, right):
        if hasattr(left_item, "shape") and hasattr(right_item, "shape"):
            if not bool((left_item == right_item).all()):
                return False
        elif left_item != right_item:
            return False
    return True


def _rng_comparison(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, bool]:
    cuda_before = before["torch_cuda"]
    cuda_after = after["torch_cuda"]
    cuda_equal = cuda_before is None and cuda_after is None
    if (
        cuda_before is not None
        and cuda_after is not None
        and len(cuda_before) == len(cuda_after)
    ):
        cuda_equal = all(
            torch.equal(left, right) for left, right in zip(cuda_before, cuda_after)
        )
    return {
        "python": before["python"] == after["python"],
        "numpy": _numpy_rng_equal(before["numpy"], after["numpy"]),
        "torch_cpu": torch.equal(before["torch_cpu"], after["torch_cpu"]),
        "torch_cuda": cuda_equal,
    }


def _default_manifest_path(output: Path) -> Path:
    return output.with_name(output.name + ".manifest.json")


def _load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path)


def _open_unique_temp(target: Path, kind: str):
    target.parent.mkdir(parents=False, exist_ok=True)
    prefix = f".{target.name}.{kind}-tmp-{os.getpid()}-{threading.get_ident()}"
    for counter in range(1000):
        candidate = target.parent / f"{prefix}-{counter}"
        try:
            return candidate, candidate.open("xb")
        except FileExistsError:
            continue
    raise TransportError(
        f"could not allocate deterministic temporary file beside {target}"
    )


def _write_checkpoint_temp(target: Path, checkpoint: Any) -> Path:
    temp_path, handle = _open_unique_temp(target, "checkpoint")
    try:
        with handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        return temp_path
    except BaseException:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_manifest_temp(target: Path, manifest: Mapping[str, Any]) -> Path:
    temp_path, handle = _open_unique_temp(target, "manifest")
    try:
        with handle:
            payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)
            handle.write((payload + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        return temp_path
    except BaseException:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _link_noreplace(temp_path: Path, target: Path) -> None:
    try:
        os.link(str(temp_path), str(target), follow_symlinks=False)
    except FileExistsError as exc:
        raise TransportError(f"refusing to overwrite existing path: {target}") from exc
    except OSError as exc:
        raise TransportError(
            f"atomic no-replace link failed for {target}; output filesystem must support hard links"
        ) from exc


def _unlink_linked_target(target: Path, temp_path: Path) -> None:
    try:
        target_stat = target.stat()
        temp_stat = temp_path.stat()
    except FileNotFoundError:
        return
    if (target_stat.st_dev, target_stat.st_ino) == (temp_stat.st_dev, temp_stat.st_ino):
        target.unlink()


def _check_adjacent_transport_sidecar(source: Path, source_sha256: str) -> None:
    sidecar = _default_manifest_path(source)
    if not sidecar.exists():
        return
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if payload.get("schema") != MANIFEST_SCHEMA:
        return
    output = payload.get("output", {})
    if output.get("sha256") == source_sha256:
        raise TransportError(
            f"source has a matching moment-transport sidecar and cannot be transformed again: {sidecar}"
        )


def transform_checkpoint(
    source: Path,
    output: Path,
    coefficient: float,
    *,
    manifest_path: Optional[Path] = None,
    optimizer_key: str = "optimizer_state",
    expected_source_sha256: Optional[str] = None,
    allow_missing_state: bool = False,
    command_line: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Create an atomically published, one-time transported checkpoint."""
    source = Path(source)
    output = Path(output)
    manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else _default_manifest_path(output)
    )

    if not source.is_file():
        raise TransportError(f"source checkpoint is not a regular file: {source}")
    if source.resolve() == output.resolve():
        raise TransportError("source and output checkpoint paths must differ")
    if output.resolve() == manifest_path.resolve():
        raise TransportError("output checkpoint and manifest paths must differ")
    if output.exists() or output.is_symlink():
        raise TransportError(
            f"refusing to overwrite existing output checkpoint: {output}"
        )
    if manifest_path.exists() or manifest_path.is_symlink():
        raise TransportError(
            f"refusing to overwrite existing manifest: {manifest_path}"
        )
    if not output.parent.is_dir():
        raise TransportError(f"output parent directory does not exist: {output.parent}")
    if not manifest_path.parent.is_dir():
        raise TransportError(
            f"manifest parent directory does not exist: {manifest_path.parent}"
        )

    rng_before = _capture_rng_state()
    source_sha_before = sha256_file(source)
    if expected_source_sha256 is not None:
        expected = expected_source_sha256.strip().lower()
        if source_sha_before != expected:
            raise TransportError(
                f"source SHA256 mismatch: observed {source_sha_before}, expected {expected}"
            )
    _check_adjacent_transport_sidecar(source, source_sha_before)

    checkpoint = _load_checkpoint(source)
    if not isinstance(checkpoint, MutableMapping):
        raise TransportError("checkpoint root must be a mutable mapping")
    if MARKER_KEY in checkpoint:
        raise TransportError(
            "checkpoint already contains a moment-transport provenance marker"
        )
    optimizer_state = checkpoint.get(optimizer_key)
    if optimizer_state is None:
        raise TransportError(f"checkpoint is missing optimizer key {optimizer_key!r}")

    optimizer_report = transform_optimizer_state(
        optimizer_state,
        coefficient,
        allow_missing_state=allow_missing_state,
    )
    marker = {
        "schema": MARKER_SCHEMA,
        "tool_version": TOOL_VERSION,
        "source_sha256": source_sha_before,
        "coefficient": optimizer_report["coefficient"],
        "squared_coefficient": optimizer_report["squared_coefficient"],
        "optimizer_key": optimizer_key,
        "parameter_association_sha256": optimizer_report[
            "parameter_association_sha256"
        ],
    }
    checkpoint[MARKER_KEY] = marker

    checkpoint_temp: Optional[Path] = None
    manifest_temp: Optional[Path] = None
    output_created = False
    manifest_created = False
    try:
        checkpoint_temp = _write_checkpoint_temp(output, checkpoint)
        output_sha = sha256_file(checkpoint_temp)
        source_sha_after = sha256_file(source)
        if source_sha_after != source_sha_before:
            raise TransportError(
                "source checkpoint changed while the transport was running"
            )

        rng_after_work = _capture_rng_state()
        rng_checks = _rng_comparison(rng_before, rng_after_work)
        if not all(rng_checks.values()):
            changed = ", ".join(
                name for name, unchanged in rng_checks.items() if not unchanged
            )
            raise TransportError(f"checkpoint transport changed RNG state: {changed}")

        manifest: Dict[str, Any] = {
            "schema": MANIFEST_SCHEMA,
            "created_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
            "tool": {
                "version": TOOL_VERSION,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "command_line": list(command_line)
                if command_line is not None
                else None,
            },
            "source": {
                "path": str(source.resolve()),
                "sha256": source_sha_before,
                "sha256_after": source_sha_after,
                "size_bytes": source.stat().st_size,
                "unchanged": True,
            },
            "output": {
                "path": str(output.resolve()),
                "sha256": output_sha,
                "size_bytes": checkpoint_temp.stat().st_size,
            },
            "manifest_path": str(manifest_path.resolve()),
            "marker": marker,
            "optimizer": {
                "key": optimizer_key,
                **optimizer_report,
            },
            "preservation": {
                "non_moment_policy": (
                    "all existing checkpoint and optimizer fields are preserved; "
                    f"only {sorted(MOMENT_FACTORS)} are scaled and {MARKER_KEY!r} is added"
                ),
                "rng_unchanged": rng_checks,
            },
        }
        manifest_temp = _write_manifest_temp(manifest_path, manifest)

        _link_noreplace(checkpoint_temp, output)
        output_created = True
        _link_noreplace(manifest_temp, manifest_path)
        manifest_created = True
        _fsync_directory(output.parent)
        if manifest_path.parent != output.parent:
            _fsync_directory(manifest_path.parent)

        # Recheck published bytes and source immutability before reporting success.
        if sha256_file(output) != output_sha:
            raise TransportError(
                "published output hash differs from the atomic temporary file"
            )
        if sha256_file(source) != source_sha_before:
            raise TransportError("source checkpoint changed during output publication")
        return manifest
    except BaseException:
        if manifest_created and manifest_temp is not None:
            _unlink_linked_target(manifest_path, manifest_temp)
        if output_created and checkpoint_temp is not None:
            _unlink_linked_target(output, checkpoint_temp)
        raise
    finally:
        for temp_path in (manifest_temp, checkpoint_temp):
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a one-time RAdam moment-scale-transported checkpoint."
    )
    parser.add_argument(
        "--source", type=Path, required=True, help="trusted source .pt checkpoint"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="new checkpoint path"
    )
    parser.add_argument(
        "--coefficient",
        "--scale",
        dest="coefficient",
        type=float,
        required=True,
        help="positive finite whole-model coefficient a_s",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="sidecar path (default: OUTPUT.manifest.json)",
    )
    parser.add_argument(
        "--optimizer-key",
        default="optimizer_state",
        help="checkpoint key containing torch optimizer.state_dict()",
    )
    parser.add_argument(
        "--expected-source-sha256",
        default=None,
        help="fail unless the source checkpoint matches this SHA256",
    )
    parser.add_argument(
        "--allow-missing-state",
        action="store_true",
        help="preserve and report parameters whose lazy optimizer state is absent",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command_line = [sys.executable, str(Path(__file__).resolve())]
    command_line.extend(list(argv) if argv is not None else sys.argv[1:])
    try:
        manifest = transform_checkpoint(
            args.source,
            args.output,
            args.coefficient,
            manifest_path=args.manifest,
            optimizer_key=args.optimizer_key,
            expected_source_sha256=args.expected_source_sha256,
            allow_missing_state=args.allow_missing_state,
            command_line=command_line,
        )
    except (OSError, TransportError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
