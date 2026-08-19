#!/usr/bin/env python3
"""Canonical, fail-closed comparison of q256 replay artifacts.

The serialized bytes of two equivalent PyTorch checkpoints need not match.
This utility loads two training-state ``.pt`` files and their paired snapshot
``.pkl`` files, hashes canonical semantic content, and reports field-level
differences.  It is intended for the a=1 no-op replay and repeated 32-step
engineering smoke tests; it never mutates an input artifact.
"""

from __future__ import annotations

import argparse
import enum
import hashlib
import json
import math
import os
import pickle
import random
import re
import struct
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA_VERSION = "ect.q256.training-state-compare/v1"

# Unknown top-level fields are errors.  Additions must be reviewed and placed
# explicitly in one of these policies before they can affect a formal replay.
TRAINING_STATE_FIELDS = frozenset(
    {
        "net",
        "ema",
        "optimizer_state",
        "loss_fn",
        "loss_fn_state",
        "gradscaler_state",
        "cur_nimg",
        "cur_tick",
        "tick_start_nimg",
        "batch_idx",
        "attempted_iteration",
        "successful_optimizer_steps",
        "adaptive_signal_window_state",
        "rng_state",
        "torch_rng_state",
        "cuda_rng_state",
        "numpy_rng_state",
        "python_rng_state",
        "data_loader_state",
        "dataloader_state",
        "sampler_state",
        "batch_sampler_state",
        "dataset_state",
    }
)
TRAINING_STATE_REQUIRED = frozenset(
    {
        "net",
        "optimizer_state",
        "loss_fn_state",
        "gradscaler_state",
        "cur_nimg",
        "cur_tick",
        "tick_start_nimg",
        "attempted_iteration",
        "successful_optimizer_steps",
    }
)
SNAPSHOT_FIELDS = frozenset(
    {
        "ema",
        "loss_fn",
        "augment_pipe",
        "dataset_kwargs",
        "training_set_kwargs",
    }
)
SNAPSHOT_REQUIRED = frozenset({"ema", "loss_fn", "augment_pipe", "dataset_kwargs"})

TOP_LEVEL_EXCLUDED_FIELDS = frozenset(
    {
        "elapsed_sec",
        "elapsed_seconds",
        "run_dir",
        "outdir",
        "output_dir",
        "resume_pkl",
        "resume_state_dump",
        "non_semantic_metadata",
        "nonsemantic_metadata",
        "_nonsemantic_metadata",
        "_q256_radam_moment_transport",
    }
)
RECURSIVE_EXCLUDED_KEYS = frozenset(
    {
        "elapsed_sec",
        "elapsed_seconds",
        "created_utc",
        "command_line",
        "hostname",
        "pid",
        "run_dir",
        "outdir",
        "output_dir",
        "source_path",
        "output_path",
        "manifest_path",
        "non_semantic_metadata",
        "nonsemantic_metadata",
        "_nonsemantic_metadata",
        "_q256_radam_moment_transport",
    }
)
WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
MODULE_STRUCTURAL_FIELDS = frozenset({"_parameters", "_buffers", "_modules"})
MODULE_HOOK_FIELDS = frozenset(
    {
        "_backward_hooks",
        "_backward_pre_hooks",
        "_forward_hooks",
        "_forward_hooks_always_called",
        "_forward_hooks_with_kwargs",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_load_state_dict_post_hooks",
        "_load_state_dict_pre_hooks",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
    }
)


class ComparisonError(RuntimeError):
    """Unreadable, unsupported, or fail-closed artifact content."""


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Read NumPy 2 pickles in environments that still expose NumPy 1 paths."""

    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            if module.startswith("numpy._core"):
                return super().find_class(
                    module.replace("numpy._core", "numpy.core", 1), name
                )
            raise


class _NumpyCompatPickleModule:
    __name__ = "pickle"
    Unpickler = _NumpyCompatUnpickler
    Pickler = pickle.Pickler
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)
    dump = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _type_name(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    persistent_source = getattr(cls, "_orig_module_src", None)
    persistent_name = getattr(cls, "_orig_class_name", None)
    if isinstance(persistent_source, str) and isinstance(persistent_name, str):
        source_hash = hashlib.sha256(persistent_source.encode("utf-8")).hexdigest()
        return f"persistent_class:{persistent_name}:source_sha256={source_hash}"
    return f"{cls.__module__}.{cls.__qualname__}"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _is_absolute_path_string(value: str) -> bool:
    if not value:
        return False
    return os.path.isabs(value) or WINDOWS_ABSOLUTE_RE.match(value) is not None


def _float_record(value: float) -> dict[str, str]:
    if math.isnan(value):
        return {"kind": "float", "value": "nan"}
    if math.isinf(value):
        return {"kind": "float", "value": "+inf" if value > 0 else "-inf"}
    return {"kind": "float", "ieee754_be": struct.pack(">d", value).hex()}


def _is_flagged_nonsemantic_metadata(key: Any, value: Any) -> bool:
    return (
        isinstance(key, str)
        and key in {"metadata", "provenance"}
        and isinstance(value, Mapping)
        and (
            value.get("semantic") is False
            or value.get("non_semantic") is True
            or value.get("nonsemantic") is True
        )
    )


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous().reshape(-1)
    if value.numel() == 0:
        return b""
    try:
        return value.view(torch.uint8).numpy().tobytes(order="C")
    except (RuntimeError, TypeError) as exc:
        raise ComparisonError(
            f"cannot obtain canonical bytes for tensor dtype {tensor.dtype}"
        ) from exc


class Canonicalizer:
    """Convert supported Python/PyTorch objects to deterministic JSON trees."""

    def __init__(self):
        self.exclusions: list[dict[str, str]] = []
        self._active: set[int] = set()

    def _exclude(self, path: str, reason: str) -> dict[str, str]:
        self.exclusions.append({"path": path, "reason": reason})
        return {"kind": "excluded", "reason": reason}

    def _enter(self, value: Any, path: str) -> int:
        identity = id(value)
        if identity in self._active:
            raise ComparisonError(f"cyclic object graph at {path}")
        self._active.add(identity)
        return identity

    def _leave(self, identity: int) -> None:
        self._active.remove(identity)

    def _tensor(self, value: torch.Tensor, path: str) -> dict[str, Any]:
        base: dict[str, Any] = {
            "kind": "torch_tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "layout": str(value.layout),
        }
        if value.is_quantized:
            base["quantized"] = True
            base["int_repr"] = self.convert(value.int_repr(), f"{path}.int_repr")
            base["qscheme"] = str(value.qscheme())
            if value.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric):
                base["q_scale"] = _float_record(float(value.q_scale()))
                base["q_zero_point"] = int(value.q_zero_point())
            else:
                base["q_per_channel_scales"] = self.convert(
                    value.q_per_channel_scales(), f"{path}.q_per_channel_scales"
                )
                base["q_per_channel_zero_points"] = self.convert(
                    value.q_per_channel_zero_points(),
                    f"{path}.q_per_channel_zero_points",
                )
                base["q_per_channel_axis"] = int(value.q_per_channel_axis())
            return base
        if value.layout == torch.strided:
            base["bytes_sha256"] = hashlib.sha256(_tensor_bytes(value)).hexdigest()
            return base
        if value.layout == torch.sparse_coo:
            coalesced = value.detach().cpu().coalesce()
            base["indices"] = self.convert(coalesced.indices(), f"{path}.indices")
            base["values"] = self.convert(coalesced.values(), f"{path}.values")
            return base

        sparse_components = {
            getattr(torch, "sparse_csr", object()): ("crow_indices", "col_indices"),
            getattr(torch, "sparse_csc", object()): ("ccol_indices", "row_indices"),
            getattr(torch, "sparse_bsr", object()): ("crow_indices", "col_indices"),
            getattr(torch, "sparse_bsc", object()): ("ccol_indices", "row_indices"),
        }
        if value.layout in sparse_components:
            first, second = sparse_components[value.layout]
            base[first] = self.convert(getattr(value, first)(), f"{path}.{first}")
            base[second] = self.convert(getattr(value, second)(), f"{path}.{second}")
            base["values"] = self.convert(value.values(), f"{path}.values")
            return base
        raise ComparisonError(f"unsupported tensor layout {value.layout} at {path}")

    def _module(self, value: torch.nn.Module, path: str) -> dict[str, Any]:
        identity = self._enter(value, path)
        try:
            topology = []
            for name, module in value.named_modules():
                active_hooks = [
                    key
                    for key in MODULE_HOOK_FIELDS
                    if key in vars(module) and bool(vars(module)[key])
                ]
                if active_hooks:
                    raise ComparisonError(
                        f"module at {path}.{name} has unsupported active hooks: "
                        f"{sorted(active_hooks)}"
                    )
                public_attributes = {
                    key: item
                    for key, item in vars(module).items()
                    if key not in MODULE_STRUCTURAL_FIELDS
                    and key not in MODULE_HOOK_FIELDS
                    and key != "training"
                }
                topology.append(
                    {
                        "name": name,
                        "type": _type_name(module),
                        "training": bool(module.training),
                        "public_attributes": self.convert(
                            public_attributes, f"{path}.modules[{name!r}].attributes"
                        ),
                        "buffers": self.convert(
                            vars(module).get("_buffers", {}),
                            f"{path}.modules[{name!r}].buffers",
                        ),
                        "parameter_slots": sorted(
                            vars(module).get("_parameters", {}).keys()
                        ),
                    }
                )
            requires_grad = {
                name: bool(parameter.requires_grad)
                for name, parameter in value.named_parameters()
            }
            return {
                "kind": "torch_module",
                "type": _type_name(value),
                "topology": topology,
                "requires_grad": self.convert(requires_grad, f"{path}.requires_grad"),
                "state_dict": self.convert(value.state_dict(), f"{path}.state_dict"),
            }
        finally:
            self._leave(identity)

    def _mapping(self, value: Mapping[Any, Any], path: str) -> dict[str, Any]:
        identity = self._enter(value, path)
        try:
            entries = []
            seen_keys: set[bytes] = set()
            for key, item in value.items():
                if (
                    isinstance(key, str) and key in RECURSIVE_EXCLUDED_KEYS
                ) or _is_flagged_nonsemantic_metadata(key, item):
                    self._exclude(f"{path}.{key}", f"explicit_nonsemantic_key:{key}")
                    continue
                canonical_key = self.convert(key, f"{path}.<key>", is_key=True)
                key_bytes = _json_bytes(canonical_key)
                if key_bytes in seen_keys:
                    raise ComparisonError(f"canonical mapping-key collision at {path}")
                seen_keys.add(key_bytes)
                entries.append(
                    (
                        key_bytes,
                        canonical_key,
                        self.convert(item, f"{path}[{key!r}]"),
                    )
                )
            entries.sort(key=lambda entry: entry[0])
            return {
                "kind": "mapping",
                "entries": [[key, item] for _, key, item in entries],
            }
        finally:
            self._leave(identity)

    def convert(self, value: Any, path: str = "$", *, is_key: bool = False) -> Any:
        if value is None:
            return {"kind": "none"}
        if isinstance(value, bool):
            return {"kind": "bool", "value": value}
        if isinstance(value, int) and not isinstance(value, enum.Enum):
            return {"kind": "int", "value": str(value)}
        if isinstance(value, float):
            return _float_record(value)
        if isinstance(value, str):
            if not is_key and _is_absolute_path_string(value):
                return self._exclude(path, "absolute_path")
            return {"kind": "str", "value": value}
        if isinstance(value, bytes):
            return {
                "kind": "bytes",
                "size": len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
            }
        if isinstance(value, bytearray):
            return self.convert(bytes(value), path)
        if isinstance(value, memoryview):
            return self.convert(value.tobytes(), path)
        if isinstance(value, Path):
            return self.convert(str(value), path, is_key=is_key)
        if isinstance(value, enum.Enum):
            return {
                "kind": "enum",
                "type": _type_name(value),
                "name": value.name,
                "value": self.convert(value.value, f"{path}.value"),
            }
        if isinstance(value, torch.Tensor):
            return self._tensor(value, path)
        if isinstance(value, torch.nn.Module):
            return self._module(value, path)
        if isinstance(value, torch.Generator):
            return {
                "kind": "torch_generator",
                "device": str(value.device),
                "state": self.convert(value.get_state(), f"{path}.state"),
            }
        if isinstance(value, torch.device):
            return {"kind": "torch_device", "value": str(value)}
        if isinstance(value, torch.dtype):
            return {"kind": "torch_dtype", "value": str(value)}
        if isinstance(value, torch.layout):
            return {"kind": "torch_layout", "value": str(value)}
        if isinstance(value, torch.memory_format):
            return {"kind": "torch_memory_format", "value": str(value)}
        if isinstance(value, torch.qscheme):
            return {"kind": "torch_qscheme", "value": str(value)}
        if isinstance(value, torch.Size):
            return {"kind": "torch_size", "value": list(value)}
        if isinstance(value, np.dtype):
            return {"kind": "numpy_dtype", "value": value.str}
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                return {
                    "kind": "numpy_array_object",
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                    "values": self.convert(value.tolist(), f"{path}.values"),
                }
            array = np.ascontiguousarray(value)
            return {
                "kind": "numpy_array",
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
        if isinstance(value, np.generic):
            scalar = np.asarray(value)
            return {
                "kind": "numpy_scalar",
                "dtype": scalar.dtype.str,
                "bytes_sha256": hashlib.sha256(scalar.tobytes()).hexdigest(),
            }
        if isinstance(value, random.Random):
            return {
                "kind": "python_random",
                "state": self.convert(value.getstate(), f"{path}.state"),
            }
        if isinstance(value, np.random.RandomState):
            return {
                "kind": "numpy_random_state",
                "state": self.convert(value.get_state(), f"{path}.state"),
            }
        if isinstance(value, Mapping):
            return self._mapping(value, path)
        if isinstance(value, tuple):
            identity = self._enter(value, path)
            try:
                return {
                    "kind": "tuple",
                    "items": [
                        self.convert(item, f"{path}[{index}]")
                        for index, item in enumerate(value)
                    ],
                }
            finally:
                self._leave(identity)
        if isinstance(value, list):
            identity = self._enter(value, path)
            try:
                return {
                    "kind": "list",
                    "items": [
                        self.convert(item, f"{path}[{index}]")
                        for index, item in enumerate(value)
                    ],
                }
            finally:
                self._leave(identity)
        if isinstance(value, (set, frozenset)):
            identity = self._enter(value, path)
            try:
                items = [self.convert(item, f"{path}.<set-item>") for item in value]
                items.sort(key=_json_bytes)
                return {"kind": _type_name(value), "items": items}
            finally:
                self._leave(identity)
        if isinstance(value, range):
            return {
                "kind": "range",
                "start": value.start,
                "stop": value.stop,
                "step": value.step,
            }
        if isinstance(value, slice):
            return {
                "kind": "slice",
                "start": self.convert(value.start, f"{path}.start"),
                "stop": self.convert(value.stop, f"{path}.stop"),
                "step": self.convert(value.step, f"{path}.step"),
            }
        if isinstance(value, type):
            return {"kind": "type", "value": _type_name(value)}
        if isinstance(
            value, (types.FunctionType, types.MethodType, types.BuiltinFunctionType)
        ):
            raise ComparisonError(f"unsupported callable {_type_name(value)} at {path}")
        if hasattr(value, "__dict__"):
            identity = self._enter(value, path)
            try:
                return {
                    "kind": "object",
                    "type": _type_name(value),
                    "attributes": self.convert(vars(value), f"{path}.__dict__"),
                }
            finally:
                self._leave(identity)
        raise ComparisonError(f"unsupported value type {_type_name(value)} at {path}")


def canonical_hash(value: Any) -> tuple[str, list[dict[str, str]]]:
    canonicalizer = Canonicalizer()
    canonical = canonicalizer.convert(value)
    return _content_hash(canonical), sorted(
        canonicalizer.exclusions, key=lambda item: (item["path"], item["reason"])
    )


def _is_explicit_top_exclusion(key: Any, value: Any) -> bool:
    return (
        isinstance(key, str)
        and (key in TOP_LEVEL_EXCLUDED_FIELDS or key.startswith("_nonsemantic_"))
    ) or _is_flagged_nonsemantic_metadata(key, value)


def summarize_payload(payload: Any, *, kind: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ComparisonError(f"{kind} root must be a mapping")
    if kind == "training_state":
        allowed = TRAINING_STATE_FIELDS
        required = TRAINING_STATE_REQUIRED
    elif kind == "snapshot":
        allowed = SNAPSHOT_FIELDS
        required = SNAPSHOT_REQUIRED
    else:
        raise ComparisonError(f"unknown artifact kind {kind!r}")

    excluded_fields = sorted(
        key for key, value in payload.items() if _is_explicit_top_exclusion(key, value)
    )
    semantic_keys = set(payload) - set(excluded_fields)
    unknown = semantic_keys - allowed
    if unknown:
        raise ComparisonError(
            f"{kind} has unknown top-level fields: {sorted(map(repr, unknown))}"
        )
    missing = required - semantic_keys
    if missing:
        raise ComparisonError(
            f"{kind} is missing required top-level fields: {sorted(missing)}"
        )

    field_hashes: dict[str, str] = {}
    nested_exclusions: dict[str, list[dict[str, str]]] = {}
    for key in sorted(semantic_keys):
        digest, exclusions = canonical_hash(payload[key])
        field_hashes[key] = digest
        if exclusions:
            nested_exclusions[key] = exclusions
    return {
        "kind": kind,
        "canonical_sha256": _content_hash({"kind": kind, "field_hashes": field_hashes}),
        "field_hashes": field_hashes,
        "excluded_top_level_fields": excluded_fields,
        "nested_exclusions": nested_exclusions,
    }


def load_training_state(path: Path) -> Any:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            pickle_module=_NumpyCompatPickleModule,
        )
    except Exception as exc:
        raise ComparisonError(f"cannot load training state {path}: {exc}") from exc


def load_snapshot(path: Path) -> Any:
    try:
        with path.open("rb") as handle:
            return _NumpyCompatUnpickler(handle).load()
    except Exception as exc:
        raise ComparisonError(f"cannot load snapshot {path}: {exc}") from exc


def summarize_file(path: Path, *, kind: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ComparisonError(f"{kind} input is not a regular file: {path}")
    stat_before = path.stat()
    signature_before = (
        stat_before.st_dev,
        stat_before.st_ino,
        stat_before.st_size,
        stat_before.st_mtime_ns,
    )
    payload = (
        load_training_state(path) if kind == "training_state" else load_snapshot(path)
    )
    summary = summarize_payload(payload, kind=kind)
    file_sha256 = sha256_file(path)
    stat_after = path.stat()
    signature_after = (
        stat_after.st_dev,
        stat_after.st_ino,
        stat_after.st_size,
        stat_after.st_mtime_ns,
    )
    if signature_before != signature_after:
        raise ComparisonError(
            f"{kind} input changed while it was being compared: {path}"
        )
    summary.update(
        {
            "path": str(path),
            "size_bytes": stat_after.st_size,
            "file_sha256": file_sha256,
        }
    )
    return summary


def compare_summaries(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    if left["kind"] != right["kind"]:
        raise ComparisonError("cannot compare summaries of different artifact kinds")
    left_fields = left["field_hashes"]
    right_fields = right["field_hashes"]
    field_names = sorted(set(left_fields) | set(right_fields))
    differences = [
        {
            "field": field,
            "left_sha256": left_fields.get(field),
            "right_sha256": right_fields.get(field),
            "reason": (
                "missing_left"
                if field not in left_fields
                else "missing_right"
                if field not in right_fields
                else "content_mismatch"
            ),
        }
        for field in field_names
        if left_fields.get(field) != right_fields.get(field)
    ]
    return {
        "kind": left["kind"],
        "equal": not differences,
        "left_canonical_sha256": left["canonical_sha256"],
        "right_canonical_sha256": right["canonical_sha256"],
        "differences": differences,
    }


def compare_artifacts(
    left_state: Path,
    right_state: Path,
    left_snapshot: Path,
    right_snapshot: Path,
) -> tuple[dict[str, Any], int]:
    specifications = (
        ("left_training_state", left_state, "training_state"),
        ("right_training_state", right_state, "training_state"),
        ("left_snapshot", left_snapshot, "snapshot"),
        ("right_snapshot", right_snapshot, "snapshot"),
    )
    summaries: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    for label, path, kind in specifications:
        try:
            summaries[label] = summarize_file(path, kind=kind)
        except ComparisonError as exc:
            errors.append({"artifact": label, "error": str(exc)})
    if errors:
        return (
            {
                "schema": SCHEMA_VERSION,
                "status": "CORRUPT",
                "equal": False,
                "errors": errors,
            },
            3,
        )

    state_comparison = compare_summaries(
        summaries["left_training_state"], summaries["right_training_state"]
    )
    snapshot_comparison = compare_summaries(
        summaries["left_snapshot"], summaries["right_snapshot"]
    )
    equal = state_comparison["equal"] and snapshot_comparison["equal"]
    result = {
        "schema": SCHEMA_VERSION,
        "status": "EQUAL" if equal else "NOT_EQUAL",
        "equal": equal,
        "comparison": {
            "training_state": state_comparison,
            "snapshot": snapshot_comparison,
        },
        "artifacts": summaries,
        "policy": {
            "unknown_top_level_fields": "fail_closed",
            "excluded_top_level_fields": sorted(TOP_LEVEL_EXCLUDED_FIELDS),
            "absolute_paths": "excluded_from_canonical_content",
            "tensor_hash": "dtype+shape+layout+canonical_value_bytes",
            "mapping_order": "normalized",
        },
    }
    return result, 0 if equal else 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--left-state",
        "--left-training-state",
        dest="left_state",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--right-state",
        "--right-training-state",
        dest="right_state",
        type=Path,
        required=True,
    )
    parser.add_argument("--left-snapshot", type=Path, required=True)
    parser.add_argument("--right-snapshot", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="also write the strict JSON report to this path",
    )
    return parser.parse_args(argv)


def _emit_json(payload: Mapping[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is not None:
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(text)
    sys.stdout.write(text)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        payload, exit_code = compare_artifacts(
            args.left_state,
            args.right_state,
            args.left_snapshot,
            args.right_snapshot,
        )
    except Exception as exc:
        payload = {
            "schema": SCHEMA_VERSION,
            "status": "CORRUPT",
            "equal": False,
            "errors": [{"artifact": "comparison", "error": str(exc)}],
        }
        exit_code = 3
    payload["exit_code"] = exit_code
    try:
        _emit_json(payload, args.output)
    except Exception as exc:
        fallback = {
            "schema": SCHEMA_VERSION,
            "status": "CORRUPT",
            "equal": False,
            "exit_code": 3,
            "errors": [{"artifact": "json_output", "error": str(exc)}],
        }
        sys.stdout.write(json.dumps(fallback, sort_keys=True, allow_nan=False) + "\n")
        return 3
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
