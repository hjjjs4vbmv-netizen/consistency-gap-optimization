#!/usr/bin/env python3
"""Fail-closed compatibility audit for q=256 continuation controls.

The input is a JSON manifest with one ``reference`` entry per seed and an
entry for each existing ``controls.F`` and ``controls.G`` candidate::

    {"reference": {"3": {"source": {...}, "run": {...}}, ...},
     "controls": {"F": {"3": {...}, ...}, "G": {"3": {...}, ...}}}

An artifact ``source`` may name ``run_dir`` or explicit ``training_state``,
``checkpoint`` and ``config`` paths.  It may instead contain precomputed
fields (the same flat names emitted by :func:`inspect_source`).  Paths are
resolved relative to the manifest.  A ``run`` may name a config/run directory
or provide a ``protocol`` mapping.  Artifact files are trusted experiment
outputs: loading them uses pickle/torch deserialization.

Exit status 0 means every F/G candidate is reusable, 2 means at least one
candidate must be relaunched, and 3 means the manifest itself is malformed.
The report is written for both status 0 and status 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


SEEDS = ("3", "4", "5")
ARMS = ("F", "G")
EXPECTED_REFERENCE = {
    "q": 256,
    "schedule": "global_sigmoid",
    "gap_scale": 1.10,
    "start_kimg": 256,
    "endpoints_kimg": [512, 768, 1024],
}
ARM_OVERRIDES = {
    "F": {"schedule": "sigmoid", "gap_scale": 1.0},
    "G": {"schedule": "global_sigmoid", "gap_scale": 1.10},
}
RNG_ALIASES = {
    "python": ("python", "python_rng_state", "random_state"),
    "numpy": ("numpy", "numpy_rng_state", "np_rng_state"),
    "torch_cpu": ("torch_cpu", "torch_rng_state", "cpu_rng_state"),
    "torch_cuda": ("torch_cuda", "cuda_rng_state_all", "cuda_rng_state"),
}
SAMPLER_ALIASES = (
    "sampler_state",
    "data_sampler_state",
    "dataloader_state",
    "data_loader_state",
    "infinite_sampler_state",
    "sampler_cursor",
)
SOURCE_FIELDS = (
    "full_state_sha256",
    "checkpoint_sha256",
    "model_sha256",
    "ema_sha256",
    "optimizer_sha256",
    "optimizer_step",
    "gradscaler.present",
    "gradscaler.sha256",
    "rng.python.present",
    "rng.python.sha256",
    "rng.numpy.present",
    "rng.numpy.sha256",
    "rng.torch_cpu.present",
    "rng.torch_cpu.sha256",
    "rng.torch_cuda.present",
    "rng.torch_cuda.sha256",
    "sampler.present",
    "sampler.sha256",
    "config_sha256",
    "config_canonical_sha256",
    "start_kimg",
)
RUN_COMMON_FIELDS = (
    "q",
    "batch.batch_size",
    "batch.batch_gpu",
    "augmentation",
    "precision.use_fp16",
    "precision.enable_amp",
    "precision.enable_tf32",
    "precision.loss_scaling",
    "checkpoint_cadence.kimg_per_tick",
    "checkpoint_cadence.snapshot_ticks",
    "checkpoint_cadence.state_dump_ticks",
    "checkpoint_cadence.ckpt_ticks",
    "checkpoint_cadence.sample_ticks",
    "checkpoint_cadence.eval_ticks",
    "data.byte_sha256",
    "data.semantic_sha256",
    "execution_core_sha256",
    "start_kimg",
    "endpoints_kimg",
)
MISSING = object()


class ManifestError(ValueError):
    """The manifest structure cannot be audited."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_value(digest: "hashlib._Hash", value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: repr(item)):
            _hash_value(digest, key)
            _hash_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("ascii") + b"\0")
        for item in value:
            _hash_value(digest, item)
    elif isinstance(value, (bytes, bytearray)):
        digest.update(b"bytes\0" + bytes(value))
    elif (
        hasattr(value, "dtype")
        and hasattr(value, "shape")
        and hasattr(value, "tobytes")
    ):
        digest.update(b"array\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.tobytes())
    elif value is None:
        digest.update(b"none\0")
    else:
        digest.update((type(value).__qualname__ + ":" + repr(value)).encode("utf-8"))


def state_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path(value: Any, base: Path) -> Path | None:
    if not isinstance(value, (str, Path)) or not str(value):
        return None
    result = Path(value).expanduser()
    return result if result.is_absolute() else base / result


def _nested(mapping: Mapping[str, Any], dotted: str, default: Any = MISSING) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def _put(result: dict[str, Any], dotted: str, value: Any) -> None:
    cursor = result
    parts = dotted.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _declared(spec: Mapping[str, Any], dotted: str) -> Any:
    value = _nested(spec, dotted)
    if value is not MISSING:
        return value
    hashes = spec.get("hashes", {})
    if isinstance(hashes, Mapping):
        return hashes.get(dotted, MISSING)
    return MISSING


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("JSON root is not an object")
    return value


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before the weights_only keyword.
        return torch.load(path, map_location="cpu")


def _state_dict(value: Any) -> Any:
    return value.state_dict() if hasattr(value, "state_dict") else value


def _optimizer_step(state: Any) -> Any:
    if not isinstance(state, Mapping) or not isinstance(state.get("state"), Mapping):
        return None
    steps = []
    for slot in state["state"].values():
        if isinstance(slot, Mapping) and "step" in slot:
            step = slot["step"]
            if isinstance(step, torch.Tensor):
                step = (
                    step.detach().cpu().item() if step.numel() == 1 else step.tolist()
                )
            steps.append(step)
    if not steps:
        return None
    if all(step == steps[0] for step in steps):
        return steps[0]
    return {
        "count": len(steps),
        "min": min(steps),
        "max": max(steps),
        "unique": sorted(set(steps)),
    }


def _first_present(mapping: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        if alias in mapping:
            return mapping[alias]
    return MISSING


def _read_commit(spec: Mapping[str, Any], run_dir: Path | None, base: Path) -> Any:
    direct = spec.get("execution_commit", spec.get("commit", MISSING))
    if direct is not MISSING:
        return direct
    commit_path = _path(spec.get("commit_file"), base)
    if commit_path is None and run_dir is not None:
        commit_path = run_dir / "commit_sha.txt"
    if commit_path is None or not commit_path.is_file():
        return None
    text = commit_path.read_text(encoding="utf-8").strip()
    match = re.search(r"\b[0-9a-fA-F]{7,64}\b", text)
    return match.group(0) if match else text or None


def _read_core_hash(spec: Mapping[str, Any], run_dir: Path | None, base: Path) -> Any:
    for key in ("execution_core_sha256", "core_sha256", "source_sha256"):
        if key in spec:
            return spec[key]
    core_files = spec.get("core_files")
    if isinstance(core_files, Mapping):
        table = {}
        for name, item in core_files.items():
            path = _path(item, base)
            if path is None or not path.is_file():
                return None
            table[str(name)] = sha256_file(path)
        return canonical_sha256(table)
    meta_path = _path(spec.get("experiment_meta"), base)
    if meta_path is None and run_dir is not None:
        meta_path = run_dir / "experiment_meta.env"
    if meta_path is not None and meta_path.is_file():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("source_sha256="):
                return line.split("=", 1)[1].strip()
    return None


def inspect_source(spec: Mapping[str, Any], base: Path) -> dict[str, Any]:
    """Return flattened load-bearing observations for one 256 kimg source."""
    if not isinstance(spec, Mapping):
        return {
            "values": {},
            "missing": list(SOURCE_FIELDS),
            "errors": ["source is not an object"],
        }
    values: dict[str, Any] = {}
    errors: list[str] = []
    run_dir = _path(spec.get("run_dir"), base)
    state_path = _path(
        spec.get("training_state", spec.get("training_state_path")), base
    )
    checkpoint_path = _path(spec.get("checkpoint", spec.get("snapshot")), base)
    config_path = _path(spec.get("config", spec.get("training_options")), base)
    if run_dir is not None:
        state_path = state_path or run_dir / "training-state-latest.pt"
        checkpoint_path = checkpoint_path or run_dir / "network-snapshot-latest.pkl"
        config_path = config_path or run_dir / "training_options.json"

    # Precomputed values make receipt-only audits possible; real artifacts win.
    for field in SOURCE_FIELDS:
        declared = _declared(spec, field)
        if declared is not MISSING:
            _put(values, field, declared)

    state = None
    if state_path is not None:
        if not state_path.is_file():
            errors.append(f"missing training state: {state_path}")
        else:
            try:
                state = _torch_load(state_path)
                if not isinstance(state, Mapping):
                    raise ValueError("training state root is not a mapping")
                values["full_state_sha256"] = sha256_file(state_path)
                net = state.get("net", state.get("model", MISSING))
                if net is not MISSING:
                    values["model_sha256"] = state_sha256(_state_dict(net))
                optimizer = state.get(
                    "optimizer_state", state.get("optimizer", MISSING)
                )
                if optimizer is not MISSING:
                    optimizer = _state_dict(optimizer)
                    values["optimizer_sha256"] = state_sha256(optimizer)
                    values["optimizer_step"] = _optimizer_step(optimizer)
                scaler = _first_present(
                    state, ("gradscaler_state", "scaler_state", "grad_scaler_state")
                )
                values["gradscaler"] = {
                    "present": scaler is not MISSING and scaler is not None,
                    "sha256": None
                    if scaler is MISSING or scaler is None
                    else state_sha256(scaler),
                }
                rng_container = state.get("rng_state", {})
                if not isinstance(rng_container, Mapping):
                    rng_container = {}
                rng_values = {}
                for kind, aliases in RNG_ALIASES.items():
                    rng = _first_present(rng_container, aliases)
                    if rng is MISSING:
                        rng = _first_present(state, aliases)
                    rng_values[kind] = {
                        "present": rng is not MISSING and rng is not None,
                        "sha256": None
                        if rng is MISSING or rng is None
                        else state_sha256(rng),
                    }
                values["rng"] = rng_values
                sampler = _first_present(state, SAMPLER_ALIASES)
                values["sampler"] = {
                    "present": sampler is not MISSING and sampler is not None,
                    "sha256": None
                    if sampler is MISSING or sampler is None
                    else state_sha256(sampler),
                }
                cur_nimg = state.get("cur_nimg")
                if isinstance(cur_nimg, (int, float)) and math.isfinite(
                    float(cur_nimg)
                ):
                    values["start_kimg"] = cur_nimg / 1000
            except Exception as exc:
                errors.append(f"cannot inspect training state {state_path}: {exc}")

    if checkpoint_path is not None:
        if not checkpoint_path.is_file():
            errors.append(f"missing checkpoint: {checkpoint_path}")
        else:
            try:
                with checkpoint_path.open("rb") as handle:
                    checkpoint = pickle.load(handle)
                if not isinstance(checkpoint, Mapping):
                    raise ValueError("checkpoint root is not a mapping")
                values["checkpoint_sha256"] = sha256_file(checkpoint_path)
                ema = checkpoint.get("ema", checkpoint.get("ema_state", MISSING))
                if ema is not MISSING:
                    values["ema_sha256"] = state_sha256(_state_dict(ema))
            except Exception as exc:
                errors.append(f"cannot inspect checkpoint {checkpoint_path}: {exc}")

    if config_path is not None:
        if not config_path.is_file():
            errors.append(f"missing config: {config_path}")
        else:
            try:
                config = _load_json(config_path)
                values["config_sha256"] = sha256_file(config_path)
                values["config_canonical_sha256"] = canonical_sha256(config)
            except Exception as exc:
                errors.append(f"cannot inspect config {config_path}: {exc}")

    missing = [
        field
        for field in SOURCE_FIELDS
        if _nested(values, field) is MISSING or _nested(values, field) is None
    ]
    return {
        "values": values,
        "missing": missing,
        "errors": errors,
        "paths": {
            "state": str(state_path) if state_path else None,
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "config": str(config_path) if config_path else None,
        },
    }


def _config_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    loss = config.get("loss_kwargs", {})
    network = config.get("network_kwargs", {})
    dataset = config.get("dataset_kwargs", {})
    return {
        "q": loss.get("q") if isinstance(loss, Mapping) else None,
        "schedule": loss.get("adj") if isinstance(loss, Mapping) else None,
        "gap_scale": loss.get("global_gap_scale")
        if isinstance(loss, Mapping)
        else None,
        "batch": {
            "batch_size": config.get("batch_size"),
            "batch_gpu": config.get("batch_gpu"),
        },
        "augmentation": {
            "augment_kwargs": config.get("augment_kwargs"),
            "dataset_xflip": dataset.get("xflip")
            if isinstance(dataset, Mapping)
            else None,
        },
        "precision": {
            "use_fp16": network.get("use_fp16")
            if isinstance(network, Mapping)
            else None,
            "enable_amp": config.get("enable_amp"),
            "enable_tf32": config.get("enable_tf32"),
            "loss_scaling": config.get("loss_scaling"),
        },
        "checkpoint_cadence": {
            key: config.get(key)
            for key in (
                "kimg_per_tick",
                "snapshot_ticks",
                "state_dump_ticks",
                "ckpt_ticks",
                "sample_ticks",
                "eval_ticks",
            )
        },
    }


def _normal_endpoints(value: Any) -> Any:
    if isinstance(value, Mapping):
        value = list(value.keys())
    if not isinstance(value, (list, tuple)):
        return value
    result = []
    for item in value:
        if isinstance(item, Mapping):
            item = item.get("kimg", item.get("endpoint_kimg"))
        if isinstance(item, float) and item.is_integer():
            item = int(item)
        result.append(item)
    return (
        sorted(result)
        if all(isinstance(item, (int, float)) for item in result)
        else result
    )


def inspect_run(spec: Mapping[str, Any], base: Path) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        return {
            "values": {},
            "missing": list(RUN_COMMON_FIELDS),
            "errors": ["run is not an object"],
        }
    values: dict[str, Any] = {}
    errors: list[str] = []
    run_dir = _path(spec.get("run_dir"), base)
    config_path = _path(spec.get("config", spec.get("training_options")), base)
    if config_path is None and run_dir is not None:
        config_path = run_dir / "training_options.json"
    config = None
    if config_path is not None:
        if not config_path.is_file():
            errors.append(f"missing run config: {config_path}")
        else:
            try:
                config = _load_json(config_path)
                values.update(_config_protocol(config))
                values["config_sha256"] = sha256_file(config_path)
                values["config_canonical_sha256"] = canonical_sha256(config)
            except Exception as exc:
                errors.append(f"cannot inspect run config {config_path}: {exc}")
    protocol = spec.get("protocol", {})
    if not isinstance(protocol, Mapping):
        errors.append("run.protocol is not an object")
        protocol = {}
    aliases = {
        "gap_scale": ("gap_scale", "global_gap_scale"),
        "schedule": ("schedule", "adj"),
        "q": ("q",),
        "batch": ("batch",),
        "augmentation": ("augmentation",),
        "precision": ("precision",),
        "checkpoint_cadence": ("checkpoint_cadence", "cadence"),
    }
    for field, names in aliases.items():
        for container in (protocol, spec):
            found = _first_present(container, names)
            if found is not MISSING:
                values[field] = found
                break
    data = spec.get("data", protocol.get("data", {}))
    if isinstance(data, Mapping):
        byte_hash = data.get("byte_sha256", data.get("sha256", MISSING))
        semantic_hash = data.get("semantic_sha256", MISSING)
        if byte_hash is not MISSING:
            _put(values, "data.byte_sha256", byte_hash)
        if semantic_hash is not MISSING:
            _put(values, "data.semantic_sha256", semantic_hash)
    data_path = _path(spec.get("data_path"), base)
    if data_path is not None and data_path.is_file():
        _put(values, "data.byte_sha256", sha256_file(data_path))
    values["execution_commit"] = _read_commit(spec, run_dir, base)
    values["execution_core_sha256"] = _read_core_hash(spec, run_dir, base)
    values["start_kimg"] = spec.get("start_kimg", protocol.get("start_kimg"))
    endpoints = spec.get(
        "endpoints_kimg", spec.get("endpoints", protocol.get("endpoints_kimg"))
    )
    values["endpoints_kimg"] = _normal_endpoints(endpoints)
    declared_config = spec.get("config_sha256")
    if declared_config is not None and "config_sha256" not in values:
        values["config_sha256"] = declared_config
    protocol_hash_fields = (
        "q",
        "schedule",
        "gap_scale",
        "batch",
        "augmentation",
        "precision",
        "checkpoint_cadence",
        "data",
        "start_kimg",
        "endpoints_kimg",
        "execution_core_sha256",
    )
    protocol_for_hash = {field: values.get(field) for field in protocol_hash_fields}
    values["protocol_config_sha256"] = canonical_sha256(protocol_for_hash)
    missing = [
        field
        for field in RUN_COMMON_FIELDS
        if _nested(values, field) is MISSING or _nested(values, field) is None
    ]
    if values.get("execution_commit") is None:
        missing.append("execution_commit")
    return {
        "values": values,
        "missing": sorted(set(missing)),
        "errors": errors,
        "paths": {"config": str(config_path) if config_path else None},
    }


def _json_value(value: Any) -> Any:
    return None if value is MISSING else value


def _row(
    rows: list[dict[str, Any]],
    seed: str,
    arm: str,
    scope: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    blocking: bool = True,
    reason: str | None = None,
    equivalent: bool = False,
) -> dict[str, Any]:
    missing = (
        expected is MISSING
        or observed is MISSING
        or expected is None
        or observed is None
    )
    if missing:
        status = "missing"
        reason = reason or "required_field_missing_or_not_serialized"
    elif expected == observed:
        status = "match"
        reason = reason or "exact_match"
    elif equivalent:
        status = "equivalent"
        reason = reason or "different_commit_same_execution_core"
    else:
        status = "mismatch"
        reason = reason or "load_bearing_field_mismatch"
    row = {
        "seed": int(seed),
        "arm": arm,
        "scope": scope,
        "field": field,
        "expected": _json_value(expected),
        "observed": _json_value(observed),
        "status": status,
        "blocking": bool(blocking and status in {"missing", "mismatch"}),
        "reason": reason,
    }
    rows.append(row)
    return row


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{label} must be an object")
    return value


def _validate_shape(
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    reference = _require_mapping(manifest.get("reference"), "reference")
    controls = _require_mapping(manifest.get("controls"), "controls")
    for seed in SEEDS:
        if seed not in reference:
            raise ManifestError(f"reference is missing seed {seed}")
        _require_mapping(reference[seed], f"reference.{seed}")
    for arm in ARMS:
        arm_map = _require_mapping(controls.get(arm), f"controls.{arm}")
        for seed in SEEDS:
            if seed not in arm_map:
                raise ManifestError(f"controls.{arm} is missing seed {seed}")
            _require_mapping(arm_map[seed], f"controls.{arm}.{seed}")
    return reference, controls


def build_report(
    manifest: Mapping[str, Any], manifest_dir: Path = Path(".")
) -> dict[str, Any]:
    """Audit all three seeds and return a JSON-serializable compatibility table."""
    reference, controls = _validate_shape(manifest)
    rows: list[dict[str, Any]] = []
    missing_fields: list[dict[str, Any]] = []
    artifact_errors: list[dict[str, Any]] = []
    seed_reports: dict[str, Any] = {}

    for seed in SEEDS:
        ref_item = reference[seed]
        reference_row_start = len(rows)
        ref_source = inspect_source(
            _require_mapping(ref_item.get("source"), f"reference.{seed}.source"),
            manifest_dir,
        )
        ref_run = inspect_run(
            _require_mapping(ref_item.get("run"), f"reference.{seed}.run"), manifest_dir
        )
        for scope, inspected in (("source", ref_source), ("run", ref_run)):
            for field in inspected["missing"]:
                missing_fields.append(
                    {
                        "seed": int(seed),
                        "arm": "T",
                        "scope": scope,
                        "field": field,
                        "reason": "missing_or_not_serialized",
                    }
                )
            for error in inspected["errors"]:
                artifact_errors.append(
                    {"seed": int(seed), "arm": "T", "scope": scope, "error": error}
                )
        # The proposed transported arm must itself be the intended protocol.
        for field, expected in EXPECTED_REFERENCE.items():
            observed = _nested(ref_run["values"], field)
            _row(rows, seed, "T", "reference", field, expected, observed)
        reference_invalid = (
            any(row["blocking"] for row in rows[reference_row_start:])
            or bool(ref_source["errors"])
            or bool(ref_run["errors"])
        )

        controls_for_seed: dict[str, Any] = {}
        inspected_controls: dict[str, Any] = {}
        for arm in ARMS:
            item = controls[arm][seed]
            source = inspect_source(
                _require_mapping(item.get("source"), f"controls.{arm}.{seed}.source"),
                manifest_dir,
            )
            run = inspect_run(
                _require_mapping(item.get("run"), f"controls.{arm}.{seed}.run"),
                manifest_dir,
            )
            inspected_controls[arm] = (source, run)
            first_row = len(rows)
            if reference_invalid:
                _row(
                    rows,
                    seed,
                    arm,
                    "reference",
                    "valid",
                    True,
                    False,
                    reason="reference_protocol_or_artifact_invalid",
                )
            for scope, inspected in (("source", source), ("run", run)):
                for field in inspected["missing"]:
                    missing_fields.append(
                        {
                            "seed": int(seed),
                            "arm": arm,
                            "scope": scope,
                            "field": field,
                            "reason": "missing_or_not_serialized",
                        }
                    )
                for error in inspected["errors"]:
                    artifact_errors.append(
                        {"seed": int(seed), "arm": arm, "scope": scope, "error": error}
                    )
            for field in SOURCE_FIELDS:
                _row(
                    rows,
                    seed,
                    arm,
                    "source",
                    field,
                    _nested(ref_source["values"], field),
                    _nested(source["values"], field),
                )

            expected_run = dict(ref_run["values"])
            expected_run.update(ARM_OVERRIDES[arm])
            for field in RUN_COMMON_FIELDS + ("schedule", "gap_scale"):
                _row(
                    rows,
                    seed,
                    arm,
                    "run",
                    field,
                    _nested(expected_run, field),
                    _nested(run["values"], field),
                )
            core_equal = _nested(
                expected_run, "execution_core_sha256"
            ) is not MISSING and _nested(
                expected_run, "execution_core_sha256"
            ) == _nested(run["values"], "execution_core_sha256")
            _row(
                rows,
                seed,
                arm,
                "run",
                "execution_commit",
                _nested(expected_run, "execution_commit"),
                _nested(run["values"], "execution_commit"),
                equivalent=core_equal,
            )
            expected_protocol = {
                key: expected_run.get(key)
                for key in (
                    "q",
                    "schedule",
                    "gap_scale",
                    "batch",
                    "augmentation",
                    "precision",
                    "checkpoint_cadence",
                    "data",
                    "start_kimg",
                    "endpoints_kimg",
                    "execution_core_sha256",
                )
            }
            expected_protocol_hash = canonical_sha256(expected_protocol)
            _row(
                rows,
                seed,
                arm,
                "run",
                "protocol_config_sha256",
                expected_protocol_hash,
                run["values"].get("protocol_config_sha256"),
            )
            # Raw config hashes are retained as evidence; normalized protocol hash above
            # is the comparable value because F intentionally differs in schedule/gap.
            rows.append(
                {
                    "seed": int(seed),
                    "arm": arm,
                    "scope": "run",
                    "field": "config_sha256",
                    "expected": ref_run["values"].get("config_sha256"),
                    "observed": run["values"].get("config_sha256"),
                    "status": "reported",
                    "blocking": False,
                    "reason": "raw_config_hash_evidence",
                }
            )

            arm_rows = rows[first_row:]
            blockers = [row for row in arm_rows if row["blocking"]]
            # Artifact errors are also fail-closed even if a precomputed field happened
            # to be present in the same receipt.
            arm_errors = [
                entry
                for entry in artifact_errors
                if entry["seed"] == int(seed) and entry["arm"] == arm
            ]
            reusable = not blockers and not arm_errors
            controls_for_seed[arm] = {
                "reusable": reusable,
                "reuse_decision": "reusable" if reusable else "fresh_required",
                "action": (
                    f"reuse_existing_{arm}_control"
                    if reusable
                    else f"launch_fresh_paired_{arm}_control"
                ),
                "blocking_fields": sorted(
                    {row["scope"] + "." + row["field"] for row in blockers}
                ),
                "reasons": sorted(
                    {row["reason"] for row in blockers}
                    | ({"artifact_inspection_error"} if arm_errors else set())
                ),
            }

        f_source = inspected_controls["F"][0]["values"]
        g_source = inspected_controls["G"][0]["values"]
        f_hash = _nested(f_source, "full_state_sha256")
        g_hash = _nested(g_source, "full_state_sha256")
        legacy_same = (
            f_hash is not MISSING and g_hash is not MISSING and f_hash == g_hash
        )
        _row(
            rows,
            seed,
            "F/G",
            "source",
            "legacy_shared_256k_source",
            f_hash,
            g_hash,
            blocking=False,
            reason=(
                "legacy_F_G_controls_share_256k_source"
                if legacy_same
                else "legacy_F_G_controls_use_different_256k_source"
            ),
        )
        seed_reports[seed] = {
            "legacy_F_G_same_256k_source": legacy_same,
            "controls": controls_for_seed,
        }

    all_reusable = all(
        seed_reports[seed]["controls"][arm]["reusable"]
        for seed in SEEDS
        for arm in ARMS
    )
    blockers = [row for row in rows if row["blocking"]]
    return {
        "schema_version": 1,
        "status": "PASS" if all_reusable and not artifact_errors else "FAIL",
        "reusable_controls": all_reusable and not artifact_errors,
        "rows": rows,
        "seeds": seed_reports,
        "blockers": blockers,
        "missing_nonserialized_fields": missing_fields,
        "artifact_errors": artifact_errors,
    }


def audit_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = _load_json(path)
    except Exception as exc:
        raise ManifestError(f"cannot load manifest {path}: {exc}") from exc
    return build_report(manifest, path.parent.resolve())


def _write_report(report: Mapping[str, Any], output: str) -> None:
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output == "-":
        sys.stdout.write(text)
        return
    path = Path(output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        "--report",
        dest="output",
        required=True,
        help="JSON report path, or - for stdout",
    )
    args = parser.parse_args(argv)
    try:
        report = audit_manifest(args.manifest.expanduser().resolve())
    except ManifestError as exc:
        _write_report(
            {"schema_version": 1, "status": "ERROR", "error": str(exc)}, args.output
        )
        return 3
    _write_report(report, args.output)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
