#!/usr/bin/env python3
"""Fail-closed validation for one q256 continuation segment.

The inputs are trusted experiment pickles produced by this repository.  This
validator deliberately uses the same unrestricted PyTorch/pickle loaders as
the training loop, but only after constraining both artifacts to ``run_dir``.
Every scientific or serialization mismatch produces a NO_GO receipt and exit
status 2.  A successful immutable receipt contains ``"verdict": "GO"``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402


SCHEMA = "ect.q256.continuation-validation/v1"
CANONICAL_DATA_PATH = "/data/raw/ECT/datasets/cifar10-32x32-canonical-08c9ed1b2b1c.zip"
CANONICAL_DATA_SHA256 = (
    "08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372"
)
CANONICAL_DATA_SIZE_BYTES = 166000134
DATA_IDENTITY_DECLARATION = (
    REPO_ROOT / "analysis" / "q256_g110_moment_transport" / "data_identity.json"
)

ARM_IDENTITIES = {
    "F": ("sigmoid", 1.0),
    "G": ("global_sigmoid", 1.10),
    "T": ("global_sigmoid", 1.10),
    # Engineering-only 32-attempt replays are all launched with the G command.
    "noop-direct": ("global_sigmoid", 1.10),
    "noop-rewrite": ("global_sigmoid", 1.10),
    "transport-repeat1": ("global_sigmoid", 1.10),
    "transport-repeat2": ("global_sigmoid", 1.10),
}

TRAIN_SUMMARY_FIELDS = (
    "attempted_iteration",
    "successful_optimizer_steps",
    "processed_nimg",
    "processed_kimg",
    "loss",
    "grad_scale",
    "step_skipped",
    "schedule",
    "stage",
    "next_loop_cur_tick",
    "loss_ema",
    "loss_reference",
    "correction",
    "signal_updates",
    "adaptive_active",
    "r_over_t_mean",
    "gap_mean",
    "gap_over_sigmoid_gap_mean",
    "lower_gap_clip_rate",
    "upper_gap_clip_rate",
    "elapsed_sec",
    "peak_vram_gb",
)

STATE_FIELDS = frozenset(
    {
        "net",
        "optimizer_state",
        "attempted_iteration",
        "successful_optimizer_steps",
        "cur_nimg",
        "cur_tick",
        "tick_start_nimg",
        "elapsed_sec",
        "loss_fn_state",
        "gradscaler_state",
    }
)
SNAPSHOT_FIELDS = frozenset({"ema", "loss_fn", "augment_pipe", "dataset_kwargs"})
OPTION_FIELDS = frozenset(
    {
        "dataset_kwargs",
        "data_loader_kwargs",
        "network_kwargs",
        "loss_kwargs",
        "optimizer_kwargs",
        "total_kimg",
        "ema_halflife_kimg",
        "ema_beta",
        "batch_size",
        "batch_gpu",
        "loss_scaling",
        "cudnn_benchmark",
        "enable_tf32",
        "enable_amp",
        "kimg_per_tick",
        "snapshot_ticks",
        "state_dump_ticks",
        "ckpt_ticks",
        "double_ticks",
        "adaptive_update_kimg",
        "mid_t",
        "metrics",
        "sample_ticks",
        "eval_ticks",
        "seed",
        "resume_pkl",
        "resume_tick",
        "resume_state_dump",
        "run_dir",
    }
)


class ValidationError(RuntimeError):
    """A corrupt artifact or a fail-closed protocol mismatch."""


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Read NumPy 2 pickles on a runtime that still exposes NumPy 1 paths."""

    def find_class(self, module: str, name: str) -> Any:
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


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be a mapping")
    return value


def _exact_keys(value: Mapping[Any, Any], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(str(item) for item in actual - expected)
        raise ValidationError(
            f"{label} key mismatch: missing={missing}, unknown={unknown}"
        )


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValidationError(f"{label} must be >= {minimum}")
    return result


def _csv_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} must be an integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < minimum:
        raise ValidationError(f"{label} must be an integer >= {minimum}")
    return int(numeric)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValidationError(f"{label} must be finite")
    return result


def _equal_number(value: Any, expected: float, label: str) -> float:
    result = _finite(value, label)
    if result != expected:
        raise ValidationError(f"{label}={result!r}, expected {expected!r}")
    return result


def _bool(value: Any, expected: bool, label: str) -> None:
    if value is not expected:
        raise ValidationError(f"{label}={value!r}, expected {expected!r}")


def _attribute(value: Any, name: str, label: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise ValidationError(f"{label} is missing {name!r}")
        return value[name]
    if not hasattr(value, name):
        raise ValidationError(f"{label} is missing attribute {name!r}")
    return getattr(value, name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(path: Path) -> tuple[int, int, int, int]:
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _regular_file(path: Path, label: str, *, nonempty: bool = True) -> Path:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ValidationError(f"cannot stat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValidationError(f"{label} is not a non-symlink regular file: {path}")
    if nonempty and info.st_size <= 0:
        raise ValidationError(f"{label} is empty: {path}")
    return path


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot load {label} {path}: {exc}") from exc
    return _mapping(payload, label)


def _load_state(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            pickle_module=_NumpyCompatPickleModule,
        )
    except Exception as exc:
        raise ValidationError(f"cannot load training state {path}: {exc}") from exc
    return _mapping(payload, "training state")


def _load_snapshot(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("rb") as handle:
            payload = _NumpyCompatUnpickler(handle).load()
    except Exception as exc:
        raise ValidationError(f"cannot load snapshot {path}: {exc}") from exc
    return _mapping(payload, "network snapshot")


def _tensor_is_finite(tensor: torch.Tensor) -> bool:
    if not (tensor.is_floating_point() or tensor.is_complex()):
        return True
    if tensor.layout == torch.strided:
        return bool(torch.isfinite(tensor).all().item())
    if tensor.layout in {
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }:
        return bool(torch.isfinite(tensor.values()).all().item())
    raise ValidationError(f"unsupported tensor layout {tensor.layout}")


def inspect_finite_tensors(value: Any, label: str) -> dict[str, int]:
    """Walk a trusted object graph and prove every tensor/ndarray is finite."""

    stack: list[tuple[str, Any]] = [(label, value)]
    seen: set[int] = set()
    tensors = 0
    tensor_numel = 0
    arrays = 0
    while stack:
        path, item = stack.pop()
        if isinstance(item, torch.Tensor):
            tensors += 1
            tensor_numel += item.numel()
            try:
                finite = _tensor_is_finite(item)
            except (RuntimeError, TypeError) as exc:
                raise ValidationError(
                    f"cannot inspect tensor at {path}: {exc}"
                ) from exc
            if not finite:
                raise ValidationError(f"non-finite tensor at {path}")
            continue
        if isinstance(item, np.ndarray):
            arrays += 1
            if item.dtype.hasobject:
                stack.extend(
                    (f"{path}[{index}]", nested)
                    for index, nested in enumerate(item.flat)
                )
            elif np.issubdtype(item.dtype, np.inexact) and not bool(
                np.isfinite(item).all()
            ):
                raise ValidationError(f"non-finite NumPy array at {path}")
            continue
        if isinstance(item, np.generic):
            if np.issubdtype(item.dtype, np.inexact) and not bool(np.isfinite(item)):
                raise ValidationError(f"non-finite NumPy scalar at {path}")
            continue
        if item is None or isinstance(item, (str, bytes, int, float, bool, type)):
            continue
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(item, Mapping):
            stack.extend((f"{path}.{key}", nested) for key, nested in item.items())
        elif isinstance(item, (list, tuple, set, frozenset)):
            stack.extend(
                (f"{path}[{index}]", nested) for index, nested in enumerate(item)
            )
        elif hasattr(item, "__dict__"):
            stack.append((f"{path}.__dict__", vars(item)))
    return {"tensors": tensors, "tensor_numel": tensor_numel, "numpy_arrays": arrays}


def _expected_identity(arm: str) -> tuple[str, float]:
    try:
        return ARM_IDENTITIES[arm]
    except KeyError as exc:
        raise ValidationError(f"unsupported expected arm {arm!r}") from exc


def _validate_identity_declaration() -> dict[str, Any]:
    path = _regular_file(DATA_IDENTITY_DECLARATION, "data identity declaration")
    declaration = _load_json(path, "data identity declaration")
    if declaration.get("schema") != "ect.q256.dataset-identity/v1":
        raise ValidationError("data identity declaration schema mismatch")
    canonical = _mapping(
        declaration.get("canonical_training_archive"),
        "data identity canonical_training_archive",
    )
    expected = {
        "path": CANONICAL_DATA_PATH,
        "sha256": CANONICAL_DATA_SHA256,
        "size_bytes": CANONICAL_DATA_SIZE_BYTES,
    }
    for key, expected_value in expected.items():
        if canonical.get(key) != expected_value:
            raise ValidationError(
                f"canonical data declaration {key}={canonical.get(key)!r}, "
                f"expected {expected_value!r}"
            )
    return {
        "declaration_path": str(path.resolve()),
        "declaration_sha256": sha256_file(path),
        **expected,
    }


def _validate_dataset_kwargs(value: Any, label: str) -> None:
    data = _mapping(value, label)
    expected_keys = {
        "class_name",
        "path",
        "use_labels",
        "xflip",
        "cache",
        "resolution",
        "max_size",
    }
    if set(data) != expected_keys:
        raise ValidationError(
            f"{label} has unsupported keys: {sorted(set(data) ^ expected_keys)}"
        )
    if data["class_name"] != "training.dataset.ImageFolderDataset":
        raise ValidationError(f"{label}.class_name mismatch")
    if data["path"] != CANONICAL_DATA_PATH:
        raise ValidationError(f"{label}.path is not the canonical archive")
    _bool(data["use_labels"], False, f"{label}.use_labels")
    _bool(data["xflip"], False, f"{label}.xflip")
    if not isinstance(data["cache"], bool):
        raise ValidationError(f"{label}.cache must be boolean")
    if _integer(data["resolution"], f"{label}.resolution", minimum=1) != 32:
        raise ValidationError(f"{label}.resolution must equal 32")
    if _integer(data["max_size"], f"{label}.max_size", minimum=1) != 50000:
        raise ValidationError(f"{label}.max_size must equal 50000")


def validate_options(
    options: Mapping[str, Any],
    *,
    run_dir: Path,
    expected_nimg: int,
    expected_seed: int,
    expected_arm: str,
    expected_mapping_override: str | None = None,
    expected_gap_override: float | None = None,
) -> dict[str, Any]:
    _exact_keys(options, OPTION_FIELDS, "training options")
    schedule, gap_scale = _expected_identity(expected_arm)
    if expected_mapping_override is not None and expected_mapping_override != schedule:
        raise ValidationError(
            "--expected-mapping conflicts with the frozen arm identity"
        )
    if expected_gap_override is not None and expected_gap_override != gap_scale:
        raise ValidationError(
            "--expected-gap-scale conflicts with the frozen arm identity"
        )

    if options["run_dir"] != str(run_dir):
        raise ValidationError(
            f"training options run_dir={options['run_dir']!r}, expected {str(run_dir)!r}"
        )
    budget_nimg = (
        _integer(options["total_kimg"], "options.total_kimg", minimum=1) * 1000
    )
    # ct_train stores an integer-kimg stopping threshold.  The 32-attempt
    # smoke target is 260096 images, so its frozen CLI budget is 260 kimg and
    # the final batch legally crosses that threshold by 96 images.
    if not budget_nimg <= expected_nimg <= budget_nimg + 127:
        raise ValidationError(
            "options.total_kimg threshold is inconsistent with expected_nimg"
        )
    if _integer(options["seed"], "options.seed") != expected_seed:
        raise ValidationError("training seed mismatch")
    if _integer(options["batch_size"], "options.batch_size", minimum=1) != 128:
        raise ValidationError("batch_size must equal 128")
    if _integer(options["batch_gpu"], "options.batch_gpu", minimum=1) != 16:
        raise ValidationError("batch_gpu must equal 16")
    _equal_number(options["ema_beta"], 0.9993, "options.ema_beta")
    if options["ema_halflife_kimg"] is not None:
        raise ValidationError("options.ema_halflife_kimg must be null")
    _equal_number(options["loss_scaling"], 1.0, "options.loss_scaling")
    _bool(options["enable_amp"], True, "options.enable_amp")
    _bool(options["enable_tf32"], False, "options.enable_tf32")
    if not isinstance(options["cudnn_benchmark"], bool):
        raise ValidationError("options.cudnn_benchmark must be boolean")

    cadence = {
        "kimg_per_tick": 10.0,
        "snapshot_ticks": None,
        "state_dump_ticks": None,
        "ckpt_ticks": 10,
        "double_ticks": 10000,
        "adaptive_update_kimg": 0.5,
        "sample_ticks": 26,
        "eval_ticks": 50,
    }
    for key, expected in cadence.items():
        value = options[key]
        if expected is None:
            if value is not None:
                raise ValidationError(f"options.{key} must be null")
        elif _finite(value, f"options.{key}") != expected:
            raise ValidationError(f"options.{key}={value!r}, expected {expected!r}")
    if options["metrics"] != []:
        raise ValidationError("formal metrics must be disabled during training")
    if options["mid_t"] != [0.821]:
        raise ValidationError("options.mid_t must equal [0.821]")

    _validate_dataset_kwargs(options["dataset_kwargs"], "options.dataset_kwargs")
    loader = _mapping(options["data_loader_kwargs"], "options.data_loader_kwargs")
    if set(loader) != {"pin_memory", "num_workers", "prefetch_factor"}:
        raise ValidationError("options.data_loader_kwargs key mismatch")
    _bool(loader["pin_memory"], True, "options.data_loader_kwargs.pin_memory")
    _integer(loader["num_workers"], "options.data_loader_kwargs.num_workers", minimum=1)
    if (
        _integer(
            loader["prefetch_factor"],
            "options.data_loader_kwargs.prefetch_factor",
            minimum=1,
        )
        != 2
    ):
        raise ValidationError("options.data_loader_kwargs.prefetch_factor must equal 2")

    network = _mapping(options["network_kwargs"], "options.network_kwargs")
    required_network = {
        "class_name": "training.networks.ECMPrecond",
        "model_type": "SongUNet",
        "embedding_type": "positional",
        "encoder_type": "standard",
        "decoder_type": "standard",
        "channel_mult_noise": 1,
        "resample_filter": [1, 1],
        "model_channels": 128,
        "channel_mult": [2, 2, 2],
        "dropout": 0.2,
        "use_fp16": True,
    }
    if set(network) != set(required_network):
        raise ValidationError("options.network_kwargs key mismatch")
    for key, expected in required_network.items():
        if network[key] != expected:
            raise ValidationError(f"options.network_kwargs.{key} mismatch")

    optimizer = _mapping(options["optimizer_kwargs"], "options.optimizer_kwargs")
    if set(optimizer) != {"class_name", "lr", "betas", "eps"}:
        raise ValidationError("options.optimizer_kwargs key mismatch")
    if optimizer["class_name"] != "torch.optim.RAdam":
        raise ValidationError("optimizer must be torch.optim.RAdam")
    _equal_number(optimizer["lr"], 1e-4, "options.optimizer_kwargs.lr")
    if not isinstance(optimizer["betas"], Sequence) or isinstance(
        optimizer["betas"], (str, bytes)
    ):
        raise ValidationError("options.optimizer_kwargs.betas must be a pair")
    if len(optimizer["betas"]) != 2:
        raise ValidationError("options.optimizer_kwargs.betas must be a pair")
    _equal_number(optimizer["betas"][0], 0.9, "options.optimizer_kwargs.betas[0]")
    _equal_number(optimizer["betas"][1], 0.999, "options.optimizer_kwargs.betas[1]")
    _equal_number(optimizer["eps"], 1e-8, "options.optimizer_kwargs.eps")

    loss = _mapping(options["loss_kwargs"], "options.loss_kwargs")
    required_loss_keys = {
        "class_name",
        "P_mean",
        "P_std",
        "q",
        "c",
        "k",
        "b",
        "adj",
        "adaptive_loss_ema_beta",
        "adaptive_warmup_updates",
        "adaptive_max_adjust",
        "adaptive_min_gap",
        "local_tbin_num_bins",
        "local_tbin_short_beta",
        "local_tbin_long_beta",
        "local_tbin_warmup_updates",
        "local_tbin_gain",
        "local_tbin_min_scale",
        "local_tbin_max_scale",
        "local_tbin_deadband",
        "local_tbin_min_gap",
        "global_gap_scale",
    }
    if set(loss) != required_loss_keys:
        raise ValidationError("options.loss_kwargs key mismatch")
    if loss["class_name"] != "training.loss.ECMLoss":
        raise ValidationError("loss must be training.loss.ECMLoss")
    _equal_number(loss["q"], 256.0, "options.loss_kwargs.q")
    _equal_number(loss["k"], 8.0, "options.loss_kwargs.k")
    _equal_number(loss["b"], 1.0, "options.loss_kwargs.b")
    _equal_number(loss["c"], 0.0, "options.loss_kwargs.c")
    if loss["adj"] != schedule:
        raise ValidationError(
            f"options schedule={loss['adj']!r}, expected {schedule!r}"
        )
    _equal_number(
        loss["global_gap_scale"], gap_scale, "options.loss_kwargs.global_gap_scale"
    )

    for path_key in ("resume_state_dump", "resume_pkl"):
        value = options[path_key]
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValidationError(f"options.{path_key} must be an absolute path")
    resume_state = Path(options["resume_state_dump"])
    resume_snapshot = Path(options["resume_pkl"])
    if resume_state.name not in {
        "training-state-latest.pt"
    } and not resume_state.name.startswith("training-state-"):
        raise ValidationError(
            "options.resume_state_dump has an invalid checkpoint name"
        )
    suffix = resume_state.name.removeprefix("training-state-").removesuffix(".pt")
    if resume_snapshot != resume_state.with_name(f"network-snapshot-{suffix}.pkl"):
        raise ValidationError("options resume state/snapshot pairing mismatch")
    _integer(options["resume_tick"], "options.resume_tick")

    return {"schedule": schedule, "global_gap_scale": gap_scale}


def _module_state(value: Any, label: str) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, torch.nn.Module):
        raise ValidationError(f"{label} must be a torch.nn.Module")
    try:
        state = value.state_dict()
    except Exception as exc:
        raise ValidationError(f"cannot read {label}.state_dict(): {exc}") from exc
    if not isinstance(state, Mapping) or not state:
        raise ValidationError(f"{label}.state_dict() must be non-empty")
    if not all(
        isinstance(key, str) and isinstance(tensor, torch.Tensor)
        for key, tensor in state.items()
    ):
        raise ValidationError(f"{label}.state_dict() contains unsupported entries")
    return state


def validate_net_and_ema(net: Any, ema: Any) -> dict[str, Any]:
    net_state = _module_state(net, "training state net")
    ema_state = _module_state(ema, "snapshot ema")
    net_keys = list(net_state)
    ema_keys = list(ema_state)
    if net_keys != ema_keys:
        raise ValidationError("strict net/EMA state_dict key order mismatch")
    for key in net_keys:
        left = net_state[key]
        right = ema_state[key]
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or left.layout != right.layout
        ):
            raise ValidationError(f"net/EMA tensor metadata mismatch for key {key!r}")
    for label, module in (("net", net), ("ema", ema)):
        _bool(_attribute(module, "use_fp16", label), True, f"{label}.use_fp16")
        if (
            _integer(
                _attribute(module, "img_resolution", label),
                f"{label}.img_resolution",
                minimum=1,
            )
            != 32
        ):
            raise ValidationError(f"{label}.img_resolution must equal 32")
        if _integer(_attribute(module, "label_dim", label), f"{label}.label_dim") != 0:
            raise ValidationError(f"{label}.label_dim must equal 0")
    return {"state_dict_keys": len(net_keys), "strict_keys_equal": True}


def validate_snapshot(
    snapshot: Mapping[str, Any], schedule: str, gap_scale: float
) -> dict[str, Any]:
    _exact_keys(snapshot, SNAPSHOT_FIELDS, "network snapshot")
    if snapshot["augment_pipe"] is not None:
        raise ValidationError("snapshot augment_pipe must be null")
    _validate_dataset_kwargs(snapshot["dataset_kwargs"], "snapshot.dataset_kwargs")
    loss = snapshot["loss_fn"]
    _equal_number(
        _attribute(loss, "q", "snapshot loss_fn"), 256.0, "snapshot loss_fn.q"
    )
    _equal_number(_attribute(loss, "k", "snapshot loss_fn"), 8.0, "snapshot loss_fn.k")
    _equal_number(_attribute(loss, "b", "snapshot loss_fn"), 1.0, "snapshot loss_fn.b")
    _equal_number(_attribute(loss, "c", "snapshot loss_fn"), 0.0, "snapshot loss_fn.c")
    schedule_object = _attribute(loss, "schedule", "snapshot loss_fn")
    actual_schedule = _attribute(schedule_object, "name", "snapshot schedule")
    if actual_schedule != schedule:
        raise ValidationError(
            f"snapshot schedule={actual_schedule!r}, expected {schedule!r}"
        )
    _equal_number(
        _attribute(schedule_object, "q", "snapshot schedule"),
        256.0,
        "snapshot schedule.q",
    )
    actual_gap = (
        _attribute(schedule_object, "global_gap_scale", "snapshot schedule")
        if schedule == "global_sigmoid"
        else 1.0
    )
    _equal_number(actual_gap, gap_scale, "snapshot global_gap_scale")
    stage = _integer(
        _attribute(loss, "stage", "snapshot loss_fn"), "snapshot loss_fn.stage"
    )
    ratio = _finite(
        _attribute(loss, "ratio", "snapshot loss_fn"), "snapshot loss_fn.ratio"
    )
    if not 0.0 <= ratio < 1.0:
        raise ValidationError("snapshot loss_fn.ratio must be in [0,1)")
    return {
        "schedule": schedule,
        "global_gap_scale": gap_scale,
        "stage": stage,
        "ratio": ratio,
    }


def validate_gradscaler(value: Any) -> dict[str, Any]:
    state = _mapping(value, "gradscaler_state")
    expected_keys = {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    }
    if set(state) != expected_keys:
        raise ValidationError("gradscaler_state key mismatch")
    scale = _finite(state["scale"], "gradscaler_state.scale")
    if scale <= 0:
        raise ValidationError("gradscaler_state.scale must be positive")
    _equal_number(state["growth_factor"], 2.0, "gradscaler_state.growth_factor")
    _equal_number(state["backoff_factor"], 0.5, "gradscaler_state.backoff_factor")
    if (
        _integer(
            state["growth_interval"], "gradscaler_state.growth_interval", minimum=1
        )
        != 2000
    ):
        raise ValidationError("gradscaler growth_interval must equal 2000")
    tracker = _integer(state["_growth_tracker"], "gradscaler_state._growth_tracker")
    return {"scale": scale, "growth_tracker": tracker}


def _step_number(value: Any, label: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or not _tensor_is_finite(value):
            raise ValidationError(f"{label} must be one finite scalar")
        value = value.item()
    numeric = _finite(value, label)
    if not numeric.is_integer() or numeric <= 0:
        raise ValidationError(f"{label} must be a positive integer")
    return int(numeric)


def validate_optimizer(
    value: Any, net: torch.nn.Module, successful_steps: int
) -> dict[str, Any]:
    optimizer = _mapping(value, "optimizer_state")
    if set(optimizer) != {"state", "param_groups"}:
        raise ValidationError("optimizer_state key mismatch")
    states = _mapping(optimizer["state"], "optimizer_state.state")
    groups = optimizer["param_groups"]
    if not isinstance(groups, list) or not groups:
        raise ValidationError("optimizer_state.param_groups must be a non-empty list")
    allowed_group_keys = {
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "maximize",
        "foreach",
        "capturable",
        "decoupled_weight_decay",
        "differentiable",
        "params",
    }
    optional_group_keys = {"maximize", "capturable"}
    required_group_keys = allowed_group_keys - optional_group_keys
    parameter_ids: list[Any] = []
    for index, raw_group in enumerate(groups):
        group = _mapping(raw_group, f"optimizer param_group[{index}]")
        actual_group_keys = set(group)
        missing = required_group_keys - actual_group_keys
        unknown = actual_group_keys - allowed_group_keys
        if missing or unknown:
            raise ValidationError(
                f"optimizer param_group[{index}] key mismatch: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        _equal_number(group["lr"], 1e-4, f"optimizer param_group[{index}].lr")
        betas = group["betas"]
        if not isinstance(betas, Sequence) or len(betas) != 2:
            raise ValidationError(f"optimizer param_group[{index}].betas mismatch")
        _equal_number(betas[0], 0.9, f"optimizer param_group[{index}].betas[0]")
        _equal_number(betas[1], 0.999, f"optimizer param_group[{index}].betas[1]")
        _equal_number(group["eps"], 1e-8, f"optimizer param_group[{index}].eps")
        _equal_number(
            group["weight_decay"], 0.0, f"optimizer param_group[{index}].weight_decay"
        )
        if "maximize" in group:
            _bool(
                group["maximize"],
                False,
                f"optimizer param_group[{index}].maximize",
            )
        if group["foreach"] is not None:
            raise ValidationError(
                f"optimizer param_group[{index}].foreach must be null"
            )
        if "capturable" in group:
            _bool(
                group["capturable"],
                False,
                f"optimizer param_group[{index}].capturable",
            )
        _bool(
            group["decoupled_weight_decay"],
            False,
            f"optimizer param_group[{index}].decoupled_weight_decay",
        )
        _bool(
            group["differentiable"],
            False,
            f"optimizer param_group[{index}].differentiable",
        )
        if not isinstance(group["params"], list) or not group["params"]:
            raise ValidationError(
                f"optimizer param_group[{index}].params must be non-empty"
            )
        parameter_ids.extend(group["params"])
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValidationError("optimizer parameter identifiers are duplicated")
    parameters = list(net.parameters())
    if len(parameter_ids) != len(parameters) or set(states) != set(parameter_ids):
        raise ValidationError("optimizer state/model parameter association mismatch")
    for index, (parameter_id, parameter) in enumerate(zip(parameter_ids, parameters)):
        item = _mapping(states[parameter_id], f"optimizer state[{parameter_id!r}]")
        if set(item) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValidationError(f"optimizer state[{parameter_id!r}] key mismatch")
        step = _step_number(item["step"], f"optimizer state[{parameter_id!r}].step")
        if step != successful_steps:
            raise ValidationError(
                f"optimizer step {step} does not match successful_optimizer_steps={successful_steps}"
            )
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = item[key]
            if not isinstance(tensor, torch.Tensor):
                raise ValidationError(
                    f"optimizer state[{parameter_id!r}].{key} must be a tensor"
                )
            if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                raise ValidationError(
                    f"optimizer state[{parameter_id!r}].{key} metadata mismatch"
                )
        if not bool((item["exp_avg_sq"] >= 0).all().item()):
            raise ValidationError(
                f"optimizer state[{parameter_id!r}].exp_avg_sq is negative"
            )
    return {"parameter_states": len(parameter_ids), "step": successful_steps}


def validate_state(
    state: Mapping[str, Any], *, expected_nimg: int, schedule: str
) -> dict[str, Any]:
    _exact_keys(state, STATE_FIELDS, "training state")
    cur_nimg = _integer(state["cur_nimg"], "state.cur_nimg", minimum=1)
    if not expected_nimg <= cur_nimg <= expected_nimg + 127:
        raise ValidationError(
            f"state.cur_nimg={cur_nimg} outside [{expected_nimg},{expected_nimg + 127}]"
        )
    attempted = _integer(
        state["attempted_iteration"], "state.attempted_iteration", minimum=1
    )
    successful = _integer(
        state["successful_optimizer_steps"],
        "state.successful_optimizer_steps",
        minimum=1,
    )
    if successful > attempted:
        raise ValidationError("successful_optimizer_steps exceeds attempted_iteration")
    if attempted * 128 != cur_nimg:
        raise ValidationError(
            "attempted_iteration does not match cur_nimg / batch_size"
        )
    cur_tick = _integer(state["cur_tick"], "state.cur_tick", minimum=1)
    if _integer(state["tick_start_nimg"], "state.tick_start_nimg") != cur_nimg:
        raise ValidationError("state.tick_start_nimg must equal state.cur_nimg")
    elapsed = _finite(state["elapsed_sec"], "state.elapsed_sec")
    if elapsed < 0:
        raise ValidationError("state.elapsed_sec must be non-negative")
    loss_state = _mapping(state["loss_fn_state"], "state.loss_fn_state")
    if set(loss_state) != {"schedule_name", "stage", "ratio", "schedule"}:
        raise ValidationError("state.loss_fn_state key mismatch")
    if loss_state["schedule_name"] != schedule:
        raise ValidationError("state loss schedule mismatch")
    stage = _integer(loss_state["stage"], "state.loss_fn_state.stage")
    ratio = _finite(loss_state["ratio"], "state.loss_fn_state.ratio")
    if not 0 <= ratio < 1:
        raise ValidationError("state.loss_fn_state.ratio must be in [0,1)")
    if _mapping(loss_state["schedule"], "state.loss_fn_state.schedule"):
        raise ValidationError("fixed schedule serialized state must be empty")
    scaler = validate_gradscaler(state["gradscaler_state"])
    optimizer = validate_optimizer(state["optimizer_state"], state["net"], successful)
    return {
        "cur_nimg": cur_nimg,
        "cur_tick": cur_tick,
        "attempted_iteration": attempted,
        "successful_optimizer_steps": successful,
        "stage": stage,
        "ratio": ratio,
        "gradscaler": scaler,
        "optimizer": optimizer,
    }


def validate_summary(
    path: Path,
    *,
    state_summary: Mapping[str, Any],
    schedule: str,
    state_scaler_scale: float,
) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = tuple(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, csv.Error, UnicodeError) as exc:
        raise ValidationError(f"cannot read train_summary.csv: {exc}") from exc
    if fieldnames != TRAIN_SUMMARY_FIELDS:
        raise ValidationError("train_summary.csv schema mismatch")
    if len(rows) < 2:
        raise ValidationError(
            "train_summary.csv must contain at least two continuation rows"
        )

    previous: dict[str, int] | None = None
    skipped_total = 0
    final_grad_scale = 0.0
    final_skipped = 0
    for offset, row in enumerate(rows, start=2):
        attempted = _csv_integer(
            row["attempted_iteration"],
            f"summary row {offset} attempted_iteration",
            minimum=1,
        )
        successful = _csv_integer(
            row["successful_optimizer_steps"],
            f"summary row {offset} successful_optimizer_steps",
            minimum=1,
        )
        processed = _csv_integer(
            row["processed_nimg"], f"summary row {offset} processed_nimg", minimum=1
        )
        if attempted * 128 != processed:
            raise ValidationError(f"summary row {offset} counter/progress mismatch")
        processed_kimg = _finite(
            row["processed_kimg"], f"summary row {offset} processed_kimg"
        )
        if processed_kimg != processed / 1000:
            raise ValidationError(f"summary row {offset} processed_kimg mismatch")
        _finite(row["loss"], f"summary row {offset} loss")
        grad_scale = _finite(row["grad_scale"], f"summary row {offset} grad_scale")
        if grad_scale <= 0:
            raise ValidationError(f"summary row {offset} grad_scale must be positive")
        skipped = _csv_integer(
            row["step_skipped"], f"summary row {offset} step_skipped"
        )
        if skipped not in (0, 1):
            raise ValidationError(f"summary row {offset} step_skipped must be 0 or 1")
        if row["schedule"] != schedule:
            raise ValidationError(f"summary row {offset} schedule mismatch")
        stage = _csv_integer(row["stage"], f"summary row {offset} stage")
        next_tick = _csv_integer(
            row["next_loop_cur_tick"], f"summary row {offset} next_loop_cur_tick"
        )
        if successful > attempted:
            raise ValidationError(
                f"summary row {offset} successful count exceeds attempts"
            )
        if previous is not None:
            if (
                attempted != previous["attempted"] + 1
                or processed != previous["processed"] + 128
            ):
                raise ValidationError(
                    f"summary row {offset} attempt/progress is not strictly consecutive"
                )
            if successful != previous["successful"] + (1 - skipped):
                raise ValidationError(
                    f"summary row {offset} successful counter transition mismatch"
                )
            if next_tick < previous["next_tick"] or stage < previous["stage"]:
                raise ValidationError(f"summary row {offset} tick/stage regressed")
        previous = {
            "attempted": attempted,
            "successful": successful,
            "processed": processed,
            "next_tick": next_tick,
            "stage": stage,
        }
        for optional in ("loss_ema", "loss_reference"):
            if row[optional].strip():
                _finite(row[optional], f"summary row {offset} {optional}")
        for field in (
            "correction",
            "r_over_t_mean",
            "gap_mean",
            "gap_over_sigmoid_gap_mean",
            "lower_gap_clip_rate",
            "upper_gap_clip_rate",
            "elapsed_sec",
            "peak_vram_gb",
        ):
            value = _finite(row[field], f"summary row {offset} {field}")
            if (
                field in {"lower_gap_clip_rate", "upper_gap_clip_rate"}
                and not 0 <= value <= 1
            ):
                raise ValidationError(f"summary row {offset} {field} outside [0,1]")
            if (
                field
                in {
                    "elapsed_sec",
                    "peak_vram_gb",
                    "gap_mean",
                    "gap_over_sigmoid_gap_mean",
                }
                and value < 0
            ):
                raise ValidationError(
                    f"summary row {offset} {field} must be non-negative"
                )
        _csv_integer(row["signal_updates"], f"summary row {offset} signal_updates")
        active = row["adaptive_active"].strip().lower()
        if active not in {"0", "false"}:
            raise ValidationError(f"summary row {offset} adaptive_active must be false")
        skipped_total += skipped
        final_grad_scale = grad_scale
        final_skipped = skipped

    assert previous is not None
    expected_final = {
        "attempted": state_summary["attempted_iteration"],
        "successful": state_summary["successful_optimizer_steps"],
        "processed": state_summary["cur_nimg"],
        "next_tick": state_summary["cur_tick"],
        "stage": state_summary["stage"],
    }
    if previous != expected_final:
        raise ValidationError(
            f"train_summary final progress/counters do not match training state: "
            f"{previous!r} != {expected_final!r}"
        )
    # The serialized scaler is after update(); account for a skip or a growth
    # interval boundary while checking the final row's pre-update scale.
    allowed_scales = {
        final_grad_scale,
        final_grad_scale * 0.5,
        final_grad_scale * 2.0,
    }
    if state_scaler_scale not in allowed_scales:
        raise ValidationError(
            "final train_summary grad_scale is inconsistent with GradScaler state"
        )
    if final_skipped and state_scaler_scale != final_grad_scale * 0.5:
        raise ValidationError("final skipped step did not apply the GradScaler backoff")
    return {
        "rows": len(rows),
        "skipped_steps_in_segment": skipped_total,
        "final_loss": _finite(rows[-1]["loss"], "summary final loss"),
        "final_grad_scale": final_grad_scale,
        "final_step_skipped": final_skipped,
        "final_processed_nimg": previous["processed"],
    }


def _validate_log(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValidationError(f"cannot read log {path}: {exc}") from exc
    if "Exiting..." not in text:
        raise ValidationError("log.txt lacks the clean Exiting marker")
    if "Traceback (most recent call last)" in text or "Aborting..." in text:
        raise ValidationError("log.txt records a traceback or aborted run")
    if "Loading training state from" not in text:
        raise ValidationError("log.txt does not record a continuation-state load")
    return {"clean_exit": True, "resume_load_recorded": True}


def validate_run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise ValidationError(f"run_dir is not a directory: {run_dir}")
    state_path = args.state.expanduser().resolve()
    snapshot_path = args.snapshot.expanduser().resolve()
    if state_path.parent != run_dir or state_path.name != "training-state-latest.pt":
        raise ValidationError("--state must name run_dir/training-state-latest.pt")
    if (
        snapshot_path.parent != run_dir
        or snapshot_path.name != "network-snapshot-latest.pkl"
    ):
        raise ValidationError(
            "--snapshot must name run_dir/network-snapshot-latest.pkl"
        )
    paths = {
        "state": _regular_file(state_path, "training state"),
        "snapshot": _regular_file(snapshot_path, "network snapshot"),
        "options": _regular_file(run_dir / "training_options.json", "training options"),
        "summary": _regular_file(run_dir / "train_summary.csv", "train summary"),
        "log": _regular_file(run_dir / "log.txt", "training log"),
    }
    numbered = sorted(
        run_dir.glob("network-snapshot-[0-9][0-9][0-9][0-9][0-9][0-9].pkl")
    )
    numbered += sorted(run_dir.glob("training-state-[0-9][0-9][0-9][0-9][0-9][0-9].pt"))
    if numbered:
        raise ValidationError(
            f"numbered checkpoint cadence must be disabled: {numbered[:4]}"
        )
    signatures = {name: _signature(path) for name, path in paths.items()}

    expected_nimg = _integer(args.expected_nimg, "expected_nimg", minimum=1)
    expected_seed = _integer(args.expected_seed, "expected_seed")
    options = _load_json(paths["options"], "training options")
    option_summary = validate_options(
        options,
        run_dir=run_dir,
        expected_nimg=expected_nimg,
        expected_seed=expected_seed,
        expected_arm=args.expected_arm,
        expected_mapping_override=args.expected_mapping,
        expected_gap_override=args.expected_gap_scale,
    )
    identity = _validate_identity_declaration()
    data_path = _regular_file(Path(CANONICAL_DATA_PATH), "canonical dataset")
    if data_path.stat().st_size != CANONICAL_DATA_SIZE_BYTES:
        raise ValidationError("canonical dataset size does not match its declaration")
    observed_data_hash = sha256_file(data_path)
    if observed_data_hash != CANONICAL_DATA_SHA256:
        raise ValidationError("canonical dataset SHA256 mismatch")

    state = _load_state(paths["state"])
    snapshot = _load_snapshot(paths["snapshot"])
    finite_state = inspect_finite_tensors(state, "training_state")
    finite_snapshot = inspect_finite_tensors(snapshot, "network_snapshot")
    state_summary = validate_state(
        state, expected_nimg=expected_nimg, schedule=option_summary["schedule"]
    )
    snapshot_summary = validate_snapshot(
        snapshot, option_summary["schedule"], option_summary["global_gap_scale"]
    )
    if (
        snapshot_summary["stage"] != state_summary["stage"]
        or snapshot_summary["ratio"] != state_summary["ratio"]
    ):
        raise ValidationError("snapshot and training-state loss progress mismatch")
    strict_keys = validate_net_and_ema(state["net"], snapshot["ema"])
    summary = validate_summary(
        paths["summary"],
        state_summary=state_summary,
        schedule=option_summary["schedule"],
        state_scaler_scale=state_summary["gradscaler"]["scale"],
    )
    log = _validate_log(paths["log"])

    file_records: dict[str, Any] = {}
    for name, path in paths.items():
        digest = sha256_file(path)
        if _signature(path) != signatures[name]:
            raise ValidationError(f"{name} changed while it was being validated")
        file_records[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": digest,
        }
    return {
        "schema": SCHEMA,
        "verdict": "GO",
        "expected": {
            "arm": args.expected_arm,
            "seed": expected_seed,
            "nimg": expected_nimg,
            "schedule": option_summary["schedule"],
            "global_gap_scale": option_summary["global_gap_scale"],
        },
        "observed": {
            "state": state_summary,
            "snapshot": snapshot_summary,
            "train_summary": summary,
            "net_ema": strict_keys,
            "finite_training_state": finite_state,
            "finite_snapshot": finite_snapshot,
            "log": log,
        },
        "dataset_identity": {
            **identity,
            "observed_sha256": observed_data_hash,
        },
        "artifacts": file_records,
    }


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    if not path.parent.is_dir():
        raise ValidationError(f"result receipt parent does not exist: {path.parent}")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ValidationError(f"refusing to overwrite result receipt: {path}") from exc
    except OSError as exc:
        raise ValidationError(
            f"cannot exclusively create result receipt {path}: {exc}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--expected-nimg", type=int, required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument("--expected-arm", required=True)
    parser.add_argument("--expected-mapping")
    parser.add_argument("--expected-gap-scale", type=float)
    parser.add_argument("--result-receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt_path = args.result_receipt.expanduser().resolve()
    if receipt_path.exists() or receipt_path.is_symlink():
        payload = {
            "schema": SCHEMA,
            "verdict": "NO_GO",
            "error": f"refusing to overwrite result receipt: {receipt_path}",
        }
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
        return 2
    try:
        payload = validate_run(args)
    except Exception as exc:
        payload = {"schema": SCHEMA, "verdict": "NO_GO", "error": str(exc)}
        try:
            _exclusive_json(receipt_path, payload)
        except Exception as write_exc:
            payload["receipt_error"] = str(write_exc)
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
        return 2
    try:
        _exclusive_json(receipt_path, payload)
    except Exception as exc:
        failure = {"schema": SCHEMA, "verdict": "NO_GO", "error": str(exc)}
        print(json.dumps(failure, sort_keys=True, allow_nan=False))
        return 2
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
