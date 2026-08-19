#!/usr/bin/env python3
"""Execute a trusted repository script with NumPy 2 -> 1 pickle aliases.

The formal q256 checkpoints were serialized by an environment whose pickle
module paths begin with ``numpy._core``.  The recovered A100 runtime exposes
NumPy 1.x, where the byte-compatible implementation lives at ``numpy.core``.
This launcher changes only class lookup while deserializing trusted artifacts;
it does not rewrite a source file, change NumPy values, or patch serialization.

Usage: ``python numpy_pickle_compat_exec.py SCRIPT [SCRIPT_ARGS ...]``.
"""

from __future__ import annotations

import pickle
import runpy
import sys
from pathlib import Path
from typing import Any

import torch


class NumpyPathCompatUnpickler(pickle.Unpickler):
    """Map NumPy 2's private package spelling to NumPy 1's spelling."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core") :]
        return super().find_class(module, name)


_ORIGINAL_PICKLE_LOAD = pickle.load
_ORIGINAL_TORCH_LOAD = torch.load


class NumpyPathCompatPickleModule:
    """Minimal pickle-module interface accepted by ``torch.load``."""

    __name__ = "pickle"
    Unpickler = NumpyPathCompatUnpickler
    Pickler = pickle.Pickler
    load = staticmethod(_ORIGINAL_PICKLE_LOAD)
    loads = staticmethod(pickle.loads)
    dump = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)


def _compat_pickle_load(file, *args, **kwargs):
    return NumpyPathCompatUnpickler(file, *args, **kwargs).load()


def _compat_torch_load(*args, **kwargs):
    if kwargs.get("weights_only") is True:
        return _ORIGINAL_TORCH_LOAD(*args, **kwargs)
    kwargs.setdefault("pickle_module", NumpyPathCompatPickleModule)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


def install_compatibility_hooks() -> None:
    """Install process-local load-only hooks; safe to call once."""

    if (
        torch.load is not _ORIGINAL_TORCH_LOAD
        or pickle.load is not _ORIGINAL_PICKLE_LOAD
    ):
        raise RuntimeError(
            "pickle compatibility hooks are already installed or replaced"
        )
    torch.load = _compat_torch_load
    pickle.load = _compat_pickle_load


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        raise SystemExit("usage: numpy_pickle_compat_exec.py SCRIPT [SCRIPT_ARGS ...]")
    target = Path(arguments[0]).expanduser().resolve()
    if not target.is_file() or target.is_symlink():
        raise SystemExit(f"target is not a regular non-symlink file: {target}")
    if target == Path(__file__).resolve():
        raise SystemExit("refusing recursive compatibility-launcher execution")
    install_compatibility_hooks()
    # Match normal ``python SCRIPT`` import resolution.  ``runpy`` otherwise
    # leaves this launcher's ``scripts/`` directory at sys.path[0], which would
    # make repository imports such as ``torch_utils`` unavailable to ct_train.
    sys.path.insert(0, str(target.parent))
    current_directory = str(Path.cwd().resolve())
    if current_directory not in sys.path:
        sys.path.insert(1, current_directory)
    sys.argv = [str(target), *arguments[1:]]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
