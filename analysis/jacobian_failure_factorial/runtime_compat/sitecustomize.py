"""Read NumPy-2 pickles with the NumPy-1 ABI used by the audit container.

NumPy 2 moved the private ``numpy.core`` package to ``numpy._core``.  The
trusted training-state pickle records those import paths, while the frozen
PyTorch 24.04 container must retain NumPy 1.x for its compiled ABI.  Aliasing
the private module names is sufficient for unpickling and does not alter array
values or numerical operations.
"""

from __future__ import annotations

import importlib
import sys

import numpy


sys.modules.setdefault("numpy._core", numpy.core)
for _submodule in (
    "_dtype",
    "_internal",
    "_methods",
    "_multiarray_umath",
    "_type_aliases",
    "fromnumeric",
    "multiarray",
    "numeric",
    "numerictypes",
    "overrides",
    "shape_base",
    "umath",
):
    try:
        _module = importlib.import_module(f"numpy.core.{_submodule}")
    except ImportError:
        continue
    sys.modules.setdefault(f"numpy._core.{_submodule}", _module)
