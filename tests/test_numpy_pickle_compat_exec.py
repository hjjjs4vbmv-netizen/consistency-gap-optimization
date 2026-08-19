from __future__ import annotations

import importlib
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scripts import numpy_pickle_compat_exec as compat


class NumpyPickleCompatExecTests(unittest.TestCase):
    def test_numpy2_class_path_maps_to_numpy1_class(self):
        expected = importlib.import_module("numpy.core.multiarray")._reconstruct
        with tempfile.TemporaryFile() as handle:
            unpickler = compat.NumpyPathCompatUnpickler(handle)
            observed = unpickler.find_class("numpy._core.multiarray", "_reconstruct")
        self.assertIs(observed, expected)

    def test_ordinary_non_numpy_lookup_is_unchanged(self):
        with tempfile.TemporaryFile() as handle:
            unpickler = compat.NumpyPathCompatUnpickler(handle)
            self.assertIs(unpickler.find_class("builtins", "dict"), dict)

    def test_direct_pickle_load_preserves_array_content(self):
        value = np.arange(12, dtype=np.float32).reshape(3, 4)
        with tempfile.TemporaryFile() as handle:
            pickle.dump(value, handle)
            handle.seek(0)
            loaded = compat._compat_pickle_load(handle)
        np.testing.assert_array_equal(loaded, value)

    def test_torch_load_injects_only_pickle_module(self):
        sentinel = object()
        with mock.patch.object(
            compat, "_ORIGINAL_TORCH_LOAD", return_value=sentinel
        ) as load:
            self.assertIs(
                compat._compat_torch_load("state.pt", weights_only=False), sentinel
            )
        kwargs = load.call_args.kwargs
        self.assertIs(kwargs["pickle_module"], compat.NumpyPathCompatPickleModule)
        self.assertFalse(kwargs["weights_only"])

    def test_weights_only_path_is_not_given_custom_unpickler(self):
        with mock.patch.object(compat, "_ORIGINAL_TORCH_LOAD", return_value=7) as load:
            self.assertEqual(
                compat._compat_torch_load("state.pt", weights_only=True), 7
            )
        self.assertNotIn("pickle_module", load.call_args.kwargs)

    def test_launcher_rejects_itself_and_missing_target(self):
        with self.assertRaises(SystemExit):
            compat.main([str(Path(compat.__file__))])
        with self.assertRaises(SystemExit):
            compat.main(["/definitely/missing/q256-script.py"])

    def test_launcher_exposes_target_directory_to_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sibling_module.py").write_text("VALUE = 17\n", encoding="utf-8")
            target = root / "target.py"
            target.write_text(
                "import sibling_module\nassert sibling_module.VALUE == 17\n",
                encoding="utf-8",
            )
            with mock.patch.object(compat, "install_compatibility_hooks"):
                self.assertEqual(compat.main([str(target)]), 0)


if __name__ == "__main__":
    unittest.main()
