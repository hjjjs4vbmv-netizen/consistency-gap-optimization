import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from scripts import verify_fixed_seed_determinism


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class VerifyFixedSeedDeterminismTest(unittest.TestCase):
    def make_result(self, root: Path) -> None:
        entries = []
        for nfe in (1, 2):
            mode = root / f"nfe{nfe}"
            images = mode / "images"
            images.mkdir(parents=True)
            for seed in range(64):
                path = images / f"seed{seed:06d}.png"
                Image.new("RGB", (1, 1), (seed, nfe, 0)).save(path)
                entries.append((sha256_file(path), path.relative_to(root).as_posix()))
            grid = mode / "grid_8x8.png"
            Image.new("RGB", (8, 8), (nfe, 0, 0)).save(grid)
            entries.append((sha256_file(grid), grid.relative_to(root).as_posix()))
        metadata = {
            "checkpoint_id": "test-checkpoint",
            "checkpoint_sha256": "a" * 64,
            "seed_list": list(range(64)),
            "seed_count": 64,
            "nfe_modes": [1, 2],
            "mid_t_by_mode": {"nfe1": [], "nfe2": [0.821]},
            "precision": "fp32",
            "model_forward_batch_size": 1,
            "work_group_sizes_verified": [8, 16],
            "repeat_runs_verified": 2,
            "determinism_passed": True,
            "image_count_by_mode": {"nfe1": 64, "nfe2": 64},
            "image_count_total": 128,
        }
        metadata_path = root / "metadata.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        entries.append((sha256_file(metadata_path), "metadata.json"))
        (root / "sha256_manifest.txt").write_text(
            "".join(f"{digest}  {name}\n" for digest, name in entries),
            encoding="utf-8",
        )

    def test_valid_artifact_passes_and_writes_report(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_result(root)
            report = verify_fixed_seed_determinism.verify(root)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["manifest_entry_count"], 131)
            verify_fixed_seed_determinism.main(["--result-dir", str(root)])
            self.assertTrue((root / "determinism_verification.json").is_file())

    def test_manifest_mismatch_fails(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_result(root)
            (root / "nfe1" / "images" / "seed000000.png").write_bytes(b"corrupt")
            with self.assertRaisesRegex(SystemExit, "manifest SHA256 mismatch"):
                verify_fixed_seed_determinism.verify(root)


if __name__ == "__main__":
    unittest.main()
