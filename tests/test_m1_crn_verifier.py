import csv
import json
import tempfile
import unittest
from pathlib import Path

from analysis.q256_optimizer_restart_ema_rebuild_v1 import verify_crn


class M1CrnVerifierTests(unittest.TestCase):
    def write_telemetry(
        self, path, branch, *, seed=50, mode="formal", corrupt=False
    ):
        fields = ("branch", "continuation_arm", *verify_crn.CRN_FIELDS)
        first, last = verify_crn.WINDOWS[mode]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for attempt in range(first, last + 1):
                common = f"h{attempt}"
                row = {
                    "branch": branch,
                    "continuation_arm": "A",
                    "seed": seed,
                    "attempted_iteration": attempt,
                    "batch_sha256": common,
                    "t_sha256": common,
                    "base_r_sha256": common,
                    "eps_sha256": common,
                    "dropout_rng_sha256": common,
                    "online_input_sha256": common,
                    "target_input_sha256": common,
                }
                if corrupt and attempt == first:
                    row["online_input_sha256"] = "different"
                writer.writerow(row)

    def test_explicit_gate_and_formal_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in verify_crn.WINDOWS:
                paths = {}
                for branch in verify_crn.BRANCHES:
                    path = Path(directory) / f"{mode}-{branch}.csv"
                    self.write_telemetry(path, branch, mode=mode)
                    paths[branch] = path
                self.assertEqual(len(verify_crn.verify(paths, 50, mode)), 64)

    def test_input_hash_or_seed_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for branch in verify_crn.BRANCHES:
                path = Path(directory) / f"{branch}.csv"
                self.write_telemetry(path, branch, mode="gate16")
                paths[branch] = path
            self.write_telemetry(
                paths["R_B"], "R_B", mode="gate16", corrupt=True
            )
            with self.assertRaisesRegex(RuntimeError, "CRN mismatch"):
                verify_crn.verify(paths, 50, "gate16")
            with self.assertRaisesRegex(RuntimeError, "seed"):
                verify_crn.verify(paths, 51, "gate16")

    def test_formal_pair_can_be_verified_without_other_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            pair = ("R_A", "R_B")
            paths = {}
            for branch in pair:
                path = Path(directory) / f"{branch}.csv"
                self.write_telemetry(path, branch, mode="formal")
                paths[branch] = path
            self.assertEqual(
                len(verify_crn.verify_pair(paths, 50, "formal", pair)), 64
            )

    def test_four_manifests_bind_one_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for branch in verify_crn.BRANCHES:
                path = Path(directory) / f"{branch}.json"
                path.write_text(json.dumps({
                    "schema": verify_crn.RUN_MANIFEST_SCHEMA,
                    "experiment_protocol": verify_crn.PROTOCOL_ID,
                    "branch": branch,
                    "continuation_arm": "A",
                    "seed": 50,
                }))
                paths[branch] = path
            seed, hashes = verify_crn.load_manifest_bindings(paths)
            self.assertEqual(seed, 50)
            self.assertEqual(set(hashes), set(verify_crn.BRANCHES))
            changed = json.loads(paths["R_B"].read_text())
            changed["seed"] = 51
            paths["R_B"].write_text(json.dumps(changed))
            with self.assertRaisesRegex(RuntimeError, "common seed"):
                verify_crn.load_manifest_bindings(paths)


if __name__ == "__main__":
    unittest.main()
