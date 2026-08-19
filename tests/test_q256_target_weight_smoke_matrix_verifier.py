import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

from training import reproducibility
from scripts import verify_q256_target_weight_arm as arm_verifier
from scripts import verify_q256_target_weight_smoke_matrix as matrix_verifier


TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
from test_q256_target_weight_verifier import RunFixture  # noqa: E402


class Q256TargetWeightSmokeMatrixVerifierTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def make_fixture(self, name, *, arm="A", skip_attempts=(2, 7)):
        root = self.root / name
        root.mkdir()
        return RunFixture(root, arm=arm, skip_attempts=skip_attempts)

    def finalize_arm(self, fixture):
        return arm_verifier.verify_run(
            fixture.root,
            arm=fixture.arm,
            seed=3,
            mode="smoke",
        )

    def make_matrix(self, *, skip_by_arm=None, mutators=None):
        skip_by_arm = skip_by_arm or {}
        mutators = mutators or {}
        fixtures = {}
        for arm in matrix_verifier.ARMS:
            fixture = self.make_fixture(
                f"arm{arm}",
                arm=arm,
                skip_attempts=skip_by_arm.get(arm, (2, 7)),
            )
            if arm in mutators:
                mutators[arm](fixture)
            self.finalize_arm(fixture)
            fixtures[arm] = fixture
        return fixtures, {arm: fixture.root for arm, fixture in fixtures.items()}

    def test_valid_matrix_emits_one_immutable_cross_arm_pass(self):
        _, run_dirs = self.make_matrix()
        receipt = self.root / "matrix-pass.json"
        report = matrix_verifier.verify_smoke_matrix(
            run_dirs, receipt_path=receipt
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["amp_skip_attempts"], [2, 7])
        self.assertTrue(
            report["trajectory_checks"]["native_A_target_equals_denominator"]
        )
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], matrix_verifier.VALIDATION_SCHEMA)
        self.assertNotIn("validation_receipt", payload)
        with self.assertRaisesRegex(
            matrix_verifier.MatrixVerificationError,
            "immutable matrix PASS receipt already exists",
        ):
            matrix_verifier.verify_smoke_matrix(
                run_dirs, receipt_path=receipt
            )

    def test_common_batch_and_factor_specific_trajectories_are_exact(self):
        def change_batch(fixture):
            rows = fixture.telemetry_rows()
            rows[4]["batch_sha256"] = "9" * 64
            fixture.write_telemetry(rows)

        _, run_dirs = self.make_matrix(mutators={"D": change_batch})
        with self.assertRaisesRegex(
            matrix_verifier.MatrixVerificationError,
            "all-arms common trajectory.batch_sha256 mismatch",
        ):
            matrix_verifier.verify_smoke_matrix(run_dirs, write_receipt=False)

        self.root = self.root / "second"
        self.root.mkdir()

        def change_target(fixture):
            rows = fixture.telemetry_rows()
            rows[10]["target_r_sha256"] = "9" * 64
            fixture.write_telemetry(rows)

        _, run_dirs = self.make_matrix(mutators={"C": change_target})
        with self.assertRaisesRegex(
            matrix_verifier.MatrixVerificationError,
            "target scale 1.1.target_r_sha256 mismatch",
        ):
            matrix_verifier.verify_smoke_matrix(run_dirs, write_receipt=False)

        self.root = self.root / "denominator"
        self.root.mkdir()

        def change_denominator(fixture):
            rows = fixture.telemetry_rows()
            rows[6]["denominator_r_sha256"] = "9" * 64
            fixture.write_telemetry(rows)

        _, run_dirs = self.make_matrix(mutators={"D": change_denominator})
        with self.assertRaisesRegex(
            matrix_verifier.MatrixVerificationError,
            "denominator scale 1.1.denominator_r_sha256 mismatch",
        ):
            matrix_verifier.verify_smoke_matrix(run_dirs, write_receipt=False)

    def test_native_arms_require_all_target_denominator_statistics_equal(self):
        def change_native_statistic(fixture):
            rows = fixture.telemetry_rows()
            rows[1]["denominator_delta_mean"] = "0.3"
            fixture.write_telemetry(rows)

        # Keep the denominator-scale A=C relation valid so this fixture reaches
        # the stricter within-native-arm target=denominator comparison.
        _, run_dirs = self.make_matrix(
            mutators={"A": change_native_statistic, "C": change_native_statistic}
        )
        with self.assertRaisesRegex(
            matrix_verifier.MatrixVerificationError,
            "native arm A target/denominator delta_mean mismatch at attempt 2",
        ):
            matrix_verifier.verify_smoke_matrix(run_dirs, write_receipt=False)

    def test_initial_components_final_rank_state_and_skip_signature_match(self):
        def change_initial(fixture):
            receipt = fixture.initial_receipt()
            receipt["hashes"]["optimizer"] = "9" * 64
            receipt["common_initial_state_sha256"] = reproducibility.state_sha256(
                receipt["hashes"]
            )
            fixture.write_json("initial_state_receipt_v1.json", receipt)

        _, run_dirs = self.make_matrix(mutators={"C": change_initial})
        with self.assertRaisesRegex(
            matrix_verifier.MatrixVerificationError,
            "common initial model/EMA/optimizer/GradScaler/RNG/sampler hashes mismatch",
        ):
            matrix_verifier.verify_smoke_matrix(run_dirs, write_receipt=False)

        self.root = self.root / "rng"
        self.root.mkdir()

        def change_rng(fixture):
            state = fixture.state()
            state["rank_states"][0]["rng_state"]["python"] = (
                random.Random(99).getstate()
            )
            fixture.write_state(state)

        _, run_dirs = self.make_matrix(mutators={"D": change_rng})
        with self.assertRaisesRegex(
            matrix_verifier.MatrixVerificationError,
            "final rank RNG state mismatch",
        ):
            matrix_verifier.verify_smoke_matrix(run_dirs, write_receipt=False)

        self.root = self.root / "skips"
        self.root.mkdir()
        _, run_dirs = self.make_matrix(skip_by_arm={"B": (2, 8)})
        with self.assertRaisesRegex(
            matrix_verifier.MatrixVerificationError,
            "AMP skip-attempt signature mismatch",
        ):
            matrix_verifier.verify_smoke_matrix(run_dirs, write_receipt=False)

    def test_existing_single_arm_pass_receipt_must_still_bind_artifacts(self):
        _, run_dirs = self.make_matrix()
        telemetry = run_dirs["A"] / "factorial_training_telemetry_v1.csv"
        telemetry.write_bytes(telemetry.read_bytes() + b"\n")
        with self.assertRaisesRegex(
            matrix_verifier.MatrixVerificationError,
            "PASS-bound artifact changed",
        ):
            matrix_verifier.verify_smoke_matrix(run_dirs, write_receipt=False)

    def test_exact_resume_ignores_only_wall_clock_fields(self):
        _, run_dirs = self.make_matrix()
        resumed = self.make_fixture("resumedA", arm="A")
        rows = resumed.telemetry_rows()
        for index, row in enumerate(rows, start=1):
            elapsed = 100.0 + index
            row["elapsed_sec"] = f"{elapsed:.6f}"
            row["gpu_hours_cumulative"] = f"{elapsed / 3600:.9f}"
        resumed.write_telemetry(rows)
        state = resumed.state()
        state["elapsed_sec"] = 132.0
        resumed.write_state(state)
        self.finalize_arm(resumed)

        report = matrix_verifier.verify_smoke_matrix(
            run_dirs,
            resume_pair=(run_dirs["A"], resumed.root, "A"),
            write_receipt=False,
        )
        comparison = report["exact_resume"]
        self.assertEqual(comparison["status"], "passed")
        self.assertEqual(
            comparison["excluded_noncomputational_fields"],
            ["elapsed_sec", "gpu_hours_cumulative"],
        )

    def test_exact_resume_rejects_each_computational_difference(self):
        _, run_dirs = self.make_matrix()
        resumed = self.make_fixture("resumedA", arm="A")
        state = resumed.state()
        state["optimizer_state"]["param_groups"][0]["lr"] = 9e-5
        resumed.write_state(state)
        self.finalize_arm(resumed)
        with self.assertRaisesRegex(
            matrix_verifier.MatrixVerificationError,
            "exact-resume final optimizer mismatch",
        ):
            matrix_verifier.verify_smoke_matrix(
                run_dirs,
                resume_pair=(run_dirs["A"], resumed.root, "A"),
                write_receipt=False,
            )

        self.root = self.root / "telemetry"
        self.root.mkdir()
        _, run_dirs = self.make_matrix()
        resumed = self.make_fixture("resumedA", arm="A")
        rows = resumed.telemetry_rows()
        rows[12]["loss"] = "1.5"
        resumed.write_telemetry(rows)
        self.finalize_arm(resumed)
        with self.assertRaisesRegex(
            matrix_verifier.MatrixVerificationError,
                r"computational telemetry mismatch at attempt 13, fields=\['loss'\]",
        ):
            matrix_verifier.verify_smoke_matrix(
                run_dirs,
                resume_pair=(run_dirs["A"], resumed.root, "A"),
                write_receipt=False,
            )


if __name__ == "__main__":
    unittest.main()
