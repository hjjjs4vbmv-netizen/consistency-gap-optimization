import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "analysis" / "q256_target_weight_factorial" / "extension_support"
DRIVER = SUPPORT / "run_q256_seed14_18_1024k_frozen_evaluation.py"
WORKER = SUPPORT / "run_q256_seed14_18_1024k_worker.sh"
LAUNCHER = SUPPORT / "launch_q256_seed14_18_1024k.sh"
RECOVERY_WORKER = SUPPORT / "run_q256_seed14_18_1024k_recovery_v2_worker.sh"
RECOVERY_LAUNCHER = SUPPORT / "launch_q256_seed14_18_1024k_recovery_v2.sh"


def load_driver():
    spec = importlib.util.spec_from_file_location("q256_seed14_18_1024k_eval", DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_summary(path: Path, *, attempted: int, processed_kimg: float) -> None:
    path.write_text(
        "attempted_iteration,successful_optimizer_steps,processed_kimg,loss,step_skipped\n"
        f"{attempted},{attempted - 10},{processed_kimg:.6f},12.5,0\n",
        encoding="utf-8",
    )


def test_evaluator_accepts_only_the_1024k_endpoint(tmp_path: Path) -> None:
    driver = load_driver()
    summary = tmp_path / "train_summary.csv"
    write_summary(summary, attempted=8000, processed_kimg=1024.0)
    endpoint = driver.latest_summary(summary)
    assert endpoint["attempted_iteration"] == 8000
    assert endpoint["processed_kimg"] == 1024.0

    write_summary(summary, attempted=2000, processed_kimg=256.0)
    with pytest.raises(driver.EvaluationFailure, match="training endpoint mismatch"):
        driver.latest_summary(summary)


def test_training_worker_freezes_budget_only_resume_contract() -> None:
    text = WORKER.read_text(encoding="utf-8")
    assert "--duration=1.024" in text
    assert '--resume="${resume_state}"' in text
    assert "--transfer=" not in text
    assert "s['cur_nimg']==1024000" in text
    assert "s['attempted_iteration']==8000" in text
    assert text.index("run_arm A") < text.index("run_arm B") < text.index("run_arm C") < text.index("run_arm D")
    assert "run_q256_seed14_18_1024k_native_evaluation_worker.sh" in text


def test_launcher_is_fail_closed_and_uses_a_new_root() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "seed14-18-1024k-from-v2-4582051-v1" in text
    assert "refusing existing run root" in text
    assert "source_256k_training_states.sha256" in text
    assert "q1024_s${seed}" in text
    assert "evaluation_chained=true" in text


def test_evaluation_contract_matches_seed3_5_metrics() -> None:
    driver = load_driver()
    assert driver.TRAINING_COMMIT == "458205192722883df393a8d017c26e6fa46f48f7"
    assert driver.PARENT_TRAINING_COMMIT == "dcca41b19e7c45512b5fbe98776520396a1bf9ac"
    assert driver.METRICS == ("kid50k_full", "fid50k_full")
    assert driver.SAMPLE_COUNT == 50_000
    assert driver.METRIC_SEED == 20_260_730
    assert driver.NFE_SETTINGS == {1: [], 2: [0.821]}


def test_recovery_v2_repairs_only_the_post_training_verifier() -> None:
    worker = RECOVERY_WORKER.read_text(encoding="utf-8")
    launcher = RECOVERY_LAUNCHER.read_text(encoding="utf-8")
    assert 'PYTHONPATH="${repo}"' in worker
    assert "adopt_completed_arm_a" in worker
    assert "run_arm A" not in worker
    assert worker.index("adopt_completed_arm_a") < worker.index("run_arm B")
    assert "hash_identical_adoption_after_post_training_verifier_failure" in worker
    assert "seed14-18-1024k-from-v2-4582051-recovery-v2" in worker
    assert "ModuleNotFoundError: No module named 'torch_utils'" in launcher
    assert "failed_v1_completed_armA_artifacts.sha256" in launcher
    assert "q1024r2_s${seed}" in launcher


def test_learning_curve_milestones_are_output_only_and_exact() -> None:
    cli = (ROOT / "ct_train.py").read_text(encoding="utf-8")
    loop = (ROOT / "training" / "ct_training_loop.py").read_text(
        encoding="utf-8"
    )
    assert "--checkpoint-milestone-kimg" in cli
    assert "checkpoint_milestone_kimg=tuple(opts.checkpoint_milestone_kimg)" in cli
    assert "_FORMAL_REPLAY_MILESTONE_KIMG = (384, 512, 640, 768, 896, 1024)" in loop
    assert "milestone_nimg % batch_size" in loop
    assert "f'kimg{milestone_kimg:04d}'" in loop
    assert "network-snapshot.pkl" in loop
    assert "training-state.pt" in loop
    assert "ect.q256.learning-curve-milestone/v1" in loop
    trajectory_block = loop.split(
        "strict_trajectory_config = reproducibility.canonical_json_data({", 1
    )[1].split("strict_trajectory_config_sha256 =", 1)[0]
    assert "checkpoint_milestone_kimg" not in trajectory_block
    milestone_block = loop.split(
        "# Save immutable, independently loadable replay milestone artifacts.", 1
    )[1].split("# Sample Img", 1)[0]
    assert "dist.barrier" not in milestone_block


def test_learning_curve_evaluation_is_nfe1_first(tmp_path: Path) -> None:
    driver_path = (
        ROOT
        / "analysis"
        / "q256_target_weight_factorial"
        / "learning_curve_replay"
        / "run_learning_curve_frozen_evaluation.py"
    )
    spec = importlib.util.spec_from_file_location("q256_learning_curve_eval", driver_path)
    assert spec is not None and spec.loader is not None
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    cells = [
        {
            "seed": 14,
            "arm": arm,
            "budget_kimg": budget,
            "checkpoint": str(tmp_path / f"{arm}-{budget}.pkl"),
            "checkpoint_sha256": "0" * 64,
            "run_dir": str(tmp_path / arm),
        }
        for arm in "ABCD"
        for budget in driver.BUDGETS
    ]
    jobs = driver.build_jobs(
        source_root=tmp_path,
        output_root=tmp_path / "out",
        dataset=tmp_path / "data.zip",
        cells=cells,
        base_port=50140,
    )
    assert len(jobs) == 48
    assert [job["nfe"] for job in jobs[:24]] == [1] * 24
    assert [job["nfe"] for job in jobs[24:]] == [2] * 24
    assert [job["budget_kimg"] for job in jobs[:6]] == list(driver.BUDGETS)
    assert jobs[0]["job_id"] == "seed14-armA-kimg0384-nfe1"
    assert jobs[-1]["job_id"] == "seed14-armD-kimg1024-nfe2"


def test_learning_curve_replay_launcher_freezes_scope() -> None:
    support = (
        ROOT
        / "analysis"
        / "q256_target_weight_factorial"
        / "learning_curve_replay"
    )
    worker = (support / "run_learning_curve_replay_worker.sh").read_text(
        encoding="utf-8"
    )
    launcher = (support / "launch_learning_curve_replay.sh").read_text(
        encoding="utf-8"
    )
    assert worker.count("--checkpoint-milestone-kimg=") == 6
    assert "run_arm A 1.0 1.0" in worker
    assert "run_arm B 1.1 1.1" in worker
    assert "run_arm C 1.1 1.0" in worker
    assert "run_arm D 1.0 1.1" in worker
    assert "--duration=1.024" in worker
    assert '--resume="${resume_state}"' in worker
    assert "--transfer=" not in worker
    assert "seed14-18-256to1024-learning-curve-replay-v1" in launcher
    assert "checkpoints=120 evaluation_jobs=240" in launcher
    assert "q256lc_s${seed}" in launcher


def test_learning_curve_recovery_adopts_exact_384_without_retraining() -> None:
    support = (
        ROOT
        / "analysis"
        / "q256_target_weight_factorial"
        / "learning_curve_replay"
    )
    worker = (
        support / "run_learning_curve_replay_recovery_v2_worker.sh"
    ).read_text(encoding="utf-8")
    launcher = (
        support / "launch_learning_curve_replay_recovery_v2.sh"
    ).read_text(encoding="utf-8")
    adopter = (support / "adopt_failed_384_milestone.py").read_text(
        encoding="utf-8"
    )
    assert "adopt_failed_384_milestone.py" in worker
    assert "resume_state=${outdir}/training-state-latest.pt" in worker
    assert "formal_source_state" in worker
    assert "q256lcr2_s${seed}" in launcher
    assert "armA_384_policy" in launcher
    assert "int(state[\"attempted_iteration\"]) != 3000" in adopter
    assert "adopted 384 training state is not byte-identical" in adopter
