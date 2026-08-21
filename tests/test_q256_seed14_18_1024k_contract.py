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
