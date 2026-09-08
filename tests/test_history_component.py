import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from training import history_component as hc, m1, schedule_switch as switch
from training.ct_training_loop import validate_planned_pause, strict_attempt_invariant_failures
from training.loss import resolve_target_weight_factorial, compute_target_weight_times
from analysis.q256_history_component_chase_v1 import protocol as p, analyze, worker
from tests import test_m1_training_state as fixtures


class HistoryComponentTests(unittest.TestCase):
    def fixture(self, directory, arm="C"):
        manifest = p.manifest(50, arm+"A", Path(directory)/"source.pt", Path(directory)/"suffix")
        source = fixtures.M1TrainingStateTests().source_state()
        source["factorial"].update(arm=arm, target_gap_scale=p.FACTORS[arm][0], denominator_gap_scale=p.FACTORS[arm][1])
        source["history_component_prefix"] = {"protocol_id":p.EXPERIMENT_ID,"seed":50,"source_history":arm,"total_kimg":1024}
        branch = copy.deepcopy(source)
        branch["ema_512"] = hc.initialize_ema_512(source["net"])
        branch["schedule_switch"] = switch.state_metadata(manifest)
        branch["history_component"] = hc.initial_metadata(manifest,0,3999)
        return manifest, source, branch

    def test_factors_are_actual_realized_times_not_scalar_loss(self):
        t, r = torch.tensor([1., .0021]), torch.tensor([.8, .002])
        c = compute_target_weight_times(t,r,target_gap_scale=1.1,denominator_gap_scale=1.)
        d = compute_target_weight_times(t,r,target_gap_scale=1.,denominator_gap_scale=1.1)
        self.assertTrue(torch.equal(c[1], r))
        self.assertTrue(torch.equal(d[0], r))
        self.assertTrue(torch.equal(c[0],d[1]))
        for arm in ("C","D"):
            factor = resolve_target_weight_factorial("q256_target_weight_v1", *p.FACTORS[arm], adj="sigmoid", global_gap_scale=1.,q=256,c=0)
            self.assertEqual(factor["arm"],arm)

    def test_formal_manifest_rejects_old_sources_and_outside_roster(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, source, _ = self.fixture(d)
            path = Path(d)/"manifest.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(switch.load_run_manifest(path),manifest)
            switch.verify_source_state(source,manifest)
            del source["history_component_prefix"]
            with self.assertRaisesRegex(RuntimeError,"own new C/D prefix"):
                switch.verify_source_state(source,manifest)
            manifest["seed"] = 66
            path.write_text(json.dumps(manifest))
            with self.assertRaises(RuntimeError): switch.load_run_manifest(path)

    def test_4000_pause_plan_is_not_half_duration_and_does_not_allow_4001(self):
        kwargs = dict(stop_after_attempts=4000,planned_pause_protocol=p.EXPERIMENT_ID,
            strict_reproducibility=True,seed=50,total_kimg=1024,resume_state_dump=None,schedule_switch_manifest=None)
        self.assertEqual(validate_planned_pause(**kwargs),4000)
        for changes in ({"stop_after_attempts":4001},{"total_kimg":512},{"seed":66}):
            with self.assertRaises(ValueError): validate_planned_pause(**{**kwargs,**changes})
        self.assertEqual(validate_planned_pause(**{**kwargs,"resume_state_dump":"own-checkpoint.pt"}),4000)

    def test_keep_optimizer_rng_buffers_and_once_only_ema(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, source, branch = self.fixture(d)
            rng = torch.get_rng_state().clone()
            other_net = copy.deepcopy(source["net"])
            optimizer = torch.optim.RAdam(other_net.parameters(),lr=1e-4)
            for v in other_net.parameters(): v.grad = torch.ones_like(v)
            optimizer.step()
            old = copy.deepcopy(optimizer.state_dict())
            self.assertEqual(hc.apply_optimizer_intervention(optimizer,"CA"),0)
            self.assertTrue(m1._equal_state(old, optimizer.state_dict()))
            self.assertTrue(torch.equal(torch.get_rng_state(),rng))
            self.assertTrue(hc.validate_branch_init_against_source(branch,source,manifest))
            out = hc.save_branch_init_state(branch,d)
            restored = torch.load(out,weights_only=False)
            hc.validate_resumed_state(restored,manifest)
            self.assertTrue(m1._equal_state(restored["ema_512"].state_dict(),source["net"].state_dict()))
            with self.assertRaises(FileExistsError): hc.save_branch_init_state(branch,d)
            restored["history_component"]["ema_512_init_count"] = 2
            with self.assertRaisesRegex(RuntimeError,"ema_512_init_count"):
                hc.validate_resumed_state(restored,manifest)
            self.assertNotIn("m1",branch)

    def test_late_amp_skip_allowed_but_nonfinite_loss_is_not_managed(self):
        inputs = dict(sample_count=128,batch_size=128,consumed_samples=700032,processed_nimg=700032,
            loss_nonfinite_count=0,raw_grad_nonfinite_count=3,sanitized_grad_nonfinite_count=0,
            update_nonfinite_count=0,model_nonfinite_count=0,ema_nonfinite_count=0,
            factor_nonfinite_count=0,nonpositive_denominator_count=0,step_skipped=1,update_norm=0)
        self.assertEqual(strict_attempt_invariant_failures(**inputs),([],0,False))
        inputs["loss_nonfinite_count"] = 1
        failures, _, managed = strict_attempt_invariant_failures(**inputs)
        self.assertIn("non-finite loss",failures)
        self.assertFalse(managed)

    def test_exact_matrix_and_original_only(self):
        rows = p.training_queue()
        self.assertEqual(len(rows),32)
        for seed in p.SEEDS:
            paths = [r["path"] for r in rows if r["seed"]==seed]
            self.assertEqual(paths,["CA","DA"] if seed%2==0 else ["DA","CA"])
        self.assertEqual(len(p.evaluation_slots()),160)
        old = p.import_old_controls()
        self.assertEqual(sum(r["status"]=="PASS" for r in old),145)
        self.assertEqual({r["original_branch"] for r in old},{"K_A","K_B"})
        self.assertEqual(analyze.summarize(old+p.evaluation_slots())["status"],"INCOMPLETE_TECHNICAL")

    def complete_rows(self):
        new = p.evaluation_slots()
        for r in new:
            r.update(status="PASS",FID=5.+r["seed"]*.07+(0 if r["path"]=="CA" else 1)+int(r["block"][1])*2,KID=-.001)
        return p.import_old_controls()+new

    def test_same_set_log_before_mean_and_closure(self):
        result = analyze.summarize(self.complete_rows())
        self.assertEqual(result["complete_four_path_seeds"],[s for s in p.SEEDS if s not in (58,65)])
        row = result["per_seed"][0]
        expected = sum(math.log(5+50*.07+b*2) for b in range(3))/3
        self.assertAlmostEqual(row["CA"],expected)
        self.assertNotAlmostEqual(row["CA"],math.log(5+50*.07+2),places=4)
        for row in result["per_seed"]:
            self.assertAlmostEqual(row["H_T"]+row["H_W"]+row["I"],row["H_J"])
        self.assertEqual({v["n"] for v in result["primary"].values()},{14})

    def test_holm_and_degenerate_status(self):
        self.assertEqual(analyze.holm_three([.01,.04,.03]),[.03,.06,.06])
        self.assertEqual(analyze.holm_three([.01,None,.04]),[.03,None,.08])
        self.assertIsNone(analyze.interval([1,1])["p_raw"])
        self.assertIsNone(analyze.interval([1])["p_raw"])

    def test_no_endpoint_imputation_and_technical_pending(self):
        rows = self.complete_rows()
        missing = next(r for r in rows if r["status"]=="NO_ENDPOINT")
        missing["FID"] = 10
        with self.assertRaises(ValueError): analyze.summarize(rows)
        rows = self.complete_rows()
        rows[-1]["status"] = "TECHNICAL_FAILURE"
        self.assertEqual(analyze.summarize(rows)["primary"],{})

    def test_completed_job_is_not_executed_again(self):
        with tempfile.TemporaryDirectory() as d:
            row = p.training_queue()[0]
            directory = Path(d)/row["slot"]/row["path"]/"prefix"
            directory.mkdir(parents=True)
            (directory/"stage_status.json").write_text('{"status":"PASS"}')
            with patch.object(worker.subprocess,"Popen",side_effect=AssertionError("duplicate process")):
                self.assertEqual(worker.run_stage({"output_root":d},row,"prefix",0)["status"],"PASS")

    def test_budget_pause_uses_cost_and_keeps_matrix(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"cost.json"
            self.assertTrue(worker.budget_admit(path,"S01/CA/prefix"))
            worker.budget_finish(path,"S01/CA/prefix",{"status":"PASS","process_gpuh":4.,"processed_attempts":4000,"end_attempt":4000})
            self.assertFalse(worker.budget_admit(path,"S01/CA/suffix"))
            self.assertEqual(json.loads(path.read_text())["status"],"INCOMPLETE_BUDGET")
            self.assertEqual(len(p.training_queue()),32)


if __name__ == "__main__": unittest.main()
