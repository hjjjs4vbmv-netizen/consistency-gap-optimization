import copy, hashlib, json, tempfile, unittest
from pathlib import Path
import torch
from training import reproducibility, schedule_switch

class ParityContractTest(unittest.TestCase):
    def state(self, arm="A", seed=999):
        net=torch.nn.Linear(2,2); ema=copy.deepcopy(net)
        traj={"seed":seed,"total_kimg":1024,"batch_size":128,"loss_kwargs":{"factorial_protocol":"q128_matched_spacing_v1","target_gap_scale":1.0 if arm=="A" else 1.1,"denominator_gap_scale":1.0 if arm=="A" else 1.1}}
        return {"net":net,"ema":ema,"optimizer_state":{"state":{},"param_groups":[]},"gradscaler_state":{},"attempted_iteration":4000,"successful_optimizer_steps":4000,"cur_nimg":512000,"rank_states":[{"rng_state":{},"sampler_state":{"consumed_samples":512000}}],"factorial":{"enabled":True,"protocol":"q128_matched_spacing_v1","arm":arm,"target_gap_scale":1.0 if arm=="A" else 1.1,"denominator_gap_scale":1.0 if arm=="A" else 1.1},"trajectory_config":traj,"trajectory_config_sha256":reproducibility.state_sha256(traj)}
    def manifest(self,path,state,branch,origin,continuation):
        return {"schema":schedule_switch.Q128_RUN_MANIFEST_SCHEMA,"experiment_protocol":schedule_switch.Q128_FRESH_PROTOCOL,"run_kind":"parity","branch":branch,"seed":999,"origin_arm":origin,"continuation_arm":continuation,"switch_kimg":512,"final_kimg":640,"protocol_sha256":"1"*64,"implementation_commit":"2"*40,"source_checkpoint_manifest_sha256":"3"*64,"source_state":{"path":str(path.resolve()),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"internal_state_sha256":schedule_switch.internal_state_hashes(state)}}
    def test_a_and_b_noop_manifests_validate_and_bind_all_state(self):
        for arm,branch in (("A","A_to_A"),("Bsame","Bsame_to_Bsame")):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as d:
                f=Path(d)/"state.pt"; f.write_bytes(b"immutable"); state=self.state(arm)
                m=self.manifest(f,state,branch,arm,arm); p=Path(d)/"m.json"; p.write_text(json.dumps(m))
                loaded=schedule_switch.load_run_manifest(p)
                self.assertEqual(schedule_switch.continuation_factorial(loaded)["protocol"],"q128_matched_spacing_v1")
                self.assertEqual(schedule_switch.verify_source_state(state,loaded),m["source_state"]["internal_state_sha256"])

if __name__=="__main__": unittest.main()
