import csv,json,math,statistics,unittest
from pathlib import Path
from scipy import stats
from analysis.q256_history_component_chase_v1.results.reproduce import compare
from analysis.q256_history_component_chase_v1 import analyze
ROOT=Path(__file__).resolve().parents[1]/'analysis/q256_history_component_chase_v1/results'

class PublishedComponentResults(unittest.TestCase):
    def test_complete_roster_and_original_controls(self):
        rows=json.loads((ROOT/'quality_slots_320.json').read_text());mapping=analyze.validate_rows(rows)
        self.assertEqual(len(mapping),320)
        self.assertEqual(sum(r['status']=='PASS' for r in rows),300)
        self.assertEqual({(r['seed'],r['path']) for r in rows if r['status']=='NO_ENDPOINT'}, {(58,'AA'),(58,'BA'),(65,'AA'),(58,'CA')})
        result=analyze.summarize(rows)
        self.assertEqual(result['complete_four_path_seeds'],list(range(50,58))+list(range(59,65)))
        compare(json.loads(json.dumps(result)),json.loads((ROOT/'statistics.json').read_text()))

    def test_independent_seed_arithmetic_and_holm(self):
        rows=json.loads((ROOT/'quality_slots_320.json').read_text());saved=json.loads((ROOT/'statistics.json').read_text())
        table={(r['seed'],r['path'],r['readout'],r['block']):r for r in rows};values={k:[] for k in ('H_T','H_W','I')}
        for seed in saved['complete_four_path_seeds']:
            y={p:sum(math.log(table[seed,p,'E_512',b]['FID']) for b in ('B0','B1','B2'))/3 for p in ('AA','BA','CA','DA')}
            values['H_T'].append(y['CA']-y['AA']);values['H_W'].append(y['DA']-y['AA']);values['I'].append(y['BA']-y['CA']-y['DA']+y['AA'])
        raw={}
        for k,v in values.items():
            mean=sum(v)/14;sd=math.sqrt(sum((x-mean)**2 for x in v)/13);se=sd/math.sqrt(14)
            raw[k]=2*stats.t.sf(abs(mean/se),13)
            self.assertAlmostEqual(mean,saved['primary'][k]['mean'],places=12)
            self.assertAlmostEqual(sd,saved['primary'][k]['sd'],places=12)
            self.assertAlmostEqual(raw[k],saved['primary'][k]['p_raw'],places=12)
        ordered=sorted(raw,key=raw.get);running=0
        for i,k in enumerate(ordered):
            running=max(running,min(1,(3-i)*raw[k]))
            self.assertAlmostEqual(running,saved['primary'][k]['p_holm'],places=12)

    def test_failure_attempt_and_cost_accounting(self):
        with (ROOT/'training_trajectories.csv').open() as f:rows=list(csv.DictReader(f))
        failed=[r for r in rows if r['status']!='PASS'];self.assertEqual(len(failed),1)
        self.assertEqual((failed[0]['seed'],failed[0]['path'],failed[0]['attempts'],failed[0]['successful_updates']),('58','CA','7592','7576'))
        cost=json.loads((ROOT/'cost_summary.json').read_text())
        self.assertEqual(sum(int(r['attempts']) for r in rows),255592)
        self.assertEqual(sum(int(r['skips']) for r in rows),407)
        self.assertAlmostEqual(sum(float(r['process_gpuh']) for r in rows),cost['training_process_gpuh'],places=10)
        self.assertLess(cost['total_recorded_new_process_gpuh'],200)

if __name__=='__main__':unittest.main()
