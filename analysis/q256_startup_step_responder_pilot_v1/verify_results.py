"""Independent checks of the frozen roster, outcome provenance, and paired arithmetic."""
import argparse,csv,json,math
from pathlib import Path
from . import protocol as p


def verify(root):
    matrix=json.loads((root/'training_matrix_frozen.json').read_text());outcomes=matrix['outcomes']
    assert len(outcomes)==16 and {(x['seed'],x['arm']) for x in outcomes}=={(s,a) for s in p.SEEDS for a in p.ARMS}
    assert all(x['status'] in p.SPEC['terminal_statuses'] for x in outcomes)
    slots=json.loads((root/'evaluation_slots.json').read_text())
    assert len(slots)==48 and {(x['seed'],x['arm'],x['block']) for x in slots}=={(s,a,b) for s in p.SEEDS for a in p.ARMS for b in ('B0','B1','B2')}
    for x in slots:
        assert x['readout']=='E_512'
        if x['status']=='PASS':
            assert x['evaluator_commit']==p.EVALUATOR_COMMIT and x['precision']=='fp32' and x['nfe']==1
            assert x.get('feature_sha256') and x.get('checkpoint_sha256') and math.isfinite(x['FID']) and x['FID']>0
    rows=list(csv.DictReader((root/'per-seed.csv').open()));assert [int(x['seed']) for x in rows]==list(p.SEEDS)
    for r in rows:
        if r['finite_paired']=='True':
            ys={a:sum(math.log(float(r[a+'_'+b+'_FID'])) for b in ('B0','B1','B2'))/3 for a in ('AA','DA')+p.ARMS}
            c=ys[p.ARMS[0]]-ys['DA'];m=ys[p.ARMS[1]]-ys['AA']
            assert math.isclose(float(r['C']),c,abs_tol=1e-12) and math.isclose(float(r['M']),m,abs_tol=1e-12)
            assert math.isclose(float(r['S']),(c-m)/2,abs_tol=1e-12)
    return dict(status='PASS',planned_seeds=8,training_outcomes=16,evaluation_slots=48)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);x=a.parse_args();print(json.dumps(verify(x.root)))
