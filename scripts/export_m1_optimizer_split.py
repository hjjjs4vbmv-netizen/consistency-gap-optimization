"""Export compact diagnostic evidence, excluding checkpoints and host credentials."""
import contextlib
import csv
import io
import json
import runpy
import sys
from pathlib import Path


def main():
    root, destination = map(Path, sys.argv[1:])
    buffer = io.StringIO()
    sys.argv = ['summarize_m1_optimizer_split', str(root)]
    with contextlib.redirect_stdout(buffer):
        runpy.run_module('scripts.summarize_m1_optimizer_split', run_name='__main__')
    summary = json.loads(buffer.getvalue())
    evidence = []
    for s in summary['observed_cells']:
        p = root / f"seed{s['seed']}/{s['arm']}/{s['operation']}"
        cell = dict(summary=s, relative_path=str(p.relative_to(root)))
        for name in ('intervention', 'first_forward', 'first_nonfinite_forward'):
            path = p / f'{name}.json'
            cell[name] = json.loads(path.read_text()) if path.exists() else None
        cell['telemetry'] = list(csv.DictReader((p/'schedule_switch_training_telemetry_v1.csv').open()))
        cell['optimizer_steps'] = [json.loads(line) for line in (p/'optimizer_steps.jsonl').read_text().splitlines()]
        evidence.append(cell)
    if len(evidence) != 32:
        raise ValueError('expected 32 terminal diagnostic cells')
    destination.write_text(json.dumps(dict(analysis_role=summary['analysis_role'], cells=evidence),
                                     ensure_ascii=False, indent=2)+'\n')
    print('Exported',len(evidence),'cells to',destination)


if __name__ == '__main__':
    main()
