"""Summarize bounded diagnostic cells without statistical population inference."""

import argparse
import csv
import json
from pathlib import Path

import torch


def first_actual(rows):
    return next((r for r in rows if r['step_skipped'] == '0'), None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    args = p.parse_args()
    reports = []
    for path in sorted(args.root.glob('seed*/*/*/status.json')):
        folder = path.parent
        report = json.loads(path.read_text())
        report.pop('command')
        telemetry = folder / 'schedule_switch_training_telemetry_v1.csv'
        if not telemetry.exists():
            reports.append(report)
            continue
        rows = list(csv.DictReader(telemetry.open()))
        if not rows:
            reports.append(report)
            continue
        first = first_actual(rows)
        report.update(observed_attempts=len(rows), last_observed_attempt=int(rows[-1]['attempted_iteration']),
                      first_actual_attempt=int(first['attempted_iteration']) if first else None,
                      first_actual_update_norm=float(first['update_norm']) if first else None,
                      max_update_norm=max(float(r['update_norm']) for r in rows),
                      skipped_attempts=sum(int(r['step_skipped']) for r in rows))
        report['first_nonfinite'] = next((dict(attempt=int(r['attempted_iteration']),
                                              fields=[k for k in r if k.endswith('nonfinite_count') and int(r[k])])
                                         for r in rows if any(int(r[k]) for k in r if k.endswith('nonfinite_count'))), None)
        reference = folder.parent / 'K'
        reference_telemetry = reference / telemetry.name
        if first and reference_telemetry.exists() and (reference / 'first_actual_update.pt').exists():
            reference_rows = list(csv.DictReader(reference_telemetry.open()))
            ref = first_actual(reference_rows)
            crn_fields = [k for k in first if k.endswith('sha256')]
            matched = bool(ref and first['attempted_iteration'] == ref['attempted_iteration'] and
                           all(first[k] == ref[k] for k in crn_fields))
            report['first_update_inputs_match_K'] = matched
            if matched:
                updates = torch.load(folder / 'first_actual_update.pt', map_location='cpu', weights_only=True)
                baseline = torch.load(reference / 'first_actual_update.pt', map_location='cpu', weights_only=True)
                if len(updates) != len(baseline):
                    raise ValueError('parameter alignment mismatch')
                dot = sum(float((a.double()*b.double()).sum()) for a, b in zip(updates, baseline))
                a2 = sum(float(a.double().square().sum()) for a in updates)
                b2 = sum(float(b.double().square().sum()) for b in baseline)
                report['first_update_cosine_to_K'] = dot/(a2*b2)**.5 if a2*b2 else None
                report['first_update_norm_ratio_to_K'] = (a2/b2)**.5 if b2 else None
        reports.append(report)
    print(json.dumps(dict(analysis_role='POST_OUTCOME_OPTIMIZER_DIAGNOSTIC',
                          planned_cells=32, observed_cells=reports), indent=2))


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
