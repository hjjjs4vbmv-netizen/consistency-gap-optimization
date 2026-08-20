# q256 formal run record — dcca41b

Created: 2026-08-20 19:16 CST; updated: 2026-08-20 19:55 CST. This is a durable run record, not a new gate or audit framework.

## Source and resume disposition

The planned-pause failure in `exact-resume-25c3d22-A-v1` had a validator-only trigger: `snapshot_grid_size=(16, 16)` contained `numpy.int64` values while the embedded probe accepted only Python `int`. Commit `3f3d204` fixed that stale type check, and the unchanged attempt-16 artifact passed the complete read-only planned-pause revalidation.

The old uninterrupted and paused fresh trajectories nevertheless diverged before resume, first at attempt 4. Model, EMA, and optimizer therefore differed at attempt 32, while GradScaler, RNG, sampler, loss state, counters, and preview state matched. This established a separate training replay issue rather than a missing checkpoint field.

The minimal passing replay configuration was:

- `torch.use_deterministic_algorithms(True)`;
- deterministic cuDNN;
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- `cudnn_benchmark=False`.

Commit `dcca41b` implements that configuration for the q256 factorial path. Two targeted launcher tests passed; the 162-item suite was not rerun.

## Compact resume result

The `dcca41b` 16→32 continuation matches its uninterrupted 32-attempt reference exactly for model, EMA, optimizer, GradScaler, RNG, sampler, loss state, control/counters, preview state, and all 32 rows of computational telemetry.

- Uninterrupted: `/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819/smoke/exact-resume-dcca41b-uninterrupted`
- Resumed: `/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819/smoke/exact-resume-dcca41b-resumed`
- Pause state SHA256: `bb49a42eed441fd62725a131404d240115f34d82208630e96bfff7f7c21bcbfe`
- Computational telemetry SHA256: `6aa942ba61d57aaf7e0f893aab9f610768f95146cbdef16a465504cadfcb0397`

All exact component digests and file digests are in `formal_run_record_dcca41b.json`.

## Formal launch

The fresh 12-cell queues started at 2026-08-20 16:35:23 CST from clean source commit `dcca41b19e7c45512b5fbe98776520396a1bf9ac`.

- Root: `/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819/formal/formal-direct-dcca41b-deterministic-v1`
- GPU0 queue: seed3 A/B/C/D, then seed5 A/B/C/D
- GPU1 queue: seed4 A/B/C/D
- tmux: `q256_dcca_gpu0`, `q256_dcca_gpu1`
- Launcher SHA256: `41c0f96108d128d54164ce4c163eed985926bd857c36dccfbee25dc58230eda5`

The queue is direct rather than the canonical receipt-producing launcher. It does not create per-cell `launch_manifest.json` or `runner_completion.json`. The source head is bound in both worker logs and the JSON record; each cell still writes training options, initial-state receipt, telemetry, checkpoints, and network snapshots.

Two pre-gate A cells under the older nondeterministic source were stopped and remain preserved under `formal-direct-25c3d22-resume-override-v1`. No later cells in that queue started.

## Snapshot at 2026-08-20 19:13:38 CST

| Cell | Status | Attempts | Accepted | kimg | AMP skips |
|---|---:|---:|---:|---:|---|
| seed3/A | complete | 2000 | 1990 | 256.0 | 1–8, 12, 1317 |
| seed3/B | complete | 2000 | 1990 | 256.0 | 1–8, 11, 1326 |
| seed3/C | running | 1234 | 1225 | 157.952 | 1–8, 11 so far |
| seed3/D | not started | — | — | — | — |
| seed4/A | complete | 2000 | 1990 | 256.0 | 1–8, 11, 956 |
| seed4/B | complete | 2000 | 1991 | 256.0 | 1–8, 10 |
| seed4/C | running | 1229 | 1219 | 157.312 | 1–8, 11, 956 so far |
| seed4/D | not started | — | — | — | — |
| seed5/A–D | not started | — | — | — | — |

There were no loss, sanitized-gradient, update, model, EMA, factorial, or denominator non-finite counts, and every raw-gradient non-finite event matched an AMP skip.

Open observation: seed4/A has 10 skips and 1990 accepted updates, while seed4/B has 9 skips and 1991 accepted updates. This one-update difference is recorded as a potential strict within-seed endpoint-comparability issue. It did not trigger an AMP mechanism audit or interrupt the running queue; it must be reassessed after seed4 C/D complete.

## Infrastructure stop detected at 2026-08-20 19:55:38 CST

Both formal workers stopped at approximately 19:37 CST while running their C cells. The only formal-log errors are DataLoader worker `Bus error` failures. At detection, `/dev/shm` was 100% full: 252 GiB used with only 8.1 MiB available; inode use was 1%. `/data/raw` still had 33 TiB available and `/data/temp` had 5.0 TiB available. Both GPUs were idle, no training PID remained, and the two formal tmux sessions were gone.

| Cell | Current status | Last attempt | Accepted | kimg | Remaining | AMP skips |
|---|---:|---:|---:|---:|---:|---|
| seed3/C | failed/incomplete | 1984 | 1974 | 253.952 | 16 attempts / 2.048 kimg | 1–8, 11, 1418 |
| seed4/C | failed/incomplete | 1984 | 1974 | 253.952 | 16 attempts / 2.048 kimg | 1–8, 11, 956 |

The latest persisted C checkpoints are the tick-20 checkpoints at approximately 202.4 kimg; final artifacts are absent. Seed3/4 A and B remain complete. Seed3/4 D and all four seed5 cells never started. Current matrix count: 4 complete, 2 incomplete, 6 not started.

This is classified as infrastructure shared-memory exhaustion, not an AMP, loss, CUDA, model-state, or resume-state failure. Detection was read-only: no process, artifact, shared-memory object, or queue was modified, and recovery has not been started.
