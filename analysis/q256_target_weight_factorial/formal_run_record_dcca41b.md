# q256 formal run record — dcca41b

Created: 2026-08-20 19:16 CST; updated: 2026-08-20 22:40 CST. This is a durable run record, not a new gate or audit framework.

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

This is classified as infrastructure shared-memory exhaustion, not an AMP, loss, CUDA, model-state, or resume-state failure. The initial detection was read-only: no process, artifact, shared-memory object, or queue was modified at that stage.

## Recovery started at 2026-08-20 20:04:35 CST

The global shared-memory occupants were not owned by this experiment: approximately 224 GiB under `/dev/shm/OmniDance` belonged to `niezichao`, and approximately 11 GiB under `/dev/shm/cuisen_suny_ray` belonged to `cuisen`. No global shared-memory object was deleted or modified.

Each GPU worker now binds its own ECT001-owned mode-700 directory to the container's `/dev/shm`. Both private directories passed a disposable 64 MiB allocation test. Workers, sampler configuration, deterministic settings, and source commit remain unchanged.

Before resume, the complete failed CSV/log/stats files were copied into per-cell `incident-20260820T1937-shm-bus-error` directories and verified byte-for-byte by SHA-256. The active train summary and factorial telemetry were then aligned to the authoritative checkpoint at attempt 1581, accepted update 1571, `202.368 kimg`, tick 21; active stats were aligned to the same tick. Exact hashes are in the JSON record.

Both C cells passed strict-resume configuration checks and produced new telemetry:

| Cell | Resume PID | Checkpoint | Observed progress at 20:05:27 CST |
|---|---:|---:|---:|
| seed3/C | 112042 | attempt 1581 / 202.368 kimg | attempt 1600 / 204.800 kimg |
| seed4/C | 112062 | attempt 1581 / 202.368 kimg | attempt 1601 / 204.928 kimg |

The remaining queues are seed3/C → seed3/D → seed5 A/B/C/D on GPU0 and seed4/C → seed4/D on GPU1. Recovery tmux sessions are `q256_dcca_recovery_gpu0` and `q256_dcca_recovery_gpu1`. A five-minute heartbeat monitor is active under automation ID `q256-formal-runs`; it checks processes, GPUs, cell progress, artifacts, errors, shared memory, and disk, and is authorized to apply only minimal experiment-scoped fixes.

## Recovery completion observed at 2026-08-20 20:18:25 CST

Both resumed C cells reached 2000 attempts, 1990 accepted updates, and exactly 256.000 kimg. Their final state, network snapshot, telemetry, initial receipt, and final image are present; both recovery workers emitted explicit PASS lines. Final artifact hashes are recorded in the JSON record.

| Cell | Final attempts | Accepted | kimg | AMP skips |
|---|---:|---:|---:|---|
| seed3/C | 2000 | 1990 | 256.000 | 1–8, 11, 1418 |
| seed4/C | 2000 | 1990 | 256.000 | 1–8, 11, 956 |

The queues automatically advanced without intervention. At this milestone, seed3/D was at attempt 13 (`1.664 kimg`, PID 834) and seed4/D was at attempt 17 (`2.176 kimg`, PID 114544). No new Traceback, bus error, OOM, CUDA error, or semantic non-finite event was present.

## Seed3/4 matrix completion observed at 2026-08-20 21:19:05 CST

Seed3/D and seed4/D both completed with 2000 attempts, 1990 accepted updates, and exactly 256.000 kimg. Their final artifacts are present, both workers emitted PASS, and GPU1 emitted `WORKER_PASS` before becoming idle. Artifact hashes are recorded in the JSON record.

| Cell | Attempts | Accepted | kimg | AMP skips |
|---|---:|---:|---:|---|
| seed3/D | 2000 | 1990 | 256.000 | 1–8, 12, 1317 |
| seed4/D | 2000 | 1990 | 256.000 | 1–8, 10, 979 |

Eight of twelve formal cells are now complete. GPU0 automatically started seed5/A; at this milestone it was at attempt 10 (`1.280 kimg`, PID 9144). GPU1 is expectedly idle because its assigned seed4 queue is complete.

Final seed4 accepted-update counts are A=1990, B=1991, C=1990, D=1990. The one-update B difference remains recorded only as an endpoint-analysis covariate; it did not trigger an AMP mechanism audit, interrupt the queue, or add replacement runs.

## Seed5/A completion observed at 2026-08-20 22:40:50 CST

Seed5/A completed with 2000 attempts, 1990 accepted updates, and exactly 256.000 kimg. Its final state, network snapshot, telemetry, initial-state receipt, and final image are present, and the recovery worker emitted an explicit PASS line. The AMP skips were attempts 1–8, 13, and 14; no semantic non-finite event or raw-gradient/skip inconsistency was observed. Artifact hashes are recorded in the JSON record.

Nine of twelve formal cells are now complete. The queue automatically advanced to seed5/B; at this milestone it was at attempt 642, accepted update 633, and `82.176 kimg` (PID 26095). GPU0 was active at 30% utilization with 3549 MiB allocated; GPU1 remained expectedly idle. No new Traceback, bus error, OOM, CUDA error, or semantic non-finite event was present. The private GPU0 shared-memory directory used 44 KiB, `/data/raw` had 33 TiB available, and `/data/temp` had 5.0 TiB available.
