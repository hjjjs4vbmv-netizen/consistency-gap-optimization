# Cross-Training-Seed Replication Protocol

## Status and purpose

This protocol extends the completed formal experiment
`gap_lr_matched_q128_s3_v1` along exactly one axis: training seed. Seed 3 is
retained as the existing formal result. Seeds 4 and 5 are new replications,
giving `n=3` training seeds after completion.

The replication does not reopen model, optimizer, learning-rate, gap,
schedule, dataset, transfer-checkpoint, precision, or budget selection. The
source experiment's machine-audited fresh-linearized coefficient remains
frozen at `c0_star=1.2963523762588691`.

## Frozen arms within each new seed

| Arm | Global gap | RAdam learning rate |
| --- | ---: | ---: |
| A | 1.0 | `1.0e-4` |
| B | 1.3 | `1.0e-4` |
| C | 1.3 | `1.2963523762588691e-4` |

Arm A is a required paired baseline. A seed group is incomplete unless A, B,
and C all complete and pass integrity verification.

## Seed-only replication contract

The following settings are inherited byte-for-byte or value-for-value from
the seed-3 formal launcher:

- training implementation `2357bb1d2531a343bdb4397f5a08f4d42a2d135b` for
  `ct_train.py` and `training/`;
- CIFAR-10 archive SHA256
  `a469a9f1b89d43a4a5a0fea42a351b6f107800fc32712881ea3d0ee8cc3a88c1`;
- EDM transfer checkpoint SHA256
  `4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da`;
- `q=128`, `k=8`, `b=1`, `c=0`, global-sigmoid mapping;
- batch 128, per-GPU batch 16, RAdam, dropout 0.2, augmentation disabled;
- `ema_beta=0.9993`, FP16 and AMP enabled, TF32 disabled;
- 256 kimg budget, 32-kimg ticks, numbered snapshots and full training states
  at every tick.

Within one seed, normalized A/B/C options may differ only in run directory,
global gap scale, and learning rate. Across seeds 4 and 5, the same arm may
differ only in run directory and seed.

The initialization convention is the same pretrained transfer checkpoint plus
fresh RAdam and GradScaler state. No arm resumes an already-trained state.

## Machine authorization

The launcher requires an external replication receipt and the immutable
source audit receipt. Its validator fails closed unless:

1. the checkout HEAD equals the replication receipt's protocol commit;
2. the working `ct_train.py` and `training/` match the frozen training commit;
3. the matrix, dataset, transfer checkpoint, and source audit receipt hashes
   match their frozen values;
4. the source receipt still records a passed three-arm mechanism audit and the
   exact frozen `c0_star`;
5. the only authorized new seeds are 4 and 5, with all A/B/C arms present.

The replication receipt extends execution scope only. It does not replace or
reinterpret the independent mechanism audit used by the source experiment.

## Resource and failure policy

One seed group runs at a time. Seed 4 executes A, B, and C sequentially; seed 5
is queued behind the complete seed-4 group and likewise executes A, B, and C.
The launcher stops on the first failure and never retries automatically.

## Required retained evidence

Every run retains:

- all numbered `network-snapshot-*.pkl` files and
  `network-snapshot-latest.pkl` containing final EMA weights;
- all numbered `training-state-*.pt` files and
  `training-state-latest.pt` containing model, RAdam, GradScaler, loss, and
  progress state;
- `training_options.json`, `stats.jsonl`, `train_summary.csv`, `log.txt`, and
  launcher stdout/stderr;
- source commits, input hashes, runtime hardware/software provenance;
- SHA256 values for every numbered/final snapshot and training state and all
  lightweight logs/configuration artifacts;
- per-run and cross-arm integrity receipts proving the allowed-difference
  contract.

## Explicit exclusions

No q expansion, optimizer comparison, additional gap grid, schedule change,
controller change, or budget expansion is authorized by this protocol.
