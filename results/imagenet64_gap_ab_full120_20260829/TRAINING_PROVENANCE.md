# ImageNet-64 training provenance

This note records the external training-state evidence behind the full-120
evaluation. It is a compact transcription of the archived checkpoints,
receipts, logs, and final training summaries; Git remains the integrity record
for the files in this repository.

## Provenance conclusion

All six trajectories have a verified full-state parent at the resume boundary
and a verified full-state archive point at 10,240 kimg. The three runs in group
A resumed from 7,680 kimg; the three runs in group B resumed from 8,960 kimg.
Every parent load is recorded in the archived training log.

The twelve inspected states use schema `ect.exact-training-state/v1` and
contain the network, EMA, optimizer, PowerEMA, loss-schedule state, progress
counters, snapshot grid, and ordered per-rank RNG and sampler states. Each
state has two rank records. At the parent boundary and at 10,240 kimg, each
rank's sampler consumption equals `cur_nimg / 2` exactly. AMP was disabled, so
the absence of a GradScaler state is expected.

`training_provenance.csv` gives the file identity, receipt SHA-256, serialized
counters, schedule state, and state-presence checks for all twelve files.

## Paired configuration equality

The IA and IB full states for each seed were compared twice: at the resume
parent and at 10,240 kimg. After flattening the serialized
`trajectory_config`, the only differing field in all six comparisons was:

| field | IA | IB |
|---|---:|---:|
| `loss_kwargs.global_gap_scale` | 1.0 | 1.1 |

The archived `training_options.json` files also differ in `run_dir`, which is
output bookkeeping rather than a scientific setting. Batch size 128,
per-GPU batch 32, two ranks, one data-loader worker, FP32, TF32 disabled,
cuDNN benchmark disabled, global-batch mean, dataset, seed pairing, stopping
budget, schedule, optimizer, model, and checkpoint milestones otherwise
match within every IA/IB pair.

The transfer source is the same archived
`edm2-img64-s-1073741-0.075.pkl` for all six trajectories, with SHA-256
`044f4aeb819b2a4164b796a5f4944c4fadca5d789a13b7a72ddc079a7c7fc434`.
The archived training source snapshot is `b07effd`.

## Resume and endpoint continuity

The serialized parent counters agree with the intended resume boundaries:

| run group | trajectories | parent kimg | attempted/successful steps | sampler samples per rank |
|---|---|---:|---:|---:|
| A | seed101-IB, seed103-IA, seed103-IB | 7,680 | 60,000 / 60,000 | 3,840,000 |
| B | seed101-IA, seed102-IA, seed102-IB | 8,960 | 70,000 / 70,000 | 4,480,000 |

For every trajectory, the parent and 10,240-kimg states have the same
`trajectory_config_sha256`. The 10,240-kimg state records 80,000 attempted
iterations, 80,000 successful optimizer steps, 10,240,000 processed images,
and 5,120,000 sampler samples per rank.

The checksum-verified final metadata bundle contains a 100,000-row training
summary for each trajectory. All six summaries are sequential from attempt 1
through 100,000, end at exactly 12,800,000 processed images, and contain zero
skipped steps and zero non-finite loss rows. Thus the seed103 degradation after
7,680 kimg occurs after distinct, correctly paired exact-resume states; the
provenance does not support attributing it to cross-seed checkpoint mixing or
an IA/IB configuration mismatch.

## Receipt basis and scope

- The six parent states and their receipt files passed the private pull's
  637-entry exact-size and SHA-256 verification (`sha256_mismatch=0`).
- The six 10,240-kimg states passed their dedicated size and SHA-256 receipts.
- The final metadata bundle, including logs, options, and final summaries,
  passed its 56-file SHA-256 verification.
- 12,800-kimg full training states were not retained. The latest archived
  exact-resume point is 10,240 kimg; 12,800 kimg is represented by compact
  PowerEMA 0.050 checkpoints and the final metadata.

The hashes here identify external artifacts that are not stored in Git. They
are not a second signing or versioning system.
