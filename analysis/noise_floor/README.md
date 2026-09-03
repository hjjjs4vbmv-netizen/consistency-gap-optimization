# Generation-block sensitivity

This post-seal analysis covers 16 prespecified checkpoints with five disjoint
50k generation blocks (`B1`–`B5`) at NFE 1 and 2. The historical `B0` values
are anchors only and are excluded from every mean and SD. The initial 13
accessible checkpoints completed 130/130 jobs. The three `BA@1024` checkpoints
remain pending an administrator-mediated copy from `/root`; after their hashes
match the manifest, the same runner resumes the existing output and executes
only their 30 missing jobs.

Published summaries are grouped under `results/noise_floor/`:

- `checkpoint_matrix/` contains the completed 130-job evaluation of the 13
  accessible checkpoints from this manifest;
- `n30_companion/` contains a separate 60-job evaluation of the mechanically
  selected seed-50, seed-51 and seed-52 AA/BA pairs from the n=30 companion
  cohort. Its manifest is `n30_companion.json`.

The companion cohort does not replace NF-02, NF-04 or NF-12. It estimates the
generation-block sensitivity of the same BA-minus-AA contrast on checkpoints
that were directly available from the companion handoff.

The eligible contrasts are exactly:

- q256 seed-5 `B-A@512`;
- q256 seed-3/4/5 `BA-AA@1024`;
- q256 target-weight `B-D@256` and `B-D@1024`;
- q128 seed-3 `Bsame-A@1024`;
- q128 seed-3 `Cmatch-Bmatch@1024`.

The two sign-change summaries are the matched-block target-weight `B-D`
256-to-1024 comparison and the seed-5 `B-A@512` to `BA-AA@1024` delayed-reversal
check. FID is analyzed on the natural-log scale; KID remains on its raw scale.
`2SD` is a descriptive generation-block variation scale; contrast rows compute
it from paired block differences. It is not a confidence limit, equivalence
bound, or `TIE` rule.

## Completed companion cohort

The first three complete AA/BA pairs in ascending seed order were fixed as
seeds 50, 51 and 52 before their generation-block results were examined. All
60 jobs passed. At NFE1, BA-minus-AA log FID was negative in all five blocks for
all three seeds. At NFE2 it was negative in all five blocks for seeds 50 and 51;
seed 52 changed sign across blocks and its mean magnitude was below its paired
`2SD` scale. The seed-52 NFE2 ordering is therefore reported as `not
interpreted`, not as a winner.

These checkpoints were serialized with a NumPy module path absent from the
frozen evaluator image. `sitecustomize.py` restores that import alias only; it
does not modify the checkpoint or evaluator. Use it for the companion run:

```text
export SINGULARITYENV_PYTHONPATH="$PWD/analysis/noise_floor"
python3 -m analysis.noise_floor.run_matrix --manifest analysis/noise_floor/n30_companion.json --mode canary --gpus 0,1
python3 -m analysis.noise_floor.run_matrix --manifest analysis/noise_floor/n30_companion.json --mode run --gpus 0,1
python3 -m analysis.noise_floor.analyze --manifest analysis/noise_floor/n30_companion.json --outdir /data/raw/ECT/q256_n30_companion/generation_blocks/analysis
```

## Protocol amendment: no pooled TIE rule

TASK 7 estimates checkpoint- and contrast-specific generation-block variation
only. No threshold derived from `2SD` or from the selected checkpoints may be
transferred to other winner-table cells, used as a TIE or equivalence rule, or
used to relabel B0 results. Frozen inference is unchanged.

## Administrator transfer required before the supplement

Copy the three original files without substituting another snapshot:

| ID | Source | Destination | Expected SHA256 |
|---|---|---|---|
| NF-02 | `/root/q256_switch_seed3_7_v3_formal_20260830_v1/seed3/B_to_A/kimg1024/network-snapshot.pkl` | `/data/raw/ECT/ect_runs/noise_floor/checkpoints/seed3/B_to_A/kimg1024/network-snapshot.pkl` | `e0f8a365f3e9aedd0547e3e2dc777869286f0e61599da28da913020eb436f6d5` |
| NF-04 | `/root/q256_switch_seed3_7_v3_formal_20260830_v1/seed4/B_to_A/kimg1024/network-snapshot.pkl` | `/data/raw/ECT/ect_runs/noise_floor/checkpoints/seed4/B_to_A/kimg1024/network-snapshot.pkl` | `126a408c5cac08bddc528d7e7b167fab7434619554fa8d01f85025fdd04ecaa7` |
| NF-12 | `/root/q256_switch_seed3_7_v3_formal_20260830_v1/seed5/B_to_A/kimg1024/network-snapshot.pkl` | `/data/raw/ECT/ect_runs/noise_floor/checkpoints/seed5/B_to_A/kimg1024/network-snapshot.pkl` | `325924ec48fd1381bbfd4af5ef0c9644b2158511979976e14f4ffeb46e0aa611` |

The administrator should create the destination directory for `ECT002`, copy
the files, make them readable by `ECT002`, and run `sha256sum` on all three
destinations. The supplement must not start unless all hashes match this table.

For q256 replay controls, the archived B0 evaluation pickle and the accessible
server snapshot have different file hashes but the same canonical EMA hash.
The runner validates the accessible snapshot hash; B0 provenance is recorded
separately in `manifest.json`.

Run `run_matrix.py --mode list --checkpoint-ids NF-02,NF-04,NF-12` before GPU
execution. The four canaries are part of the 30-job supplement. A full filtered
run keeps the original global job indices 130--159 and does not scan the 130
existing retained feature files; any stale or non-PASS supplement receipt stops
execution.

Invoke both commands as modules from the repository/source root:

```text
python3 -m analysis.noise_floor.run_matrix --mode canary --gpus 0,1 --checkpoint-ids NF-02,NF-04,NF-12
python3 -m analysis.noise_floor.run_matrix --mode run --gpus 0,1 --checkpoint-ids NF-02,NF-04,NF-12
python3 -m analysis.noise_floor.analyze --outdir /data/raw/ECT/ect_runs/noise_floor/output/analysis
```

Until the 30-job supplement completes, q256 history contrasts and delayed
reversal remain `NOT_EVALUATED` in the published 130-job output. They become
reportable only after all three copied checkpoints pass the manifest hashes and
all 30 receipts validate.
