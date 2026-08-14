# Gap evidence asset audit — 2026-08-14

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-15
- Verification Status: ANALYZED
- Version Label: gap_evidence_asset_audit_v2

## Verdict

The same-trajectory longitudinal bundle in PR #49 is internally consistent and
uses four states from one uninterrupted seed-3 Arm A run.  The fixed PR #47
plot also reads the four numbered-state result receipts that bind to those same
training-state hashes.  It does not read the stale mutable-`latest` duplicate.

PR #58 closes the h=20 portion of B002: all four K headlines now recompute from
Git-committed per-coordinate predictions, targets, and weights.  It does not
close full-matrix provenance: horizons outside h=20 still depend on node-local
paths with hashes and sizes but no durable externally resolvable locator.

The evidence estate is not yet publication-ready under the requested
commit -> run -> checkpoint -> raw artifact -> plotting-script standard.
Four unresolved blockers remain after this audit branch adds a deterministic
PR #53 aggregation reproducer and quarantines the mutable-`latest` duplicate.

## Audited PR boundary

| PR | Canonical head | Role in the evidence chain | Audit result |
|---|---|---|---|
| #47 | `6bb913d` | Prospective scalar-history receipts and K plot | Canonical numbered-state receipts pass |
| #49 | `3d2c33e` | Same-trajectory longitudinal receipts, CSV, and figure | Pass |
| #50 | `54e6681` | Seed-4/5 numbered state and snapshot receipts | Pass with documented execution deviations |
| #51 | `e954a72` | Exact numbered-checkpoint evaluation handoff | Pass for seed 4/5 endpoint binding |
| #52 | `4f6b056` | Corrected interpretation wording merged into #47 | Pass; no numerical artifact changed |
| #53 | `fbdcb13` | 54 disjoint FID/KID-5k JSONLs and summaries | JSONL/CSV hashes pass; sample-level and two checkpoint hashes remain incomplete |
| #58 | `ff7a49b` | Cross-K h20 raw predictions and full R2(K,h) matrix | h20 is Git-self-contained and recomputes; full matrix remains node-local-only |

## Canonical longitudinal chain

All four PR #49 states share:

- experiment: `gap_lr_matched_q128_s3_v1`;
- run: `arm_a_g1_0_lr_fixed_s3`;
- training seed: 3;
- arm/gap/LR: A / 1.0 / 1e-4;
- training code commit: `2357bb1d2531a343bdb4397f5a08f4d42a2d135b`;
- one uninterrupted trajectory: true.

| K (kimg) | State | Optimizer steps | Training-state SHA256 | Network-snapshot SHA256 |
|---:|---|---:|---|---|
| 32.128 | `000001` | 243 | `dcda9f67...1eb67a` | `70473352...315894` |
| 64.128 | `000002` | 493 | `91763eea...734f4e2` | `962ecd95...08e5c` |
| 128.128 | `000004` | 993 | `cd49080a...7650d` | `107622af...9d0c2` |
| 256.000 | `000008` | 1991 | `a5401395...9bbc8` | `fa48bf5a...f1c32b` |

The handoff receipt, four stateful-audit JSONs, four layerwise CSVs, and
`longitudinal_summary.csv` agree on these state identities.  The source state
is marked preserved and both virtual branches execute rather than skipping the
optimizer step at every K.

## Figure input audit

| Figure | Script | Actual input set | Result |
|---|---|---|---|
| PR #49 `same_trajectory_residuals.pdf` | `scripts/plot_same_trajectory_longitudinal.py` at `3d2c33e` | Only `analysis/same_trajectory_longitudinal/longitudinal_summary.csv` at `3d2c33e` | Pass; CSV is cross-checked against four JSON receipts |
| PR #47 `k_horizon_R2_Ropt.pdf` | `analysis/plot_k_curve.py` at `6bb913d` | Only `analysis/real_history/k{32,64,128,256}/scalar_prediction.json` | Pass; every result binds a numbered training state matching PR #49 |
| PR #58 h20 figures 1 and 3 | `analysis/plot_crossk.py` at `ff7a49b` | `summary.json` plus Git-committed `{h_pred,h_actual,weights}_h20.npy` arrays | Pass; 2 tests / 8 subtests independently recompute R2 and Corr at all four K |
| PR #58 full-matrix figure 2 | `analysis/plot_crossk.py` at `ff7a49b` | `summary.json`, derived from external gradient/update/moment histories | Blocked for off-node re-derivation; manifest has hashes/sizes but only node-local paths |
| PR #53 seedwise `delta_ctrl` table | none in PR #53 | 54 metric JSONLs -> two CSVs | Repaired in this audit branch with `scripts/rebuild_disjoint_5k_summary.py`; exact byte reproduction passes |

## Noncanonical artifact quarantine

`analysis/real_history/scalar_prediction.json` at `6bb913d` is forbidden for
claims.  It records `training-state-latest.pt`, has no seed, and binds source
state SHA256 `a23cacc8...d81f78`, whereas the numbered canonical state
`training-state-000008.pt` is `a5401395...9bbc8`.  The PR #47 plotting script
does not read this file.

The canonical K=256 mechanism receipt is
`analysis/real_history/k256/scalar_prediction.json`.

## Remaining blockers

| ID | Severity | Required closure |
|---|---|---|
| B002 | High | For the full R2(K,h) matrix outside the Git-self-contained h20 headline, publish the hash-bound histories at a durable externally resolvable URI/object version. |
| B003 | High | Retain or regenerate hash-bound sample/feature artifacts for each PR #53 FID/KID cell. |
| B005 | High | Publish the exact `network-snapshot-000008.pkl` SHA256 values for seed-3 Arms B and C and bind them to the metric cells. |
| B006 | Medium | Add checkpoint SHA256 and explicit ordered sample-seed range to every metric receipt, or add a signed cell manifest carrying those fields. |

The h20 subfinding `B002-H20` is closed at PR #58 head `ff7a49b`: four sets of
Git-committed NPY predictions/targets/weights reproduce weighted R2 and Corr.
The remaining B002 is deliberately narrower; `/data/raw/ECT/...` plus SHA256
and size is an identity receipt, not a durable external locator.

The former mutable-latest and aggregation blockers are closed locally: commit
`9307a76` replaces the duplicate payload with a tombstone, and the new
reproducer rebuilds both PR #53 CSVs byte-for-byte from all 54 SHA-bound
JSONLs.

The source-host rehash for B005 was attempted twice read-only on 2026-08-14;
the SSH endpoint closed both connections before the two files could be read.
No checkpoint hash was inferred from a path, alias, or unrelated receipt.

## Reproduction commands

```bash
python3 scripts/verify_gap_artifact_manifest.py
python3 scripts/rebuild_disjoint_5k_summary.py --verify
python3 -m pytest tests/test_crossk_h20_recompute.py tests/test_gap_artifact_manifest.py
```

Use the stricter publication gate when preparing manuscript numbers:

```bash
python3 scripts/verify_gap_artifact_manifest.py --require-publication-ready
```

It must exit nonzero until all blocking findings in
`evidence/gap_artifact_manifest_v1.json` are closed.
