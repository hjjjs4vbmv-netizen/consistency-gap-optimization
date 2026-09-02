# Appendix: seed-4/5 replication provenance and documented deviations

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-13
- Verification Status: ANALYZED
- Version Label: gap_lr_seed45_provenance_appendix_v1

Two additional training-seed replications were executed for the fixed
three-arm comparison: A (`g=1.0`, LR `1.0e-4`), B (`g=1.3`, LR `1.0e-4`),
and C (`g=1.3`, LR `1.2963523762588691e-4`). All six runs reached 256 kimg
and 2,000 attempted iterations. Five runs recorded 1,992 successful optimizer
steps and eight AMP skips; seed 5 arm C recorded 1,991 successful steps and
nine AMP skips. All counts were within the frozen allowance, and strengthened
post-run checks found finite final training states and EMA snapshots.

The execution was not protocol-exact. The following deviations are retained
rather than normalized away:

| ID | Planned | Observed | Evidential consequence |
| --- | --- | --- | --- |
| D1 | One fail-stop launcher executed all runs. | The original launcher stopped after seed 4 arm A; two manual recovery launchers freshly started the remaining arms, with no trained-state resume. | Launcher continuity changed, but retained artifacts support fresh-start rather than resume execution. |
| D2 | Seed groups ran fully serially. | Seed 4 B/C overlapped seed 5 runs on different logged GPU indices. | Runtime and performance comparisons are excluded; concurrent execution is retained as provenance. |
| D3 | All runs used logged GPU index 1. | Seed 4 B/C used logged index 0. A single pre-launch sidecar records the same A100 model, memory, and driver for indices 0 and 1, but no per-run CUDA UUID was retained. | Cross-device bitwise equivalence is not claimed; a low-order numerical environment effect cannot be excluded. |
| D4 | Diagnostic initialization previews were byte-identical. | Seed 5 arm A's `model_init.png` differed from B/C by at most one 8-bit level. | The preview is not a parameter hash. A deterministic post-run reconstruction produced the same expected transferred tensor-state hash for all six runs, but historical in-process pre-update identity was not observed. |
| D5 | Seed 4 arm A was verified inline before launcher continuation. | The original verifier output and exit status were not retained; the unchanged artifact set later passed strengthened post-hoc verification. | Current artifact integrity is supported, while the exact original verifier failure mechanism remains unknown. |

The six reconstructed expected transferred tensor states share canonical
SHA256
`cabefcddf9ca190fe50eee05714f2cd5d92680b6974a6d8bd2be3be42a1ada19`.
This value is a deterministic reconstruction from the frozen transfer
checkpoint, implementation, dataset interface, and receipt-bound options. It
is not a historical in-process pre-update attestation and does not reconstruct
RNG state.

The quality-blind machine adjudicator recommended acceptance with zero failed
conditions and no rerun-designated arm. However, no independent quality-blind
review was recorded before PR #50 merged, and the PR conversation had already
exposed seed-4/5 quality results. Accordingly, this appendix describes the
runs as completed replications with documented deviations, not as
protocol-exact executions or independently blind-accepted replications. The
subsequent disjoint 5k evaluation proceeds under the Leader's explicit
2026-08-13 directive and remains analytically separate from the post-run
integrity audit.

## Frozen provenance bindings

| Artifact | SHA256 |
| --- | --- |
| Execution protocol commit | `583c2fe0f914fc1191903d747737fd54b4ba1eef` |
| Training-code commit | `2357bb1d2531a343bdb4397f5a08f4d42a2d135b` |
| Replication matrix | `113a4676916e045f95a1928dd6fa163552515ce589a3721b8873bb72f389ad77` |
| Source audit receipt | `6487fbcc5f63817c8e3a91968f45fb13437d1c580afa73966bdf0ad8061bb9fa` |
| Dataset | `a469a9f1b89d43a4a5a0fea42a351b6f107800fc32712881ea3d0ee8cc3a88c1` |
| Transfer checkpoint | `4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da` |
| Objective blind evidence | `8a6ee20300056a4353851ba919134486313629749d541c892d3a7b8fc8c82c12` |
| Initialization reconstruction | `b2e53f2ac865f8729ac7566c8c165c4bb773262399363da53085085b44a17d18` |
| Machine adjudication candidate | `e7a158f235e2644c244297758b293e04701714d904707ce637e841225284901f` |

Raw checkpoints, training states, logs, datasets, absolute server paths, and
stable device identifiers remain external to Git. The sanitized public
receipts bind retained artifacts by SHA256 and byte size; the Role E handoff
additionally freezes the six exact numbered final EMA checkpoints used for
the disjoint-block evaluation.
