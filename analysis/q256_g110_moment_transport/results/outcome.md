# q256 g=1.10 moment-transport continuation outcome

Final status: **scientific NO-GO before continuation training**.

The frozen held-out manipulation gate completed on `gpu0003` from
`2026-08-19T04:20:35Z` to `2026-08-19T04:55:19Z`. It evaluated all twelve
preregistered held-out rows (four rows for each training seed 3, 4, and 5),
wrote an empty stderr log, and returned the intentional scientific exit code
4. `formal_training_authorized` is false. Consequently, no 32-step smoke,
continuation training, sampling, FID, or KID was started.

## Compatibility decision

The artifact-backed compatibility audit required fresh F and fresh G for all
three seeds. Existing G checkpoints used a different 256 kimg source. Existing
F checkpoints could not prove the missing serialized RNG/sampler state and
the complete protocol identity. Had the manipulation gate passed, the frozen
formal design would therefore have been fresh F/G/T from the common fixed
256 kimg source.

## Held-out gate result

| Seed | Frozen `a_s` | median `R_opt_G` | median `R_opt_T` | Suppression | median exact residual | median `||U_T||/||U_F||` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.8370121196598016 | 0.08302803446060611 | 0.08725688495820916 | -0.05093280269821987 | 0.000027978676718216554 | 1.0123058479690594 |
| 4 | 0.8073491626309143 | 0.09143990170885782 | 0.10880718291755766 | -0.1899311010197362 | 0.00003985941326622518 | 1.0248249657078135 |
| 5 | 0.8233457134218897 | 0.07300412304151258 | 0.08995829986922899 | -0.23223588095258263 | 0.00005650710222066433 | 1.0212817589529748 |

All finite-value, no-skip, AMP, deterministic rerun, branch-order, source
preservation, exact-scalar residual, and update-norm gates passed. The
mechanism gate failed because every seed had negative rather than positive
suppression. The cross-seed median suppression was
`-0.1899311010197362`, below the preregistered minimum `0.50`.

This was not a median-only edge case: all 12 held-out batches had
`R_opt_T > R_opt_G`, with per-batch worsening of approximately 1.9% to 29.8%.

The exact-scalar residuals were all far below `0.01` and the norm ratios were
inside `[0.90, 1.10]`; these checks make a scale/sign, branch-order, or gross
update-magnitude failure an implausible explanation for the result. The
one-time scalar transport simply did not suppress the held-out residual under
the frozen definition.

Two independent read-only audits recomputed the seed medians and gates from
the raw batch CSV. One additionally reconstructed `R_opt` from the recorded
norm/cosine quantities, reaggregated all 208 layer summaries per arm and row,
and compared ordinary-G outputs against the original factorial receipts. No
sign, arm-label, split, aggregation, cloning, AMP, or optimizer-state error was
found. See `independent_audit.md`.

## Integrity and provenance

- Execution commit: `250bd759a561718a502ab9577c140dd609c27031`.
- Preflight runner SHA256:
  `e2ca6ad845bc3a653c29c6502898b7a57a858a2d2bc2dadcc46786906f1e93e0`.
- Stateful audit library SHA256:
  `dc105a0dcfd36b4c984b3c8cf36abcf462ab2a7d2d9fadb44a8aa90f4a275fc0`.
- Final verdict SHA256:
  `0cc66f77e4fedd4fb2c86bf46f450a518cc4666b7b7c65b3ca5559b0830e01a7`.
- Compatibility report SHA256:
  `31f33042b7c173a14ea0ea7e41edee2a87ef59ea59dbf5257328339bd0a03bd6`.
- Runtime: Python 3.10.12, PyTorch `2.2.0a0+81ea7a4`, CUDA 12.3, AMP enabled.
- The six source state/snapshot file hashes were recomputed after the gate and
  exactly matched their frozen pre-run hashes.
- The server worktree remained clean at the execution commit. No `ct_train.py`
  process or formal run root existed at postcheck time.

Two launcher-only engineering failures are retained separately. Neither
evaluated a scientific gate or started continuation training. The successful
attempt used a SHA256-bound host-tmux launcher and produced all raw batch,
seed, verdict, log, timestamp, and exit artifacts. `evidence_manifest.json`
binds their archived copies.
