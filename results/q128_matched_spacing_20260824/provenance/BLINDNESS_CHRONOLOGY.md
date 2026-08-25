# q128 matched-spacing blindness chronology

The five-arm extension was not fully blind to all prior q128 evidence. The
historical canonical `A/Bsame` learning-curve result was committed before both
the matched-spacing calibration and the five-arm protocol freeze. What remained
unknown was the quality of the fresh matched arms and the fresh 210-job matrix.

The legacy frozen-config field `quality_unblinded_before_freeze=false` must
therefore be read narrowly: no quality from the fresh matched-arm extension had
been unblinded before its freeze. It does **not** mean that historical q128
`A/Bsame` results were unknown.

| Event | UTC | Evidence |
| --- | --- | --- |
| Historical q128 `A/Bsame` results committed | 2026-08-23 16:52:37 | `659ae769` |
| Quality-blind matched-spacing calibration frozen | 2026-08-24 02:56:55.697462 | calibration manifest SHA256 `08b84aa6...` |
| Five-arm protocol artifact frozen | 2026-08-24 02:58:36 | frozen config |
| Five-arm protocol committed | 2026-08-24 03:01:49 | `da3d979` after rebase; original `7c7b05d` |
| First formal training launch | 2026-08-24 03:14:22.229256 | preserved launch-record mtime |
| Multi-GPU runtime amendment | 2026-08-24 04:58:06 | frozen config + `bfd5f67` |
| Checkpoint-streaming amendment | 2026-08-24 05:04:00 | frozen config + `1618f9c` |
| Last formal training completion | 2026-08-24 08:24:37.812337 | preserved log mtime |
| First fresh-matrix evaluation log | 2026-08-24 08:28:35.045375 | server archive |
| First fresh-matrix metric written | 2026-08-24 08:30:47.150041 | embedded metric timestamp |
| Matrix reached 210/210 sealed jobs | 2026-08-24 11:45:02.527241 | latest sealed-receipt mtime |
| First metric unblind | after 11:45:02 and by 15:58:00 | conversation authorization was not independently timestamped |
| Analysis code and result package first committed | 2026-08-24 15:58:00 | `1b5be42` after rebase; original `b953809` |

Because AULC first appears in the post-unblind analysis/result commit, it is a
deterministic descriptive full-curve summary, not a frozen or preregistered
primary outcome.
