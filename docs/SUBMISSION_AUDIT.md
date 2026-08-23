# Submission Audit

Audited on 2026-08-23 against `docs/OVERLEAF_ICLR2027_MAIN.tex` and the rendered submission PDF.

| Gate | Status | Evidence |
|---|---|---|
| Official style and anonymity | PASS | `iclr2027_conference.sty`; `Anonymous Authors`; `\iclrfinalcopy` remains commented. |
| Main-text page limit | PASS | 11 total PDF pages; Discussion/statements end on page 8; References begin on page 9; main text = 8 pages. |
| Abstract length | PASS | 224 words, within the 180–230 target. |
| Citation integrity | PASS | 15 unique cited keys; no missing or unused BibTeX entries; all entries traced to primary sources. |
| Cross-references | PASS | No undefined citation/reference, duplicate label, missing figure, or LaTeX error in the final log. |
| Boxes/layout | PASS | No overfull boxes; underfull-box warnings only. All 11 rendered pages inspected. |
| Figure readability | PASS | Four vector figures inspected at manuscript size; seed-level evidence and censoring remain legible. |
| Numerical provenance | PASS | Headline values map to the claim/evidence matrix, source map, and compact CSV/JSON artifacts. |
| Evidence status | PASS | Formal, replay, secondary, post-hoc descriptive, diagnostic, and NO-GO evidence are explicitly separated. |
| Statistical unit | PASS | Training seed is the independent unit; budgets/NFE and FID/KID pairs are repeated readouts. |
| Claim ceiling | PASS | No universal best-gap, universal compute-saving, cross-$q$, cross-dataset, target-dominance, or optimizer-causality claim. |
| AI disclosure | PASS | Covers hypothesis, design feedback, code/derivation review, interpretation, writing, and figure assistance; executed pipelines and human responsibility stated. |
| Ethics/reproducibility | PASS | Both statements included before the bibliography. |
| Appendix order | PASS | Bibliography precedes appendices A–D. |
| Submission anonymity/privacy | PASS | TeX, BibTeX, and extracted PDF text contain no GitHub username, PR number, collaborator name, absolute local path, `/data/`, or `/mnt/`. |

## Build

Command:

```text
tectonic -X compile OVERLEAF_ICLR2027_MAIN.tex -o build -k --keep-logs -p
```

Outputs:

- `docs/build/iclr2027_submission.pdf`
- `docs/build/iclr2027_submission.log`

Tectonic reports benign font substitutions for the local XeTeX validation environment, underfull boxes, and PDF-version notices for two upstream vector figures. It reports no overfull box, undefined reference/citation, or compilation error. The same source remains compatible with the official Overleaf/pdfLaTeX template workflow.

## Repository-wide marker scan

The required recursive scan was executed. Submission-bearing TeX/BibTeX/PDF text is clean. Internal historical audit documents elsewhere under `docs/` retain development markers and server paths as provenance; they are not included by the submission source. The compiled PDF metadata contains no author identity and no custom metadata stream.
