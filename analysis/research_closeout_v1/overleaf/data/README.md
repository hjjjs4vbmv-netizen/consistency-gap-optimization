# Archived measurements and generated summaries

`raw/` contains the earlier result records. `../sources/pr112/` contains the pinned PR112 analysis/result tree and two earlier control-source documents, preserving the structure required by the unchanged verifier. `source_manifest.json` has 186 immutable file records (26 earlier + 160 added), with original repository paths, commits, and hashes. Original source reports remain unedited.

`../scripts/plot_figures.py` reconstructs the component, history, startup, ImageNet, longitudinal, and restore/hold figures and six corresponding table-row files. For PR111 it uses all 96 full-precision block FIDs, 48 new and 48 reused AA/DA controls, checks scalar/checkpoint bindings, and recomputes nominal tests and the two-simple-effect Holm adjustment.

`../scripts/plot_windows.py` starts from PR112's 72 full-precision FID blocks, reconciles all 48 reused controls with PR111, and recomputes T/L/M, paired-t intervals/tests, the T sign-flip sensitivity, and TOST. It writes two new figures, three new table-row files, and `window_recomputation.json`. First-16 successful-update diagnostics use the 16 raw JSONL prefixes, containing 406 attempts and 256 actual updates. Historical early rho is derived from the observed clock and native formula; early net displacement remains missing.

`startup_summary.json`, `plot_recomputation.json`, and `window_recomputation.json` are generated summaries. Nine `*_rows.tex` files provide all table values. Display rounding is applied only after statistical calculations. Three evaluation blocks do not represent three independent training seeds.

`archived_interim/` retains the obsolete screenshot transcription for traceability; it is excluded from all current plots and tables.

No outcome from the new response-unselected q128 campaign is present. Appendix G's design description is not data. Earlier protocol-limited q128 history evidence remains explicitly separate in the scope appendix.
