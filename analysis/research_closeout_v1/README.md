# ECT evidence closeout (0 new GPUh)

Read [REPORT_ZH.md](REPORT_ZH.md) for scope, PASS/UNVERIFIED findings, statistical interpretation and remaining limits. The English manuscript is in `overleaf/`; the ZIP uploads directly to Overleaf. All changes are additive to PR113 `42871769be317f6baa1bd6e030bec120c22d9371`.

`SOURCE_BINDINGS.json` maps original full-file hashes, extracted raw hashes, public hashes and extraction boundaries. Publication is post-hoc. `evidence/` contains only small redacted records. No checkpoint, generated image or feature arrays were transferred. `GPU_LEDGER.json` separates zero new work from original campaign costs.

CPU reproduction, from repository root:

```sh
python analysis/q128_startup_fresh8_v1/verify_results.py
python analysis/q256_startup_delayed6_10_v1/verify_results.py
python analysis/research_closeout_v1/scripts/audit_existing.py
python analysis/research_closeout_v1/scripts/verify_source_bindings.py
python analysis/research_closeout_v1/scripts/make_tables_figures.py
cd analysis/research_closeout_v1/overleaf
bash scripts/build.sh
python scripts/validate_artifact.py
```

The source-binding check needs scientific commit `fcc94d26b1085011ac935084f7c1cf4c149e2947` in local Git. Python package versions actually used are in `validation/cpu_environment.json`; these CPU dependencies do not replace the archived training environment. Original engineering checks are reused, not rerun. `publish_evidence.py` is the publication transform for a separately held private collection and is not required to analyze the public supplement.

The existing verifier checks original statistics; the additive audit covers newly published receipt/config/clock/LR/AMP records and duplicate chronology. No additional optimizer tests or training implementation changes are introduced. Figures/tables are in `figures/` and `tables/`; missing q256 early net displacement stays empty. No fitted update–FID relationship is presented.

Final PDF QA: 29 pages, 11 figures, 19 tables; original and added source-hash checks pass. The bibliography-heading check accepts both title case and uppercase to support the Tectonic preview. The manuscript remains an anonymous research draft, not a venue-ready page-limit submission.
