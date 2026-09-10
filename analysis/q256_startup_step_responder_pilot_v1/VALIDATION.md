# Completed execution and validation

The fixed matrix completed with 8 planned seeds, 16/16 PASS training trajectories,
48/48 PASS new endpoint evaluations, and 8 finite complete seed pairs. There were
no scientific failures, failed evaluation slots, or missing endpoints.

The archived CPU regression log records 43 passing tests. The sole GPU preflight
used 0.20301482354808187 process GPUh: all 64 original-A rows matched across 14
telemetry fields; both interventions' pause/resume checks at successful updates
3, 5 and 6 matched uninterrupted full states. This short preflight does not claim
to exercise the full suffix. All formal trajectories reached attempt 8000.

The frozen `verify_results` check passed for the complete roster, evaluation
settings, source identities and paired arithmetic. Additional delivery checks
matched all 48 new evaluation checkpoint hashes to their training outcomes,
matched all 48 old AA/DA FID and KID values exactly to the historical results,
checked every generation-block range and metric seed, and independently
recomputed the three contrast means. PNG/PDF figures were rendered and visually
checked; the local delivery ZIP passed its integrity check.

## Provenance and operational amendments

- Base: PR110 head `d9e0021f7e6a8049a642f81bd22390aec0b383ac`.
- GPU preflight execution: `cdd1326fea08548cd2a4b403d9c02cfd97bb37b4`.
- Initial formal training implementation: `4a023f3ac1a5e68300c0e54affa56d314df7b964`.
- Final dispatch controls: `b564131e9f6dd2d94e42f73da1b03184cdc4c9db`.
- Frozen evaluator: `d6aba02fb88e9db0993623895eb2228ed717d810`.

The user authorized evaluation after each execution group completed, early
archive/return of each paid host, relocation of only the never-started seed57 D
trajectory, evaluation-only copying of the completed seed57 A endpoint, and
removal of financial dispatch cutoffs. These amendments did not change the
scientific matrix, startup multipliers, sample blocks or statistics. Original
queues/commands in `results/execution` retain their historical assignments;
the placement amendment and final outcomes identify actual execution. The old
ECT seed57 D dispatcher never ran and was disabled. Imported seed57 A was not
retrained, and no active trajectory migrated.

The report stage initially lacked matplotlib. Fixed plotting dependencies were
installed in an isolated report directory and exposed only to the CPU analysis
command: matplotlib 3.10.6, contourpy 1.3.2, cycler 0.12.1, fonttools 4.59.2,
kiwisolver 1.4.9, packaging 25.0, pyparsing 3.2.3, python-dateutil 2.9.0.post0 and
six 1.17.0. Frozen NumPy/SciPy, training, evaluation and analysis source were
unchanged. The original analysis and verifier then completed successfully.

## Archive and cost accounting

Both paid-host archives were returned to ECT and every content-addressed object
passed SHA256 verification before release readiness. The original six-card host
has 4,543 logical files / 3,545 unique objects (62,410,934,224 unique bytes); the
additional host has 2,371 logical files / 2,187 objects (9,477,321,771 unique bytes).
Owned-host experiment files are indexed and hashed in place. Archive indices
preserve relative paths, lengths, object hashes, modes and symlink identities.

Measured experiment process use was 66.0708175153 GPUh: paid hosts 51.1876705826,
owned ECT 14.8831469326. Allocated idle capacity is reported separately; these
numbers are not verified rental-account billing totals. No financial cutoff
remained active under the user's completion-priority instruction.

## Interpretation and artifact boundaries

This is a responder-enriched exploratory mechanism pilot. A user-requested
six-seed interim summary was viewed before the complete eight-seed analysis;
ordinary intervals and p-values are nominal and have no sequential-look
adjustment. C/M Holm adjustment covers those two prespecified comparisons only.
No scientific settings were adapted after viewing intermediate metrics.

`results/execution` contains path-redacted receipts, manifests and frozen
commands. `${...}` placeholders require local path binding before reuse; they
are not claims that the redacted bytes retain the original receipt hash.
Hashes inside immutable receipts and `raw_artifact_hashes.json` refer to raw
archived artifacts. `PUBLIC_ARTIFACT_SHA256.json` hashes the published files.
Large checkpoints, generated samples, feature arrays, credentials and private
absolute paths are excluded. PR110 engineering-only guards remain unchanged.
