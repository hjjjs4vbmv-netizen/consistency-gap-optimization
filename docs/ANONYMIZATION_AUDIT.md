# Anonymous Repository Audit

## Scope

The anonymous submission repository is a clean export, not a rename of the current collaboration repository. The public collaboration history, owner name, pull requests, issue discussions, and contributor metadata must not be copied into the anonymous release.

## Preserve

- source code required to train and evaluate the declared methods,
- environment specifications and setup commands,
- dataset and checkpoint download instructions with cryptographic hashes,
- configuration files and exact commands,
- lightweight tables, figures, manifests, and machine-readable summaries,
- third-party copyright notices, licenses, and academic citations.

## Exclude

- `.git/` history from the collaboration repository,
- checkpoints, raw image sets, caches, and full logs,
- private server paths and SSH hostnames,
- usernames, personal email addresses, access tokens, and editor metadata,
- internal chat exports, reviewer identities, and non-public storage links,
- abandoned exploratory results that are not cited or documented.

## Repository layout

```text
anonymous-submission/
├── README.md
├── LICENSE
├── environment/
│   ├── environment.yml
│   └── versions.md
├── configs/
│   ├── primary/
│   └── generalization/
├── docs/
│   ├── evaluation_protocol.md
│   ├── reproducibility_checklist.md
│   └── asset_manifest.md
├── scripts/
│   ├── prepare_data.sh
│   ├── train.sh
│   ├── evaluate.sh
│   └── reproduce_tables.sh
├── results/
│   ├── primary/
│   └── generalization/
├── supplementary/
└── src/
```

## Automated scan patterns

The release candidate must be scanned for:

- Windows user paths: `C:\Users\`
- Linux private roots: `/root/`, `/home/<name>/`, and project-specific `/mnt/` paths
- credential variables and token-like strings
- private keys and credential files
- GitHub owner/repository URLs belonging to the collaboration repository
- personal email addresses
- VS Code, Jupyter, and shell-history artifacts

An automated scan is a warning system, not proof of anonymity. Every match must be reviewed manually because third-party citations and license contacts may be legitimate.

## Release procedure

1. Create a new empty private repository controlled by the submission lead.
2. Export the approved source tree without `.git` history.
3. Copy only files listed in the release manifest.
4. Replace machine-specific paths with documented environment variables.
5. Run the automated scan and resolve every finding.
6. Build the environment and run smoke tests from a clean clone.
7. Generate tables and figures from tracked lightweight inputs.
8. Ask a team member who did not build the export to perform the clean-clone audit.
9. Freeze the anonymous release commit and record its SHA in the private submission record.
10. Keep the mapping between anonymous and collaboration commits outside the anonymous repository.

## Current audit status

| Item | Status | Notes |
| --- | --- | --- |
| Clean export repository | pending | Do not reuse collaboration Git history |
| Anonymous README | pending | Current README is upstream-oriented |
| Identity/path scan | pending | Scanner to be added |
| Asset manifest | pending | Dataset and transfer SHA values required |
| Clean-clone smoke | pending | Run only after export exists |
| Independent reviewer | pending | Assign before Week 6 freeze |
