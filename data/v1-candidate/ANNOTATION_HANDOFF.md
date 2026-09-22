# V1.0 annotation handoff

This directory is a **candidate**, not a released V1.0 dataset. Production-profile
structure validation passes, but independent human review is still missing.

## Secondary annotator package

Give the second human only:

- `tools.json`;
- `annotations/secondary-assignment.csv`; and
- `../../docs/annotation-guide.md` from the repository.

Do **not** give them `queries.jsonl`, `annotations/primary.jsonl`,
`secondary-sample-manifest.csv`, or prior benchmark outputs. The manifest contains
gold strata and is coordinator-only.

The annotator must fill every blank label field, use a stable non-primary
`annotator_id`, preserve `query_id` and this corpus hash, and separate multiple Tool
IDs with `|`:

`sha256:8983296fa7cc6343c94a11fe15307e87f3bf5c6f03bcd036e971b9e9e18de069`

Allowed abstention reasons are blank for routeable cases, `ambiguous`,
`insufficient_information`, and `no_match`. A No-match record uses an empty relevant
set. Return the completed CSV without sorting or deleting rows.

Freeze the returned file before discussing disagreements:

```powershell
python scripts/process_v1_annotations.py prepare `
  --completed path/to/completed-secondary.csv
```

This writes immutable `secondary.jsonl`, `agreement.json`, and a disagreement-only
`adjudication-assignment.csv`. The command refuses to overwrite existing review
artifacts.

## Primary self-check

`annotations/primary-recheck-assignment.csv` contains all 180 Clarify, Hard Negative,
and No-match base cases without their existing labels. Complete it as a separate pass
without opening `primary.jsonl`, then compare and adjudicate changes explicitly.

```powershell
python scripts/process_v1_annotations.py recheck `
  --completed path/to/completed-primary-recheck.csv
```

After an adjudicator fills every blank adjudicated field in
`adjudication-assignment.csv`, freeze the final disagreement records with:

```powershell
python scripts/process_v1_annotations.py finalize `
  --completed path/to/completed-adjudication.csv
```

## Release gate

Do not rename this directory or promote its files into `data/` until:

1. all 100 secondary rows are complete and validated;
2. raw primary and secondary labels are frozen before adjudication;
3. route/abstain Cohen's Kappa and Tool-set Jaccard are calculated;
4. disagreement rows and adjudicator decisions are retained; and
5. any Test correction increments the dataset version and reruns every experiment.
