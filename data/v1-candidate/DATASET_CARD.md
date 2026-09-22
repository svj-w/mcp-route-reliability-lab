# V1.0 candidate Dataset Card

## Status

This directory is a **review candidate**, not the stable V1.0 benchmark. It passes
the production structural profile, but no independent secondary labels,
adjudications, agreement measurements, or frozen V1.0 benchmark results exist yet.
Do not cite candidate counts as a released human-reviewed dataset.

## Composition

| Item | Count |
|---|---:|
| Normalized MCP tools | 40 |
| Base queries | 300 |
| English / Chinese bases | 150 / 150 |
| Development / Calibration / Test bases | 100 / 100 / 100 |
| Clear / Hard Negative / Multi-valid / Clarify / No-match | 100 / 60 / 20 / 60 / 60 |
| Paraphrase / cross-language / noise variants | 30 / 30 / 30 |
| Total query records | 390 |

All variants share their base query's split. The immutable candidate Tool Corpus
hash is `sha256:8983296fa7cc6343c94a11fe15307e87f3bf5c6f03bcd036e971b9e9e18de069`.

## Construction and labels

The first 100 bases come from V0.1. The additional 200 bases are synthetic candidate
records generated deterministically from project-authored intent templates and
provisional labels in `scripts/build_v1_candidate.py`. Scenario suffixes make record
identity explicit; they do not represent natural production traffic. The 90 paired
variants are also deterministic. These labels must be treated as provisional until
the documented human review is complete.

The secondary sample contains exactly 100 base queries: 10 English and 10 Chinese
examples from each of the five case types, selected while balancing Tool domain and
difficulty. Its gold strata are kept in a coordinator-only manifest and must not be
given to the blinded annotator. All 180 Clarify, Hard Negative, and No-match bases
also have a blank primary self-check assignment.

Follow `ANNOTATION_HANDOFF.md`. Raw primary, secondary, and recheck files are frozen
before discussion. Route/abstain agreement uses Cohen's Kappa; relevant Tool sets use
per-query and mean Jaccard. Constant-label Kappa is reported as unavailable. Label
disagreements retain both raw records and a separate adjudication record.

## Tool provenance and licensing

All normalized Tool records derive from the official
`modelcontextprotocol/servers` repository at commit `d73f99efbfd40c3aa1b61e88728b3d49fb52608f`. `sources.csv`
records the exact source file, upstream identifier, access date, license transition,
and transformation for every Tool. No upstream implementation code is redistributed.

Project code is Apache-2.0. Original queries, provisional annotations,
documentation, and normalized descriptions are CC BY 4.0. Upstream schemas retain
their recorded Apache-2.0/MIT terms; consult `sources.csv` before redistribution.

## Intended use and limitations

This candidate evaluates single-step routing, clarification, and abstention over one
known Tool Corpus. It does not execute tools, generate arguments, measure downstream
task success, test unseen-Tool cold start, represent production traffic, or provide a
formal safety guarantee. The deliberately balanced stress-test priors differ from a
deployment; calibrated probabilities and thresholds can shift with class and query
distribution. Synthetic wording and repeated intent families can inflate apparent
regularity and must be disclosed in any derived report.

## Release gate

Stable V1.0 requires all of the following: complete 100-query independent review;
complete 180-query primary recheck; frozen raw labels; agreement report; explicit
adjudication of every change/disagreement; final corpus/query hashes; one frozen Test
evaluation per configuration; inspected manifests, metrics, error slices, policies,
figures, and reliability cards; and a clean CI/release rehearsal. Any correction to
frozen Test labels increments the dataset version and reruns every final experiment.
