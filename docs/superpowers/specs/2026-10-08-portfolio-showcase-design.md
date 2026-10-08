# AI Agent Engineering Portfolio Showcase Design

## Purpose and audience

Make MCP Route Reliability Lab understandable to an AI application/Agent engineering recruiter within 30 seconds and demonstrable after setup in about five minutes. The showcase is an evidence-based engineering case study, not a product launch or a research paper. The repository remains an evaluation toolkit; no new Agent runtime, UI, model, benchmark metric, or paid service is in scope.

The repository-facing entry point stays in English. Job-application and interview materials are in Chinese. The existing V0.1 and V1 AI experimental dataset, source CSVs, reports, and code remain unchanged. There is no GitHub push or public release in this work.

## Content components

1. **README landing section (English).** Put the latest V1 AI experimental milestone above the V0.1 history. State the problem, the evaluation pipeline, the practical engineering choices (versioned Tool Corpus, provenance-bound predictions, family-isolated splits, calibration-only fitting, fail-closed policy), and the release boundary. Show a small measured BM25-versus-embedding comparison and link to existing reports. Keep the V0.1 section, but make the timeline and differences explicit. The first screen must say that V1 AI experimental is AI-annotated, not independently human-reviewed, and not a production safety certification.
2. **Demo guide (`docs/portfolio/demo.md`, English).** Give a reproducible offline fixture route through existing CLI commands after installation: validate fixture, run BM25 into a fresh ignored `runs/` directory, inspect the generated card/metrics/errors, then inspect the frozen V1 AI experimental report and its unavailable 5% policy. Include expected checkpoints and common failure recovery (missing dependency, occupied run directory, embedding model cache/download). Do not pretend the fixture output equals V1 results; the fixture demonstrates the mechanism while committed V1 reports supply the portfolio evidence.
3. **Interview guide (`docs/portfolio/interview-zh.md`, Chinese).** Provide a truthful one-line project description, adaptable resume bullets, a 60-second walkthrough, a short architecture explanation, and likely follow-up questions. Distinguish verified repository facts from personal contribution claims that the student must tailor to what they personally did. Include explicit red lines: do not claim independent human review, a safe deployment threshold, 190 independent intents, or that this repository already runs a production Agent.

These components link outward to the existing Dataset Card, annotation report, reliability cards, and source manifests rather than duplicating detailed evidence. The README stays the navigation hub; the guides remain focused and independently readable.

## Evidence and claim rules

- Read all displayed numbers from the frozen `reports/v1-ai-experimental/{bm25,embedding}/metrics.json` and the dataset `source-manifest.json`; do not calculate a new leaderboard or reuse V0.1 numbers for V1.
- Explain that Test has 100 base queries plus 90 paired variants (190 records, not 190 independent intents). Note that 112 base labels remain generator-provisional.
- State both policies are unavailable at the configured 5% target risk. Do not use “safe,” “production-ready,” “human-reviewed,” or “stable V1.0” as release claims.
- Use a real, inspectable error example from `errors.csv`; identify the gold label and predicted ranking accurately.
- The resume text may describe the project architecture and demonstrated skills, but must not assert the user's sole authorship, team size, internship impact, or deployment experience without their verification.

## Verification and failure handling

The documentation change is accepted when: (a) all links resolve locally, (b) copied figures and numbers match committed artifacts, (c) demo commands run from a clean setup into a fresh output path without touching frozen datasets or reports, (d) no claim crosses the release boundary above, and (e) `git diff` contains only the agreed documentation files. If a demo step depends on a missing optional embedding model or network access, show the offline BM25 path as the default and label the embedding step optional. Existing automated tests and lint still run to guard against incidental changes, but documentation validation is primarily link, command, and claim review.
