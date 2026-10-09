# Release readiness design

The user approved the complete paired-robustness repair and independent repository
preparation on 2026-10-09. Execute locally; external publication remains a separate
final step.

## Scope and approach

Choose the complete repair over a wording-only correction: the existing paired
metric implementation must consume the frozen Test predictions without refitting
or selecting thresholds on Test. Report overall results and paraphrase,
cross_language, and noise slices, each with its pair count. V1 contains 90 pairs,
30 in each condition, within 190 Test records.

An unavailable selected policy has no defined accept/abstain decision for this
analysis. Represent accepted decisions and decision-flip metrics as null rather
than manufacturing a zero flip rate. Keep ranking correctness defined by the
existing routing-error contract. Confidence and accuracy shifts are variant minus
base. Report, figure, and metrics.json must agree.

## Components

- evaluation/robustness.py: permit explicitly unavailable decisions, reject mixed
  availability, and aggregate pairs by stable base_query_id.
- cli.py: construct paired cases from already-frozen Test outputs and emit the
  same overall and condition metrics to reporting and machine-readable artifacts.
- reporting.py: include pair counts, display null as not_available, omit undefined
  decision bars, and describe an empty result as not computed.
- provenance.py: capture an exact Git commit; identify a dirty tree or an absent
  repository honestly. Reproduction commands use the actual input/config paths
  and a concrete fresh run directory.

## Repository isolation

Import only the agent/ path history into a temporary local repository, remove the
agent/ prefix from imported paths, and compare its HEAD tree against the original
179 project files before installing its Git metadata in agent/.git. Preserve the
original Desktop Git repository. Verify all imported historical paths are within
the project and omit the unrelated dsh/ document. Do not configure or contact a
remote during this work.

## Validation and report publication

Add meaningful tests for missing pairs, null decisions, 90 total pairs and three
30-pair slices, report/JSON agreement, provenance, and concrete commands. Use
project-owned temporary paths where Windows sandbox rename permissions require
them. Run the full suite, Ruff on src/tests/scripts, and mypy on src.

Commit the implementation before running both V1 baselines. Use offline cached
embedding weights with the pinned model revision where available. Generate new
run directories and check dataset/model bindings, lifecycle, policy status, and
artifact checksums before replacing the frozen reports. Preserve the unchanged
dataset bytes and AI-only experimental status. Finally verify an isolated local
clone and document the project-history migration and publication handoff.
