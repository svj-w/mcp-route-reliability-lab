# Release Readiness Implementation Plan

> **For agentic workers:** Execute natively in this session. The optional
> superpowers:executing-plans package is not installed; preserve the task-by-task
> test and review workflow described below.

**Goal:** Produce truthful paired V1 reports in an independent local Git repository.

**Architecture:** Reuse frozen predictions and the existing paired evaluator.
Nullable decisions express unavailable policies. Filter project history before
committing the implementation and regenerating provenance-bound reports.

**Tech Stack:** Python 3.11, Pydantic, Typer, NumPy, Matplotlib, pytest, Git.

**Spec:** ../specs/2026-10-09-release-readiness-design.md

## Global Constraints

- Preserve the unchanged dataset bytes and AI-only experimental status.
- Confidence and accuracy shifts are variant minus base.
- Do not configure or contact a remote during this work.
- Report, figure, and metrics.json must agree.

## Review Focus

- Missing or duplicate bases must fail instead of silently reducing pair counts.
- Unavailable policies must yield null decision flips, including in figures.
- No Test variants must yield an explicit unavailable result without invented data.
- Reproduction paths with spaces must remain valid command arguments.
- Filtered history must exclude every Desktop path outside agent/.

### Task 1: Isolate project history

**Files:** Temporary migration helper under .deps; new agent/.git metadata.
**Interfaces:** Git fast-export scoped to agent/; fast-import with stripped prefix.

- [x] Export the current project branch with full filtered trees, preserving data
  blocks unchanged and rejecting paths outside agent/.
- [x] Import into a fresh repository; compare `git ls-tree -r HEAD` object IDs to
  the original project tree. Inspect all historical paths and refs.
- [x] Install verified Git metadata and commit the reviewed design and plan.

### Task 2: Connect paired evaluation and truthful reporting

**Files:** src/mcp_route_reliability/evaluation/robustness.py, cli.py,
reporting.py, provenance.py; tests/unit/test_robustness.py,
tests/integration/test_reporting.py, tests/integration/test_end_to_end.py,
tests/unit/test_provenance.py; golden report and example.

**Interfaces:** `EvaluatedCase.accepted: bool | None`;
`RobustnessMetrics.decision_flip_rate: float | None`;
`RobustnessSlice.pair_count: int`; `resolve_source_commit(root: Path) -> str`.

- [x] Add and run regression assertions:
  ```python
  assert result.overall.decision_flip_rate is None
  assert result.by_perturbation["noise"].pair_count == 30
  assert "Paired robustness not computed" in empty_card
  assert "not_available" in unavailable_card
  ```
- [x] Permit all-null decisions and reject mixed availability in the paired
  evaluator. Build evaluated cases from Test top-1 IDs, calibrated confidence,
  routing correctness, and the already-selected policy.
- [x] Emit overall/by-condition/count/deltas under metrics.json robustness;
  convert those same aggregates into report rows. Show unavailable decision
  metrics honestly in both table and figure.
- [x] Capture Git commit and actual input/config arguments; use a fresh concrete
  reproduction directory and test command round-tripping with spaced paths.
- [x] Run focused tests, then full pytest, Ruff, and mypy; commit implementation.

### Task 3: Rebuild and verify frozen V1 reports

**Files:** reports/v1-ai-experimental/{bm25,embedding}/;
tests/integration/test_v1_ai_benchmark.py; README.md.
**Interfaces:** Existing benchmark CLI and checksum-bound RunManifest.

- [x] Run both commands from the committed implementation:
  ```console
  python -m mcp_route_reliability.cli benchmark --config configs/v1-ai-bm25.yaml --tools data/v1-ai-experimental/tools.json --queries data/v1-ai-experimental/queries.jsonl --run-dir runs/release-bm25 --json
  python -m mcp_route_reliability.cli benchmark --config configs/v1-ai-embedding.yaml --tools data/v1-ai-experimental/tools.json --queries data/v1-ai-experimental/queries.jsonl --run-dir runs/release-embedding --json
  ```
- [x] Check both outputs have 190 Test records, 90 pairs, three 30-pair conditions,
  null decision flips where policy unavailable, exact commit provenance, and
  correct input hashes. Verify every manifest checksum.
- [x] Replace frozen report artifacts only after verification. Add frozen report
  assertions for these invariants and update README with paired results.
- [x] Run affected tests and commit the verified reports and documentation.

### Task 4: Verify publication handoff

**Files:** docs/portfolio/release-preparation.md.
**Interfaces:** Independent repository root and no configured remote.

- [x] Document migration provenance, report source commit, AI experimental status,
  reproduction instructions, and remaining external publication step.
- [x] Clone the independent repository into a fresh ignored local directory;
  verify historical path boundaries and run tests from the clone.
- [x] Check links, artifact checksums, clean Git status, and no remote; finish
  with concrete local results for the user's review.

## Verification Notes

- Both reports were generated from clean commit
  `6319120606b19f34802d0c9cad4dbf65866921cd` and retain that exact provenance.
- Independently reconstructed all 90 pairs from each run's predictions,
  confidence features, and persisted calibrator; verified all 24 artifact hashes.
- Ranking and candidate order stayed unchanged. MiniLM score drift versus the
  previous report was at most 1.4758793615010646e-7; this is disclosed, not hidden
  behind a byte-identical reproducibility claim.
- Fresh read-only review found PowerShell apostrophe and @-prefix path issues;
  both are fixed and tested against real native Python argument parsing.
- Full local suite: 187 passed; Ruff passed; mypy passed (32 source modules).
- Local README/portfolio links: 42 targets checked. Independent local clone:
  187 passed, Ruff passed, mypy passed (32 source modules); all 24 artifact hashes
  and all 42 links passed again. Confirmed imports use cloned src, not the
  original project tree; existing installed dependencies were reused.
- Project root is agent/, branch main, with no remote. Dataset diff from the
  imported starting commit is empty. Final documentation-only commit is audited
  in the project and fast-forwarded into the clean local verification clone.
- The Desktop parent repository received a concurrent dsh/ commit; it was
  inspected read-only and left untouched rather than reverted to the snapshot.
