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

- [ ] Export the current project branch with full filtered trees, preserving data
  blocks unchanged and rejecting paths outside agent/.
- [ ] Import into a fresh repository; compare `git ls-tree -r HEAD` object IDs to
  the original project tree. Inspect all historical paths and refs.
- [ ] Install verified Git metadata and commit the reviewed design and plan.

### Task 2: Connect paired evaluation and truthful reporting

**Files:** src/mcp_route_reliability/evaluation/robustness.py, cli.py,
reporting.py, provenance.py; tests/unit/test_robustness.py,
tests/integration/test_reporting.py, tests/integration/test_end_to_end.py,
tests/unit/test_provenance.py; golden report and example.

**Interfaces:** `EvaluatedCase.accepted: bool | None`;
`RobustnessMetrics.decision_flip_rate: float | None`;
`RobustnessSlice.pair_count: int`; `resolve_source_commit(root: Path) -> str`.

- [ ] Add and run regression assertions:
  ```python
  assert result.overall.decision_flip_rate is None
  assert result.by_perturbation["noise"].pair_count == 30
  assert "Paired robustness not computed" in empty_card
  assert "not_available" in unavailable_card
  ```
- [ ] Permit all-null decisions and reject mixed availability in the paired
  evaluator. Build evaluated cases from Test top-1 IDs, calibrated confidence,
  routing correctness, and the already-selected policy.
- [ ] Emit overall/by-condition/count/deltas under metrics.json robustness;
  convert those same aggregates into report rows. Show unavailable decision
  metrics honestly in both table and figure.
- [ ] Capture Git commit and actual input/config arguments; use a fresh concrete
  reproduction directory and test command round-tripping with spaced paths.
- [ ] Run focused tests, then full pytest, Ruff, and mypy; commit implementation.

### Task 3: Rebuild and verify frozen V1 reports

**Files:** reports/v1-ai-experimental/{bm25,embedding}/;
tests/integration/test_v1_ai_benchmark.py; README.md.
**Interfaces:** Existing benchmark CLI and checksum-bound RunManifest.

- [ ] Run both commands from the committed implementation:
  ```console
  python -m mcp_route_reliability.cli benchmark --config configs/v1-ai-bm25.yaml --tools data/v1-ai-experimental/tools.json --queries data/v1-ai-experimental/queries.jsonl --run-dir runs/release-bm25 --json
  python -m mcp_route_reliability.cli benchmark --config configs/v1-ai-embedding.yaml --tools data/v1-ai-experimental/tools.json --queries data/v1-ai-experimental/queries.jsonl --run-dir runs/release-embedding --json
  ```
- [ ] Check both outputs have 190 Test records, 90 pairs, three 30-pair conditions,
  null decision flips where policy unavailable, exact commit provenance, and
  correct input hashes. Verify every manifest checksum.
- [ ] Replace frozen report artifacts only after verification. Add frozen report
  assertions for these invariants and update README with paired results.
- [ ] Run affected tests and commit the verified reports and documentation.

### Task 4: Verify publication handoff

**Files:** docs/portfolio/release-preparation.md.
**Interfaces:** Independent repository root and no configured remote.

- [ ] Document migration provenance, report source commit, AI experimental status,
  reproduction instructions, and remaining external publication step.
- [ ] Clone the independent repository into a fresh ignored local directory;
  verify historical path boundaries and run tests from the clone.
- [ ] Check links, artifact checksums, clean Git status, and no remote; finish
  with concrete local results for the user's review.
