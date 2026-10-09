# CI Portability Implementation Plan

> **For agentic workers:** Execute natively in this session. The optional
> superpowers:executing-plans package is not installed; preserve task-by-task
> regression testing and independent final review.

**Goal:** Make both existing public CI jobs pass without changing experimental data.

**Architecture:** Align CI mypy's target with its Python matrix and generate
candidate CSVs with explicit LF terminators. Pin these contracts with offline tests.

**Tech Stack:** Python 3.11/3.12, csv, pytest, mypy, GitHub Actions, Git.

**Spec:** ../specs/2026-10-09-ci-portability-design.md

## Global Constraints

- Preserve all committed data/ and reports/ bytes; retain AI experimental status.
- Retain the existing dependency constraints and both Python matrix jobs.
- Preserve assignment CSV BOM encodings and blank review fields.
- Push only to svj-w/mcp-route-reliability-lab, main, without force.

## Review Focus

- Windows CRLF checkout must not invalidate snapshot comparison.
- Candidate rebuild must run in a temporary root, not overwrite frozen files.
- Sources CSV is UTF-8; two assignment CSVs keep their UTF-8 BOM.
- Neither disabled tests nor ignored dependency errors may conceal the failure.
- Actual Python 3.12 success requires remote CI, not a local 3.11 target simulation.

### Task 1: Repair portability with two failing regressions

**Files:** .github/workflows/ci.yml, scripts/build_v1_candidate.py,
tests/integration/test_v1_candidate.py.

**Interfaces:** Existing builder main() consumes data/tools.json,
data/queries.jsonl, data/sources.csv; writes the same data/v1-candidate/ snapshot.

- [ ] Add workflow regression:
  ```python
  assert "python -m mypy --python-version ${{ matrix.python-version }} src" in workflow
  ```
- [ ] Add isolated rebuild regression. Copy the builder and its three source
  files into tmp_path, run it with sys.executable and PYTHONPATH=ROOT/src, then
  compare all output files to the frozen candidate:
  ```python
  assert actual == expected.replace(b"\r\n", b"\n")
  assert b"\r\n" not in generated_csv
  ```
  Preserve BOM bytes explicitly for the two assignment files.
- [ ] Run python -m pytest tests/integration/test_v1_candidate.py -q with a fresh
  .deps/ basetemp and confirm both new regressions fail.
- [ ] Change the workflow command to the asserted matrix-aware command and
  add lineterminator="\n" to all four existing csv.DictWriter constructors.
- [ ] Run focused and full pytest, Ruff, mypy; production validation and offline
  fixture benchmark. Check git diff --exit-code -- data reports.
- [ ] Obtain a fresh read-only review and resolve any in-scope findings.

### Task 2: Publish and verify both CI jobs

**Files:** This plan and docs/portfolio/release-preparation.md.

**Interfaces:** Normal main push to the existing origin; read-only GitHub CI API.

- [ ] Record the minimal repair and publication status in the release notes.
- [ ] Stage only the named files plus the design; check staged diff and commit.
- [ ] Push main without force, then verify remote SHA equals local HEAD.
- [ ] Read the push-triggered run for that SHA and require Python 3.11 and 3.12
  to complete successfully, including the candidate diff gate and fixture steps.
- [ ] Record exact local/remote results, leaving a clean synchronized worktree.
