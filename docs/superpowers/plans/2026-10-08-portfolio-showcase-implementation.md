# AI Agent Portfolio Showcase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing evaluation toolkit easy to assess for an AI application/Agent engineering role without overstating its maturity or the student's personal contribution.

**Architecture:** The English README becomes the navigation hub; an English offline demo guide demonstrates the existing CLI, and a Chinese interview guide translates verified evidence into adaptable talking points. All numbers and claims are read from already frozen artifacts. This is documentation-only; no dataset, benchmark, model, or application code changes.

**Tech Stack:** Markdown, PowerShell, the existing Python 3.11/3.12 `mrrl` CLI and committed JSON/CSV report artifacts.

**Spec:** `docs/superpowers/specs/2026-10-08-portfolio-showcase-design.md`

## Global Constraints

- Target audience: AI application/Agent engineering recruiters; English repository entry point, Chinese job-application and interview material.
- Keep the project positioned as an evaluation toolkit, not a production Agent or a research paper.
- Do not modify V0.1 or V1 AI experimental data, source CSVs, reports, metrics, code, or dependencies; do not push to GitHub.
- Cite frozen V1 facts only: 40 tools, 300 bases, 90 variants; Test has 100 bases plus 90 paired variants; 112 base labels remain generator-provisional.
- BM25 Test Top-1 is `0.173684210526316` and multilingual MiniLM Test Top-1 is `0.321052631578947`; both 5% target-risk policies are unavailable. Round displayed values to four decimals and link to the report files.
- Never claim independent human review, a safe deployment threshold, 190 independent intents, stable V1.0, sole authorship, team size, internship impact, or real-world deployment.
- The default demo must work offline after installation using the fixture BM25 path; embedding is optional and may need a model download.

## Review Focus

1. A recruiter opens the README first screen: Task 1 checks that the V1 AI experimental boundary appears before the V0.1 historical banner.
2. A number from V0.1 is accidentally presented as V1: Task 1 compares the displayed Top-1 numbers and sample counts with frozen V1 `metrics.json` and `source-manifest.json`.
3. A demo run directory already exists or a model is not cached: Task 2 runs the BM25 fixture into a fresh `runs/` path and explains how to choose another path; embedding is explicitly optional.
4. A relative Markdown link breaks after publication: Task 3 checks local links from all three portfolio documents.
5. A resume bullet implies unsupported authorship or production safety: Task 3 compares every claim with repository evidence and marks personal-contribution language as editable only after user verification.

---

### Task 1: Reframe the English README landing section

**Files:**
- Modify: `README.md`
- Read: `data/v1-ai-experimental/source-manifest.json`
- Read: `reports/v1-ai-experimental/{bm25,embedding}/metrics.json`

**Interfaces:**
- Consumes: existing frozen metrics, Dataset Card, annotation report, and reliability-card links.
- Produces: an English entry point that Task 2 and Task 3 can link back to without duplicating details.

- [ ] **Step 1: Confirm the current landing issue.** Run `Get-Content README.md -TotalCount 20`; the first milestone banner currently describes V0.1 rather than V1 AI experimental. Record the old order before editing.
- [ ] **Step 2: Independently read the source values.** Run the following from the repository root; expect `300`, `90`, `112`, `0.173684210526316`, and `0.321052631578947` before rounding:

```powershell
$manifest = Get-Content -Raw data/v1-ai-experimental/source-manifest.json | ConvertFrom-Json
$bm25 = Get-Content -Raw reports/v1-ai-experimental/bm25/metrics.json | ConvertFrom-Json
$embedding = Get-Content -Raw reports/v1-ai-experimental/embedding/metrics.json | ConvertFrom-Json
$manifest.base_count, $manifest.variant_count, $manifest.provenance_counts.generator_provisional
$bm25.ranking.top1_accuracy, $embedding.ranking.top1_accuracy
```

- [ ] **Step 3: Edit only `README.md`.** Put an immediately visible V1 AI experimental banner and short value proposition below the title, followed by a compact “What this demonstrates” paragraph. Move or summarize the existing V1 result table near the top, keeping numbers and caveats unchanged. Keep the V0.1 benchmark and candidate sections as historical milestones rather than deleting them. Link the architecture, Dataset Card, annotation report, both reliability cards, and one existing figure. Task 3 adds links to the two new guides after those files exist, so this task does not ship broken links.

  Suggested opening copy: “An evidence-first toolkit for evaluating whether an MCP tool router should route, clarify, or abstain. Latest milestone: V1 AI experimental (AI-annotated, not independently human-reviewed, not a production safety certification).” The visible engineering proof points are provenance-bound predictions, template-family split isolation, calibration-only fitting, and fail-closed policy output.
- [ ] **Step 4: Verify section order and claims.** `Select-String -Path README.md -Pattern 'V1 AI experimental','V0.1 benchmark'` must show V1 first. The first 30 lines must contain `AI-annotated`, `not independently human-reviewed`, and `not a production safety` or clearly equivalent wording. Compare the displayed Top-1 values to Step 2 and read the report sample size (`190`) while explaining 100 bases plus 90 variants.
- [ ] **Step 5: Commit only the README.** Run `git add -- README.md`, `git diff --cached --check`, and `git commit -m "docs: foreground AI experimental engineering case"` from the `agent` project root or use the parent Git root with the `agent/` prefix. Do not stage other Desktop paths.

### Task 2: Add the offline demonstration guide

**Files:**
- Create: `docs/portfolio/demo.md`
- Read: `configs/fixture-bm25.yaml`, `tests/fixtures/tools.jsonl`, `tests/fixtures/queries.jsonl`
- Read: `reports/v1-ai-experimental/{bm25,embedding}/policy.json`

**Interfaces:**
- Consumes: the existing `mrrl validate` and `mrrl benchmark` commands and a fresh ignored `runs/` directory.
- Produces: a guided, post-installation, offline five-minute demonstration; it does not regenerate the V1 frozen reports.

- [ ] **Step 1: Verify the guide does not exist and the fixture is intact.** Run `Test-Path docs/portfolio/demo.md` (expect `False`) and `Test-Path tests/fixtures/tools.jsonl,tests/fixtures/queries.jsonl,configs/fixture-bm25.yaml` (all `True`).
- [ ] **Step 2: Create `docs/portfolio/demo.md`.** State “after installation” in the time claim. Include these exact default commands and explain the output paths and expected checks:

```powershell
mrrl validate --tools tests/fixtures/tools.jsonl `
  --queries tests/fixtures/queries.jsonl --profile fixture
mrrl benchmark --config configs/fixture-bm25.yaml `
  --tools tests/fixtures/tools.jsonl `
  --queries tests/fixtures/queries.jsonl `
  --run-dir runs/portfolio-fixture
```

  After the commands, direct readers to `runs/portfolio-fixture/reliability-card.md`, `metrics.json`, `errors.csv`, and `manifest.json`. Then link to both committed V1 reliability cards and their `policy.json` files, explicitly distinguishing fixture demonstration from V1 evidence. Include a short failure-recovery section: install dependencies in Python 3.11/3.12, use a new run name if `runs/portfolio-fixture` exists, and treat the embedding baseline as optional because its pinned model may need a download.
- [ ] **Step 3: Exercise the commands in a fresh path.** If `runs/portfolio-fixture-verification` already exists, choose a different unused name rather than deleting it. Execute the same validate command, then the benchmark command with `--run-dir runs/portfolio-fixture-verification`. Check that its four named artifacts exist and inspect `reports/v1-ai-experimental/{bm25,embedding}/policy.json`: both `available` values must be `false`.
- [ ] **Step 4: Check the guide is self-contained.** A reader must find installation prerequisites by link, understand that fixture numbers differ from V1, and know how to inspect one routing error without a network connection. Commit only `docs/portfolio/demo.md` with message `docs: add offline portfolio demo`.

### Task 3: Add the Chinese interview guide and audit the documentation

**Files:**
- Create: `docs/portfolio/interview-zh.md`
- Modify: `README.md` (add the two portfolio-guide links)
- Verify: `docs/portfolio/demo.md`
- Read: `data/v1-ai-experimental/ANNOTATION_REPORT.md`, both V1 `errors.csv` files

**Interfaces:**
- Consumes: the final README framing and the frozen V1 reports.
- Produces: reusable, truthful Chinese application material; no personal work history is invented.

- [ ] **Step 1: Confirm the gap.** Run `Test-Path docs/portfolio/interview-zh.md` (expect `False`). Read one real example from `reports/v1-ai-experimental/embedding/errors.csv`; for `v01_q_015`, gold is `git.log` and `git.show` ranks ahead of it.
- [ ] **Step 2: Create `docs/portfolio/interview-zh.md`.** Include one-line positioning, two adaptable resume bullets, a 60-second explanation, a four-stage architecture explanation (Tool Corpus → routing → calibration → frozen Test/report), and three likely questions with answers: why abstention matters, how to interpret the low embedding Top-1 without inventing a cause, and why this is not stable V1.0. Mark personal-contribution verbs as templates that the user must rewrite to match what they personally implemented. Include the `v01_q_015` error as an inspectable example and the explicit claim red lines from Global Constraints.

  One-line positioning: “MCP Route Reliability Lab 是一个评测 Agent 工具路由是否应调用、澄清或拒绝的可复现实验项目。” The two resume bullets should describe (1) the provenance-bound evaluation pipeline and (2) the 40-tool/390-record AI experimental dataset plus BM25/embedding comparison. Prefix the section with “仅保留你能亲自解释并证明参与的工作；以下不是可直接照抄的个人经历”。
- [ ] **Step 3: Add both guide links to the README.** Link `docs/portfolio/demo.md` and `docs/portfolio/interview-zh.md` from the landing section once both files exist.
- [ ] **Step 4: Audit evidence and links.** Confirm all local Markdown destinations in the three documents resolve from their containing directories; inspect every displayed number against the V1 manifest/metrics; ensure the interview guide does not claim sole authorship, human review, production deployment, or a usable risk threshold. The negative-claims section may name those phrases only to reject them. Run this read-only link check from the project root:

```powershell
$docs = @('README.md', 'docs/portfolio/demo.md', 'docs/portfolio/interview-zh.md')
foreach ($doc in $docs) {
  $content = Get-Content -Raw -LiteralPath $doc
  foreach ($match in [regex]::Matches($content, '\]\(([^)#]+)')) {
    $target = $match.Groups[1].Value
    if ($target -match '^(https?://|#)') { continue }
    $resolved = Join-Path (Split-Path (Resolve-Path -LiteralPath $doc).Path -Parent) $target
    if (-not (Test-Path -LiteralPath $resolved)) { throw "Broken link in ${doc}: $target" }
  }
}
```
- [ ] **Step 5: Run the repository guardrail suite.** Run `py -3.11 -m pytest -q`, `py -3.11 -m ruff check .`, and `py -3.11 -m mypy src` in a supported environment. Review `git -C D:\lenovo\Documents\OneDrive\Desktop status --short -- agent` and `git -C D:\lenovo\Documents\OneDrive\Desktop diff --check -- agent` for unintended changes. Commit only `docs/portfolio/interview-zh.md` and `README.md` with message `docs: add truthful Agent engineering interview guide`.

## Completion gate

The final handoff links README, demo guide, and interview guide; reports the actual demo and guardrail results; and explicitly says that neither independent human review nor a GitHub push was performed. If a prerequisite or command cannot run locally, report the exact step and keep the guide's claims narrow instead of inventing a passing result.
