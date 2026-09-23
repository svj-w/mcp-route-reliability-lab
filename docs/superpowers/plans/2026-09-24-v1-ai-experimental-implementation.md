# V1 AI Experimental Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a reproducible AI-annotated V1 experiment without representing AI work as independent human review.

**Architecture:** A new dataset builder consumes immutable copies of the V1 candidate and two AI CSVs, applies an explicit decision log, assigns intent families atomically to splits, then writes a derived dataset. The existing benchmark pipeline reads that frozen snapshot with a dedicated experimental validation profile. Source import, label decisions, split allocation, and evaluation remain separate units.

**Tech Stack:** Python 3.11/3.12, Pydantic v2, pytest, Typer, existing BM25 and sentence-transformers routers.

**Spec:** `docs/superpowers/specs/2026-09-24-v1-ai-experimental-design.md`

## Global Constraints

- Output name is `v1-ai-experimental`; never call it human-reviewed or stable V1.0.
- Keep `data/v1-candidate/`, `data/`, and `reports/v0.1/` unchanged.
- Raw AI inputs are immutable copies. Primary SHA-256 is `4497cdb9e0e53ab5d351e93fd5af7990cebc900e95216d6342d67e8b8d339377`; secondary SHA-256 is `9aefbd47bd2cec803da3aedd50a65a9c75bb67370c88e8b7f66fcd2fc24a7990`.
- Tool Corpus hash remains `sha256:8983296fa7cc6343c94a11fe15307e87f3bf5c6f03bcd036e971b9e9e18de069` unless Tool records change; this plan does not change them.
- Dataset acceptance requires 40 tools, 300 bases, 90 variants, 150 Chinese and 150 English bases, 100 bases per split, and zero cross-split normalized intent-family collisions. Report actual case counts after semantic corrections.
- Only 85 secondary rows share the revised primary text before post-hoc editing: 45 compare with an AI primary recheck and 40 compare with generator-provisional primary labels. Never call the combined 85 or the 15 revised rows independent inter-annotator agreement.
- No new paid service, external publication, GitHub push, or Test-driven retuning.

## Review Focus

1. Altered raw bytes or corpus hash must fail before labels are materialized (Task 1 test).
2. The 15 revised secondary rows must remain traceable to their original, mismatched-text rows (Task 2 test).
3. `v01_q_039` and the eight disputed `multi_valid` records must not retain schema-inconsistent routeable gold (Task 2 test).
4. `v1_q_183` variants must preserve the final base intent and split (Task 3 test).
5. A repeated template with different scenario numbers must never land on both Calibration and Test (Task 3 test).

---

### Task 1: Freeze and validate source annotations

**Files:**
- Create: `data/v1-ai-inputs/raw/completed-primary-recheck.csv`
- Create: `data/v1-ai-inputs/raw/completed-secondary.csv`
- Create: `src/mcp_route_reliability/ai_annotation_sources.py`
- Test: `tests/unit/test_ai_annotation_sources.py`

**Interfaces:**
- Consumes: `ToolRecord`, `Annotation`, `tool_corpus_sha256`, the candidate assignment CSVs.
- Produces: `CompletedRow(query_id, text, language, annotation)` and `load_completed_rows(path, expected_ids, tools, expected_sha256) -> dict[str, CompletedRow]`.

- [ ] **Step 1: Write a failing source-integrity test.**

```python
def test_rejects_tampered_ai_csv(tmp_path: Path, tiny_tools: list[ToolRecord]) -> None:
    source = tmp_path / "primary.csv"
    source.write_text("query_id,text\nchanged,changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_completed_rows(source, ("v01_q_001",), tiny_tools, "0" * 64)
```

- [ ] **Step 2: Run `py -3.11 -m pytest tests/unit/test_ai_annotation_sources.py -q` and confirm the missing-module failure.**
- [ ] **Step 3: Implement a strict CSV loader.** Read UTF-8 with BOM tolerance, require the ten assignment columns, compare exact ID sets and row counts, reject duplicate IDs, parse `|`-separated Tool IDs and booleans, instantiate `Annotation`, and compare the actual file SHA-256 before parsing. Keep `text` and `language` alongside the parsed annotation.

```python
import csv
import hashlib
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from mcp_route_reliability.annotation_workflow import ASSIGNMENT_FIELDS
from mcp_route_reliability.hashing import tool_corpus_sha256
from mcp_route_reliability.models import Annotation, Language, ToolRecord


@dataclass(frozen=True)
class CompletedRow:
    query_id: str
    text: str
    language: Language
    annotation: Annotation


def load_completed_rows(
    path: Path,
    expected_ids: Collection[str],
    tools: Sequence[ToolRecord],
    expected_sha256: str,
) -> dict[str, CompletedRow]:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"{path}: SHA-256 mismatch: {actual}")
    corpus_hash = tool_corpus_sha256(tools)
    valid_tools = {tool.tool_id for tool in tools}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != set(ASSIGNMENT_FIELDS):
            raise ValueError(f"{path}: unexpected CSV columns")
        source_rows = list(reader)
    rows: dict[str, CompletedRow] = {}
    for source in source_rows:
        query_id = source["query_id"].strip()
        if query_id in rows:
            raise ValueError(f"{path}: duplicate query_id {query_id}")
        if source["tool_corpus_hash"] != corpus_hash:
            raise ValueError(f"{path}: {query_id} has a wrong Tool Corpus hash")
        relevant = tuple(filter(None, source["relevant_tool_ids"].split("|")))
        if len(relevant) != len(set(relevant)) or not set(relevant) <= valid_tools:
            raise ValueError(f"{path}: {query_id} has invalid Tool IDs")
        value = source["should_abstain"].strip().lower()
        if value not in {"true", "false"}:
            raise ValueError(f"{path}: {query_id} has invalid should_abstain")
        language = source["language"].strip()
        if language not in {"zh", "en"}:
            raise ValueError(f"{path}: {query_id} has invalid language")
        annotation = Annotation.model_validate({
            "annotation_id": f"{path.stem}-{query_id}",
            "annotator_id": source["annotator_id"].strip(),
            "query_id": query_id,
            "relevant_tool_ids": relevant,
            "should_abstain": value == "true",
            "abstain_reason": source["abstain_reason"].strip() or None,
            "label_rationale": source["label_rationale"].strip(),
            "candidate_conflict_note": source["candidate_conflict_note"].strip() or None,
            "tool_corpus_hash": corpus_hash,
        })
        rows[query_id] = CompletedRow(
            query_id, source["text"], cast(Language, language), annotation
        )
    if set(rows) != set(expected_ids):
        raise ValueError(f"{path}: query IDs differ from the assignment")
    return rows
```

- [ ] **Step 4: Copy the two user CSVs byte-for-byte into `data/v1-ai-inputs/raw/`; compare `Get-FileHash -Algorithm SHA256` with the two Global Constraints values.** Use explicit file paths; do not edit the supplied originals.
- [ ] **Step 5: Add tests for valid 180/100 ID coverage, a duplicate ID, unknown Tool ID, and changed corpus hash; rerun the focused tests.**
- [ ] **Step 6: Stage only `agent/data/v1-ai-inputs/raw/`, `agent/src/mcp_route_reliability/ai_annotation_sources.py`, and `agent/tests/unit/test_ai_annotation_sources.py`, then commit with `git commit -m "feat: freeze AI annotation sources"`.**

### Task 2: Record post-hoc revisions and semantic decisions

**Files:**
- Create: `src/mcp_route_reliability/ai_label_decisions.py`
- Create: `data/v1-ai-inputs/decisions.jsonl`
- Test: `tests/unit/test_ai_label_decisions.py`

**Interfaces:**
- Consumes: `CompletedRow`, candidate `QueryCase`, candidate `Annotation`, `ToolRecord`.
- Produces: `ResolvedBase(query: QueryCase, final_annotation: Annotation, provenance: str)` and `resolve_bases(candidate_bases: Sequence[QueryCase], candidate_primary: Mapping[str, Annotation], primary_rows: Mapping[str, CompletedRow], secondary_rows: Mapping[str, CompletedRow], decisions: Sequence[Decision]) -> tuple[ResolvedBase, ...]`, plus 15 separately labeled post-hoc secondary revisions.

- [ ] **Step 1: Write failing tests for the known conflicts.**

```python
def test_known_semantic_corrections() -> None:
    root = Path(__file__).resolve().parents[2]
    decisions = load_decisions(root / "data" / "v1-ai-inputs" / "decisions.jsonl")
    gold = {row.query_id: row for row in decisions if row.target == "final_gold"}
    missing_root = gold["v01_q_039"]
    assert (missing_root.case_type, missing_root.should_abstain) == ("clarify", True)
    assert missing_root.abstain_reason == "insufficient_information"
    assert gold["v1_q_101"].relevant_tool_ids == ("memory.open_nodes",)
    assert gold["v1_q_195"].should_abstain is True  # repo_path is absent


def test_posthoc_revisions_are_not_raw_secondary() -> None:
    root = Path(__file__).resolve().parents[2]
    decisions = load_decisions(root / "data" / "v1-ai-inputs" / "decisions.jsonl")
    revisions = [row for row in decisions if row.target == "secondary_posthoc"]
    assert len(revisions) == 15
    assert all(row.reviewer_type == "ai_posthoc" for row in revisions)
```

- [ ] **Step 2: Run `py -3.11 -m pytest tests/unit/test_ai_label_decisions.py -q` and confirm failure.**
- [ ] **Step 3: Implement explicit decision parsing and application.** A JSONL row has `query_id`, `target` (`final_gold` or `secondary_posthoc`), `text`, `case_type`, `relevant_tool_ids`, `should_abstain`, `abstain_reason`, `rationale`, `reviewer_type`, `source_role`, and `source_query_id`. Reject unknown targets, duplicate `(query_id,target)`, wrong Tool IDs, or a secondary revision whose text differs from final text. Never overwrite the raw `CompletedRow`.

```python
class Decision(StrictModel):
    query_id: str
    target: Literal["final_gold", "secondary_posthoc"]
    text: str
    case_type: CaseType
    relevant_tool_ids: tuple[str, ...]
    should_abstain: bool
    abstain_reason: AbstainReason | None
    rationale: str
    reviewer_type: Literal["ai_posthoc"] = "ai_posthoc"
    source_role: Literal["primary_recheck", "secondary_raw", "candidate_primary"]
    source_query_id: str


def load_decisions(path: Path) -> tuple[Decision, ...]:
    values = tuple(
        Decision.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    keys = [(row.query_id, row.target) for row in values]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate AI decision target")
    return values
```
- [ ] **Step 4: Populate and review `decisions.jsonl`.** Add `final_gold` rows for the union of all 28 primary text rewrites, every primary-recheck label change against the candidate, and later adjudications; do not auto-accept a changed label without a decision. For secondary post-hoc rows, use the final primary text and these 15 ID/tool-set groups: `v01_q_001/004/041` → `git.diff_staged|git.diff_unstaged`; `009/012/028/065` → `filesystem.directory_tree|filesystem.list_directory`; `017/020/057/060` → `memory.open_nodes|memory.search_nodes`; `033/036/073/076` → `time.convert_time|time.get_current_time`. All 15 abstain as `ambiguous`; write a query-specific rationale. Keep each original secondary row unchanged.
- [ ] **Step 5: Decide the nine named structural conflicts from the Tool schemas.** `v01_q_039` stays text-unchanged and becomes `clarify / insufficient_information / filesystem.search_files`. `v1_q_101/127/142/205` become exact-name `hard_negative / memory.open_nodes`. `v1_q_195/197/213/238` become `clarify / insufficient_information / git.show`, since `repo_path` is required and absent. Review all 28 rewritten texts and every primary-recheck label change against the required Tool inputs; add explicit decisions for any further conflicts. Preserve unchanged candidate labels as `generator_provisional` when neither AI source covers them.
- [ ] **Step 6: Add a test that all 180 primary recheck rows are consumed, all 100 secondary source rows remain present, 15 revisions have distinct provenance, every final base label validates as `QueryCase`, and every changed label has a decision-log row; rerun focused tests.**
- [ ] **Step 7: Commit only Task 2 files with `git commit -m "feat: resolve AI annotation conflicts"`.**

### Task 3: Group intent families and synchronize variants

**Files:**
- Create: `src/mcp_route_reliability/intent_families.py`
- Test: `tests/unit/test_intent_families.py`
- Modify: `src/mcp_route_reliability/ai_label_decisions.py`

**Interfaces:**
- Consumes: final base `QueryCase` records, candidate variants, `normalize_query_text`.
- Produces: `intent_family_key(text: str) -> str`, `assign_family_splits(bases: Sequence[QueryCase], pinned_test_ids: set[str], seed: int = 20260924) -> dict[str, Split]`, and `synchronize_variants(candidate_variants: Sequence[QueryCase], final_bases: Mapping[str, QueryCase], split_by_id: Mapping[str, Split]) -> tuple[QueryCase, ...]`.

- [ ] **Step 1: Write failing family-isolation tests.**

```python
def test_scenario_numbers_do_not_separate_a_family() -> None:
    first = "Show /project (scenario 1)"
    second = "Show /project (scenario 81)"
    assert intent_family_key(first) == intent_family_key(second)


def test_family_allocator_never_crosses_splits() -> None:
    root = Path(__file__).resolve().parents[2]
    rows = load_queries(root / "data" / "v1-candidate" / "queries.jsonl")
    bases = [row for row in rows if row.perturbation_type is None]
    pinned = {row.base_query_id for row in rows if row.perturbation_type is not None}
    allocated = assign_family_splits(bases, pinned_test_ids=pinned, seed=20260924)
    assert len(allocated) == 300
    assert set(Counter(allocated.values()).values()) == {100}
    by_family: dict[str, set[Split]] = defaultdict(set)
    for row in bases:
        by_family[intent_family_key(row.text)].add(allocated[row.query_id])
    assert all(len(splits) == 1 for splits in by_family.values())
```

- [ ] **Step 2: Run `py -3.11 -m pytest tests/unit/test_intent_families.py -q` and confirm failure.**
- [ ] **Step 3: Implement suffix removal for English `(scenario N)`, `[V1 scenario N]` and Chinese `（场景 N）`, `（V1 场景 N）`, then call `normalize_query_text`.** A checked-in override map may join cross-language equivalents; it may not split identical normalized texts.

```python
SCENARIO_SUFFIX = re.compile(
    r"\s*(?:\((?:v1\s*)?(?:scenario|场景)\s*\d+\)|"
    r"\[(?:v1\s*)?scenario\s*\d+\])$",
    flags=re.IGNORECASE,
)


def intent_family_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return normalize_query_text(SCENARIO_SUFFIX.sub("", normalized))
```
- [ ] **Step 4: Implement deterministic whole-family allocation.** Pin families containing selected variant bases to Test (the current candidate pins 47 base rows in 29 families), then use bounded subset-sum allocation by group size to fill Test and Calibration to exactly 100 each; the remainder is Development. Sort keys before seeded tie-breaking. Fail with a capacity diagnostic if impossible. Record family key and allocation in `split-manifest.csv`.

```python
def choose_families(
    groups: Mapping[str, tuple[str, ...]], target: int, seed: int
) -> set[str]:
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    ordered = sorted(groups)
    random.Random(seed).shuffle(ordered)
    for key in ordered:
        size = len(groups[key])
        for count, selected in sorted(reachable.items(), reverse=True):
            new_count = count + size
            if new_count <= target and new_count not in reachable:
                reachable[new_count] = (*selected, key)
    if target not in reachable:
        raise ValueError(f"intent families cannot fill split capacity {target}")
    return set(reachable[target])
```
- [ ] **Step 5: Synchronize 90 variants.** Every variant inherits final base label and split. Rewrite `v1_q_183__paraphrase`, `__cross_language`, and `__noise` to express the new browse-versus-glob-search ambiguity, without Tool IDs in the text. If a variant base moves out of Test, select a replacement Test base in the same case/language stratum and regenerate its three variants; retain 30 of each perturbation type.
- [ ] **Step 6: Add tests for zero family collisions, exact 100/100/100 counts, `v1_q_183` semantic and label consistency, 90 variants, and deterministic allocation across two runs; rerun focused tests.**
- [ ] **Step 7: Commit Task 3 code and tests with `git commit -m "feat: isolate intent families across splits"`.**

### Task 4: Build and validate the derived snapshot

**Files:**
- Create: `src/mcp_route_reliability/ai_dataset.py`
- Create: `scripts/build_v1_ai_experimental.py`
- Modify: `src/mcp_route_reliability/validation.py`
- Test: `tests/integration/test_v1_ai_experimental.py`
- Generated: `data/v1-ai-experimental/tools.json`, `queries.jsonl`, `annotations/final.jsonl`, `annotations/secondary-posthoc.jsonl`, `raw/` copies, `decisions.jsonl` copy, `split-manifest.csv`, `source-manifest.json`, `DATASET_CARD.md`.

**Interfaces:**
- Consumes: Tasks 1–3 and `validate_dataset`.
- Produces: `build_ai_dataset(candidate_dir: Path, input_dir: Path, output_dir: Path) -> DatasetSummary` and `get_validation_profile("ai-experimental")`.

- [ ] **Step 1: Write a failing integration test.**

```python
def test_derived_snapshot_is_valid_and_not_human_reviewed(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    CANDIDATE = root / "data" / "v1-candidate"
    INPUTS = root / "data" / "v1-ai-inputs"
    summary = build_ai_dataset(CANDIDATE, INPUTS, tmp_path / "out")
    queries = load_queries(tmp_path / "out" / "queries.jsonl")
    assert summary.base_query_count == 300
    assert summary.variant_count == 90
    assert len({query.query_id for query in queries}) == 390
    card = (tmp_path / "out" / "DATASET_CARD.md").read_text(encoding="utf-8")
    assert "AI experimental" in card
    assert "not human-reviewed" in card
```

- [ ] **Step 2: Run `py -3.11 -m pytest tests/integration/test_v1_ai_experimental.py -q` and confirm failure.**
- [ ] **Step 3: Add `AI_EXPERIMENTAL_PROFILE` with 40 tools, 100/100/100 bases, 90 variants, 150/150 languages, and 30 of each perturbation type; leave `case_type_base_counts=None`.** Register it as `ai-experimental`. Keep `production` behavior unchanged.

```python
AI_EXPERIMENTAL_PROFILE = ValidationProfile(
    name="ai-experimental",
    minimum_tools=40,
    maximum_tools=40,
    split_base_counts={"development": 100, "calibration": 100, "test": 100},
    variant_count=90,
    allowed_case_types=ALL_CASE_TYPES,
    language_base_counts={"zh": 150, "en": 150},
    case_type_base_counts=None,
    perturbation_counts={"paraphrase": 30, "cross_language": 30, "noise": 30},
)
```
- [ ] **Step 4: Implement the builder.** Fail before writes on invalid inputs or unresolved decisions; stage a complete output directory including exact raw-input and decision-log copies; validate all 390 rows and family isolation; emit canonical JSONL, manifest hashes, provenance counts (`180` primary rechecked, `120` not primary rechecked, `100` secondary raw, `15` post-hoc, plus the computed count still generator-provisional), actual case counts, and the Dataset Card; atomically rename the stage to a previously nonexistent output. A second build to a separate output must produce byte-identical data files. Do not embed timestamps or machine-local source paths in deterministic files.
- [ ] **Step 5: Add negative tests for edited raw bytes, duplicate/missing decision, wrong hash, family crossing, mismatched variant label, and refusing to overwrite an existing output. Add a test that `data/v1-candidate` files are unchanged.**
- [ ] **Step 6: Generate `data/v1-ai-experimental/` in a fresh path and run `py -3.11 -m mcp_route_reliability.cli validate --tools data/v1-ai-experimental/tools.json --queries data/v1-ai-experimental/queries.jsonl --profile ai-experimental`; verify all tests, `ruff`, and `mypy`.**
- [ ] **Step 7: Commit Task 4 code, tests, and deterministic data with `git commit -m "data: freeze V1 AI experimental snapshot"`.**

### Task 5: Re-run both baselines and document the claims

**Files:**
- Modify: `src/mcp_route_reliability/cli.py`
- Create: `configs/v1-ai-bm25.yaml`, `configs/v1-ai-embedding.yaml`
- Modify: `README.md`
- Create: `data/v1-ai-experimental/ANNOTATION_REPORT.md`
- Generated: `reports/v1-ai-experimental/bm25/`, `reports/v1-ai-experimental/embedding/`
- Test: `tests/integration/test_v1_ai_benchmark.py`

**Interfaces:**
- Consumes: Task 4 frozen dataset and the existing `mrrl benchmark` command.
- Produces: two provenance-bound benchmark reports and documentation that distinguishes raw 85-row diagnostics from 100-row post-hoc concordance.

- [ ] **Step 1: Write failing CLI/report and comparison-population tests.**

```python
def test_ai_profile_is_explicitly_non_human() -> None:
    raw = json.loads(Path("configs/bm25.yaml").read_text(encoding="utf-8"))
    config = BenchmarkConfig.model_validate({**raw, "profile": "ai-experimental"})
    assert config.profile == "ai-experimental"
    assert "AI" in profile_limitation(config.profile)


def test_comparable_source_populations() -> None:
    root = Path(__file__).resolve().parents[2]
    tools = load_tools(root / "data" / "v1-candidate" / "tools.json")
    inputs = root / "data" / "v1-ai-inputs" / "raw"
    assignments = root / "data" / "v1-candidate" / "annotations"
    def ids(path: Path) -> set[str]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return {row["query_id"] for row in csv.DictReader(handle)}
    primary = load_completed_rows(
        inputs / "completed-primary-recheck.csv",
        ids(assignments / "primary-recheck-assignment.csv"),
        tools,
        "4497cdb9e0e53ab5d351e93fd5af7990cebc900e95216d6342d67e8b8d339377",
    )
    secondary = load_completed_rows(
        inputs / "completed-secondary.csv",
        ids(assignments / "secondary-assignment.csv"),
        tools,
        "9aefbd47bd2cec803da3aedd50a65a9c75bb67370c88e8b7f66fcd2fc24a7990",
    )
    same_recheck = {
        query_id for query_id, row in secondary.items()
        if query_id in primary and primary[query_id].text == row.text
    }
    changed_text = {
        query_id for query_id, row in secondary.items()
        if query_id in primary and primary[query_id].text != row.text
    }
    candidate_only = len(secondary) - len(primary.keys() & secondary.keys())
    assert (len(same_recheck), len(changed_text), candidate_only) == (45, 15, 40)
```

- [ ] **Step 2: Run `py -3.11 -m pytest tests/integration/test_v1_ai_benchmark.py -q` and confirm failure.**
- [ ] **Step 3: Extend `BenchmarkConfig.profile` and the report limitation branch to support `ai-experimental` explicitly.** Do not route it through generic `production` wording. Add two configs by copying existing V0.1 settings and changing only profile and seed for this dataset.

```python
def profile_limitation(profile: str) -> str:
    if profile == "ai-experimental":
        return (
            "AI-annotated synthetic experiment with post-hoc revisions and "
            "generator-provisional labels; not independent human review or a safety guarantee."
        )
    if profile == "fixture":
        return "The fixture uses minimum accepted N=3 and is explicitly non-production."
    if profile == "v0.1":
        return (
            "V0.1 contains 100 author-created, class-balanced queries and is explicitly "
            "non-production."
        )
    return (
        "The production-size profile remains a benchmark artifact, not a safety "
        "certification."
    )
```
- [ ] **Step 4: Compute annotation diagnostics from same-text raw rows only.** For the 45 same-text IDs covered by the AI primary recheck, compare secondary against that recheck; for the remaining 40 clear/multi-valid IDs, compare secondary against generator-provisional primary labels. Use `cohen_kappa` and `mean_set_jaccard` on each subgroup and optionally on the combined 85, with sample size and source mixture explicit. Report constant-label Kappa as unavailable, not perfect. Separately report 100-row post-hoc concordance as descriptive, with 15 revised IDs and eight named multi-valid disagreements; do not call any of these a human or blinded agreement estimate. Include the computed generator-provisional count, the 120 not-primary-rechecked count, and actual case distribution.
- [ ] **Step 5: Freeze configs and run BM25, then embedding, each into a fresh run directory under `reports/v1-ai-experimental/`.** Use `mrrl benchmark --config configs/v1-ai-bm25.yaml --tools data/v1-ai-experimental/tools.json --queries data/v1-ai-experimental/queries.jsonl --run-dir reports/v1-ai-experimental/bm25` and the analogous embedding paths. If the pinned embedding model is unavailable in the restricted environment, leave that report absent and mark the release incomplete; never fabricate metrics or silently switch models.
- [ ] **Step 6: Check each manifest's Tool Corpus and query-set hashes, source counts, Test-once lifecycle, policy binding, figure files, error slices, and reliability-card limitations. Run `py -3.11 -m pytest -q`, `py -3.11 -m ruff check .`, and `py -3.11 -m mypy src`.**
- [ ] **Step 7: Update README with exact reproducible commands, real measured metrics, failure examples, AI provenance, template-family caveat, and non-production scope. Commit only Task 5 files with `git commit -m "docs: report V1 AI experimental results"`.**

## Completion gate

Inspect `git status --short -- agent` for unintended changes, verify the source CSV hashes again, and compare the generated dataset hash against both run manifests. If any Task 4 or Task 5 acceptance check fails, report the specific missing artifact or invariant rather than marking V1 AI experimental complete.
