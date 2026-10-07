from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_route_reliability.ai_annotation_sources import load_completed_rows
from mcp_route_reliability.ai_label_decisions import (
    build_secondary_posthoc,
    load_decisions,
    resolve_bases,
)
from mcp_route_reliability.annotation_workflow import load_manifest_query_ids
from mcp_route_reliability.datasets import load_annotations, load_queries, load_tools

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "data" / "v1-candidate"
INPUTS = ROOT / "data" / "v1-ai-inputs"
PRIMARY_SHA256 = "4497cdb9e0e53ab5d351e93fd5af7990cebc900e95216d6342d67e8b8d339377"
SECONDARY_SHA256 = "9aefbd47bd2cec803da3aedd50a65a9c75bb67370c88e8b7f66fcd2fc24a7990"


def load_sources():  # type: ignore[no-untyped-def]
    tools = load_tools(CANDIDATE / "tools.json")
    queries = load_queries(CANDIDATE / "queries.jsonl")
    primary = load_annotations(CANDIDATE / "annotations" / "primary.jsonl")
    recheck = load_completed_rows(
        INPUTS / "raw" / "completed-primary-recheck.csv",
        load_manifest_query_ids(CANDIDATE / "annotations" / "primary-recheck-assignment.csv"),
        tools,
        PRIMARY_SHA256,
    )
    secondary = load_completed_rows(
        INPUTS / "raw" / "completed-secondary.csv",
        load_manifest_query_ids(CANDIDATE / "annotations" / "secondary-sample-manifest.csv"),
        tools,
        SECONDARY_SHA256,
    )
    return tools, queries, primary, recheck, secondary


def test_decisions_cover_known_conflicts_and_posthoc_rows() -> None:
    decisions = load_decisions(INPUTS / "decisions.jsonl")
    final = {row.query_id: row for row in decisions if row.target == "final_gold"}
    posthoc = [row for row in decisions if row.target == "secondary_posthoc"]
    assert len(posthoc) == 15
    assert final["v01_q_039"].case_type == "clarify"
    assert final["v01_q_039"].abstain_reason == "insufficient_information"
    for query_id in ("v1_q_101", "v1_q_127", "v1_q_142", "v1_q_205"):
        assert final[query_id].relevant_tool_ids == ("memory.open_nodes",)
        assert final[query_id].case_type == "hard_negative"
    for query_id in ("v1_q_195", "v1_q_197", "v1_q_213", "v1_q_238"):
        assert final[query_id].relevant_tool_ids == ("git.show",)
        assert final[query_id].abstain_reason == "insufficient_information"


def test_resolved_bases_keep_original_and_posthoc_distinct() -> None:
    tools, queries, primary, recheck, secondary = load_sources()
    decisions = load_decisions(INPUTS / "decisions.jsonl")
    bases = [row for row in queries if row.perturbation_type is None]
    resolved = resolve_bases(bases, primary, recheck, decisions, tools)
    by_id = {row.query.query_id: row for row in resolved}
    assert len(by_id) == 300
    assert by_id["v01_q_039"].query.should_abstain is True
    assert by_id["v1_q_101"].query.case_type == "hard_negative"
    assert by_id["v1_q_195"].query.case_type == "clarify"
    assert by_id["v01_q_001"].query.text == recheck["v01_q_001"].text
    assert by_id["v01_q_001"].query.text != secondary["v01_q_001"].text
    revisions = build_secondary_posthoc(secondary, by_id, decisions, tools)
    assert len(revisions) == 15
    assert revisions["v01_q_001"].annotator_id.startswith("codex_ai_posthoc_")
    assert revisions["v01_q_001"].relevant_tool_ids == (
        "git.diff_staged", "git.diff_unstaged"
    )


def test_duplicate_decision_target_is_rejected(tmp_path: Path) -> None:
    row = {
        "query_id": "q_001",
        "target": "final_gold",
        "text": "Read a file",
        "case_type": "clear",
        "relevant_tool_ids": ["filesystem.read_text_file"],
        "should_abstain": False,
        "abstain_reason": None,
        "rationale": "The file path is supplied.",
        "reviewer_type": "ai_posthoc",
        "source_role": "candidate_primary",
        "source_query_id": "q_001",
    }
    path = tmp_path / "decisions.jsonl"
    path.write_text("\n".join((json.dumps(row), json.dumps(row))) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_decisions(path)
