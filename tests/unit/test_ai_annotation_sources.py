from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from mcp_route_reliability.ai_annotation_sources import load_completed_rows
from mcp_route_reliability.annotation_workflow import ASSIGNMENT_FIELDS, load_manifest_query_ids
from mcp_route_reliability.datasets import load_tools
from mcp_route_reliability.hashing import tool_corpus_sha256
from mcp_route_reliability.models import ToolRecord


def write_source(path: Path, tools: list[ToolRecord], **changes: str) -> str:
    row = {
        "query_id": "q_001",
        "text": "Read /project/notes.txt",
        "language": "en",
        "relevant_tool_ids": tools[0].tool_id,
        "should_abstain": "false",
        "abstain_reason": "",
        "label_rationale": "The named Tool can read the file.",
        "candidate_conflict_note": "",
        "annotator_id": "ai-reviewer",
        "tool_corpus_hash": tool_corpus_sha256(tools),
    }
    row.update(changes)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ASSIGNMENT_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_completed_rows_preserves_text_and_label(
    tmp_path: Path, tiny_tools: list[ToolRecord]
) -> None:
    source = tmp_path / "primary.csv"
    digest = write_source(source, tiny_tools)
    rows = load_completed_rows(source, {"q_001"}, tiny_tools, digest)
    assert rows["q_001"].text == "Read /project/notes.txt"
    assert rows["q_001"].annotation.relevant_tool_ids == (tiny_tools[0].tool_id,)


def test_rejects_tampered_file_before_parsing(
    tmp_path: Path, tiny_tools: list[ToolRecord]
) -> None:
    source = tmp_path / "primary.csv"
    write_source(source, tiny_tools)
    with pytest.raises(ValueError, match="SHA-256"):
        load_completed_rows(source, {"q_001"}, tiny_tools, "0" * 64)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"relevant_tool_ids": "unknown.tool"}, "unknown Tool"),
        ({"should_abstain": "maybe"}, "should_abstain"),
        ({"tool_corpus_hash": "sha256:wrong"}, "Tool Corpus"),
    ],
)
def test_rejects_invalid_label_fields(
    tmp_path: Path,
    tiny_tools: list[ToolRecord],
    changes: dict[str, str],
    expected: str,
) -> None:
    source = tmp_path / "primary.csv"
    digest = write_source(source, tiny_tools, **changes)
    with pytest.raises(ValueError, match=expected):
        load_completed_rows(source, {"q_001"}, tiny_tools, digest)


def test_rejects_missing_or_duplicate_ids(
    tmp_path: Path, tiny_tools: list[ToolRecord]
) -> None:
    source = tmp_path / "primary.csv"
    digest = write_source(source, tiny_tools)
    with pytest.raises(ValueError, match="query IDs"):
        load_completed_rows(source, {"q_001", "q_002"}, tiny_tools, digest)
    with source.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "q_001", "Read /project/notes.txt", "en", tiny_tools[0].tool_id,
                "false", "", "The named Tool can read the file.", "", "ai-reviewer",
                tool_corpus_sha256(tiny_tools),
            ]
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="duplicate query_id"):
        load_completed_rows(source, {"q_001"}, tiny_tools, digest)


def test_frozen_ai_sources_cover_the_assignments() -> None:
    root = Path(__file__).resolve().parents[2]
    candidate = root / "data" / "v1-candidate"
    inputs = root / "data" / "v1-ai-inputs" / "raw"
    tools = load_tools(candidate / "tools.json")
    primary = load_completed_rows(
        inputs / "completed-primary-recheck.csv",
        load_manifest_query_ids(candidate / "annotations" / "primary-recheck-assignment.csv"),
        tools,
        "4497cdb9e0e53ab5d351e93fd5af7990cebc900e95216d6342d67e8b8d339377",
    )
    secondary = load_completed_rows(
        inputs / "completed-secondary.csv",
        load_manifest_query_ids(candidate / "annotations" / "secondary-sample-manifest.csv"),
        tools,
        "9aefbd47bd2cec803da3aedd50a65a9c75bb67370c88e8b7f66fcd2fc24a7990",
    )
    assert len(primary) == 180
    assert len(secondary) == 100
    same_text_recheck = {
        query_id
        for query_id, row in secondary.items()
        if query_id in primary and row.text == primary[query_id].text
    }
    changed_text = {
        query_id
        for query_id, row in secondary.items()
        if query_id in primary and row.text != primary[query_id].text
    }
    assert len(same_text_recheck) == 45
    assert len(changed_text) == 15
