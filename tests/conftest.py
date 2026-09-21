from __future__ import annotations

from collections.abc import Sequence

import pytest

from mcp_route_reliability.hashing import tool_corpus_sha256
from mcp_route_reliability.models import QueryCase, ToolRecord


def make_tiny_tools(count: int = 10) -> list[ToolRecord]:
    return [
        ToolRecord(
            tool_id=f"synthetic.tool_{index:02d}",
            name=f"tool_{index:02d}",
            description=f"Synthetic capability number {index} for deterministic tests.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
            server="synthetic",
            domain=f"domain_{index % 5}",
            source_url="https://example.invalid/synthetic-fixture",
            source_commit=None,
            license="CC0-1.0",
        )
        for index in range(count)
    ]


def make_tiny_queries(tools: Sequence[ToolRecord]) -> list[QueryCase]:
    corpus_hash = tool_corpus_sha256(tools)
    split_by_index = ["development"] * 8 + ["calibration"] * 6 + ["test"] * 6
    cases: list[QueryCase] = []
    for index, split in enumerate(split_by_index):
        kind = index % 5
        language = "zh" if index % 2 == 0 else "en"
        common = {
            "query_id": f"q_{index:03d}",
            "base_query_id": f"q_{index:03d}",
            "text": (
                f"请执行第 {index} 个合成能力"
                if language == "zh"
                else f"Run synthetic capability request {index}"
            ),
            "language": language,
            "difficulty": ("easy", "medium", "hard")[index % 3],
            "split": split,
            "label_rationale": f"Deterministic fixture label {index}.",
            "tool_corpus_hash": corpus_hash,
        }
        if kind == 0:
            case = QueryCase(
                **common,
                case_type="clear",
                relevant_tool_ids=(tools[index % len(tools)].tool_id,),
                should_abstain=False,
                abstain_reason=None,
            )
        elif kind == 1:
            case = QueryCase(
                **common,
                case_type="hard_negative",
                relevant_tool_ids=(tools[index % len(tools)].tool_id,),
                should_abstain=False,
                abstain_reason=None,
                candidate_conflict_note="A neighboring synthetic tool is lexically similar.",
            )
        elif kind == 2:
            case = QueryCase(
                **common,
                case_type="multi_valid",
                relevant_tool_ids=(
                    tools[index % len(tools)].tool_id,
                    tools[(index + 1) % len(tools)].tool_id,
                ),
                should_abstain=False,
                abstain_reason=None,
            )
        elif kind == 3:
            case = QueryCase(
                **common,
                case_type="clarify",
                relevant_tool_ids=(tools[index % len(tools)].tool_id,),
                should_abstain=True,
                abstain_reason="insufficient_information",
            )
        else:
            case = QueryCase(
                **common,
                case_type="no_match",
                relevant_tool_ids=(),
                should_abstain=True,
                abstain_reason="no_match",
            )
        cases.append(case)
    return cases


@pytest.fixture
def tiny_tools() -> list[ToolRecord]:
    return make_tiny_tools()


@pytest.fixture
def tiny_queries(tiny_tools: list[ToolRecord]) -> list[QueryCase]:
    return make_tiny_queries(tiny_tools)
