from __future__ import annotations

import pytest

from mcp_route_reliability.hashing import tool_corpus_sha256
from mcp_route_reliability.models import InferenceQuery, ToolRecord
from mcp_route_reliability.routers.bm25 import BM25Router, tokenize


def make_tool(tool_id: str, description: str) -> ToolRecord:
    return ToolRecord(
        tool_id=tool_id,
        name=tool_id.rsplit(".", maxsplit=1)[-1],
        description=description,
        input_schema={
            "type": "object",
            "properties": {"search_query": {"type": "string"}},
        },
        server="test",
        domain="developer_tools",
        source_url="https://example.invalid/router-test",
        license="CC0-1.0",
    )


def make_query(text: str = "search Python code") -> InferenceQuery:
    return InferenceQuery(query_id="q_001", text=text, language="en")


def test_tokenize_mixed_chinese_and_english() -> None:
    tokens = tokenize("GitHub 搜索代码 repo_name")

    assert "github" in tokens
    assert "repo_name" in tokens
    assert {"搜", "索", "搜索", "代", "码", "代码"}.issubset(tokens)


def test_empty_corpus_and_invalid_top_k_are_rejected() -> None:
    router = BM25Router()
    with pytest.raises(ValueError, match="at least one tool"):
        router.rank(make_query(), [], top_k=1, query_set_hash="sha256:queries")
    with pytest.raises(ValueError, match="top_k"):
        router.rank(
            make_query(),
            [make_tool("tool.a", "Search source code")],
            top_k=0,
            query_set_hash="sha256:queries",
        )


def test_rank_returns_contiguous_unique_candidates_and_metadata() -> None:
    tools = [
        make_tool("tool.repo", "Search repositories by topic"),
        make_tool("tool.code", "Search Python source code by query"),
        make_tool("tool.issue", "Search issue tickets"),
    ]

    prediction = BM25Router().rank(
        make_query(), tools, top_k=2, query_set_hash="sha256:queries"
    )

    assert prediction.candidates[0].tool_id == "tool.code"
    assert [candidate.rank for candidate in prediction.candidates] == [1, 2]
    assert len({candidate.tool_id for candidate in prediction.candidates}) == 2
    assert prediction.tool_corpus_hash == tool_corpus_sha256(tools)
    assert prediction.query_set_hash == "sha256:queries"
    assert prediction.score_metadata.model_dump() == {
        "score_type": "similarity",
        "higher_is_better": True,
        "comparable_across_queries": False,
    }


def test_tied_scores_use_tool_id_as_stable_tie_break() -> None:
    tools = [
        make_tool("tool.z", "alpha capability"),
        make_tool("tool.a", "beta capability"),
    ]

    prediction = BM25Router().rank(
        make_query("unseen-token"), tools, top_k=5, query_set_hash="sha256:queries"
    )

    assert [candidate.tool_id for candidate in prediction.candidates] == ["tool.a", "tool.z"]
    assert [candidate.score for candidate in prediction.candidates] == [0.0, 0.0]


def test_negative_scores_remain_valid_and_sorted() -> None:
    tools = [
        make_tool("tool.a", "common alpha"),
        make_tool("tool.b", "common beta"),
    ]

    prediction = BM25Router().rank(
        make_query("common"), tools, top_k=2, query_set_hash="sha256:queries"
    )
    scores = [candidate.score for candidate in prediction.candidates]

    assert all(score is not None for score in scores)
    assert scores == sorted(scores, reverse=True)
    assert any(score < 0 for score in scores if score is not None)


def test_repeated_runs_are_identical() -> None:
    tools = [
        make_tool("tool.a", "Search code"),
        make_tool("tool.b", "Search repositories"),
    ]
    router = BM25Router()

    first = router.rank(make_query(), tools, top_k=2, query_set_hash="sha256:queries")
    second = router.rank(make_query(), tools, top_k=2, query_set_hash="sha256:queries")

    assert first == second
