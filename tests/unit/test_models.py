from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from mcp_route_reliability.models import (
    InferenceQuery,
    QueryCase,
    RankedCandidate,
    RouterPrediction,
    ScoreMetadata,
)


def make_case() -> QueryCase:
    return QueryCase(
        query_id="q_001",
        base_query_id="q_001",
        text="在仓库里搜索 FastAPI 代码",
        language="zh",
        case_type="clear",
        difficulty="medium",
        relevant_tool_ids=("github.search_code",),
        should_abstain=False,
        abstain_reason=None,
        split="test",
        label_rationale="The query explicitly asks for code search.",
        candidate_conflict_note=None,
        tool_corpus_hash="sha256:corpus",
    )


def similarity_metadata() -> ScoreMetadata:
    return ScoreMetadata(
        score_type="similarity",
        higher_is_better=True,
        comparable_across_queries=False,
    )


def make_prediction(*candidates: RankedCandidate, error: str | None = None) -> RouterPrediction:
    return RouterPrediction(
        query_id="q_001",
        router_id="bm25",
        router_version="1.0.0",
        model_version=None,
        tool_corpus_hash="sha256:corpus",
        query_set_hash="sha256:queries",
        score_metadata=similarity_metadata(),
        candidates=candidates,
        latency_ms=1.5,
        error=error,
    )


def test_query_case_requires_explicit_inference_conversion() -> None:
    case = make_case()

    assert not isinstance(case, InferenceQuery)
    inference = case.to_inference_query()

    assert type(inference) is InferenceQuery
    assert inference.model_dump() == {
        "schema_version": "1.0",
        "query_id": "q_001",
        "text": "在仓库里搜索 FastAPI 代码",
        "language": "zh",
    }
    forbidden = {"relevant_tool_ids", "case_type", "should_abstain", "abstain_reason"}
    assert forbidden.isdisjoint(inference.model_dump())


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_candidate_rejects_non_finite_numeric_scores(value: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        RankedCandidate(tool_id="tool.a", score=value, rank=1)


def test_prediction_rejects_duplicate_tool_ids() -> None:
    candidates = (
        RankedCandidate(tool_id="tool.a", score=0.8, rank=1),
        RankedCandidate(tool_id="tool.a", score=0.7, rank=2),
    )

    with pytest.raises(ValidationError, match="unique"):
        make_prediction(*candidates)


def test_prediction_rejects_non_contiguous_ranks() -> None:
    candidates = (
        RankedCandidate(tool_id="tool.a", score=0.8, rank=1),
        RankedCandidate(tool_id="tool.b", score=0.7, rank=3),
    )

    with pytest.raises(ValidationError, match="contiguous"):
        make_prediction(*candidates)


def test_distance_scores_must_be_lower_is_better() -> None:
    with pytest.raises(ValidationError, match="distance"):
        ScoreMetadata(
            score_type="distance",
            higher_is_better=True,
            comparable_across_queries=False,
        )


def test_rank_only_requires_null_scores() -> None:
    metadata = ScoreMetadata(
        score_type="rank_only",
        higher_is_better=False,
        comparable_across_queries=False,
    )

    valid = RouterPrediction(
        query_id="q_001",
        router_id="external",
        router_version="1.0.0",
        tool_corpus_hash="sha256:corpus",
        query_set_hash="sha256:queries",
        score_metadata=metadata,
        candidates=(RankedCandidate(tool_id="tool.a", score=None, rank=1),),
        latency_ms=None,
        error=None,
    )
    assert valid.candidates[0].score is None

    with pytest.raises(ValidationError, match="rank_only"):
        RouterPrediction(
            **{
                **valid.model_dump(),
                "candidates": [{"tool_id": "tool.a", "score": -1.0, "rank": 1}],
            }
        )


def test_empty_candidates_require_router_error() -> None:
    with pytest.raises(ValidationError, match="error"):
        make_prediction()

    failed = make_prediction(error="router timeout")
    assert failed.candidates == ()
    assert failed.error == "router timeout"
