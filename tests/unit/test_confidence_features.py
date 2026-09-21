from __future__ import annotations

import math
from typing import cast

import pytest
from pydantic import ValidationError

from mcp_route_reliability.confidence.features import (
    FeatureNormalizer,
    extract_confidence_features,
)
from mcp_route_reliability.models import (
    InferenceQuery,
    QueryCase,
    RankedCandidate,
    RouterPrediction,
    ScoreMetadata,
    ToolRecord,
)


def metadata(score_type: str = "similarity") -> ScoreMetadata:
    return ScoreMetadata(
        score_type=score_type,
        higher_is_better=score_type != "distance",
        comparable_across_queries=False,
    )


def prediction(
    scores: tuple[float, ...],
    *,
    score_metadata: ScoreMetadata | None = None,
    query_id: str = "q1",
) -> RouterPrediction:
    candidates = tuple(
        RankedCandidate(tool_id=f"tool.{index}", score=score, rank=index + 1)
        for index, score in enumerate(scores)
    )
    return RouterPrediction(
        query_id=query_id,
        router_id="router",
        router_version="1.0",
        model_version="model:1",
        tool_corpus_hash="sha256:corpus",
        query_set_hash="sha256:queries",
        score_metadata=score_metadata or metadata(),
        candidates=candidates,
        latency_ms=None,
        error=None,
    )


def rank_only_prediction() -> RouterPrediction:
    return RouterPrediction(
        query_id="q1",
        router_id="router",
        router_version="1.0",
        model_version="model:1",
        tool_corpus_hash="sha256:corpus",
        query_set_hash="sha256:queries",
        score_metadata=ScoreMetadata(
            score_type="rank_only",
            higher_is_better=False,
            comparable_across_queries=False,
        ),
        candidates=(
            RankedCandidate(tool_id="tool.0", score=None, rank=1),
            RankedCandidate(tool_id="tool.1", score=None, rank=2),
        ),
        latency_ms=None,
        error=None,
    )


def tool(
    tool_id: str, *, name: str = "search_code", description: str = "Search code"
) -> ToolRecord:
    return ToolRecord(
        tool_id=tool_id,
        name=name,
        description=description,
        input_schema={"type": "object", "properties": {}},
        server="test",
        domain="developer_tools",
        source_url="https://example.invalid/features",
        license="CC0-1.0",
    )


def query(text: str = "search code", language: str = "en") -> InferenceQuery:
    return InferenceQuery(query_id="q1", text=text, language=language)


def tools_by_id(*tools: ToolRecord) -> dict[str, ToolRecord]:
    return {item.tool_id: item for item in tools}


def test_query_case_requires_explicit_conversion(
    tiny_queries: list[QueryCase],
) -> None:
    route = prediction((0.8, 0.4), query_id=tiny_queries[0].query_id)
    normalizer = FeatureNormalizer.fit([route])
    corpus = tools_by_id(tool("tool.0"), tool("tool.1"))

    with pytest.raises(TypeError, match="InferenceQuery"):
        extract_confidence_features(
            cast(InferenceQuery, tiny_queries[0]), route, corpus, normalizer
        )


def test_similarity_features_and_label_isolation() -> None:
    calibration = [prediction((0.6, 0.2)), prediction((0.8, 0.3), query_id="q2")]
    normalizer = FeatureNormalizer.fit(calibration)
    route = calibration[1].model_copy(update={"query_id": "q1"})

    features = extract_confidence_features(
        query(), route, tools_by_id(tool("tool.0"), tool("tool.1")), normalizer
    )

    assert features.top1_score == 0.8
    assert features.top1_score_standardized == pytest.approx(1.0)
    assert features.top1_minus_top2 == pytest.approx(0.5)
    assert features.score_entropy is not None
    assert features.lexical_coverage == 1.0
    forbidden = {"relevant_tool_ids", "case_type", "should_abstain", "abstain_reason"}
    assert forbidden.isdisjoint(features.model_dump())


def test_distance_direction_is_normalized_before_margin() -> None:
    distance = metadata("distance")
    calibration = [
        prediction((0.4, 0.7), score_metadata=distance),
        prediction((0.2, 0.5), score_metadata=distance, query_id="q2"),
    ]
    normalizer = FeatureNormalizer.fit(calibration)
    route = calibration[1].model_copy(update={"query_id": "q1"})

    features = extract_confidence_features(
        query(), route, tools_by_id(tool("tool.0"), tool("tool.1")), normalizer
    )

    assert features.top1_score == 0.2
    assert features.top1_score_standardized == pytest.approx(1.0)
    assert features.top1_minus_top2 == pytest.approx(0.3)


def test_singleton_and_zero_variance_have_defined_values() -> None:
    route = prediction((-1.0,))
    normalizer = FeatureNormalizer.fit([route, route.model_copy(update={"query_id": "q2"})])
    features = extract_confidence_features(query(), route, tools_by_id(tool("tool.0")), normalizer)

    assert features.top1_score_standardized == 0.0
    assert features.top1_minus_top2 == 0.0
    assert features.score_entropy == 0.0
    assert features.score_dispersion == 0.0
    assert features.score_dispersion_standardized == 0.0


def test_tied_negative_scores_have_stable_entropy_and_dispersion() -> None:
    route = prediction((-2.0, -2.0))
    normalizer = FeatureNormalizer.fit([route])
    features = extract_confidence_features(
        query(), route, tools_by_id(tool("tool.0"), tool("tool.1")), normalizer
    )

    assert features.top1_minus_top2 == 0.0
    assert features.score_entropy == pytest.approx(math.log(2))
    assert features.score_dispersion == 0.0


def test_rank_only_exposes_no_score_derived_features() -> None:
    features = extract_confidence_features(
        query(),
        rank_only_prediction(),
        tools_by_id(tool("tool.0"), tool("tool.1")),
        normalizer=None,
    )

    assert features.top1_score is None
    assert features.top1_score_standardized is None
    assert features.top1_minus_top2 is None
    assert features.score_entropy is None
    assert features.score_dispersion is None


def test_error_prediction_and_non_finite_values_are_rejected() -> None:
    failed = RouterPrediction(
        query_id="q1",
        router_id="router",
        router_version="1.0",
        model_version=None,
        tool_corpus_hash="sha256:corpus",
        query_set_hash="sha256:queries",
        score_metadata=metadata(),
        candidates=(),
        error="timeout",
    )
    with pytest.raises(ValueError, match="successful prediction"):
        extract_confidence_features(query(), failed, {}, normalizer=None)

    with pytest.raises(ValidationError, match="finite"):
        RankedCandidate(tool_id="tool.0", score=math.nan, rank=1)


def test_missing_description_and_chinese_lexical_coverage() -> None:
    route = prediction((0.9,))
    normalizer = FeatureNormalizer.fit([route])
    missing = extract_confidence_features(
        query("anything"),
        route,
        tools_by_id(tool("tool.0", description="")),
        normalizer,
    )
    assert missing.selected_tool_description_missing is True
    assert missing.selected_tool_description_length == 0

    chinese = extract_confidence_features(
        query("搜索代码", "zh"),
        route,
        tools_by_id(tool("tool.0", name="搜索代码", description="搜索代码")),
        normalizer,
    )
    assert chinese.lexical_coverage == 1.0


def test_router_agreement_is_optional_and_validated() -> None:
    route = prediction((0.9, 0.2))
    normalizer = FeatureNormalizer.fit([route])
    corpus = tools_by_id(tool("tool.0"), tool("tool.1"))
    without_peer = extract_confidence_features(query(), route, corpus, normalizer)
    assert without_peer.router_top1_agreement is None

    same = route.model_copy(update={"router_id": "peer"})
    with_peer = extract_confidence_features(query(), route, corpus, normalizer, same)
    assert with_peer.router_top1_agreement is True

    reversed_peer = RouterPrediction(
        **{
            **same.model_dump(),
            "candidates": [
                {"tool_id": "tool.1", "score": 0.9, "rank": 1},
                {"tool_id": "tool.0", "score": 0.2, "rank": 2},
            ],
        }
    )
    disagrees = extract_confidence_features(query(), route, corpus, normalizer, reversed_peer)
    assert disagrees.router_top1_agreement is False


def test_normalizer_binding_mismatch_fails_closed() -> None:
    route = prediction((0.8, 0.4))
    normalizer = FeatureNormalizer.fit([route])
    mismatched = route.model_copy(update={"router_version": "2.0"})

    with pytest.raises(ValueError, match="binding"):
        extract_confidence_features(
            query(), mismatched, tools_by_id(tool("tool.0"), tool("tool.1")), normalizer
        )
