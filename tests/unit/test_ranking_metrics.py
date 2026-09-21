from __future__ import annotations

import pytest

from mcp_route_reliability.evaluation.ranking import (
    EvaluationInputError,
    is_routing_error,
    per_case_errors,
    ranking_metrics,
)
from mcp_route_reliability.models import QueryCase, RankedCandidate, RouterPrediction, ScoreMetadata


def case(
    query_id: str,
    case_type: str,
    relevant: tuple[str, ...],
    *,
    should_abstain: bool = False,
    reason: str | None = None,
    language: str = "en",
) -> QueryCase:
    return QueryCase(
        query_id=query_id,
        base_query_id=query_id,
        text=f"Unique query {query_id}",
        language=language,
        case_type=case_type,
        difficulty="medium",
        relevant_tool_ids=relevant,
        should_abstain=should_abstain,
        abstain_reason=reason,
        split="test",
        label_rationale="Hand-computed ranking fixture.",
        candidate_conflict_note=None,
        tool_corpus_hash="sha256:corpus",
    )


def prediction(query_id: str, tool_ids: tuple[str, ...]) -> RouterPrediction:
    return RouterPrediction(
        query_id=query_id,
        router_id="router",
        router_version="1.0",
        model_version="model:1",
        tool_corpus_hash="sha256:corpus",
        query_set_hash="sha256:queries",
        score_metadata=ScoreMetadata(
            score_type="similarity",
            higher_is_better=True,
            comparable_across_queries=False,
        ),
        candidates=tuple(
            RankedCandidate(tool_id=tool_id, score=1.0 / rank, rank=rank)
            for rank, tool_id in enumerate(tool_ids, start=1)
        ),
        latency_ms=None,
        error=None,
    )


def hand_computed_fixture() -> tuple[list[QueryCase], list[RouterPrediction]]:
    cases = [
        case("clear", "clear", ("tool.a",), language="en"),
        case("hard", "hard_negative", ("tool.b",), language="zh"),
        case("multi", "multi_valid", ("tool.c", "tool.d"), language="en"),
        case(
            "clarify",
            "clarify",
            ("tool.e",),
            should_abstain=True,
            reason="insufficient_information",
            language="zh",
        ),
        case(
            "none",
            "no_match",
            (),
            should_abstain=True,
            reason="no_match",
            language="en",
        ),
    ]
    predictions = [
        prediction("clear", ("tool.a", "tool.x")),
        prediction("hard", ("tool.x", "tool.b")),
        prediction("multi", ("tool.x", "tool.d")),
        prediction("clarify", ("tool.e", "tool.x")),
        prediction("none", ("tool.x", "tool.a")),
    ]
    return cases, predictions


def test_ranking_metrics_match_hand_calculation() -> None:
    cases, predictions = hand_computed_fixture()
    metrics = ranking_metrics(
        cases,
        predictions,
        ks=(1, 2, 5),
        known_tool_ids={"tool.a", "tool.b", "tool.c", "tool.d", "tool.e", "tool.x"},
    )

    assert metrics.sample_size == 5
    assert metrics.top1_accuracy == pytest.approx(1 / 5)
    assert metrics.recall_at_k == {
        1: pytest.approx(1 / 5),
        2: pytest.approx(3 / 5),
        5: pytest.approx(3 / 5),
    }
    assert metrics.mrr == pytest.approx(2 / 5)
    assert metrics.multi_valid_hit_rate == 1.0
    assert metrics.slices["language:en"].top1_accuracy == pytest.approx(1 / 3)
    assert metrics.slices["case_type:no_match"].top1_accuracy == 0.0


def test_per_case_errors_preserve_gold_and_top_k() -> None:
    cases, predictions = hand_computed_fixture()
    rows = per_case_errors(cases, predictions)
    by_id = {row.query_id: row for row in rows}

    assert by_id["clear"].is_correct is True
    assert by_id["clear"].is_routing_error is False
    assert by_id["hard"].top1_tool_id == "tool.x"
    assert by_id["hard"].is_routing_error is True
    assert by_id["multi"].first_relevant_rank == 2
    assert by_id["clarify"].is_routing_error is True
    assert by_id["none"].relevant_tool_ids == ()


def test_is_routing_error_handles_abstain_and_router_failure() -> None:
    routeable = case("q", "clear", ("tool.a",))
    abstain = case("n", "no_match", (), should_abstain=True, reason="no_match")
    assert is_routing_error(routeable, prediction("q", ("tool.a",))) is False
    assert is_routing_error(routeable, prediction("q", ("tool.b",))) is True
    assert is_routing_error(abstain, prediction("n", ("tool.a",))) is True

    failed = prediction("q", ("tool.a",)).model_copy(update={"candidates": (), "error": "timeout"})
    assert is_routing_error(routeable, failed) is True


def test_missing_extra_or_duplicate_predictions_are_rejected() -> None:
    cases, predictions = hand_computed_fixture()
    with pytest.raises(EvaluationInputError, match="missing prediction"):
        ranking_metrics(cases, predictions[:-1])
    with pytest.raises(EvaluationInputError, match="extra prediction"):
        ranking_metrics(
            cases,
            [*predictions, prediction("extra", ("tool.a",))],
        )
    with pytest.raises(EvaluationInputError, match="duplicate prediction"):
        ranking_metrics(cases, [*predictions, predictions[0]])


def test_unknown_candidate_tool_is_rejected() -> None:
    cases, predictions = hand_computed_fixture()
    with pytest.raises(EvaluationInputError, match="unknown candidate Tool"):
        ranking_metrics(cases, predictions, known_tool_ids={"tool.a"})


def test_invalid_or_large_k_values_have_deterministic_behavior() -> None:
    cases, predictions = hand_computed_fixture()
    metrics = ranking_metrics(cases, predictions, ks=(10, 1, 10))
    assert list(metrics.recall_at_k) == [1, 10]
    assert metrics.recall_at_k[10] == pytest.approx(3 / 5)

    with pytest.raises(ValueError, match="positive"):
        ranking_metrics(cases, predictions, ks=(0,))


def test_query_case_and_prediction_ids_must_match() -> None:
    routeable = case("q1", "clear", ("tool.a",))
    with pytest.raises(ValueError, match="query IDs"):
        is_routing_error(routeable, prediction("q2", ("tool.a",)))
