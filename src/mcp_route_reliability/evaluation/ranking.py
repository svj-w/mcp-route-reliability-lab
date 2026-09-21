"""Raw Top-K routing metrics and per-query error records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Sequence

from pydantic import Field

from mcp_route_reliability.models import QueryCase, RouterPrediction, StrictModel


class EvaluationInputError(ValueError):
    """Cases and predictions cannot be joined without ambiguity."""


class CaseError(StrictModel):
    query_id: str
    base_query_id: str
    language: str
    case_type: str
    should_abstain: bool
    relevant_tool_ids: tuple[str, ...]
    top_k_tool_ids: tuple[str, ...]
    top1_tool_id: str | None
    first_relevant_rank: int | None
    router_error: str | None
    is_correct: bool
    is_routing_error: bool


class SliceMetrics(StrictModel):
    sample_size: int = Field(ge=1)
    top1_accuracy: float = Field(ge=0, le=1)
    routing_error_rate: float = Field(ge=0, le=1)


class RankingMetrics(StrictModel):
    sample_size: int = Field(ge=1)
    top1_accuracy: float = Field(ge=0, le=1)
    recall_at_k: dict[int, float]
    mrr: float = Field(ge=0, le=1)
    multi_valid_hit_rate: float | None = Field(default=None, ge=0, le=1)
    slices: dict[str, SliceMetrics]


def is_routing_error(case: QueryCase, prediction: RouterPrediction) -> bool:
    if case.query_id != prediction.query_id:
        raise ValueError("case and prediction query IDs do not match")
    if case.tool_corpus_hash != prediction.tool_corpus_hash:
        raise ValueError("case and prediction Tool Corpus hashes do not match")
    if prediction.error is not None or not prediction.candidates:
        return True
    if case.should_abstain:
        return True
    return prediction.candidates[0].tool_id not in case.relevant_tool_ids


def _index_predictions(
    cases: Sequence[QueryCase], predictions: Sequence[RouterPrediction]
) -> dict[str, RouterPrediction]:
    case_ids = [case.query_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise EvaluationInputError("duplicate QueryCase query_id")

    indexed: dict[str, RouterPrediction] = {}
    for prediction in predictions:
        if prediction.query_id in indexed:
            raise EvaluationInputError(f"duplicate prediction for {prediction.query_id!r}")
        indexed[prediction.query_id] = prediction

    missing = sorted(set(case_ids) - set(indexed))
    extra = sorted(set(indexed) - set(case_ids))
    if missing:
        raise EvaluationInputError(f"missing prediction query IDs: {missing}")
    if extra:
        raise EvaluationInputError(f"extra prediction query IDs: {extra}")
    return indexed


def _validate_known_tools(
    predictions: Sequence[RouterPrediction], known_tool_ids: Collection[str] | None
) -> None:
    if known_tool_ids is None:
        return
    known = set(known_tool_ids)
    for prediction in predictions:
        for candidate in prediction.candidates:
            if candidate.tool_id not in known:
                raise EvaluationInputError(
                    f"query {prediction.query_id!r} uses unknown candidate Tool "
                    f"{candidate.tool_id!r}"
                )


def per_case_errors(
    cases: Sequence[QueryCase],
    predictions: Sequence[RouterPrediction],
    *,
    known_tool_ids: Collection[str] | None = None,
) -> list[CaseError]:
    indexed = _index_predictions(cases, predictions)
    _validate_known_tools(predictions, known_tool_ids)
    rows: list[CaseError] = []
    for case in cases:
        prediction = indexed[case.query_id]
        error = is_routing_error(case, prediction)
        ranked_ids = tuple(candidate.tool_id for candidate in prediction.candidates)
        first_relevant_rank = next(
            (
                candidate.rank
                for candidate in prediction.candidates
                if candidate.tool_id in case.relevant_tool_ids
            ),
            None,
        )
        rows.append(
            CaseError(
                query_id=case.query_id,
                base_query_id=case.base_query_id,
                language=case.language,
                case_type=case.case_type,
                should_abstain=case.should_abstain,
                relevant_tool_ids=case.relevant_tool_ids,
                top_k_tool_ids=ranked_ids,
                top1_tool_id=ranked_ids[0] if ranked_ids else None,
                first_relevant_rank=first_relevant_rank,
                router_error=prediction.error,
                is_correct=not error,
                is_routing_error=error,
            )
        )
    return rows


def _slice_metrics(rows: Sequence[CaseError]) -> SliceMetrics:
    correct = sum(row.is_correct for row in rows)
    return SliceMetrics(
        sample_size=len(rows),
        top1_accuracy=correct / len(rows),
        routing_error_rate=1.0 - correct / len(rows),
    )


def ranking_metrics(
    cases: Sequence[QueryCase],
    predictions: Sequence[RouterPrediction],
    ks: Sequence[int] = (1, 3, 5),
    *,
    known_tool_ids: Collection[str] | None = None,
) -> RankingMetrics:
    if not cases:
        raise ValueError("ranking evaluation requires at least one case")
    normalized_ks = sorted(set(ks))
    if not normalized_ks or any(k < 1 for k in normalized_ks):
        raise ValueError("Recall@K values must be positive")

    rows = per_case_errors(cases, predictions, known_tool_ids=known_tool_ids)
    indexed = _index_predictions(cases, predictions)
    sample_size = len(cases)

    recall_at_k: dict[int, float] = {}
    for k in normalized_ks:
        hits = 0
        for case in cases:
            if case.should_abstain:
                continue
            candidates = indexed[case.query_id].candidates[:k]
            if any(candidate.tool_id in case.relevant_tool_ids for candidate in candidates):
                hits += 1
        recall_at_k[k] = hits / sample_size

    reciprocal_ranks = [
        0.0
        if case.should_abstain or row.first_relevant_rank is None
        else 1 / row.first_relevant_rank
        for case, row in zip(cases, rows, strict=True)
    ]
    multi_rows = [
        row for case, row in zip(cases, rows, strict=True) if case.case_type == "multi_valid"
    ]
    multi_hit_rate = (
        sum(row.first_relevant_rank is not None for row in multi_rows) / len(multi_rows)
        if multi_rows
        else None
    )

    grouped: dict[str, list[CaseError]] = defaultdict(list)
    for case, row in zip(cases, rows, strict=True):
        grouped[f"language:{case.language}"].append(row)
        grouped[f"case_type:{case.case_type}"].append(row)

    return RankingMetrics(
        sample_size=sample_size,
        top1_accuracy=sum(row.is_correct for row in rows) / sample_size,
        recall_at_k=recall_at_k,
        mrr=sum(reciprocal_ranks) / sample_size,
        multi_valid_hit_rate=multi_hit_rate,
        slices={key: _slice_metrics(value) for key, value in sorted(grouped.items())},
    )
