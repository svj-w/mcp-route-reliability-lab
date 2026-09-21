"""Fail-fast JSONL importer for external Top-K router predictions."""

from __future__ import annotations

import json
from collections.abc import Collection
from itertools import pairwise
from pathlib import Path

from pydantic import ValidationError

from mcp_route_reliability.models import RouterPrediction


class PredictionImportError(ValueError):
    """An external prediction file violates its declared routing contract."""


def _load_jsonl(path: Path) -> list[RouterPrediction]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as error:
        raise PredictionImportError(f"cannot read {path}: {error}") from error

    predictions: list[RouterPrediction] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise PredictionImportError(f"{path} line {line_number}: {error.msg}") from error
        try:
            prediction = RouterPrediction.model_validate(raw)
        except ValidationError as error:
            raise PredictionImportError(f"{path} line {line_number}: {error}") from error
        predictions.append(prediction)
    return predictions


def _validate_score_order(path: Path, prediction: RouterPrediction) -> None:
    if prediction.score_metadata.score_type == "rank_only" or len(prediction.candidates) < 2:
        return

    for previous, current in pairwise(prediction.candidates):
        if previous.score is None or current.score is None:
            raise PredictionImportError(
                f"{path} query {prediction.query_id}: scored candidates require numeric values"
            )
        if prediction.score_metadata.higher_is_better:
            incorrectly_ordered = previous.score < current.score
        else:
            incorrectly_ordered = previous.score > current.score
        if incorrectly_ordered:
            raise PredictionImportError(
                f"{path} query {prediction.query_id}: score ordering contradicts higher_is_better"
            )
        if previous.score == current.score and previous.tool_id > current.tool_id:
            raise PredictionImportError(
                f"{path} query {prediction.query_id}: tie ordering must use ascending tool_id"
            )


def import_predictions(
    path: Path,
    *,
    expected_tool_corpus_hash: str,
    expected_query_set_hash: str,
    expected_query_ids: Collection[str],
    expected_tool_ids: Collection[str],
) -> list[RouterPrediction]:
    """Load all records, validate the complete file contract, then return results."""

    predictions = _load_jsonl(path)
    expected_queries = set(expected_query_ids)
    expected_tools = set(expected_tool_ids)
    seen_queries: set[str] = set()
    reference_router: tuple[str, str, str | None] | None = None
    reference_score_metadata: object | None = None

    for prediction in predictions:
        context = f"{path} query {prediction.query_id}"
        if prediction.query_id in seen_queries:
            raise PredictionImportError(f"{context}: duplicate query_id")
        seen_queries.add(prediction.query_id)

        if prediction.tool_corpus_hash != expected_tool_corpus_hash:
            raise PredictionImportError(
                f"{context}: tool corpus hash {prediction.tool_corpus_hash!r} does not match "
                f"{expected_tool_corpus_hash!r}"
            )
        if prediction.query_set_hash != expected_query_set_hash:
            raise PredictionImportError(
                f"{context}: query set hash {prediction.query_set_hash!r} does not match "
                f"{expected_query_set_hash!r}"
            )

        router_contract = (
            prediction.router_id,
            prediction.router_version,
            prediction.model_version,
        )
        if reference_router is None:
            reference_router = router_contract
        elif router_contract != reference_router:
            raise PredictionImportError(
                f"{context}: router/model contract {router_contract!r} does not match "
                f"{reference_router!r}"
            )

        metadata_payload = prediction.score_metadata.model_dump(mode="json")
        if reference_score_metadata is None:
            reference_score_metadata = metadata_payload
        elif metadata_payload != reference_score_metadata:
            raise PredictionImportError(f"{context}: inconsistent score metadata")

        if prediction.error is not None and prediction.candidates:
            raise PredictionImportError(f"{context}: error records must not contain candidates")
        for candidate in prediction.candidates:
            if candidate.tool_id not in expected_tools:
                raise PredictionImportError(
                    f"{context}: candidate uses unknown Tool ID {candidate.tool_id!r}"
                )
        _validate_score_order(path, prediction)

    missing = sorted(expected_queries - seen_queries)
    extra = sorted(seen_queries - expected_queries)
    if missing:
        raise PredictionImportError(f"{path}: missing query IDs: {missing}")
    if extra:
        raise PredictionImportError(f"{path}: extra query IDs: {extra}")

    return sorted(predictions, key=lambda prediction: prediction.query_id)
