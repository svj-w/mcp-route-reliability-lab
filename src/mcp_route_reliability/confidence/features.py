"""Leakage-safe, inference-observable confidence features."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Self

from pydantic import Field, model_validator

from mcp_route_reliability.models import (
    ConfidenceFeatures,
    InferenceQuery,
    RouterPrediction,
    ScoreMetadata,
    StrictModel,
    ToolRecord,
)
from mcp_route_reliability.routers.bm25 import tokenize

FEATURE_VERSION = "1.0"


def _signed_score(value: float, metadata: ScoreMetadata) -> float:
    return value if metadata.higher_is_better else -value


def _require_numeric_scores(prediction: RouterPrediction) -> list[float]:
    if prediction.error is not None or not prediction.candidates:
        raise ValueError(f"query {prediction.query_id} requires a successful prediction")
    if prediction.score_metadata.score_type == "rank_only":
        raise ValueError("rank_only predictions do not have numeric confidence scores")
    scores = [candidate.score for candidate in prediction.candidates]
    if any(score is None for score in scores):
        raise ValueError("numeric predictions require every candidate score")
    return [float(score) for score in scores if score is not None]


class FeatureNormalizer(StrictModel):
    schema_version: str = "1.0"
    feature_version: str = FEATURE_VERSION
    router_id: str = Field(min_length=1)
    router_version: str = Field(min_length=1)
    model_version: str | None
    tool_corpus_hash: str = Field(min_length=1)
    score_metadata: ScoreMetadata
    top1_mean: float
    top1_standard_deviation: float = Field(ge=0)
    dispersion_mean: float
    dispersion_standard_deviation: float = Field(ge=0)
    sample_size: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_finite_parameters(self) -> Self:
        values = (
            self.top1_mean,
            self.top1_standard_deviation,
            self.dispersion_mean,
            self.dispersion_standard_deviation,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("normalizer parameters must be finite")
        return self

    @classmethod
    def fit(cls, calibration_predictions: Sequence[RouterPrediction]) -> FeatureNormalizer:
        if not calibration_predictions:
            raise ValueError("normalizer fitting requires at least one Calibration prediction")

        first = calibration_predictions[0]
        binding = (
            first.router_id,
            first.router_version,
            first.model_version,
            first.tool_corpus_hash,
            first.score_metadata,
        )
        signed_top1: list[float] = []
        dispersions: list[float] = []
        for prediction in calibration_predictions:
            current_binding = (
                prediction.router_id,
                prediction.router_version,
                prediction.model_version,
                prediction.tool_corpus_hash,
                prediction.score_metadata,
            )
            if current_binding != binding:
                raise ValueError("Calibration predictions have inconsistent normalizer bindings")
            scores = _require_numeric_scores(prediction)
            signed = [_signed_score(score, prediction.score_metadata) for score in scores]
            signed_top1.append(signed[0])
            dispersions.append(statistics.pstdev(signed) if len(signed) > 1 else 0.0)

        return cls(
            router_id=first.router_id,
            router_version=first.router_version,
            model_version=first.model_version,
            tool_corpus_hash=first.tool_corpus_hash,
            score_metadata=first.score_metadata,
            top1_mean=statistics.fmean(signed_top1),
            top1_standard_deviation=(
                statistics.pstdev(signed_top1) if len(signed_top1) > 1 else 0.0
            ),
            dispersion_mean=statistics.fmean(dispersions),
            dispersion_standard_deviation=(
                statistics.pstdev(dispersions) if len(dispersions) > 1 else 0.0
            ),
            sample_size=len(calibration_predictions),
        )

    def validate_prediction_binding(self, prediction: RouterPrediction) -> None:
        expected = (
            self.router_id,
            self.router_version,
            self.model_version,
            self.tool_corpus_hash,
            self.score_metadata,
        )
        actual = (
            prediction.router_id,
            prediction.router_version,
            prediction.model_version,
            prediction.tool_corpus_hash,
            prediction.score_metadata,
        )
        if actual != expected:
            raise ValueError("prediction does not match FeatureNormalizer binding")

    def standardize_top1(self, raw_score: float) -> float:
        signed = _signed_score(raw_score, self.score_metadata)
        if self.top1_standard_deviation == 0:
            return 0.0
        return (signed - self.top1_mean) / self.top1_standard_deviation

    def standardize_dispersion(self, raw_dispersion: float) -> float:
        if self.dispersion_standard_deviation == 0:
            return 0.0
        return (
            raw_dispersion - self.dispersion_mean
        ) / self.dispersion_standard_deviation


def _entropy(signed_scores: Sequence[float]) -> float:
    if len(signed_scores) < 2:
        return 0.0
    maximum = max(signed_scores)
    exponentials = [math.exp(score - maximum) for score in signed_scores]
    denominator = sum(exponentials)
    probabilities = [value / denominator for value in exponentials]
    return -sum(probability * math.log(probability) for probability in probabilities)


def _lexical_coverage(query_text: str, tool: ToolRecord) -> float:
    query_tokens = set(tokenize(query_text))
    if not query_tokens:
        return 0.0
    tool_tokens = set(tokenize(f"{tool.name} {tool.description}"))
    return len(query_tokens & tool_tokens) / len(query_tokens)


def _validate_peer(prediction: RouterPrediction, peer: RouterPrediction) -> None:
    if peer.error is not None or not peer.candidates:
        raise ValueError("peer agreement requires a successful prediction")
    if peer.query_id != prediction.query_id:
        raise ValueError("peer prediction query ID does not match")
    if peer.tool_corpus_hash != prediction.tool_corpus_hash:
        raise ValueError("peer prediction Tool Corpus hash does not match")


def extract_confidence_features(
    query: InferenceQuery,
    prediction: RouterPrediction,
    tools_by_id: Mapping[str, ToolRecord],
    normalizer: FeatureNormalizer | None,
    peer_prediction: RouterPrediction | None = None,
) -> ConfidenceFeatures:
    if type(query) is not InferenceQuery:
        raise TypeError("feature extraction requires an exact InferenceQuery")
    if query.query_id != prediction.query_id:
        raise ValueError("query and prediction IDs do not match")
    if prediction.error is not None or not prediction.candidates:
        raise ValueError("feature extraction requires a successful prediction")

    top1_tool_id = prediction.candidates[0].tool_id
    try:
        selected_tool = tools_by_id[top1_tool_id]
    except KeyError as error:
        raise ValueError(f"selected Tool ID {top1_tool_id!r} is absent from the corpus") from error

    if peer_prediction is None:
        agreement = None
    else:
        _validate_peer(prediction, peer_prediction)
        agreement = peer_prediction.candidates[0].tool_id == top1_tool_id

    if prediction.score_metadata.score_type == "rank_only":
        if normalizer is not None:
            raise ValueError("rank_only feature extraction must not receive a normalizer")
        top1_score = None
        standardized_top1 = None
        margin = None
        entropy = None
        dispersion = None
        standardized_dispersion = None
    else:
        if normalizer is None:
            raise ValueError("numeric feature extraction requires a FeatureNormalizer")
        normalizer.validate_prediction_binding(prediction)
        raw_scores = _require_numeric_scores(prediction)
        signed_scores = [
            _signed_score(score, prediction.score_metadata) for score in raw_scores
        ]
        top1_score = raw_scores[0]
        standardized_top1 = normalizer.standardize_top1(top1_score)
        margin = signed_scores[0] - signed_scores[1] if len(signed_scores) > 1 else 0.0
        entropy = _entropy(signed_scores)
        dispersion = statistics.pstdev(signed_scores) if len(signed_scores) > 1 else 0.0
        standardized_dispersion = normalizer.standardize_dispersion(dispersion)

    description = selected_tool.description.strip()
    return ConfidenceFeatures(
        feature_version=FEATURE_VERSION,
        query_id=query.query_id,
        router_id=prediction.router_id,
        router_version=prediction.router_version,
        model_version=prediction.model_version,
        tool_corpus_hash=prediction.tool_corpus_hash,
        score_metadata=prediction.score_metadata,
        top1_score=top1_score,
        top1_score_standardized=standardized_top1,
        top1_minus_top2=margin,
        score_entropy=entropy,
        score_dispersion=dispersion,
        score_dispersion_standardized=standardized_dispersion,
        query_length=len(tokenize(query.text)),
        language=query.language,
        selected_tool_description_length=len(description),
        selected_tool_description_missing=not description,
        lexical_coverage=_lexical_coverage(query.text, selected_tool),
        router_top1_agreement=agreement,
    )
