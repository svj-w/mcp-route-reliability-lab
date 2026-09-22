"""Independent abstention reason hints and their evaluation metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, field_validator

from mcp_route_reliability.models import (
    AbstainReason,
    ConfidenceFeatures,
    InferenceQuery,
    ReasonHint,
    RouterPrediction,
    StrictModel,
)

GOLD_REASON_CLASSES: tuple[AbstainReason, ...] = (
    "no_match",
    "ambiguous",
    "insufficient_information",
)
PREDICTED_REASON_CLASSES: tuple[ReasonHint, ...] = (*GOLD_REASON_CLASSES, "uncertain")


class ReasonHintConfig(StrictModel):
    """Persisted bounds selected exclusively from the Calibration split."""

    schema_version: Literal["1.0"] = "1.0"
    fitted_split: Literal["calibration"] = "calibration"
    calibration_sample_size: int = Field(ge=1)
    query_token_upper_bound: int = Field(ge=1)
    lexical_coverage_upper_bound: float = Field(ge=0, le=1)
    standardized_top1_lower_bound: float
    margin_lower_bound: float
    entropy_upper_bound: float

    @field_validator(
        "standardized_top1_lower_bound",
        "margin_lower_bound",
        "entropy_upper_bound",
    )
    @classmethod
    def validate_finite_bounds(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("reason hint bounds must be finite")
        return value


class ReasonClassMetrics(StrictModel):
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    support: int = Field(ge=0)


class ReasonMetrics(StrictModel):
    by_class: dict[str, ReasonClassMetrics]
    macro_f1: float = Field(ge=0, le=1)
    confusion_matrix: dict[str, dict[str, int]]
    uncertain_count: int = Field(ge=0)
    uncertain_rate: float = Field(ge=0, le=1)
    sample_size: int = Field(ge=1)


def _validate_observation_bindings(
    query: InferenceQuery,
    prediction: RouterPrediction,
    features: ConfidenceFeatures,
) -> None:
    if type(query) is not InferenceQuery:
        raise TypeError("reason inference requires an exact InferenceQuery")
    if query.query_id != prediction.query_id or query.query_id != features.query_id:
        raise ValueError("query, prediction, and feature IDs must match")
    prediction_binding = (
        prediction.router_id,
        prediction.router_version,
        prediction.model_version,
        prediction.tool_corpus_hash,
        prediction.score_metadata,
    )
    feature_binding = (
        features.router_id,
        features.router_version,
        features.model_version,
        features.tool_corpus_hash,
        features.score_metadata,
    )
    if prediction_binding != feature_binding:
        raise ValueError("prediction and confidence feature bindings must match")


def infer_reason_hint(
    query: InferenceQuery,
    prediction: RouterPrediction,
    features: ConfidenceFeatures,
    config: ReasonHintConfig,
) -> ReasonHint:
    """Diagnose an abstention without accessing labels or changing its decision."""

    _validate_observation_bindings(query, prediction, features)
    if (
        features.query_length < config.query_token_upper_bound
        and features.lexical_coverage < config.lexical_coverage_upper_bound
    ):
        return "insufficient_information"
    if (
        features.top1_score_standardized is not None
        and features.top1_score_standardized < config.standardized_top1_lower_bound
    ):
        return "no_match"
    if (
        features.top1_minus_top2 is not None
        and features.top1_minus_top2 < config.margin_lower_bound
    ) or (
        features.score_entropy is not None and features.score_entropy > config.entropy_upper_bound
    ):
        return "ambiguous"
    return "uncertain"


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_reason_hints(
    gold: Sequence[AbstainReason],
    predicted: Sequence[ReasonHint],
) -> ReasonMetrics:
    """Evaluate the independent diagnostic over the three gold reason classes."""

    if len(gold) != len(predicted):
        raise ValueError("gold and predicted reasons must have the same length")
    if not gold:
        raise ValueError("reason evaluation requires nonempty inputs")

    confusion: dict[str, dict[str, int]] = {
        gold_reason: {predicted_reason: 0 for predicted_reason in PREDICTED_REASON_CLASSES}
        for gold_reason in GOLD_REASON_CLASSES
    }
    for gold_reason, predicted_reason in zip(gold, predicted, strict=True):
        confusion[gold_reason][predicted_reason] += 1

    by_class: dict[str, ReasonClassMetrics] = {}
    for reason in GOLD_REASON_CLASSES:
        true_positive = confusion[reason][reason]
        predicted_positive = sum(confusion[row][reason] for row in GOLD_REASON_CLASSES)
        support = sum(confusion[reason].values())
        precision = _safe_ratio(true_positive, predicted_positive)
        recall = _safe_ratio(true_positive, support)
        f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
        by_class[reason] = ReasonClassMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
        )

    uncertain_count = sum(item == "uncertain" for item in predicted)
    return ReasonMetrics(
        by_class=by_class,
        macro_f1=sum(metric.f1 for metric in by_class.values()) / len(by_class),
        confusion_matrix=confusion,
        uncertain_count=uncertain_count,
        uncertain_rate=uncertain_count / len(predicted),
        sample_size=len(predicted),
    )
