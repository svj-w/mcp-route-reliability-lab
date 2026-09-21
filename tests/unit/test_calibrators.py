from __future__ import annotations

import numpy as np
import pytest

from mcp_route_reliability.confidence.calibrators import (
    IsotonicCalibrator,
    LogisticConfidenceModel,
    UnavailableCalibrator,
    fit_isotonic,
    fit_logistic,
)
from mcp_route_reliability.models import ConfidenceFeatures, ScoreMetadata


def feature_row(index: int, *, agreement: bool | None = None) -> ConfidenceFeatures:
    value = index / 10
    return ConfidenceFeatures(
        feature_version="1.0",
        query_id=f"q{index}",
        router_id="router",
        router_version="1.0",
        model_version="model:1",
        tool_corpus_hash="sha256:corpus",
        score_metadata=ScoreMetadata(
            score_type="similarity",
            higher_is_better=True,
            comparable_across_queries=False,
        ),
        top1_score=value,
        top1_score_standardized=value,
        top1_minus_top2=value / 2,
        score_entropy=1.0 - value,
        score_dispersion=value / 3,
        score_dispersion_standardized=value / 4,
        query_length=5 + index,
        language="en" if index % 2 else "zh",
        selected_tool_description_length=20 + index,
        selected_tool_description_missing=False,
        lexical_coverage=value,
        router_top1_agreement=agreement,
    )


def test_isotonic_is_monotonic_and_round_trips() -> None:
    rows = [feature_row(index) for index in range(6)]
    fitted = fit_isotonic(
        rows,
        [0, 0, 0, 1, 1, 1],
        feature_name="top1_score_standardized",
    )
    assert isinstance(fitted, IsotonicCalibrator)

    probabilities = fitted.predict_correctness(rows)
    assert np.all(np.diff(probabilities) >= 0)
    restored = IsotonicCalibrator.model_validate_json(fitted.model_dump_json())
    assert restored.to_payload() == fitted.to_payload()
    assert restored.predict_correctness(rows) == pytest.approx(probabilities)
    assert fitted.sample_size == 6
    assert fitted.unique_score_count == 6


@pytest.mark.parametrize(
    ("rows", "labels", "reason"),
    [
        ([feature_row(1), feature_row(1)], [0, 1], "insufficient_unique_scores"),
        ([feature_row(1), feature_row(2)], [1, 1], "single_target_class"),
    ],
)
def test_isotonic_preconditions_return_unavailable(
    rows: list[ConfidenceFeatures], labels: list[int], reason: str
) -> None:
    fitted = fit_isotonic(rows, labels, feature_name="top1_score_standardized")
    assert isinstance(fitted, UnavailableCalibrator)
    assert fitted.reason == reason


def test_logistic_uses_four_to_six_features_and_round_trips() -> None:
    rows = [
        feature_row(index, agreement=None if index == 0 else index % 2 == 0) for index in range(8)
    ]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    feature_names = (
        "top1_score_standardized",
        "top1_minus_top2",
        "score_entropy",
        "lexical_coverage",
        "router_top1_agreement",
    )

    fitted = fit_logistic(rows, labels, feature_names=feature_names)
    assert isinstance(fitted, LogisticConfidenceModel)
    probabilities = fitted.predict_correctness(rows)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert probabilities[-1] > probabilities[0]

    restored = LogisticConfidenceModel.model_validate_json(fitted.model_dump_json())
    assert restored.to_payload() == fitted.to_payload()
    assert restored.predict_correctness(rows) == pytest.approx(probabilities)
    assert restored.feature_names == feature_names


@pytest.mark.parametrize("count", [3, 7])
def test_logistic_rejects_feature_counts_outside_four_to_six(count: int) -> None:
    names = tuple(f"feature_{index}" for index in range(count))
    with pytest.raises(ValueError, match="4 to 6"):
        fit_logistic([feature_row(0), feature_row(1)], [0, 1], feature_names=names)


def test_logistic_single_class_and_all_missing_feature_are_unavailable() -> None:
    rows = [feature_row(index) for index in range(4)]
    names = (
        "top1_score_standardized",
        "top1_minus_top2",
        "score_entropy",
        "router_top1_agreement",
    )
    single = fit_logistic(rows, [1, 1, 1, 1], feature_names=names)
    assert isinstance(single, UnavailableCalibrator)
    assert single.reason == "single_target_class"

    missing = fit_logistic(rows, [0, 0, 1, 1], feature_names=names)
    assert isinstance(missing, UnavailableCalibrator)
    assert missing.reason == "feature_all_missing:router_top1_agreement"
