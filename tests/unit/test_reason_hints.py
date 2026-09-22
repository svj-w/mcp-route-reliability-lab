from __future__ import annotations

import pytest

from mcp_route_reliability.confidence.reasons import (
    ReasonHintConfig,
    evaluate_reason_hints,
    infer_reason_hint,
)
from mcp_route_reliability.models import (
    ConfidenceFeatures,
    InferenceQuery,
    RankedCandidate,
    RouterPrediction,
    ScoreMetadata,
)

METADATA = ScoreMetadata(
    score_type="similarity",
    higher_is_better=True,
    comparable_across_queries=False,
)
CONFIG = ReasonHintConfig(
    calibration_sample_size=40,
    query_token_upper_bound=3,
    lexical_coverage_upper_bound=0.2,
    standardized_top1_lower_bound=-1.0,
    margin_lower_bound=0.1,
    entropy_upper_bound=0.9,
)


def prediction() -> RouterPrediction:
    return RouterPrediction(
        query_id="q1",
        router_id="bm25",
        router_version="1.0.0",
        model_version="rank-bm25:0.2.2",
        tool_corpus_hash="sha256:corpus",
        query_set_hash="sha256:runtime",
        score_metadata=METADATA,
        candidates=(RankedCandidate(tool_id="tool.search", score=1.0, rank=1),),
    )


def features(**updates: object) -> ConfidenceFeatures:
    values: dict[str, object] = {
        "feature_version": "1.0",
        "query_id": "q1",
        "router_id": "bm25",
        "router_version": "1.0.0",
        "model_version": "rank-bm25:0.2.2",
        "tool_corpus_hash": "sha256:corpus",
        "score_metadata": METADATA,
        "top1_score": 1.0,
        "top1_score_standardized": 0.5,
        "top1_minus_top2": 0.5,
        "score_entropy": 0.2,
        "score_dispersion": 0.5,
        "score_dispersion_standardized": 0.0,
        "query_length": 5,
        "language": "en",
        "selected_tool_description_length": 20,
        "selected_tool_description_missing": False,
        "lexical_coverage": 0.8,
        "router_top1_agreement": True,
    }
    values.update(updates)
    return ConfidenceFeatures.model_validate(values)


@pytest.mark.parametrize(
    ("feature_updates", "expected"),
    [
        ({"query_length": 2, "lexical_coverage": 0.1}, "insufficient_information"),
        ({"top1_score_standardized": -2.0}, "no_match"),
        ({"top1_minus_top2": 0.05}, "ambiguous"),
        ({"score_entropy": 1.2}, "ambiguous"),
        ({}, "uncertain"),
    ],
)
def test_reason_precedence_and_fallback(feature_updates: dict[str, object], expected: str) -> None:
    assert (
        infer_reason_hint(
            InferenceQuery(query_id="q1", text="search for code", language="en"),
            prediction(),
            features(**feature_updates),
            CONFIG,
        )
        == expected
    )


def test_reason_config_is_versioned_and_calibration_bound() -> None:
    payload = CONFIG.model_dump(mode="json")
    assert payload["schema_version"] == "1.0"
    assert payload["fitted_split"] == "calibration"
    assert ReasonHintConfig.model_validate(payload) == CONFIG


def test_reason_metrics_match_hand_calculation() -> None:
    metrics = evaluate_reason_hints(
        ["no_match", "no_match", "ambiguous", "insufficient_information"],
        ["no_match", "ambiguous", "ambiguous", "uncertain"],
    )

    assert metrics.by_class["no_match"].precision == 1.0
    assert metrics.by_class["no_match"].recall == 0.5
    assert metrics.by_class["no_match"].f1 == pytest.approx(2 / 3)
    assert metrics.by_class["no_match"].support == 2
    assert metrics.by_class["ambiguous"].precision == 0.5
    assert metrics.by_class["ambiguous"].recall == 1.0
    assert metrics.by_class["insufficient_information"].f1 == 0.0
    assert metrics.macro_f1 == pytest.approx(4 / 9)
    assert metrics.confusion_matrix["no_match"]["ambiguous"] == 1
    assert metrics.confusion_matrix["insufficient_information"]["uncertain"] == 1
    assert metrics.uncertain_count == 1
    assert metrics.uncertain_rate == 0.25


def test_reason_metrics_reject_length_mismatch_and_empty_input() -> None:
    with pytest.raises(ValueError, match="same length"):
        evaluate_reason_hints(["no_match"], [])
    with pytest.raises(ValueError, match="nonempty"):
        evaluate_reason_hints([], [])
