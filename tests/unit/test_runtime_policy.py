from __future__ import annotations

import math

import pytest

from mcp_route_reliability.confidence.reasons import ReasonHintConfig
from mcp_route_reliability.models import (
    ConfidenceFeatures,
    InferenceQuery,
    PolicyArtifact,
    RankedCandidate,
    RouterPrediction,
    ScoreMetadata,
)
from mcp_route_reliability.runtime import (
    PolicyApplicationError,
    PolicyBindingError,
    PolicyUnavailableError,
    apply_policy,
)

METADATA = ScoreMetadata(
    score_type="similarity",
    higher_is_better=True,
    comparable_across_queries=False,
)


def query() -> InferenceQuery:
    return InferenceQuery(query_id="q1", text="search code", language="en")


def prediction(**updates: object) -> RouterPrediction:
    values: dict[str, object] = {
        "query_id": "q1",
        "router_id": "bm25",
        "router_version": "1.0.0",
        "model_version": "rank-bm25:0.2.2",
        "tool_corpus_hash": "sha256:corpus",
        "query_set_hash": "sha256:new-runtime-queries",
        "score_metadata": METADATA,
        "candidates": (
            RankedCandidate(tool_id="tool.search", score=0.8, rank=1),
            RankedCandidate(tool_id="tool.issue", score=0.3, rank=2),
        ),
    }
    values.update(updates)
    return RouterPrediction.model_validate(values)


def features(**updates: object) -> ConfidenceFeatures:
    values: dict[str, object] = {
        "feature_version": "1.0",
        "query_id": "q1",
        "router_id": "bm25",
        "router_version": "1.0.0",
        "model_version": "rank-bm25:0.2.2",
        "tool_corpus_hash": "sha256:corpus",
        "score_metadata": METADATA,
        "top1_score": 0.8,
        "top1_score_standardized": 0.5,
        "top1_minus_top2": 0.5,
        "score_entropy": 0.4,
        "score_dispersion": 0.25,
        "score_dispersion_standardized": 0.0,
        "query_length": 2,
        "language": "en",
        "selected_tool_description_length": 20,
        "selected_tool_description_missing": False,
        "lexical_coverage": 0.8,
        "router_top1_agreement": None,
    }
    values.update(updates)
    return ConfidenceFeatures.model_validate(values)


def policy(**updates: object) -> PolicyArtifact:
    values: dict[str, object] = {
        "policy_id": "policy-max-score",
        "policy_kind": "empirical",
        "available": True,
        "router_id": "bm25",
        "router_version": "1.0.0",
        "model_version": "rank-bm25:0.2.2",
        "tool_corpus_hash": "sha256:corpus",
        "calibration_dataset_hash": "sha256:old-calibration-dataset",
        "feature_version": "1.0",
        "score_metadata": METADATA,
        "calibration_method": "max_score",
        "threshold": 0.8,
        "target_risk": 0.05,
        "minimum_accepted": 30,
        "calibration_sample_size": 100,
        "selected_accepted_count": 50,
        "confidence_interval_method": "one-sided Wilson 95%",
        "source_commit": "abc123",
        "feature_names": ("top1_score",),
        "calibrator_payload": {"method": "max_score"},
        "empirical_risk": 0.04,
        "risk_upper_bound": 0.08,
    }
    values.update(updates)
    return PolicyArtifact.model_validate(values)


@pytest.mark.parametrize(
    ("policy_updates", "prediction_updates", "feature_updates", "field"),
    [
        ({"router_id": "other"}, {}, {}, "router_id"),
        ({"router_version": "2.0.0"}, {}, {}, "router_version"),
        ({"model_version": "other-model"}, {}, {}, "model_version"),
        ({"tool_corpus_hash": "sha256:other"}, {}, {}, "tool_corpus_hash"),
        ({"feature_version": "2.0"}, {}, {}, "feature_version"),
        (
            {
                "score_metadata": ScoreMetadata(
                    score_type="logit", higher_is_better=True, comparable_across_queries=False
                )
            },
            {},
            {},
            "score_metadata",
        ),
    ],
)
def test_binding_mismatches_fail_closed(
    policy_updates: dict[str, object],
    prediction_updates: dict[str, object],
    feature_updates: dict[str, object],
    field: str,
) -> None:
    with pytest.raises(PolicyBindingError, match=field):
        apply_policy(
            policy(**policy_updates),
            query(),
            prediction(**prediction_updates),
            features(**feature_updates),
        )


def test_unavailable_policy_fails_closed() -> None:
    unavailable = policy(
        available=False,
        unavailable_reason="no_qualifying_threshold",
        threshold=None,
        selected_accepted_count=0,
        empirical_risk=None,
        risk_upper_bound=None,
    )
    with pytest.raises(PolicyUnavailableError, match="no_qualifying_threshold"):
        apply_policy(unavailable, query(), prediction(), features())


def test_prediction_error_or_no_candidates_is_rejected() -> None:
    failed = prediction(candidates=(), error="router timeout")
    with pytest.raises(PolicyApplicationError, match="router timeout"):
        apply_policy(policy(), query(), failed, features())


def test_exact_threshold_is_accepted_and_calibration_hash_is_only_provenance() -> None:
    decision = apply_policy(policy(), query(), prediction(), features())
    assert decision.decision == "accept"
    assert decision.selected_tool_id == "tool.search"
    assert decision.reason_hint is None
    assert decision.policy_score == 0.8
    assert decision.estimated_correctness is None


def test_isotonic_missing_feature_is_rejected() -> None:
    isotonic = policy(
        policy_id="policy-isotonic",
        calibration_method="isotonic",
        threshold=0.5,
        feature_names=("top1_score_standardized",),
        calibrator_payload={
            "method": "isotonic",
            "available": True,
            "feature_name": "top1_score_standardized",
            "x_thresholds": [-1.0, 1.0],
            "y_thresholds": [0.2, 0.9],
            "sample_size": 100,
            "unique_score_count": 20,
        },
    )
    with pytest.raises(PolicyApplicationError, match="missing"):
        apply_policy(isotonic, query(), prediction(), features(top1_score_standardized=None))


def test_non_finite_calibrated_probability_is_rejected() -> None:
    logistic = policy(
        policy_id="policy-logistic",
        calibration_method="logistic",
        threshold=0.5,
        feature_names=(
            "top1_score_standardized",
            "top1_minus_top2",
            "query_length",
            "lexical_coverage",
        ),
        calibrator_payload={
            "method": "logistic",
            "available": True,
            "feature_names": [
                "top1_score_standardized",
                "top1_minus_top2",
                "query_length",
                "lexical_coverage",
            ],
            "medians": [0.0, 0.0, 2.0, 0.5],
            "means": [0.0, 0.0, 2.0, 0.5],
            "scales": [1.0, 1.0, 1.0, 1.0],
            "coefficients": [math.nan, 0.0, 0.0, 0.0],
            "intercept": 0.0,
            "sample_size": 100,
            "regularization": "l2",
            "c_value": 1.0,
        },
    )
    with pytest.raises(PolicyApplicationError, match="finite"):
        apply_policy(logistic, query(), prediction(), features())


def test_reason_config_changes_hint_but_not_abstention() -> None:
    base_config = ReasonHintConfig(
        calibration_sample_size=100,
        query_token_upper_bound=1,
        lexical_coverage_upper_bound=0.1,
        standardized_top1_lower_bound=-1.0,
        margin_lower_bound=0.1,
        entropy_upper_bound=1.0,
    )
    no_match_config = base_config.model_copy(update={"standardized_top1_lower_bound": 1.0})
    low_score_policy = policy(threshold=0.81)
    first = apply_policy(
        low_score_policy.model_copy(
            update={
                "calibrator_payload": {
                    "method": "max_score",
                    "reason_hint_config": base_config.model_dump(mode="json"),
                }
            }
        ),
        query(),
        prediction(),
        features(),
    )
    second = apply_policy(
        low_score_policy.model_copy(
            update={
                "calibrator_payload": {
                    "method": "max_score",
                    "reason_hint_config": no_match_config.model_dump(mode="json"),
                }
            }
        ),
        query(),
        prediction(),
        features(),
    )
    assert first.decision == second.decision == "abstain"
    assert first.reason_hint == "uncertain"
    assert second.reason_hint == "no_match"
