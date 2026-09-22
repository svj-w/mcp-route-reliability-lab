from __future__ import annotations

import pytest

from mcp_route_reliability.confidence.policies import (
    CalibrationRow,
    PolicyProvenance,
    build_policy_artifact,
    select_policy,
    wilson_upper_bound,
)
from mcp_route_reliability.models import PolicyArtifact, ScoreMetadata


def rows(scores_and_errors: list[tuple[float, bool]]) -> list[CalibrationRow]:
    return [
        CalibrationRow(query_id=f"q{index:03d}", score=score, is_error=is_error)
        for index, (score, is_error) in enumerate(scores_and_errors)
    ]


def test_one_sided_wilson_upper_bound_matches_reference() -> None:
    assert wilson_upper_bound(0, 30) == pytest.approx(0.082724, abs=1e-6)
    assert wilson_upper_bound(0, 100) == pytest.approx(0.026343, abs=1e-6)
    with pytest.raises(ValueError, match="accepted"):
        wilson_upper_bound(0, 0)
    with pytest.raises(ValueError, match="errors"):
        wilson_upper_bound(2, 1)


def test_fewer_than_30_samples_produces_explicit_unavailable_policies() -> None:
    selection = select_policy(
        rows([(0.9, False)] * 29),
        method="max_score",
        target_risk=0.05,
        minimum_accepted=30,
    )
    assert selection.empirical.available is False
    assert selection.empirical.reason == "insufficient_calibration_samples"
    assert selection.conservative.available is False


def test_exactly_30_zero_errors_has_empirical_but_not_conservative_policy() -> None:
    selection = select_policy(
        rows([(0.9, False)] * 30),
        method="margin",
        target_risk=0.05,
        minimum_accepted=30,
    )
    assert selection.empirical.available is True
    assert selection.empirical.candidate is not None
    assert selection.empirical.candidate.accepted == 30
    assert selection.empirical.candidate.empirical_risk == 0.0
    assert selection.conservative.available is False
    assert selection.conservative.reason == "no_qualifying_threshold"


def test_empirical_risk_equal_to_target_qualifies() -> None:
    values = [(0.8, index < 5) for index in range(100)]
    selection = select_policy(rows(values), method="isotonic", target_risk=0.05)
    assert selection.empirical.available is True
    assert selection.empirical.candidate is not None
    assert selection.empirical.candidate.empirical_risk == pytest.approx(0.05)
    assert selection.empirical.candidate.accepted == 100


def test_no_empirical_candidate_is_visible() -> None:
    selection = select_policy(
        rows([(0.9, True)] * 100),
        method="logistic",
        target_risk=0.05,
    )
    assert selection.empirical.available is False
    assert selection.empirical.reason == "no_qualifying_threshold"
    assert selection.conservative.available is False


def test_conservative_policy_becomes_available_with_enough_zero_error_samples() -> None:
    selection = select_policy(rows([(0.7, False)] * 100), method="logistic")
    assert selection.conservative.available is True
    assert selection.conservative.candidate is not None
    assert selection.conservative.candidate.accepted == 100
    assert selection.conservative.candidate.risk_upper_bound < 0.05


def test_thresholds_never_split_ties_and_choose_maximum_coverage() -> None:
    values = [(0.9, False)] * 35 + [(0.8, True)] * 10 + [(0.7, False)] * 55
    selection = select_policy(rows(values), method="max_score")
    assert selection.empirical.candidate is not None
    assert selection.empirical.candidate.threshold == 0.9
    assert selection.empirical.candidate.accepted == 35


def test_build_policy_artifact_serializes_all_bindings() -> None:
    selection = select_policy(rows([(0.7, False)] * 100), method="logistic")
    provenance = PolicyProvenance(
        router_id="embedding",
        router_version="1.0.0",
        model_version="model:abc",
        tool_corpus_hash="sha256:corpus",
        calibration_dataset_hash="sha256:calibration",
        feature_version="1.0",
        score_metadata=ScoreMetadata(
            score_type="similarity",
            higher_is_better=True,
            comparable_across_queries=False,
        ),
        source_commit="abc123",
        calibrator_payload={"method": "logistic", "coefficients": [1.0]},
        feature_names=("top1_score_standardized", "top1_minus_top2"),
    )

    artifact = build_policy_artifact(selection, provenance, policy_kind="conservative")

    assert isinstance(artifact, PolicyArtifact)
    assert artifact.available is True
    assert artifact.router_id == "embedding"
    assert artifact.router_version == "1.0.0"
    assert artifact.model_version == "model:abc"
    assert artifact.tool_corpus_hash == "sha256:corpus"
    assert artifact.calibration_dataset_hash == "sha256:calibration"
    assert artifact.feature_version == "1.0"
    assert artifact.score_metadata == provenance.score_metadata
    assert artifact.threshold == 0.7
    assert artifact.target_risk == 0.05
    assert artifact.minimum_accepted == 30
    assert artifact.calibration_sample_size == 100
    assert artifact.selected_accepted_count == 100
    assert artifact.confidence_interval_method == "one-sided Wilson 95%"
    assert artifact.source_commit == "abc123"
    assert artifact.calibrator_payload == provenance.calibrator_payload


def test_unavailable_artifact_retains_reason_and_provenance() -> None:
    selection = select_policy(rows([(0.9, False)] * 30), method="margin")
    provenance = PolicyProvenance(
        router_id="bm25",
        router_version="1.0.0",
        model_version="rank-bm25:0.2.2",
        tool_corpus_hash="sha256:corpus",
        calibration_dataset_hash="sha256:calibration",
        feature_version="1.0",
        score_metadata=ScoreMetadata(
            score_type="similarity",
            higher_is_better=True,
            comparable_across_queries=False,
        ),
        source_commit="abc123",
        calibrator_payload={"method": "margin"},
        feature_names=("top1_minus_top2",),
    )
    artifact = build_policy_artifact(selection, provenance, policy_kind="conservative")
    assert artifact.available is False
    assert artifact.threshold is None
    assert artifact.unavailable_reason == "no_qualifying_threshold"
