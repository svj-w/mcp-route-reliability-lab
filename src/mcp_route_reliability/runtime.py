"""Fail-closed runtime application of frozen reliability policies."""

from __future__ import annotations

import math

from pydantic import ValidationError

from mcp_route_reliability.confidence.calibrators import (
    IsotonicCalibrator,
    LogisticConfidenceModel,
)
from mcp_route_reliability.confidence.reasons import ReasonHintConfig, infer_reason_hint
from mcp_route_reliability.models import (
    ConfidenceFeatures,
    InferenceQuery,
    PolicyArtifact,
    ReliabilityDecision,
    RouterPrediction,
)


class PolicyApplicationError(ValueError):
    """Base error for inputs that cannot safely produce a runtime decision."""


class PolicyBindingError(PolicyApplicationError):
    """Raised when runtime inputs do not match a policy's frozen bindings."""


class PolicyUnavailableError(PolicyApplicationError):
    """Raised when calibration did not produce an available policy."""


def _binding_error(field: str, expected: object, actual: object) -> PolicyBindingError:
    return PolicyBindingError(
        f"policy binding mismatch for {field}: expected {expected!r}, got {actual!r}"
    )


def _validate_bindings(
    policy: PolicyArtifact,
    query: InferenceQuery,
    prediction: RouterPrediction,
    features: ConfidenceFeatures,
) -> None:
    if type(query) is not InferenceQuery:
        raise TypeError("policy application requires an exact InferenceQuery")
    if query.query_id != prediction.query_id or query.query_id != features.query_id:
        raise PolicyBindingError("query_id mismatch among query, prediction, and features")

    checks = (
        ("router_id", policy.router_id, prediction.router_id, features.router_id),
        (
            "router_version",
            policy.router_version,
            prediction.router_version,
            features.router_version,
        ),
        (
            "model_version",
            policy.model_version,
            prediction.model_version,
            features.model_version,
        ),
        (
            "tool_corpus_hash",
            policy.tool_corpus_hash,
            prediction.tool_corpus_hash,
            features.tool_corpus_hash,
        ),
        (
            "score_metadata",
            policy.score_metadata,
            prediction.score_metadata,
            features.score_metadata,
        ),
    )
    for field, expected, prediction_value, feature_value in checks:
        if prediction_value != expected:
            raise _binding_error(field, expected, prediction_value)
        if feature_value != expected:
            raise _binding_error(field, expected, feature_value)
    if features.feature_version != policy.feature_version:
        raise _binding_error("feature_version", policy.feature_version, features.feature_version)


def _payload_without_reason_config(policy: PolicyArtifact) -> dict[str, object]:
    payload = dict(policy.calibrator_payload)
    payload.pop("reason_hint_config", None)
    return payload


def _validate_feature_contract(policy: PolicyArtifact, expected: tuple[str, ...]) -> None:
    if policy.feature_names != expected:
        raise PolicyBindingError(
            "policy binding mismatch for feature_names: "
            f"expected {expected!r}, got {policy.feature_names!r}"
        )


def _raw_policy_score(policy: PolicyArtifact, features: ConfidenceFeatures) -> float:
    if policy.calibration_method == "max_score":
        _validate_feature_contract(policy, ("top1_score",))
        if features.top1_score is None:
            raise PolicyApplicationError("feature 'top1_score' is missing")
        return (
            features.top1_score if policy.score_metadata.higher_is_better else -features.top1_score
        )
    if policy.calibration_method == "margin":
        _validate_feature_contract(policy, ("top1_minus_top2",))
        if features.top1_minus_top2 is None:
            raise PolicyApplicationError("feature 'top1_minus_top2' is missing")
        return features.top1_minus_top2
    raise AssertionError("raw score requested for a calibrated method")


def _calibrated_policy_score(policy: PolicyArtifact, features: ConfidenceFeatures) -> float:
    payload = _payload_without_reason_config(policy)
    try:
        if policy.calibration_method == "isotonic":
            isotonic = IsotonicCalibrator.model_validate(payload)
            _validate_feature_contract(policy, (isotonic.feature_name,))
            score = float(isotonic.predict_correctness([features])[0])
        elif policy.calibration_method == "logistic":
            logistic = LogisticConfidenceModel.model_validate(payload)
            _validate_feature_contract(policy, logistic.feature_names)
            score = float(logistic.predict_correctness([features])[0])
        else:
            raise AssertionError("calibrator requested for a raw method")
    except (ValidationError, ValueError, IndexError) as error:
        raise PolicyApplicationError(f"invalid calibrator payload or feature: {error}") from error
    if not math.isfinite(score):
        raise PolicyApplicationError("calibrated probability must be finite")
    if not 0.0 <= score <= 1.0:
        raise PolicyApplicationError("calibrated probability must be between zero and one")
    return score


def _reason_config(policy: PolicyArtifact) -> ReasonHintConfig | None:
    payload = policy.calibrator_payload.get("reason_hint_config")
    if payload is None:
        return None
    try:
        return ReasonHintConfig.model_validate(payload)
    except ValidationError as error:
        raise PolicyApplicationError(f"invalid reason_hint_config: {error}") from error


def apply_policy(
    policy: PolicyArtifact,
    query: InferenceQuery,
    prediction: RouterPrediction,
    features: ConfidenceFeatures,
) -> ReliabilityDecision:
    """Apply a frozen policy without labels, I/O, tool execution, or silent fallback."""

    if not policy.available:
        raise PolicyUnavailableError(
            f"policy {policy.policy_id!r} is unavailable: {policy.unavailable_reason}"
        )
    if policy.threshold is None:
        raise PolicyUnavailableError(f"policy {policy.policy_id!r} has no threshold")
    if prediction.error is not None or not prediction.candidates:
        detail = prediction.error or "prediction has no candidates"
        raise PolicyApplicationError(f"cannot apply policy to failed prediction: {detail}")

    _validate_bindings(policy, query, prediction, features)
    if policy.calibration_method in {"max_score", "margin"}:
        score = _raw_policy_score(policy, features)
        estimated_correctness = None
    else:
        if not 0.0 <= policy.threshold <= 1.0:
            raise PolicyApplicationError("calibrated policy threshold must be between zero and one")
        score = _calibrated_policy_score(policy, features)
        estimated_correctness = score
    if not math.isfinite(score):
        raise PolicyApplicationError("policy score must be finite")

    accepted = score >= policy.threshold
    if accepted:
        reason_hint = None
    else:
        config = _reason_config(policy)
        reason_hint = (
            infer_reason_hint(query, prediction, features, config)
            if config is not None
            else "uncertain"
        )

    return ReliabilityDecision(
        query_id=query.query_id,
        top1_tool_id=prediction.candidates[0].tool_id,
        selected_tool_id=prediction.candidates[0].tool_id if accepted else None,
        estimated_correctness=estimated_correctness,
        policy_score=score,
        decision="accept" if accepted else "abstain",
        reason=("confidence_at_or_above_threshold" if accepted else "confidence_below_threshold"),
        reason_hint=reason_hint,
        policy_id=policy.policy_id,
        threshold=policy.threshold,
    )
