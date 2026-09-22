"""Risk-controlled threshold selection and version-bound policy artifacts."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from mcp_route_reliability.hashing import canonical_sha256
from mcp_route_reliability.models import (
    CalibrationMethod,
    PolicyArtifact,
    ScoreMetadata,
    StrictModel,
)


class CalibrationRow(StrictModel):
    """One calibration observation scored so larger values mean more confidence."""

    query_id: str = Field(min_length=1)
    score: float
    is_error: bool

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value


class PolicyCandidate(StrictModel):
    """Metrics for accepting every row at or above a tied threshold."""

    threshold: float
    accepted: int = Field(ge=0)
    coverage: float = Field(ge=0, le=1)
    errors: int = Field(ge=0)
    empirical_risk: float = Field(ge=0, le=1)
    risk_upper_bound: float = Field(ge=0, le=1)

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("threshold must be finite")
        return value

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.errors > self.accepted:
            raise ValueError("errors cannot exceed accepted")
        return self


class PolicyChoice(StrictModel):
    """An available threshold choice or an explicit unavailable result."""

    available: bool
    reason: str | None = None
    candidate: PolicyCandidate | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        if self.available:
            if self.candidate is None:
                raise ValueError("available choices require a candidate")
            if self.reason is not None:
                raise ValueError("available choices cannot carry a reason")
        else:
            if self.candidate is not None:
                raise ValueError("unavailable choices cannot carry a candidate")
            if not self.reason:
                raise ValueError("unavailable choices require a reason")
        return self


class PolicySelectionResult(StrictModel):
    """Empirical and conservative choices from the same calibration sample."""

    method: CalibrationMethod
    target_risk: float = Field(ge=0, le=1)
    minimum_accepted: int = Field(ge=1)
    calibration_sample_size: int = Field(ge=0)
    empirical: PolicyChoice
    conservative: PolicyChoice


class PolicyProvenance(StrictModel):
    """Bindings that prevent a policy from being reused with incompatible inputs."""

    router_id: str = Field(min_length=1)
    router_version: str = Field(min_length=1)
    model_version: str | None = None
    tool_corpus_hash: str = Field(min_length=1)
    calibration_dataset_hash: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    score_metadata: ScoreMetadata
    source_commit: str = Field(min_length=1)
    calibrator_payload: dict[str, object]
    feature_names: tuple[str, ...] = ()


def wilson_upper_bound(
    errors: int,
    accepted: int,
    *,
    confidence: float = 0.95,
) -> float:
    """Return a one-sided Wilson upper confidence bound for the error rate."""

    if accepted < 1:
        raise ValueError("accepted must be at least one")
    if errors < 0 or errors > accepted:
        raise ValueError("errors must be between zero and accepted")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")

    n = float(accepted)
    proportion = errors / n
    z = NormalDist().inv_cdf(confidence)
    z_squared = z * z
    denominator = 1.0 + z_squared / n
    center = (proportion + z_squared / (2.0 * n)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / n
            + z_squared / (4.0 * n * n)
        )
        / denominator
    )
    return min(1.0, center + radius)


def _unavailable(reason: str) -> PolicyChoice:
    return PolicyChoice(available=False, reason=reason)


def _best_candidate(candidates: list[PolicyCandidate]) -> PolicyChoice:
    if not candidates:
        return _unavailable("no_qualifying_threshold")
    selected = max(candidates, key=lambda item: (item.coverage, item.threshold))
    return PolicyChoice(available=True, candidate=selected)


def select_policy(
    calibration_rows: list[CalibrationRow],
    *,
    method: CalibrationMethod,
    target_risk: float = 0.05,
    minimum_accepted: int = 30,
) -> PolicySelectionResult:
    """Select maximum-coverage thresholds without splitting equal-score ties."""

    if not calibration_rows:
        raise ValueError("calibration_rows must not be empty")
    if not 0.0 <= target_risk <= 1.0:
        raise ValueError("target_risk must be between zero and one")
    if minimum_accepted < 1:
        raise ValueError("minimum_accepted must be at least one")

    query_ids = [row.query_id for row in calibration_rows]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("calibration query IDs must be unique")
    if method in {"isotonic", "logistic"} and any(
        not 0.0 <= row.score <= 1.0 for row in calibration_rows
    ):
        raise ValueError("calibrated confidence scores must be between zero and one")

    sample_size = len(calibration_rows)
    if sample_size < minimum_accepted:
        unavailable = _unavailable("insufficient_calibration_samples")
        return PolicySelectionResult(
            method=method,
            target_risk=target_risk,
            minimum_accepted=minimum_accepted,
            calibration_sample_size=sample_size,
            empirical=unavailable,
            conservative=unavailable,
        )

    empirical_candidates: list[PolicyCandidate] = []
    conservative_candidates: list[PolicyCandidate] = []
    for threshold in sorted({row.score for row in calibration_rows}, reverse=True):
        accepted_rows = [row for row in calibration_rows if row.score >= threshold]
        accepted = len(accepted_rows)
        if accepted < minimum_accepted:
            continue
        errors = sum(row.is_error for row in accepted_rows)
        empirical_risk = errors / accepted
        upper_bound = wilson_upper_bound(errors, accepted)
        candidate = PolicyCandidate(
            threshold=threshold,
            accepted=accepted,
            coverage=accepted / sample_size,
            errors=errors,
            empirical_risk=empirical_risk,
            risk_upper_bound=upper_bound,
        )
        if empirical_risk <= target_risk:
            empirical_candidates.append(candidate)
        if upper_bound <= target_risk:
            conservative_candidates.append(candidate)

    return PolicySelectionResult(
        method=method,
        target_risk=target_risk,
        minimum_accepted=minimum_accepted,
        calibration_sample_size=sample_size,
        empirical=_best_candidate(empirical_candidates),
        conservative=_best_candidate(conservative_candidates),
    )


def build_policy_artifact(
    selection: PolicySelectionResult,
    provenance: PolicyProvenance,
    *,
    policy_kind: Literal["empirical", "conservative"],
) -> PolicyArtifact:
    """Materialize a deterministic, fully bound policy artifact."""

    choice = selection.empirical if policy_kind == "empirical" else selection.conservative
    candidate = choice.candidate
    identity = {
        "policy_kind": policy_kind,
        "selection": selection,
        "provenance": provenance,
    }
    digest = canonical_sha256(identity).removeprefix("sha256:")[:12]

    return PolicyArtifact(
        policy_id=f"policy-{policy_kind}-{digest}",
        policy_kind=policy_kind,
        available=choice.available,
        unavailable_reason=choice.reason,
        router_id=provenance.router_id,
        router_version=provenance.router_version,
        model_version=provenance.model_version,
        tool_corpus_hash=provenance.tool_corpus_hash,
        calibration_dataset_hash=provenance.calibration_dataset_hash,
        feature_version=provenance.feature_version,
        score_metadata=provenance.score_metadata,
        calibration_method=selection.method,
        threshold=candidate.threshold if candidate else None,
        target_risk=selection.target_risk,
        minimum_accepted=selection.minimum_accepted,
        calibration_sample_size=selection.calibration_sample_size,
        selected_accepted_count=candidate.accepted if candidate else 0,
        confidence_interval_method="one-sided Wilson 95%",
        source_commit=provenance.source_commit,
        feature_names=provenance.feature_names,
        calibrator_payload=provenance.calibrator_payload,
        empirical_risk=candidate.empirical_risk if candidate else None,
        risk_upper_bound=candidate.risk_upper_bound if candidate else None,
    )
