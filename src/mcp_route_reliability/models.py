"""Versioned domain contracts shared by routers, evaluation, and runtime policy code."""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Language = Literal["zh", "en"]
Split = Literal["development", "calibration", "test"]
CaseType = Literal["clear", "hard_negative", "multi_valid", "clarify", "no_match"]
Difficulty = Literal["easy", "medium", "hard"]
AbstainReason = Literal["no_match", "ambiguous", "insufficient_information"]
Decision = Literal["accept", "abstain"]
ReasonHint = Literal["no_match", "ambiguous", "insufficient_information", "uncertain"]
ScoreType = Literal["similarity", "logit", "distance", "rank_only"]
CalibrationMethod = Literal["max_score", "margin", "isotonic", "logistic"]
PolicyKind = Literal["empirical", "conservative"]
PerturbationType = Literal["paraphrase", "cross_language", "noise"]


class StrictModel(BaseModel):
    """Immutable model with no silently ignored input fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class ToolRecord(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    tool_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str
    input_schema: dict[str, object]
    server: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    source_commit: str | None = None
    license: str = Field(min_length=1)


class InferenceQuery(StrictModel):
    """The only query shape accepted by inference-time code."""

    schema_version: Literal["1.0"] = "1.0"
    query_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    language: Language


class QueryCase(StrictModel):
    """Evaluation record containing labels that must not cross the inference boundary."""

    schema_version: Literal["1.0"] = "1.0"
    query_id: str = Field(min_length=1)
    base_query_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    language: Language
    case_type: CaseType
    difficulty: Difficulty
    relevant_tool_ids: tuple[str, ...]
    should_abstain: bool
    abstain_reason: AbstainReason | None
    split: Split
    label_rationale: str = Field(min_length=1)
    candidate_conflict_note: str | None = None
    tool_corpus_hash: str = Field(min_length=1)
    perturbation_type: PerturbationType | None = None

    @model_validator(mode="after")
    def validate_label_semantics(self) -> Self:
        if len(set(self.relevant_tool_ids)) != len(self.relevant_tool_ids):
            raise ValueError("relevant_tool_ids must be unique")

        if self.case_type == "no_match":
            if self.relevant_tool_ids:
                raise ValueError("no_match cases cannot contain relevant tools")
            if not self.should_abstain or self.abstain_reason != "no_match":
                raise ValueError("no_match cases must abstain with reason no_match")
            return self

        if self.case_type == "clarify":
            if not self.should_abstain:
                raise ValueError("clarify cases must abstain")
            if self.abstain_reason not in {"ambiguous", "insufficient_information"}:
                raise ValueError("clarify cases require an ambiguity or information reason")
            return self

        if self.should_abstain or self.abstain_reason is not None:
            raise ValueError("routeable cases cannot carry abstention labels")
        if not self.relevant_tool_ids:
            raise ValueError("routeable cases require at least one relevant tool")
        if self.case_type == "multi_valid" and len(self.relevant_tool_ids) < 2:
            raise ValueError("multi_valid cases require at least two relevant tools")
        return self

    def to_inference_query(self) -> InferenceQuery:
        return InferenceQuery(
            query_id=self.query_id,
            text=self.text,
            language=self.language,
        )


class ScoreMetadata(StrictModel):
    score_type: ScoreType
    higher_is_better: bool
    comparable_across_queries: bool

    @model_validator(mode="after")
    def validate_direction(self) -> Self:
        if self.score_type == "distance" and self.higher_is_better:
            raise ValueError("distance scores must use lower-is-better ordering")
        if self.score_type == "rank_only" and self.comparable_across_queries:
            raise ValueError("rank_only values cannot be comparable across queries")
        return self


class RankedCandidate(StrictModel):
    tool_id: str = Field(min_length=1)
    score: float | None
    rank: int = Field(ge=1)

    @field_validator("score")
    @classmethod
    def validate_finite_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("candidate score must be finite")
        return value


class RouterPrediction(StrictModel):
    schema_version: Literal["1.1"] = "1.1"
    query_id: str = Field(min_length=1)
    router_id: str = Field(min_length=1)
    router_version: str = Field(min_length=1)
    model_version: str | None = None
    tool_corpus_hash: str = Field(min_length=1)
    query_set_hash: str = Field(min_length=1)
    score_metadata: ScoreMetadata
    candidates: tuple[RankedCandidate, ...]
    latency_ms: float | None = Field(default=None, ge=0)
    error: str | None = None

    @model_validator(mode="after")
    def validate_candidates(self) -> Self:
        if not self.candidates and not (self.error and self.error.strip()):
            raise ValueError("empty candidates require a nonempty router error")

        tool_ids = [candidate.tool_id for candidate in self.candidates]
        if len(set(tool_ids)) != len(tool_ids):
            raise ValueError("candidate tool IDs must be unique")

        ranks = [candidate.rank for candidate in self.candidates]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("candidate ranks must be contiguous and start at 1")

        scores = [candidate.score for candidate in self.candidates]
        if self.score_metadata.score_type == "rank_only":
            if any(score is not None for score in scores):
                raise ValueError("rank_only candidates must use score=None")
        elif any(score is None for score in scores):
            raise ValueError("scored predictions require a numeric score for every candidate")
        return self


class ConfidenceFeatures(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    feature_version: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    router_id: str = Field(min_length=1)
    router_version: str = Field(min_length=1)
    model_version: str | None = None
    tool_corpus_hash: str = Field(min_length=1)
    score_metadata: ScoreMetadata
    top1_score: float | None
    top1_score_standardized: float | None
    top1_minus_top2: float | None
    score_entropy: float | None
    score_dispersion: float | None
    score_dispersion_standardized: float | None
    query_length: int = Field(ge=0)
    language: Language
    selected_tool_description_length: int = Field(ge=0)
    selected_tool_description_missing: bool
    lexical_coverage: float = Field(ge=0, le=1)
    router_top1_agreement: bool | None

    @field_validator(
        "top1_score",
        "top1_score_standardized",
        "top1_minus_top2",
        "score_entropy",
        "score_dispersion",
        "score_dispersion_standardized",
    )
    @classmethod
    def validate_optional_finite_value(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("confidence features must be finite")
        return value


class ReliabilityDecision(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    query_id: str = Field(min_length=1)
    top1_tool_id: str = Field(min_length=1)
    selected_tool_id: str | None
    estimated_correctness: float | None = Field(default=None, ge=0, le=1)
    policy_score: float
    decision: Decision
    reason: str = Field(min_length=1)
    reason_hint: ReasonHint | None
    policy_id: str = Field(min_length=1)
    threshold: float

    @field_validator("policy_score", "threshold")
    @classmethod
    def validate_decision_numbers(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("decision values must be finite")
        return value

    @model_validator(mode="after")
    def validate_decision_shape(self) -> Self:
        if self.decision == "accept":
            if self.selected_tool_id is None:
                raise ValueError("accepted decisions require selected_tool_id")
            if self.reason_hint is not None:
                raise ValueError("accepted decisions cannot carry a reason_hint")
        else:
            if self.selected_tool_id is not None:
                raise ValueError("abstained decisions cannot select a tool")
            if self.reason_hint is None:
                raise ValueError("abstained decisions require a reason_hint")
        return self


class PolicyArtifact(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    policy_id: str = Field(min_length=1)
    policy_kind: PolicyKind
    available: bool
    unavailable_reason: str | None = None
    router_id: str = Field(min_length=1)
    router_version: str = Field(min_length=1)
    model_version: str | None = None
    tool_corpus_hash: str = Field(min_length=1)
    calibration_dataset_hash: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    score_metadata: ScoreMetadata
    calibration_method: CalibrationMethod
    threshold: float | None
    target_risk: float = Field(ge=0, le=1)
    minimum_accepted: int = Field(ge=1)
    calibration_sample_size: int = Field(ge=0)
    selected_accepted_count: int = Field(ge=0)
    confidence_interval_method: str = Field(min_length=1)
    source_commit: str = Field(min_length=1)
    feature_names: tuple[str, ...] = ()
    calibrator_payload: dict[str, object]
    empirical_risk: float | None = Field(default=None, ge=0, le=1)
    risk_upper_bound: float | None = Field(default=None, ge=0, le=1)

    @field_validator("threshold")
    @classmethod
    def validate_optional_threshold(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("policy threshold must be finite")
        return value

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        if self.available:
            if self.threshold is None:
                raise ValueError("available policies require a threshold")
            if self.unavailable_reason is not None:
                raise ValueError("available policies cannot have an unavailable reason")
        elif not self.unavailable_reason:
            raise ValueError("unavailable policies require a reason")
        return self


class Annotation(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    annotation_id: str = Field(min_length=1)
    annotator_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    relevant_tool_ids: tuple[str, ...]
    should_abstain: bool
    abstain_reason: AbstainReason | None
    label_rationale: str = Field(min_length=1)
    candidate_conflict_note: str | None = None
    tool_corpus_hash: str = Field(min_length=1)


    @model_validator(mode="after")
    def validate_annotation_semantics(self) -> Self:
        if len(set(self.relevant_tool_ids)) != len(self.relevant_tool_ids):
            raise ValueError("annotation relevant_tool_ids must be unique")
        if self.should_abstain and self.abstain_reason is None:
            raise ValueError("abstaining annotations require a reason")
        if not self.should_abstain and self.abstain_reason is not None:
            raise ValueError("routing annotations cannot carry an abstention reason")
        if self.abstain_reason == "no_match" and self.relevant_tool_ids:
            raise ValueError("no_match annotations cannot contain relevant tools")
        return self


class DisagreementRecord(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    query_id: str = Field(min_length=1)
    primary: Annotation
    secondary: Annotation
    adjudicated: Annotation
