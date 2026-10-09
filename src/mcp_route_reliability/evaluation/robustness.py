"""Paired distribution-shift metrics joined by stable base-query identity."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from pydantic import Field, field_validator

from mcp_route_reliability.models import PerturbationType, StrictModel


class EvaluatedCase(StrictModel):
    """Inference result fields needed for paired robustness analysis."""

    query_id: str = Field(min_length=1)
    base_query_id: str = Field(min_length=1)
    perturbation_type: PerturbationType | None
    top1_tool_id: str = Field(min_length=1)
    accepted: bool | None
    confidence: float
    correct: bool

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("confidence must be finite")
        return value


class PairedDelta(StrictModel):
    base_query_id: str
    variant_query_id: str
    perturbation_type: PerturbationType
    top1_flipped: bool
    decision_flipped: bool | None
    confidence_shift: float
    accuracy_shift: int = Field(ge=-1, le=1)


class RobustnessMetrics(StrictModel):
    pair_count: int = Field(ge=1)
    top1_flip_rate: float = Field(ge=0, le=1)
    decision_flip_rate: float | None = Field(ge=0, le=1)
    mean_confidence_shift: float
    mean_accuracy_shift: float = Field(ge=-1, le=1)


class RobustnessResult(StrictModel):
    overall: RobustnessMetrics
    by_perturbation: dict[str, RobustnessMetrics]
    paired_deltas: tuple[PairedDelta, ...]


def _aggregate(deltas: Sequence[PairedDelta]) -> RobustnessMetrics:
    if not deltas:
        raise ValueError("robustness aggregation requires at least one pair")
    return RobustnessMetrics(
        pair_count=len(deltas),
        top1_flip_rate=statistics.fmean(delta.top1_flipped for delta in deltas),
        decision_flip_rate=(
            statistics.fmean(
                delta.decision_flipped for delta in deltas if delta.decision_flipped is not None
            )
            if all(delta.decision_flipped is not None for delta in deltas)
            else None
        ),
        mean_confidence_shift=statistics.fmean(delta.confidence_shift for delta in deltas),
        mean_accuracy_shift=statistics.fmean(delta.accuracy_shift for delta in deltas),
    )


def paired_robustness(
    base: Sequence[EvaluatedCase],
    variants: Sequence[EvaluatedCase],
) -> RobustnessResult:
    """Compare every perturbation with exactly one unperturbed base record."""

    if not base:
        raise ValueError("paired robustness requires nonempty base records")
    bases_by_id: dict[str, EvaluatedCase] = {}
    for row in base:
        if row.perturbation_type is not None or row.query_id != row.base_query_id:
            raise ValueError(f"malformed base record {row.query_id!r}")
        if row.base_query_id in bases_by_id:
            raise ValueError(f"duplicate base record {row.base_query_id!r}")
        bases_by_id[row.base_query_id] = row
    if not variants:
        raise ValueError("paired robustness requires nonempty variant records")
    if len({row.accepted is not None for row in (*base, *variants)}) != 1:
        raise ValueError("paired robustness rejects mixed decision availability")

    seen_query_ids: set[str] = set()
    seen_pairs: set[tuple[str, PerturbationType]] = set()
    deltas: list[PairedDelta] = []
    for variant in variants:
        if variant.perturbation_type is None or variant.query_id == variant.base_query_id:
            raise ValueError(f"malformed variant record {variant.query_id!r}")
        if variant.query_id in seen_query_ids:
            raise ValueError(f"duplicate variant query ID {variant.query_id!r}")
        seen_query_ids.add(variant.query_id)
        try:
            base_row = bases_by_id[variant.base_query_id]
        except KeyError as error:
            raise ValueError(
                f"variant {variant.query_id!r} references missing base {variant.base_query_id!r}"
            ) from error
        pair_key = (variant.base_query_id, variant.perturbation_type)
        if pair_key in seen_pairs:
            raise ValueError(f"duplicate variant pair {pair_key!r}")
        seen_pairs.add(pair_key)
        deltas.append(
            PairedDelta(
                base_query_id=variant.base_query_id,
                variant_query_id=variant.query_id,
                perturbation_type=variant.perturbation_type,
                top1_flipped=variant.top1_tool_id != base_row.top1_tool_id,
                decision_flipped=(
                    variant.accepted != base_row.accepted
                    if variant.accepted is not None and base_row.accepted is not None
                    else None
                ),
                confidence_shift=variant.confidence - base_row.confidence,
                accuracy_shift=int(variant.correct) - int(base_row.correct),
            )
        )

    by_perturbation: dict[str, RobustnessMetrics] = {}
    for perturbation in ("paraphrase", "cross_language", "noise"):
        slice_deltas = [delta for delta in deltas if delta.perturbation_type == perturbation]
        if slice_deltas:
            by_perturbation[perturbation] = _aggregate(slice_deltas)
    return RobustnessResult(
        overall=_aggregate(deltas),
        by_perturbation=by_perturbation,
        paired_deltas=tuple(deltas),
    )
