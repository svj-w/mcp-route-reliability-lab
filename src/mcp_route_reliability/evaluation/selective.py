"""Selective prediction risk/coverage curves and AURC."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from mcp_route_reliability.models import StrictModel


class RiskCoveragePoint(StrictModel):
    accepted: int = Field(ge=0)
    coverage: float = Field(ge=0, le=1)
    risk: float = Field(ge=0, le=1)
    threshold: float | None = Field(default=None, ge=0, le=1)


class RiskCoverageResult(StrictModel):
    points: tuple[RiskCoveragePoint, ...]
    aurc: float = Field(ge=0, le=1)


def risk_coverage_curve(
    y_error: NDArray[np.generic], p_correct: NDArray[np.generic]
) -> RiskCoverageResult:
    errors = np.asarray(y_error)
    confidence = np.asarray(p_correct, dtype=np.float64)
    if errors.ndim != 1 or confidence.ndim != 1:
        raise ValueError("errors and confidence must be one-dimensional")
    if errors.size == 0:
        raise ValueError("errors and confidence must be nonempty")
    if errors.size != confidence.size:
        raise ValueError("errors and confidence must have the same length")
    if not np.isin(errors, [0, 1]).all():
        raise ValueError("errors must be binary")
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("confidence must be finite and within [0, 1]")

    order = np.argsort(-confidence, kind="stable")
    sorted_errors = errors[order].astype(np.int64)
    sorted_confidence = confidence[order]
    points: list[RiskCoveragePoint] = [
        RiskCoveragePoint(accepted=0, coverage=0.0, risk=0.0, threshold=None)
    ]
    cumulative_errors = 0
    sample_size = len(errors)
    for index, (error, threshold) in enumerate(
        zip(sorted_errors, sorted_confidence, strict=True), start=1
    ):
        cumulative_errors += int(error)
        points.append(
            RiskCoveragePoint(
                accepted=index,
                coverage=index / sample_size,
                risk=cumulative_errors / index,
                threshold=float(threshold),
            )
        )

    aurc = sum(
        (right.coverage - left.coverage) * (left.risk + right.risk) / 2
        for left, right in pairwise(points)
    )
    return RiskCoverageResult(points=tuple(points), aurc=aurc)
