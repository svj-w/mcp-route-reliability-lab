"""Probability-quality metrics with explicit edge-case availability."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from sklearn.metrics import roc_auc_score

from mcp_route_reliability.models import StrictModel


class OptionalMetric(StrictModel):
    value: float | None
    available: bool
    reason: str | None


class ECEBin(StrictModel):
    lower_bound: float = Field(ge=0, le=1)
    upper_bound: float = Field(ge=0, le=1)
    count: int = Field(ge=1)
    mean_confidence: float = Field(ge=0, le=1)
    accuracy: float = Field(ge=0, le=1)
    absolute_gap: float = Field(ge=0, le=1)


class ECEResult(StrictModel):
    ece: float = Field(ge=0, le=1)
    bins: tuple[ECEBin, ...]


def _validated_binary_probabilities(
    target: NDArray[np.generic], probability: NDArray[np.generic]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    y = np.asarray(target)
    p = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1:
        raise ValueError("targets and probabilities must be one-dimensional")
    if y.size == 0:
        raise ValueError("targets and probabilities must be nonempty")
    if y.size != p.size:
        raise ValueError("targets and probabilities must have the same length")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("targets must be binary")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite and within [0, 1]")
    return y.astype(np.int64), p


def brier_score(
    y_correct: NDArray[np.generic], p_correct: NDArray[np.generic]
) -> float:
    y, p = _validated_binary_probabilities(y_correct, p_correct)
    return float(np.mean((p - y) ** 2))


def equal_frequency_ece(
    y_correct: NDArray[np.generic],
    p_correct: NDArray[np.generic],
    bins: int = 5,
) -> ECEResult:
    y, p = _validated_binary_probabilities(y_correct, p_correct)
    if bins < 1:
        raise ValueError("bins must be positive")
    if y.size < bins:
        raise ValueError(f"ECE requires at least {bins} samples")

    order = np.argsort(p, kind="stable")
    grouped_indices = np.array_split(order, bins)
    result_bins: list[ECEBin] = []
    weighted_gap = 0.0
    for indices in grouped_indices:
        bin_confidence = p[indices]
        bin_target = y[indices]
        mean_confidence = float(np.mean(bin_confidence))
        accuracy = float(np.mean(bin_target))
        gap = abs(mean_confidence - accuracy)
        weighted_gap += len(indices) / len(y) * gap
        result_bins.append(
            ECEBin(
                lower_bound=float(np.min(bin_confidence)),
                upper_bound=float(np.max(bin_confidence)),
                count=len(indices),
                mean_confidence=mean_confidence,
                accuracy=accuracy,
                absolute_gap=gap,
            )
        )
    return ECEResult(ece=weighted_gap, bins=tuple(result_bins))


def error_auroc(
    y_error: NDArray[np.generic], p_error: NDArray[np.generic]
) -> OptionalMetric:
    y, p = _validated_binary_probabilities(y_error, p_error)
    if np.unique(y).size < 2:
        return OptionalMetric(value=None, available=False, reason="single_target_class")
    return OptionalMetric(
        value=float(roc_auc_score(y, p)),
        available=True,
        reason=None,
    )
