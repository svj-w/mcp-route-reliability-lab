"""Deterministic percentile intervals with whole-group resampling."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np
from pydantic import Field, field_validator

from mcp_route_reliability.models import StrictModel


class MetricRow(StrictModel):
    row_id: str = Field(min_length=1)
    base_query_id: str = Field(min_length=1)
    value: float

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("metric row value must be finite")
        return value


class BootstrapInterval(StrictModel):
    estimate: float
    lower: float
    upper: float
    confidence_level: float = Field(ge=0, le=1)
    replicates: int = Field(ge=1)
    seed: int
    group_key: str = Field(min_length=1)
    group_count: int = Field(ge=1)

    @field_validator("estimate", "lower", "upper")
    @classmethod
    def validate_finite_metric(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("bootstrap metrics must be finite")
        return value


def _finite_metric(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("metric must return a finite value")
    return result


def grouped_bootstrap(
    rows: Sequence[MetricRow],
    metric: Callable[[Sequence[MetricRow]], float],
    *,
    group_key: str = "base_query_id",
    replicates: int = 1000,
    seed: int = 20260921,
    production_mode: bool = True,
) -> BootstrapInterval:
    """Resample group IDs with replacement and always retain complete groups."""

    if not rows:
        raise ValueError("grouped bootstrap requires nonempty rows")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if production_mode and replicates < 1000:
        raise ValueError("production mode requires at least 1000 replicates")
    if not group_key:
        raise ValueError("group_key must not be empty")

    groups: dict[str, list[MetricRow]] = defaultdict(list)
    for row in rows:
        if not hasattr(row, group_key):
            raise ValueError(f"MetricRow has no group key {group_key!r}")
        group_value = getattr(row, group_key)
        if not isinstance(group_value, str) or not group_value:
            raise ValueError("group key values must be nonempty strings")
        groups[group_value].append(row)
    group_ids = sorted(groups)

    estimate = _finite_metric(metric(rows))
    generator = random.Random(seed)
    replicate_values: list[float] = []
    for _ in range(replicates):
        sampled_ids = generator.choices(group_ids, k=len(group_ids))
        sampled_rows = [row for group_id in sampled_ids for row in groups[group_id]]
        replicate_values.append(_finite_metric(metric(sampled_rows)))

    lower, upper = np.percentile(
        np.asarray(replicate_values, dtype=np.float64),
        [2.5, 97.5],
        method="linear",
    )
    return BootstrapInterval(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        confidence_level=0.95,
        replicates=replicates,
        seed=seed,
        group_key=group_key,
        group_count=len(group_ids),
    )
