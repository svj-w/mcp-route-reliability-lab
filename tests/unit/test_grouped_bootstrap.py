from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import pytest

from mcp_route_reliability.evaluation.bootstrap import (
    MetricRow,
    grouped_bootstrap,
)


def mean_value(rows: Sequence[MetricRow]) -> float:
    return sum(row.value for row in rows) / len(rows)


def grouped_rows() -> list[MetricRow]:
    return [
        MetricRow(row_id="g1-base", base_query_id="g1", value=0.0),
        MetricRow(row_id="g1-noise", base_query_id="g1", value=1.0),
        MetricRow(row_id="g2-base", base_query_id="g2", value=1.0),
        MetricRow(row_id="g3-base", base_query_id="g3", value=0.0),
        MetricRow(row_id="g3-para", base_query_id="g3", value=0.5),
        MetricRow(row_id="g3-cross", base_query_id="g3", value=1.0),
    ]


def test_every_replicate_resamples_complete_base_query_groups() -> None:
    original_sizes = {"g1": 2, "g2": 1, "g3": 3}
    observed_replicates = 0

    def assert_complete_groups(rows: Sequence[MetricRow]) -> float:
        nonlocal observed_replicates
        observed_replicates += 1
        counts = Counter(row.base_query_id for row in rows)
        multiplicities = []
        for group_id, count in counts.items():
            assert count % original_sizes[group_id] == 0
            multiplicities.append(count // original_sizes[group_id])
        assert sum(multiplicities) == 3
        return mean_value(rows)

    grouped_bootstrap(
        grouped_rows(),
        assert_complete_groups,
        replicates=25,
        seed=17,
        production_mode=False,
    )
    assert observed_replicates == 26  # original estimate plus 25 replicates


def test_fixed_seed_is_reproducible() -> None:
    first = grouped_bootstrap(
        grouped_rows(), mean_value, replicates=50, seed=123, production_mode=False
    )
    second = grouped_bootstrap(
        grouped_rows(), mean_value, replicates=50, seed=123, production_mode=False
    )
    assert first == second


def test_production_mode_requires_at_least_one_thousand_replicates() -> None:
    with pytest.raises(ValueError, match="at least 1000"):
        grouped_bootstrap(grouped_rows(), mean_value, replicates=999)


def test_percentile_bounds_match_saved_small_example() -> None:
    rows = [
        MetricRow(row_id="a", base_query_id="a", value=0.0),
        MetricRow(row_id="b", base_query_id="b", value=1.0),
    ]
    result = grouped_bootstrap(rows, mean_value, replicates=8, seed=7, production_mode=False)
    assert result.estimate == 0.5
    assert result.lower == 0.0
    assert result.upper == 0.5
    assert result.group_count == 2
    assert result.group_key == "base_query_id"


def test_invalid_rows_and_metric_results_are_rejected() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        grouped_bootstrap([], mean_value, replicates=8, production_mode=False)
    with pytest.raises(ValueError, match="finite"):
        grouped_bootstrap(
            grouped_rows(),
            lambda _: float("nan"),
            replicates=8,
            production_mode=False,
        )
