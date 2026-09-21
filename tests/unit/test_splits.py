from __future__ import annotations

import pytest

from mcp_route_reliability.datasets import assign_grouped_splits
from mcp_route_reliability.models import QueryCase, ToolRecord


def test_grouped_split_is_reproducible_and_keeps_variants_together(
    tiny_tools: list[ToolRecord], tiny_queries: list[QueryCase]
) -> None:
    base = tiny_queries[0]
    variant = base.model_copy(
        update={
            "query_id": "q_000_noise",
            "text": "请执行第零个合成能力呀",
            "perturbation_type": "noise",
        }
    )
    cases = [*tiny_queries, variant]
    domains = {tool.tool_id: tool.domain for tool in tiny_tools}
    targets = {"development": 8, "calibration": 6, "test": 6}

    first = assign_grouped_splits(cases, seed=20260921, targets=targets, tool_domains=domains)
    second = assign_grouped_splits(cases, seed=20260921, targets=targets, tool_domains=domains)

    assert first == second
    assert set(first) == {case.base_query_id for case in cases}
    assert first[base.base_query_id] == first[variant.base_query_id]
    assert list(first.values()).count("development") == 8
    assert list(first.values()).count("calibration") == 6
    assert list(first.values()).count("test") == 6


@pytest.mark.parametrize(
    "targets",
    [
        {"development": 8, "calibration": 6, "test": 5},
        {"development": 8, "calibration": -1, "test": 13},
    ],
)
def test_impossible_split_targets_fail_clearly(
    tiny_queries: list[QueryCase], targets: dict[str, int]
) -> None:
    with pytest.raises(ValueError, match="target"):
        assign_grouped_splits(tiny_queries, seed=1, targets=targets)


def test_group_requires_exactly_one_base_record(tiny_queries: list[QueryCase]) -> None:
    without_base = [case for case in tiny_queries if case.query_id != "q_000"]
    orphan_variant = tiny_queries[0].model_copy(
        update={
            "query_id": "q_000_noise",
            "text": "orphan unique text",
            "perturbation_type": "noise",
        }
    )
    with pytest.raises(ValueError, match="base record"):
        assign_grouped_splits([*without_base, orphan_variant], seed=1, targets={
            "development": 8,
            "calibration": 6,
            "test": 6,
        })
