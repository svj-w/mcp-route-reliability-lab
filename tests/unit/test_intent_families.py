# ruff: noqa: RUF001 -- The fixture intentionally contains Chinese punctuation.

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from mcp_route_reliability.datasets import load_queries
from mcp_route_reliability.intent_families import (
    assign_family_splits,
    intent_family_key,
    synchronize_variants,
)

ROOT = Path(__file__).resolve().parents[2]


def test_scenario_numbers_share_a_family() -> None:
    assert intent_family_key("Show /project (scenario 1)") == intent_family_key(
        "Show /project (scenario 81)"
    )
    assert intent_family_key("查看目录（V1 场景 2）") == intent_family_key(
        "查看目录（V1 场景 92）"
    )


def test_family_allocation_is_exact_and_deterministic() -> None:
    rows = load_queries(ROOT / "data" / "v1-candidate" / "queries.jsonl")
    bases = [row for row in rows if row.perturbation_type is None]
    pinned = {row.base_query_id for row in rows if row.perturbation_type is not None}
    first = assign_family_splits(bases, pinned_test_ids=pinned, seed=20260924)
    second = assign_family_splits(bases, pinned_test_ids=pinned, seed=20260924)
    assert first == second
    assert Counter(first.values()) == {"development": 100, "calibration": 100, "test": 100}
    assert all(first[query_id] == "test" for query_id in pinned)
    families: dict[str, set[str]] = defaultdict(set)
    for row in bases:
        families[intent_family_key(row.text)].add(first[row.query_id])
    assert all(len(splits) == 1 for splits in families.values())


def test_v1_q183_variants_follow_rewritten_base() -> None:
    rows = load_queries(ROOT / "data" / "v1-candidate" / "queries.jsonl")
    base = next(row for row in rows if row.query_id == "v1_q_183")
    final = base.model_copy(update={
        "text": (
            "查看 /project：尚未确定是列出当前层内容，还是递归查找匹配 *.md 的路径"
            "（V1 场景 183）"
        ),
    })
    variants = [row for row in rows if row.base_query_id == base.query_id and row.perturbation_type]
    revised = synchronize_variants(variants, {base.query_id: final}, {base.query_id: "test"})
    assert len(revised) == 3
    assert all(row.relevant_tool_ids == final.relevant_tool_ids for row in revised)
    assert all(row.split == "test" for row in revised)
    assert all("/project" in row.text and "*.md" in row.text for row in revised)
    assert all("no path" not in row.text.lower() for row in revised)
