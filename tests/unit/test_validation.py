from __future__ import annotations

from pathlib import Path

import pytest

from mcp_route_reliability.datasets import DatasetLoadError, load_queries, load_tools
from mcp_route_reliability.models import QueryCase, ToolRecord
from mcp_route_reliability.validation import (
    FIXTURE_PROFILE,
    DatasetValidationError,
    validate_dataset,
)


def test_tiny_dataset_matches_fixture_profile(
    tiny_tools: list[ToolRecord], tiny_queries: list[QueryCase]
) -> None:
    summary = validate_dataset(tiny_tools, tiny_queries, FIXTURE_PROFILE)

    assert summary.tool_count == 10
    assert summary.base_query_count == 20
    assert summary.variant_count == 0
    assert summary.split_base_counts == {
        "development": 8,
        "calibration": 6,
        "test": 6,
    }


def test_duplicate_tool_id_is_rejected(
    tiny_tools: list[ToolRecord], tiny_queries: list[QueryCase]
) -> None:
    duplicate = tiny_tools[0].model_copy(update={"name": "duplicate"})
    with pytest.raises(DatasetValidationError, match="duplicate tool_id"):
        validate_dataset([*tiny_tools, duplicate], tiny_queries, FIXTURE_PROFILE)


def test_normalized_duplicate_query_text_is_rejected(
    tiny_tools: list[ToolRecord], tiny_queries: list[QueryCase]
) -> None:
    duplicate = tiny_queries[1].model_copy(
        update={"text": f"  {tiny_queries[0].text.upper()}  "}
    )
    modified = [tiny_queries[0], duplicate, *tiny_queries[2:]]
    with pytest.raises(DatasetValidationError, match="duplicate normalized query text"):
        validate_dataset(tiny_tools, modified, FIXTURE_PROFILE)


def test_unknown_relevant_tool_is_rejected(
    tiny_tools: list[ToolRecord], tiny_queries: list[QueryCase]
) -> None:
    invalid = tiny_queries[0].model_copy(update={"relevant_tool_ids": ("missing.tool",)})
    with pytest.raises(DatasetValidationError, match="unknown relevant tool"):
        validate_dataset(tiny_tools, [invalid, *tiny_queries[1:]], FIXTURE_PROFILE)


def test_no_match_with_relevant_tool_is_rejected(
    tiny_tools: list[ToolRecord], tiny_queries: list[QueryCase]
) -> None:
    original = tiny_queries[4]
    invalid = original.model_copy(update={"relevant_tool_ids": (tiny_tools[0].tool_id,)})
    modified = [invalid if case.query_id == original.query_id else case for case in tiny_queries]
    with pytest.raises(DatasetValidationError, match="no_match"):
        validate_dataset(tiny_tools, modified, FIXTURE_PROFILE)


def test_inconsistent_abstention_fields_are_rejected(
    tiny_tools: list[ToolRecord], tiny_queries: list[QueryCase]
) -> None:
    original = tiny_queries[0]
    invalid = original.model_copy(
        update={"should_abstain": True, "abstain_reason": "ambiguous"}
    )
    modified = [invalid if case.query_id == original.query_id else case for case in tiny_queries]
    with pytest.raises(DatasetValidationError, match="routeable"):
        validate_dataset(tiny_tools, modified, FIXTURE_PROFILE)


def test_base_query_group_cannot_cross_splits(
    tiny_tools: list[ToolRecord], tiny_queries: list[QueryCase]
) -> None:
    original = tiny_queries[-1]
    variant = original.model_copy(
        update={
            "query_id": "q_variant",
            "text": "A unique perturbed request",
            "base_query_id": tiny_queries[0].base_query_id,
            "perturbation_type": "noise",
            "split": "test",
        }
    )
    with pytest.raises(DatasetValidationError, match="crosses splits"):
        validate_dataset(tiny_tools, [*tiny_queries, variant], FIXTURE_PROFILE)


def test_checked_in_jsonl_fixtures_load_and_validate() -> None:
    fixture_dir = Path(__file__).parents[1] / "fixtures"
    tools = load_tools(fixture_dir / "tools.jsonl")
    queries = load_queries(fixture_dir / "queries.jsonl")

    summary = validate_dataset(tools, queries, FIXTURE_PROFILE)
    assert summary.total_query_records == 20


def test_loader_error_names_path_and_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('{"tool_id": "ok"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(DatasetLoadError) as error:
        load_tools(path)

    message = str(error.value)
    assert str(path) in message
    assert "line 1" in message or "line 2" in message
