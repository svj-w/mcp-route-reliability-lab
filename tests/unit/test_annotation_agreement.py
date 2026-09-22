from __future__ import annotations

import pytest

from mcp_route_reliability.datasets import (
    build_disagreement_records,
    cohen_kappa,
    mean_set_jaccard,
)
from mcp_route_reliability.models import Annotation


def make_annotation(
    query_id: str,
    annotator_id: str,
    *,
    relevant: tuple[str, ...] = ("tool.a",),
    should_abstain: bool = False,
    reason: str | None = None,
) -> Annotation:
    return Annotation(
        annotation_id=f"{annotator_id}:{query_id}",
        annotator_id=annotator_id,
        query_id=query_id,
        relevant_tool_ids=relevant,
        should_abstain=should_abstain,
        abstain_reason=reason,
        label_rationale=f"Rationale from {annotator_id}",
        candidate_conflict_note=None,
        tool_corpus_hash="sha256:corpus",
    )


def test_kappa_perfect_non_degenerate_agreement_is_one() -> None:
    result = cohen_kappa(["route", "abstain"], ["route", "abstain"])
    assert result.available is True
    assert result.value == 1.0
    assert result.sample_size == 2


def test_kappa_known_example() -> None:
    result = cohen_kappa(
        ["route", "route", "abstain", "abstain"],
        ["route", "abstain", "abstain", "abstain"],
    )
    assert result.value == pytest.approx(0.5)


def test_kappa_constant_labels_are_explicitly_unavailable() -> None:
    result = cohen_kappa(["route", "route"], ["route", "route"])
    assert result.available is False
    assert result.value is None
    assert result.reason == "degenerate_labels"


def test_kappa_rejects_empty_or_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        cohen_kappa([], [])
    with pytest.raises(ValueError, match="same length"):
        cohen_kappa(["route"], ["route", "abstain"])


def test_set_jaccard_includes_per_sample_values() -> None:
    result = mean_set_jaccard(
        [frozenset(), frozenset({"a", "b"}), frozenset({"a"})],
        [frozenset(), frozenset({"b", "c"}), frozenset()],
    )
    assert result.per_sample == pytest.approx((1.0, 1 / 3, 0.0))
    assert result.mean == pytest.approx(4 / 9)


def test_disagreement_records_retain_both_raw_labels_and_adjudication() -> None:
    primary = make_annotation("q1", "primary")
    secondary = make_annotation(
        "q1",
        "secondary",
        relevant=(),
        should_abstain=True,
        reason="no_match",
    )
    adjudicated = make_annotation("q1", "adjudicator")

    rows = build_disagreement_records([primary], [secondary], [adjudicated])

    assert len(rows) == 1
    assert rows[0].primary == primary
    assert rows[0].secondary == secondary
    assert rows[0].adjudicated == adjudicated


def test_agreement_export_requires_independent_annotators() -> None:
    first = make_annotation("q1", "same")
    second = first.model_copy(update={"annotation_id": "same:q1:second"})
    adjudicated = make_annotation("q1", "adjudicator")

    with pytest.raises(ValueError, match="independent"):
        build_disagreement_records([first], [second], [adjudicated])


def test_tool_set_order_is_not_a_disagreement() -> None:
    primary = make_annotation("q1", "primary", relevant=("tool.a", "tool.b"))
    secondary = make_annotation("q1", "secondary", relevant=("tool.b", "tool.a"))
    adjudicated = make_annotation("q1", "adjudicator", relevant=("tool.a", "tool.b"))

    assert build_disagreement_records([primary], [secondary], [adjudicated]) == []
