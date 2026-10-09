from __future__ import annotations

import pytest

from mcp_route_reliability.evaluation.robustness import (
    EvaluatedCase,
    paired_robustness,
)


def case(
    query_id: str,
    base_query_id: str,
    perturbation_type: str | None,
    top1: str,
    accepted: bool | None,
    confidence: float,
    correct: bool,
) -> EvaluatedCase:
    return EvaluatedCase(
        query_id=query_id,
        base_query_id=base_query_id,
        perturbation_type=perturbation_type,
        top1_tool_id=top1,
        accepted=accepted,
        confidence=confidence,
        correct=correct,
    )


def base_rows() -> list[EvaluatedCase]:
    return [
        case("b1", "b1", None, "tool.a", True, 0.9, True),
        case("b2", "b2", None, "tool.x", False, 0.4, False),
    ]


def variant_rows() -> list[EvaluatedCase]:
    return [
        case("b1-p", "b1", "paraphrase", "tool.b", False, 0.6, False),
        case("b2-p", "b2", "paraphrase", "tool.x", False, 0.5, True),
        case("b1-c", "b1", "cross_language", "tool.a", True, 0.8, True),
        case("b2-n", "b2", "noise", "tool.y", True, 0.2, False),
    ]


def test_paired_robustness_reports_overall_and_perturbation_slices() -> None:
    result = paired_robustness(base_rows(), variant_rows())

    assert result.overall.pair_count == 4
    assert result.overall.top1_flip_rate == 0.5
    assert result.overall.decision_flip_rate == 0.5
    assert result.overall.mean_confidence_shift == pytest.approx(-0.125)
    assert result.overall.mean_accuracy_shift == 0.0
    assert result.by_perturbation["paraphrase"].pair_count == 2
    assert result.by_perturbation["paraphrase"].top1_flip_rate == 0.5
    assert result.by_perturbation["cross_language"].mean_confidence_shift == pytest.approx(-0.1)
    assert result.by_perturbation["noise"].decision_flip_rate == 1.0
    assert len(result.paired_deltas) == 4
    assert result.paired_deltas[0].base_query_id == "b1"


def test_variant_with_missing_base_is_rejected() -> None:
    missing = case("missing-p", "missing", "paraphrase", "tool.a", True, 0.5, True)
    with pytest.raises(ValueError, match="missing base"):
        paired_robustness(base_rows(), [missing])


def test_duplicate_base_perturbation_pair_is_rejected() -> None:
    duplicate = case("b1-p2", "b1", "paraphrase", "tool.a", True, 0.7, True)
    with pytest.raises(ValueError, match="duplicate variant pair"):
        paired_robustness(base_rows(), [variant_rows()[0], duplicate])


def test_duplicate_or_malformed_base_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate base"):
        paired_robustness([base_rows()[0], base_rows()[0]], variant_rows())
    malformed = case("not-base", "b1", None, "tool.a", True, 0.9, True)
    with pytest.raises(ValueError, match="base record"):
        paired_robustness([malformed], [])


def test_unavailable_decisions_do_not_manufacture_zero_flip_rates() -> None:
    bases = [row.model_copy(update={"accepted": None}) for row in base_rows()]
    variants = [row.model_copy(update={"accepted": None}) for row in variant_rows()]
    result = paired_robustness(bases, variants)
    assert result.overall.top1_flip_rate == 0.5
    assert result.overall.decision_flip_rate is None
    assert all(row.decision_flipped is None for row in result.paired_deltas)
    assert all(row.decision_flip_rate is None for row in result.by_perturbation.values())


def test_mixed_decision_availability_is_rejected() -> None:
    variants = [row.model_copy(update={"accepted": None}) for row in variant_rows()]
    with pytest.raises(ValueError, match="mixed decision availability"):
        paired_robustness(base_rows(), variants)
