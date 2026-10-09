from __future__ import annotations

from pathlib import Path

import pytest

from mcp_route_reliability.evaluation.bootstrap import BootstrapInterval
from mcp_route_reliability.evaluation.calibration import ECEBin
from mcp_route_reliability.evaluation.selective import RiskCoveragePoint
from mcp_route_reliability.models import PolicyArtifact, ScoreMetadata
from mcp_route_reliability.reporting import (
    ErrorExportRow,
    EvaluationBundle,
    MethodReport,
    RobustnessSlice,
    build_reliability_card,
    errors_csv_rows,
    render_figures,
)

ROOT = Path(__file__).resolve().parents[2]
METADATA = ScoreMetadata(
    score_type="similarity",
    higher_is_better=True,
    comparable_across_queries=False,
)


def policy(kind: str, *, available: bool) -> PolicyArtifact:
    return PolicyArtifact(
        policy_id=f"policy-{kind}",
        policy_kind=kind,
        available=available,
        unavailable_reason=None if available else "no_qualifying_threshold",
        router_id="embedding",
        router_version="1.0.0",
        model_version="sentence-transformers:test-model",
        tool_corpus_hash="sha256:tools",
        calibration_dataset_hash="sha256:calibration",
        feature_version="1.0",
        score_metadata=METADATA,
        calibration_method="logistic",
        threshold=0.82 if available else None,
        target_risk=0.05,
        minimum_accepted=30,
        calibration_sample_size=100,
        selected_accepted_count=42 if available else 0,
        confidence_interval_method="one-sided Wilson 95%",
        source_commit="abc123",
        feature_names=(
            "top1_score_standardized",
            "top1_minus_top2",
            "query_length",
            "lexical_coverage",
        ),
        calibrator_payload={"method": "logistic"},
        empirical_risk=0.0476 if available else None,
        risk_upper_bound=0.1130 if available else None,
    )


def bundle() -> EvaluationBundle:
    return EvaluationBundle(
        dataset_version="v1.0",
        dataset_hash="sha256:dataset",
        tool_corpus_version="v1.0",
        tool_corpus_hash="sha256:tools",
        router_id="embedding",
        router_version="1.0.0",
        model_version="sentence-transformers:test-model",
        score_metadata=METADATA,
        feature_version="1.0",
        source_commit="abc123",
        environment={"python": "3.11", "platform": "test-os"},
        dataset_composition={"clear": 100, "hard_negative": 80, "clarify": 60, "no_match": 60},
        calibration_sample_size=100,
        test_sample_size=100,
        methods=(
            MethodReport(
                method="max_score",
                brier=0.2200,
                ece=0.1800,
                error_auroc=0.7100,
                aurc=0.1300,
                coverage=0.7000,
                selective_risk=0.0857,
                accepted=70,
            ),
            MethodReport(
                method="logistic",
                brier=0.1400,
                ece=0.0900,
                error_auroc=0.8300,
                aurc=0.0700,
                coverage=0.4200,
                selective_risk=0.0476,
                accepted=42,
            ),
        ),
        reliability_bins=(
            ECEBin(
                lower_bound=0.1,
                upper_bound=0.4,
                count=50,
                mean_confidence=0.3,
                accuracy=0.28,
                absolute_gap=0.02,
            ),
            ECEBin(
                lower_bound=0.4,
                upper_bound=0.9,
                count=50,
                mean_confidence=0.7,
                accuracy=0.76,
                absolute_gap=0.06,
            ),
        ),
        risk_coverage_points=(
            RiskCoveragePoint(accepted=0, coverage=0.0, risk=0.0, threshold=None),
            RiskCoveragePoint(accepted=42, coverage=0.42, risk=0.0476, threshold=0.82),
            RiskCoveragePoint(accepted=100, coverage=1.0, risk=0.16, threshold=0.1),
        ),
        empirical_policy=policy("empirical", available=True),
        conservative_policy=policy("conservative", available=False),
        selected_policy_kind="empirical",
        bootstrap_intervals={
            "selective_risk": BootstrapInterval(
                estimate=0.0476,
                lower=0.0200,
                upper=0.0910,
                confidence_level=0.95,
                replicates=1000,
                seed=20260921,
                group_key="base_query_id",
                group_count=100,
            )
        },
        robustness_slices=(
            RobustnessSlice(
                condition="paraphrase",
                pair_count=50,
                accuracy_shift=-0.02,
                top1_flip_rate=0.08,
                decision_flip_rate=0.05,
                confidence_shift=-0.03,
            ),
            RobustnessSlice(
                condition="cross_language",
                pair_count=50,
                accuracy_shift=-0.07,
                top1_flip_rate=0.18,
                decision_flip_rate=0.12,
                confidence_shift=-0.09,
            ),
        ),
        errors=(
            ErrorExportRow(
                query_id="q-007",
                query_text="search private code",
                gold_tool_ids=("github.search_code",),
                top_k_tool_ids=("web.search", "github.search_code"),
                decision="accept",
                confidence=0.91,
                reason_hint=None,
                language="en",
                case_type="hard_negative",
            ),
        ),
        limitations=(
            "Synthetic class balance may differ from production traffic.",
            "The threshold is invalid after router, model, corpus, or score-semantics changes.",
        ),
        commands=(
            "mrrl evaluate --config configs/v1.yaml",
            "mrrl report --run runs/example",
        ),
    )


def test_reliability_card_matches_reviewed_golden() -> None:
    actual = build_reliability_card(bundle())
    expected = (ROOT / "tests" / "golden" / "reliability-card.md").read_text(encoding="utf-8")
    example = (ROOT / "docs" / "reliability-card-example.md").read_text(encoding="utf-8")
    assert actual == expected == example


def test_render_figures_writes_four_valid_png_files(tmp_path: Path) -> None:
    paths = render_figures(bundle(), tmp_path)
    assert {path.name for path in paths} == {
        "reliability-diagram.png",
        "risk-coverage.png",
        "method-comparison.png",
        "robustness.png",
    }
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in paths)


def test_empty_robustness_is_reported_as_unavailable(tmp_path: Path) -> None:
    empty = bundle().model_copy(update={"robustness_slices": ()})
    card = build_reliability_card(empty)
    assert "Paired robustness not computed" in card
    assert "No perturbation variants" not in card
    robustness = next(
        path for path in render_figures(empty, tmp_path) if path.name == "robustness.png"
    )
    assert robustness.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_unavailable_decision_metrics_are_explicit_in_card_and_figure(tmp_path: Path) -> None:
    original = bundle()
    unavailable = original.model_copy(
        update={
            "selected_policy_kind": "conservative",
            "robustness_slices": tuple(
                row.model_copy(update={"decision_flip_rate": None})
                for row in original.robustness_slices
            ),
        }
    )
    card = build_reliability_card(unavailable)
    assert "| paraphrase | 50 | -0.0200 | 0.0800 | not_available | -0.0300 |" in card
    assert "selected policy is unavailable" in card
    assert len(render_figures(unavailable, tmp_path)) == 4


def test_bundle_rejects_numeric_decision_flips_for_unavailable_policy() -> None:
    payload = bundle().model_dump()
    payload["selected_policy_kind"] = "conservative"
    with pytest.raises(ValueError, match="unavailable policies cannot report numeric"):
        EvaluationBundle.model_validate(payload)


def test_error_csv_rows_include_required_diagnostics() -> None:
    rows = errors_csv_rows(bundle().errors)
    assert rows == [
        {
            "query_id": "q-007",
            "query_text": "search private code",
            "gold_tool_ids": "github.search_code",
            "top_k_tool_ids": "web.search|github.search_code",
            "decision": "accept",
            "confidence": "0.910000",
            "reason_hint": "",
            "language": "en",
            "case_type": "hard_negative",
        }
    ]
