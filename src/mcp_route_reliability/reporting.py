"""Reliability Card, diagnostic CSV, and headless figure rendering."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field, field_validator, model_validator

from mcp_route_reliability.evaluation.bootstrap import BootstrapInterval
from mcp_route_reliability.evaluation.calibration import ECEBin
from mcp_route_reliability.evaluation.selective import RiskCoveragePoint
from mcp_route_reliability.models import (
    CaseType,
    Decision,
    Language,
    PolicyArtifact,
    PolicyKind,
    ReasonHint,
    ScoreMetadata,
    StrictModel,
)


class MethodReport(StrictModel):
    method: str = Field(min_length=1)
    brier: float = Field(ge=0, le=1)
    ece: float = Field(ge=0, le=1)
    error_auroc: float | None = Field(default=None, ge=0, le=1)
    aurc: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    selective_risk: float = Field(ge=0, le=1)
    accepted: int = Field(ge=0)


class RobustnessSlice(StrictModel):
    condition: str = Field(min_length=1)
    accuracy_shift: float = Field(ge=-1, le=1)
    top1_flip_rate: float = Field(ge=0, le=1)
    decision_flip_rate: float = Field(ge=0, le=1)
    confidence_shift: float

    @field_validator("confidence_shift")
    @classmethod
    def validate_confidence_shift(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("confidence shift must be finite")
        return value


class ErrorExportRow(StrictModel):
    query_id: str = Field(min_length=1)
    query_text: str = Field(min_length=1)
    gold_tool_ids: tuple[str, ...]
    top_k_tool_ids: tuple[str, ...]
    decision: Decision
    confidence: float
    reason_hint: ReasonHint | None
    language: Language
    case_type: CaseType

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("error confidence must be finite")
        return value


class EvaluationBundle(StrictModel):
    dataset_version: str = Field(min_length=1)
    dataset_hash: str = Field(min_length=1)
    tool_corpus_version: str = Field(min_length=1)
    tool_corpus_hash: str = Field(min_length=1)
    router_id: str = Field(min_length=1)
    router_version: str = Field(min_length=1)
    model_version: str | None
    score_metadata: ScoreMetadata
    feature_version: str = Field(min_length=1)
    source_commit: str = Field(min_length=1)
    environment: dict[str, str]
    dataset_composition: dict[str, int]
    calibration_sample_size: int = Field(ge=1)
    test_sample_size: int = Field(ge=1)
    methods: tuple[MethodReport, ...]
    reliability_bins: tuple[ECEBin, ...]
    risk_coverage_points: tuple[RiskCoveragePoint, ...]
    empirical_policy: PolicyArtifact
    conservative_policy: PolicyArtifact
    selected_policy_kind: PolicyKind
    bootstrap_intervals: dict[str, BootstrapInterval]
    robustness_slices: tuple[RobustnessSlice, ...]
    errors: tuple[ErrorExportRow, ...]
    limitations: tuple[str, ...]
    commands: tuple[str, ...]

    @model_validator(mode="after")
    def validate_bundle(self) -> EvaluationBundle:
        if not self.methods:
            raise ValueError("reporting requires at least one method")
        if not self.reliability_bins:
            raise ValueError("reporting requires reliability bins")
        if not self.risk_coverage_points:
            raise ValueError("reporting requires risk-coverage points")
        if self.empirical_policy.policy_kind != "empirical":
            raise ValueError("empirical_policy has the wrong policy kind")
        if self.conservative_policy.policy_kind != "conservative":
            raise ValueError("conservative_policy has the wrong policy kind")
        return self


def _format_metric(value: float | None) -> str:
    return "not_available" if value is None else f"{value:.4f}"


def _policy_row(policy: PolicyArtifact) -> str:
    if policy.available:
        return (
            f"| {policy.policy_kind} | yes | {_format_metric(policy.threshold)} | "
            f"{policy.selected_accepted_count} | {_format_metric(policy.empirical_risk)} | "
            f"{_format_metric(policy.risk_upper_bound)} | — |"
        )
    return (
        f"| {policy.policy_kind} | no | not_available | 0 | not_available | "
        f"not_available | {policy.unavailable_reason} |"
    )


def build_reliability_card(bundle: EvaluationBundle) -> str:
    """Render a deterministic Markdown report without claiming safety certification."""

    lines = [
        "# MCP Route Reliability Card",
        "",
        "> This report is an evaluation artifact, not a safety certification.",
        "",
        "## Scope and provenance",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Dataset | {bundle.dataset_version} (`{bundle.dataset_hash}`) |",
        f"| Tool corpus | {bundle.tool_corpus_version} (`{bundle.tool_corpus_hash}`) |",
        f"| Router | {bundle.router_id} {bundle.router_version} |",
        f"| Model | {bundle.model_version or 'not_available'} |",
        f"| Score metadata | `{bundle.score_metadata.model_dump_json()}` |",
        f"| Feature version | {bundle.feature_version} |",
        f"| Source commit | `{bundle.source_commit}` |",
        (
            "| Calibration / frozen test N | "
            f"{bundle.calibration_sample_size} / {bundle.test_sample_size} |"
        ),
        "",
        "## Benchmark composition",
        "",
        "| Case type | Count |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {name} | {count} |" for name, count in sorted(bundle.dataset_composition.items())
    )
    lines.extend(
        [
            "",
            "## Environment",
            "",
            "| Component | Version |",
            "|---|---|",
        ]
    )
    lines.extend(f"| {name} | {version} |" for name, version in sorted(bundle.environment.items()))
    lines.extend(
        [
            "",
            "## Method comparison on frozen test",
            "",
            "| Method | Brier | ECE | Error AUROC | AURC | Coverage | "
            "Selective risk | Accepted N |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in bundle.methods:
        lines.append(
            f"| {method.method} | {_format_metric(method.brier)} | "
            f"{_format_metric(method.ece)} | {_format_metric(method.error_auroc)} | "
            f"{_format_metric(method.aurc)} | {_format_metric(method.coverage)} | "
            f"{_format_metric(method.selective_risk)} | {method.accepted} |"
        )
    lines.extend(
        [
            "",
            "## Risk-controlled policies",
            "",
            f"Configured selection: **{bundle.selected_policy_kind}**. "
            "Acceptance uses `confidence >= threshold`.",
            "",
            "| Policy | Available | Threshold | Calibration accepted N | Empirical risk | "
            "One-sided 95% risk UCB | Reason |",
            "|---|---|---:|---:|---:|---:|---|",
            _policy_row(bundle.empirical_policy),
            _policy_row(bundle.conservative_policy),
            "",
            "## Grouped bootstrap intervals",
            "",
            "Intervals resample complete `base_query_id` groups; variants are never "
            "independent units.",
            "",
            "| Metric | Estimate | 95% lower | 95% upper | Replicates | Seed | Groups |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, interval in sorted(bundle.bootstrap_intervals.items()):
        lines.append(
            f"| {name} | {_format_metric(interval.estimate)} | "
            f"{_format_metric(interval.lower)} | {_format_metric(interval.upper)} | "
            f"{interval.replicates} | {interval.seed} | {interval.group_count} |"
        )
    lines.extend(
        [
            "",
            "## Paired robustness",
            "",
            "| Condition | Accuracy shift | Top-1 flip | Decision flip | Confidence shift |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in bundle.robustness_slices:
        lines.append(
            f"| {item.condition} | {_format_metric(item.accuracy_shift)} | "
            f"{_format_metric(item.top1_flip_rate)} | "
            f"{_format_metric(item.decision_flip_rate)} | "
            f"{_format_metric(item.confidence_shift)} |"
        )
    if not bundle.robustness_slices:
        lines.append(
            "| No perturbation variants in this dataset | not_available | "
            "not_available | not_available | not_available |"
        )
    lines.extend(
        [
            "",
            "## Error diagnostics",
            "",
            f"Exported error rows: **{len(bundle.errors)}**. The CSV includes query text, "
            "gold set, Top-K, decision, confidence, reason hint, language, and case type.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in bundle.limitations)
    lines.extend(
        [
            "- Thresholds apply only to the bound router, model, tool corpus, feature "
            "version, score semantics, and a comparable query distribution.",
            "- Absence of an available conservative policy is reported directly; the "
            "target risk is never relaxed silently.",
            "- This report must not be interpreted as a safety certification.",
            "",
            "## Reproduction commands",
            "",
            "```console",
            *bundle.commands,
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def errors_csv_rows(errors: Sequence[ErrorExportRow]) -> list[dict[str, str]]:
    return [
        {
            "query_id": error.query_id,
            "query_text": error.query_text,
            "gold_tool_ids": "|".join(error.gold_tool_ids),
            "top_k_tool_ids": "|".join(error.top_k_tool_ids),
            "decision": error.decision,
            "confidence": f"{error.confidence:.6f}",
            "reason_hint": error.reason_hint or "",
            "language": error.language,
            "case_type": error.case_type,
        }
        for error in errors
    ]


def render_figures(bundle: EvaluationBundle, output_dir: Path) -> list[Path]:
    """Render four CI-safe PNG diagnostics with Matplotlib's Agg backend."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(6, 5))
    confidence = [item.mean_confidence for item in bundle.reliability_bins]
    accuracy = [item.accuracy for item in bundle.reliability_bins]
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="ideal")
    axis.plot(confidence, accuracy, marker="o", label="observed")
    axis.set(xlabel="Mean confidence", ylabel="Accuracy", title="Reliability diagram")
    axis.legend()
    path = output_dir / "reliability-diagram.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(
        [point.coverage for point in bundle.risk_coverage_points],
        [point.risk for point in bundle.risk_coverage_points],
        marker="o",
    )
    axis.set(xlabel="Coverage", ylabel="Selective risk", title="Risk-coverage")
    path = output_dir / "risk-coverage.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(7, 5))
    names = [method.method for method in bundle.methods]
    positions = list(range(len(names)))
    width = 0.25
    axis.bar(
        [value - width for value in positions],
        [item.brier for item in bundle.methods],
        width,
        label="Brier",
    )
    axis.bar(positions, [item.ece for item in bundle.methods], width, label="ECE")
    axis.bar(
        [value + width for value in positions],
        [item.aurc for item in bundle.methods],
        width,
        label="AURC",
    )
    axis.set_xticks(positions, names)
    axis.set(title="Method comparison", ylabel="Lower is better")
    axis.legend()
    path = output_dir / "method-comparison.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(7, 5))
    conditions = [item.condition for item in bundle.robustness_slices]
    positions = list(range(len(conditions)))
    width = 0.35
    if bundle.robustness_slices:
        axis.bar(
            [value - width / 2 for value in positions],
            [item.top1_flip_rate for item in bundle.robustness_slices],
            width,
            label="Top-1 flip",
        )
        axis.bar(
            [value + width / 2 for value in positions],
            [item.decision_flip_rate for item in bundle.robustness_slices],
            width,
            label="Decision flip",
        )
        axis.set_xticks(positions, conditions)
        axis.legend()
    else:
        axis.text(
            0.5,
            0.5,
            "No perturbation variants in this dataset",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_ylim(0.0, 1.0)
    axis.set(title="Paired robustness", ylabel="Rate")
    path = output_dir / "robustness.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path)

    return paths
