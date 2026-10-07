"""Typer CLI for explicit, reproducible MCP route reliability workflows."""

# ruff: noqa: B008 -- Typer declares CLI options through evaluated defaults.

from __future__ import annotations

import json
import platform
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np
import typer
from pydantic import Field, ValidationError

from mcp_route_reliability.artifacts import RunManifest, RunWriter
from mcp_route_reliability.confidence.calibrators import (
    LogisticConfidenceModel,
    fit_logistic,
)
from mcp_route_reliability.confidence.features import (
    FEATURE_VERSION,
    FeatureNormalizer,
    extract_confidence_features,
)
from mcp_route_reliability.confidence.policies import (
    CalibrationRow,
    PolicyProvenance,
    build_policy_artifact,
    select_policy,
)
from mcp_route_reliability.confidence.reasons import ReasonHintConfig, infer_reason_hint
from mcp_route_reliability.datasets import load_queries, load_tools
from mcp_route_reliability.evaluation.bootstrap import MetricRow, grouped_bootstrap
from mcp_route_reliability.evaluation.calibration import (
    brier_score,
    equal_frequency_ece,
    error_auroc,
)
from mcp_route_reliability.evaluation.ranking import (
    is_routing_error,
    per_case_errors,
    ranking_metrics,
)
from mcp_route_reliability.evaluation.selective import risk_coverage_curve
from mcp_route_reliability.hashing import canonical_json_bytes, canonical_sha256
from mcp_route_reliability.importers.jsonl import import_predictions
from mcp_route_reliability.models import (
    ConfidenceFeatures,
    InferenceQuery,
    PolicyArtifact,
    PolicyKind,
    QueryCase,
    RouterPrediction,
    StrictModel,
    ToolRecord,
)
from mcp_route_reliability.reporting import (
    ErrorExportRow,
    EvaluationBundle,
    MethodReport,
    build_reliability_card,
    errors_csv_rows,
    render_figures,
)
from mcp_route_reliability.routers.base import Router
from mcp_route_reliability.routers.bm25 import BM25Router
from mcp_route_reliability.routers.embedding import (
    EmbeddingRouter,
    SentenceTransformerProvider,
)
from mcp_route_reliability.runtime import apply_policy
from mcp_route_reliability.validation import (
    get_validation_profile,
    validate_dataset,
)

app = typer.Typer(
    name="mrrl",
    help="Reliability evaluation and risk-controlled policies for MCP tool routing.",
    no_args_is_help=True,
)

ResultT = TypeVar("ResultT")


class ConfigError(ValueError):
    """A CLI configuration file is unreadable or violates its contract."""


class BenchmarkConfig(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    router: Literal["bm25", "embedding"]
    model_name: str | None = None
    model_revision: str = "main"
    profile: Literal["fixture", "v0.1", "production", "ai-experimental"]
    seed: int
    top_k: int = Field(ge=1)
    splits: dict[str, str]
    policy_method: Literal["logistic"]
    selected_policy_kind: PolicyKind
    target_risk: float = Field(ge=0, le=1)
    minimum_accepted: int = Field(ge=1)
    fixture_minimum_accepted: int = Field(ge=1)
    bootstrap_replicates: int = Field(ge=1000)
    fixture_bootstrap_replicates: int = Field(ge=1)
    report: dict[str, object]


def _emit(payload: object) -> None:
    typer.echo(canonical_json_bytes(payload).decode("utf-8"))


def _success(value: object, json_output: bool, message: str) -> None:
    if json_output:
        if isinstance(value, Mapping):
            _emit({"ok": True, **value})
        else:
            _emit({"ok": True, "result": value})
    else:
        typer.echo(message)


def _failure(error: Exception, json_output: bool) -> None:
    payload = {
        "ok": False,
        "error_type": type(error).__name__,
        "message": str(error),
    }
    if json_output:
        _emit(payload)
    else:
        typer.echo(f"error: {error}", err=True)
    raise typer.Exit(code=1)


def _execute(
    operation: Callable[[], ResultT],
    *,
    json_output: bool,
    message: Callable[[ResultT], str],
) -> None:
    try:
        result = operation()
    except Exception as error:
        _failure(error, json_output)
        return
    _success(result, json_output, message(result))


def _load_config(path: Path) -> BenchmarkConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as error:
        raise ConfigError(f"cannot read config {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"{path}: config must be JSON-compatible YAML: {error.msg}") from error
    try:
        return BenchmarkConfig.model_validate(raw)
    except ValidationError as error:
        raise ConfigError(f"invalid config {path}: {error}") from error


def _query_set_hash(queries: Sequence[QueryCase]) -> str:
    inference_queries = [
        query.to_inference_query() for query in sorted(queries, key=lambda row: row.query_id)
    ]
    return canonical_sha256(inference_queries)


def _route_all(
    config: BenchmarkConfig,
    tools: Sequence[ToolRecord],
    queries: Sequence[QueryCase],
) -> list[RouterPrediction]:
    query_hash = _query_set_hash(queries)
    router: Router
    if config.router == "bm25":
        router = BM25Router()
    else:
        if not config.model_name:
            raise ConfigError("embedding config requires model_name")
        router = EmbeddingRouter(
            SentenceTransformerProvider(
                config.model_name,
                revision=config.model_revision,
                device="cpu",
            )
        )
    return [
        router.rank(query.to_inference_query(), tools, config.top_k, query_hash)
        for query in queries
    ]


def _write_jsonl(path: Path, rows: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


@app.command("validate")
def validate_command(
    tools: Path = typer.Option(..., "--tools"),
    queries: Path = typer.Option(..., "--queries"),
    profile: str = typer.Option(..., "--profile"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Validate a tool corpus and query set against a named profile."""

    def operation() -> dict[str, object]:
        summary = validate_dataset(
            load_tools(tools), load_queries(queries), get_validation_profile(profile)
        )
        return {"summary": summary.model_dump(mode="json")}

    _execute(
        operation,
        json_output=json_output,
        message=lambda _result: "dataset is valid",
    )


@app.command("route")
def route_command(
    config: Path = typer.Option(..., "--config"),
    tools: Path = typer.Option(..., "--tools"),
    queries: Path = typer.Option(..., "--queries"),
    output: Path = typer.Option(..., "--output"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Route every query with the configured deterministic baseline."""

    def operation() -> dict[str, object]:
        settings = _load_config(config)
        tool_rows = load_tools(tools)
        query_rows = load_queries(queries)
        validate_dataset(tool_rows, query_rows, get_validation_profile(settings.profile))
        predictions = _route_all(settings, tool_rows, query_rows)
        _write_jsonl(output, predictions)
        return {"output": str(output), "prediction_count": len(predictions)}

    _execute(
        operation,
        json_output=json_output,
        message=lambda result: f"wrote {result['prediction_count']} predictions to {output}",
    )


@app.command("import-predictions")
def import_predictions_command(
    input_path: Path = typer.Option(..., "--input"),
    tools: Path = typer.Option(..., "--tools"),
    queries: Path = typer.Option(..., "--queries"),
    output: Path = typer.Option(..., "--output"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Validate and normalize external router predictions."""

    def operation() -> dict[str, object]:
        tool_rows = load_tools(tools)
        query_rows = load_queries(queries)
        predictions = import_predictions(
            input_path,
            expected_tool_corpus_hash=canonical_sha256(
                sorted(tool_rows, key=lambda row: row.tool_id)
            ),
            expected_query_set_hash=_query_set_hash(query_rows),
            expected_query_ids={row.query_id for row in query_rows},
            expected_tool_ids={row.tool_id for row in tool_rows},
        )
        _write_jsonl(output, predictions)
        return {"output": str(output), "prediction_count": len(predictions)}

    _execute(
        operation,
        json_output=json_output,
        message=lambda result: f"imported {result['prediction_count']} predictions",
    )


def _load_bound_predictions(
    path: Path, tools: Sequence[ToolRecord], queries: Sequence[QueryCase]
) -> list[RouterPrediction]:
    from mcp_route_reliability.hashing import tool_corpus_sha256

    return import_predictions(
        path,
        expected_tool_corpus_hash=tool_corpus_sha256(tools),
        expected_query_set_hash=_query_set_hash(queries),
        expected_query_ids={row.query_id for row in queries},
        expected_tool_ids={row.tool_id for row in tools},
    )


@app.command("evaluate")
def evaluate_command(
    predictions: Path = typer.Option(..., "--predictions"),
    tools: Path = typer.Option(..., "--tools"),
    queries: Path = typer.Option(..., "--queries"),
    run_dir: Path = typer.Option(..., "--run-dir"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Evaluate a complete bound prediction file with ranking metrics."""

    def operation() -> dict[str, object]:
        tool_rows = load_tools(tools)
        query_rows = load_queries(queries)
        prediction_rows = _load_bound_predictions(predictions, tool_rows, query_rows)
        metrics = ranking_metrics(
            query_rows,
            prediction_rows,
            known_tool_ids={tool.tool_id for tool in tool_rows},
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "metrics.json").write_bytes(canonical_json_bytes(metrics) + b"\n")
        return {"run_dir": str(run_dir), "metrics": metrics.model_dump(mode="json")}

    _execute(
        operation,
        json_output=json_output,
        message=lambda result: f"evaluation written to {result['run_dir']}",
    )


@app.command("fit-policy")
def fit_policy_command(
    run_dir: Path = typer.Option(..., "--run-dir"),
    method: str = typer.Option(..., "--method"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Inspect the frozen policy of a completed benchmark run."""

    def operation() -> dict[str, object]:
        normalized = method.replace("-", "_")
        artifact = PolicyArtifact.model_validate_json(
            (run_dir / "policy.json").read_text(encoding="utf-8")
        )
        if artifact.calibration_method != normalized:
            raise ValueError(f"run policy uses {artifact.calibration_method!r}, not {normalized!r}")
        return {"policy": artifact.model_dump(mode="json")}

    _execute(
        operation,
        json_output=json_output,
        message=lambda _result: "policy is frozen",
    )


def _read_single_prediction(path: Path) -> RouterPrediction:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError("prediction file is empty")
    first_line = text.splitlines()[0]
    return RouterPrediction.model_validate_json(first_line)


@app.command("apply-policy")
def apply_policy_command(
    policy: Path = typer.Option(..., "--policy"),
    prediction: Path = typer.Option(..., "--prediction"),
    query: Path = typer.Option(..., "--query"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Apply a policy to one query bundle containing query and feature records."""

    def operation() -> dict[str, object]:
        bundle = json.loads(query.read_text(encoding="utf-8-sig"))
        if not isinstance(bundle, dict) or "query" not in bundle or "features" not in bundle:
            raise ValueError("query file must contain 'query' and 'features' objects")
        decision = apply_policy(
            PolicyArtifact.model_validate_json(policy.read_text(encoding="utf-8-sig")),
            InferenceQuery.model_validate(bundle["query"]),
            _read_single_prediction(prediction),
            ConfidenceFeatures.model_validate(bundle["features"]),
        )
        return {"decision": decision.model_dump(mode="json")}

    _execute(
        operation,
        json_output=json_output,
        message=lambda _result: "policy applied",
    )


@app.command("report")
def report_command(
    run_dir: Path = typer.Option(..., "--run-dir"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Locate the immutable Reliability Card of a completed run."""

    def operation() -> dict[str, object]:
        card = run_dir / "reliability-card.md"
        manifest = RunManifest.model_validate_json(
            (run_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.status != "complete" or not card.is_file():
            raise ValueError("run is not complete")
        return {"report": str(card)}

    _execute(
        operation,
        json_output=json_output,
        message=lambda result: f"report: {result['report']}",
    )


def _case_maps(
    queries: Sequence[QueryCase], predictions: Sequence[RouterPrediction]
) -> tuple[dict[str, QueryCase], dict[str, RouterPrediction]]:
    return (
        {row.query_id: row for row in queries},
        {row.query_id: row for row in predictions},
    )


def _feature_rows(
    tools: Sequence[ToolRecord],
    queries: Sequence[QueryCase],
    predictions: Sequence[RouterPrediction],
    normalizer: FeatureNormalizer,
) -> list[ConfidenceFeatures]:
    tools_by_id = {tool.tool_id: tool for tool in tools}
    cases, predictions_by_id = _case_maps(queries, predictions)
    return [
        extract_confidence_features(
            cases[query_id].to_inference_query(),
            predictions_by_id[query_id],
            tools_by_id,
            normalizer,
        )
        for query_id in sorted(cases)
    ]


def _fixture_reason_config(sample_size: int) -> ReasonHintConfig:
    return ReasonHintConfig(
        calibration_sample_size=sample_size,
        query_token_upper_bound=3,
        lexical_coverage_upper_bound=0.15,
        standardized_top1_lower_bound=-1.0,
        margin_lower_bound=0.1,
        entropy_upper_bound=1.5,
    )


def _selected_test_metrics(
    errors: np.ndarray[Any, np.dtype[np.int64]],
    probabilities: np.ndarray[Any, np.dtype[np.float64]],
    artifact: PolicyArtifact,
) -> tuple[float, float, int]:
    if not artifact.available or artifact.threshold is None:
        return 0.0, 0.0, 0
    accepted = probabilities >= artifact.threshold
    accepted_count = int(np.sum(accepted))
    if accepted_count == 0:
        return 0.0, 0.0, 0
    risk = float(np.mean(errors[accepted]))
    return accepted_count / len(errors), risk, accepted_count


def _build_benchmark(
    settings: BenchmarkConfig,
    tools_path: Path,
    queries_path: Path,
    run_dir: Path,
) -> dict[str, object]:
    tools = load_tools(tools_path)
    queries = load_queries(queries_path)
    summary = validate_dataset(tools, queries, get_validation_profile(settings.profile))
    predictions = _route_all(settings, tools, queries)
    cases_by_id, predictions_by_id = _case_maps(queries, predictions)
    calibration_cases = sorted(
        (case for case in queries if case.split == "calibration"),
        key=lambda row: row.query_id,
    )
    test_cases = sorted(
        (case for case in queries if case.split == "test"), key=lambda row: row.query_id
    )
    calibration_predictions = [predictions_by_id[case.query_id] for case in calibration_cases]

    normalizer = FeatureNormalizer.fit(calibration_predictions)
    feature_rows = _feature_rows(tools, queries, predictions, normalizer)
    features_by_id = {row.query_id: row for row in feature_rows}
    calibration_features = [features_by_id[case.query_id] for case in calibration_cases]
    calibration_labels = [
        int(not is_routing_error(case, predictions_by_id[case.query_id]))
        for case in calibration_cases
    ]
    feature_names = (
        "top1_score_standardized",
        "top1_minus_top2",
        "query_length",
        "lexical_coverage",
    )
    fitted = fit_logistic(
        calibration_features,
        calibration_labels,
        feature_names=feature_names,
    )
    if not isinstance(fitted, LogisticConfidenceModel):
        raise ValueError(f"fixture logistic calibrator unavailable: {fitted.reason}")
    calibration_probability = fitted.predict_correctness(calibration_features)
    minimum_accepted = (
        settings.fixture_minimum_accepted
        if settings.profile == "fixture"
        else settings.minimum_accepted
    )
    selection = select_policy(
        [
            CalibrationRow(
                query_id=case.query_id,
                score=float(probability),
                is_error=not bool(label),
            )
            for case, probability, label in zip(
                calibration_cases,
                calibration_probability,
                calibration_labels,
                strict=True,
            )
        ],
        method="logistic",
        target_risk=settings.target_risk,
        minimum_accepted=minimum_accepted,
    )
    calibrator_payload = fitted.to_payload()
    reason_hint_config = _fixture_reason_config(len(calibration_cases))
    calibrator_payload["reason_hint_config"] = reason_hint_config.model_dump(mode="json")
    source_commit = "working-tree"
    provenance = PolicyProvenance(
        router_id=predictions[0].router_id,
        router_version=predictions[0].router_version,
        model_version=predictions[0].model_version,
        tool_corpus_hash=summary.tool_corpus_hash,
        calibration_dataset_hash=canonical_sha256(calibration_cases),
        feature_version=FEATURE_VERSION,
        score_metadata=predictions[0].score_metadata,
        source_commit=source_commit,
        calibrator_payload=calibrator_payload,
        feature_names=feature_names,
    )
    empirical_policy = build_policy_artifact(selection, provenance, policy_kind="empirical")
    conservative_policy = build_policy_artifact(selection, provenance, policy_kind="conservative")
    if settings.profile == "v0.1":
        empirical_policy = PolicyArtifact.model_validate(
            empirical_policy.model_copy(
                update={
                    "policy_id": f"exploratory-v0.1-{empirical_policy.policy_id}",
                    "available": False,
                    "unavailable_reason": "exploratory_v0_1",
                    "threshold": None,
                    "selected_accepted_count": 0,
                    "empirical_risk": None,
                    "risk_upper_bound": None,
                }
            ).model_dump()
        )
        conservative_policy = PolicyArtifact.model_validate(
            conservative_policy.model_copy(
                update={
                    "policy_id": f"exploratory-v0.1-{conservative_policy.policy_id}",
                    "available": False,
                    "unavailable_reason": "exploratory_v0_1",
                    "threshold": None,
                    "selected_accepted_count": 0,
                    "empirical_risk": None,
                    "risk_upper_bound": None,
                }
            ).model_dump()
        )
    if settings.profile == "fixture":
        empirical_policy = empirical_policy.model_copy(
            update={"policy_id": f"non-production-fixture-{empirical_policy.policy_id}"}
        )
        conservative_policy = conservative_policy.model_copy(
            update={"policy_id": f"non-production-fixture-{conservative_policy.policy_id}"}
        )
    selected_policy = (
        empirical_policy if settings.selected_policy_kind == "empirical" else conservative_policy
    )

    # Configuration is frozen here. No fitting or threshold selection occurs below.
    test_predictions = [predictions_by_id[case.query_id] for case in test_cases]
    test_features = [features_by_id[case.query_id] for case in test_cases]
    test_probability = fitted.predict_correctness(test_features)
    y_correct = np.asarray(
        [
            int(not is_routing_error(case, prediction))
            for case, prediction in zip(test_cases, test_predictions, strict=True)
        ],
        dtype=np.int64,
    )
    y_error = 1 - y_correct
    ece = equal_frequency_ece(y_correct, test_probability, bins=min(5, len(test_cases)))
    auroc = error_auroc(y_error, 1.0 - test_probability)
    selective = risk_coverage_curve(y_error, test_probability)
    coverage, selected_risk, accepted = _selected_test_metrics(
        y_error, test_probability, selected_policy
    )
    ranking = ranking_metrics(
        test_cases,
        test_predictions,
        known_tool_ids={tool.tool_id for tool in tools},
    )
    bootstrap_replicates = (
        settings.fixture_bootstrap_replicates
        if settings.profile == "fixture"
        else settings.bootstrap_replicates
    )
    interval = grouped_bootstrap(
        [
            MetricRow(
                row_id=case.query_id,
                base_query_id=case.base_query_id,
                value=float(error),
            )
            for case, error in zip(test_cases, y_error, strict=True)
        ],
        lambda rows: sum(row.value for row in rows) / len(rows),
        replicates=bootstrap_replicates,
        seed=settings.seed,
        production_mode=settings.profile != "fixture",
    )
    method_report = MethodReport(
        method="logistic",
        brier=brier_score(y_correct, test_probability),
        ece=ece.ece,
        error_auroc=auroc.value,
        aurc=selective.aurc,
        coverage=coverage,
        selective_risk=selected_risk,
        accepted=accepted,
    )
    errors = per_case_errors(
        test_cases,
        test_predictions,
        known_tool_ids={tool.tool_id for tool in tools},
    )
    error_exports = tuple(
        ErrorExportRow(
            query_id=row.query_id,
            query_text=cases_by_id[row.query_id].text,
            gold_tool_ids=row.relevant_tool_ids,
            top_k_tool_ids=row.top_k_tool_ids,
            decision=(
                "accept"
                if selected_policy.available
                and selected_policy.threshold is not None
                and test_probability[index] >= selected_policy.threshold
                else "abstain"
            ),
            confidence=float(test_probability[index]),
            reason_hint=(
                None
                if selected_policy.available
                and selected_policy.threshold is not None
                and test_probability[index] >= selected_policy.threshold
                else infer_reason_hint(
                    test_cases[index].to_inference_query(),
                    test_predictions[index],
                    test_features[index],
                    reason_hint_config,
                )
            ),
            language=cases_by_id[row.query_id].language,
            case_type=cases_by_id[row.query_id].case_type,
        )
        for index, row in enumerate(errors)
        if row.is_routing_error
    )
    if settings.profile == "fixture":
        profile_limitation = (
            "The fixture uses minimum accepted N=3 and is explicitly non-production."
        )
    elif settings.profile == "v0.1":
        profile_limitation = (
            "V0.1 contains 100 author-created, class-balanced queries and is explicitly "
            "non-production."
        )
    elif settings.profile == "ai-experimental":
        profile_limitation = (
            "AI-annotated synthetic experiment with post-hoc revisions and "
            "generator-provisional labels; not independent human review or a safety guarantee."
        )
    else:
        profile_limitation = (
            "The production-size profile remains a benchmark artifact, not a safety "
            "certification."
        )
    bundle = EvaluationBundle(
        dataset_version=settings.profile,
        dataset_hash=canonical_sha256(queries),
        tool_corpus_version=settings.profile,
        tool_corpus_hash=summary.tool_corpus_hash,
        router_id=predictions[0].router_id,
        router_version=predictions[0].router_version,
        model_version=predictions[0].model_version,
        score_metadata=predictions[0].score_metadata,
        feature_version=FEATURE_VERSION,
        source_commit=source_commit,
        environment={
            "platform": platform.system(),
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        },
        dataset_composition=summary.case_type_base_counts,
        calibration_sample_size=len(calibration_cases),
        test_sample_size=len(test_cases),
        methods=(method_report,),
        reliability_bins=ece.bins,
        risk_coverage_points=selective.points,
        empirical_policy=empirical_policy,
        conservative_policy=conservative_policy,
        selected_policy_kind=settings.selected_policy_kind,
        bootstrap_intervals={"routing_error": interval},
        robustness_slices=(),
        errors=error_exports,
        limitations=(
            profile_limitation,
            "Synthetic class balance may differ from production traffic.",
        ),
        commands=(
            f"mrrl benchmark --config configs/{settings.router}.yaml --tools TOOLS "
            "--queries QUERIES --run-dir RUN",
        ),
    )
    card = build_reliability_card(bundle)
    lifecycle = {
        "calibration_only_fit": True,
        "configuration_frozen": True,
        "frozen_before_test": True,
        "test_evaluations": 1,
        "fit_query_ids": [case.query_id for case in calibration_cases],
        "test_query_ids": [case.query_id for case in test_cases],
    }
    metrics_payload = {
        "ranking": ranking.model_dump(mode="json"),
        "calibration": {
            "brier": method_report.brier,
            "ece": ece.model_dump(mode="json"),
            "error_auroc": auroc.model_dump(mode="json"),
        },
        "selective": selective.model_dump(mode="json"),
        "bootstrap": interval.model_dump(mode="json"),
        "lifecycle": lifecycle,
    }
    all_errors = {
        row.query_id: row
        for row in per_case_errors(
            queries,
            predictions,
            known_tool_ids={tool.tool_id for tool in tools},
        )
    }
    parquet_rows = [
        {
            **feature.model_dump(mode="json"),
            "is_correct": all_errors[feature.query_id].is_correct,
        }
        for feature in feature_rows
    ]
    manifest = RunManifest(
        run_id=run_dir.name,
        source_commit=source_commit,
        dataset_version=settings.profile,
        dataset_hash=canonical_sha256(queries),
        tool_corpus_version=settings.profile,
        tool_corpus_hash=summary.tool_corpus_hash,
        router_id=predictions[0].router_id,
        router_version=predictions[0].router_version,
        model_version=predictions[0].model_version,
        score_metadata=predictions[0].score_metadata,
        feature_version=FEATURE_VERSION,
        prediction_importer_version="1.0",
        random_seed=settings.seed,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        dependency_versions={
            "numpy": version("numpy"),
            "pydantic": version("pydantic"),
            "rank-bm25": version("rank-bm25"),
            "scikit-learn": version("scikit-learn"),
        },
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        commands=bundle.commands,
    )
    with RunWriter(run_dir.parent, run_dir.name) as writer:
        writer.write_manifest(manifest)
        writer.write_jsonl("predictions.jsonl", predictions)
        writer.write_features_parquet(parquet_rows)
        writer.write_json("metrics.json", metrics_payload)
        writer.write_csv("errors.csv", errors_csv_rows(error_exports))
        writer.write_json("policy.json", selected_policy)
        writer.write_json("policy-empirical.json", empirical_policy)
        writer.write_json("policy-conservative.json", conservative_policy)
        writer.write_text("reliability-card.md", card)
        with tempfile.TemporaryDirectory() as figure_directory:
            for figure in render_figures(bundle, Path(figure_directory)):
                writer.write_bytes(f"figures/{figure.name}", figure.read_bytes())
        final_path = writer.finalize()
    return {
        "run_dir": str(final_path),
        "policy_available": selected_policy.available,
        "configuration_frozen": True,
    }


@app.command("benchmark")
def benchmark_command(
    config: Path = typer.Option(..., "--config"),
    tools: Path = typer.Option(..., "--tools"),
    queries: Path = typer.Option(..., "--queries"),
    run_dir: Path = typer.Option(..., "--run-dir"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Run validation, routing, calibration, frozen Test evaluation, and reporting."""

    _execute(
        lambda: _build_benchmark(_load_config(config), tools, queries, run_dir),
        json_output=json_output,
        message=lambda result: f"benchmark complete: {result['run_dir']}",
    )


if __name__ == "__main__":
    app()
