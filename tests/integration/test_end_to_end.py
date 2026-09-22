from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from mcp_route_reliability.cli import app
from mcp_route_reliability.datasets import load_queries
from mcp_route_reliability.models import (
    ConfidenceFeatures,
    PolicyArtifact,
    RouterPrediction,
)
from mcp_route_reliability.runtime import PolicyUnavailableError, apply_policy

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
RUNNER = CliRunner()


def run_benchmark(run_dir: Path) -> dict[str, object]:
    result = RUNNER.invoke(
        app,
        [
            "benchmark",
            "--config",
            str(ROOT / "configs" / "bm25.yaml"),
            "--tools",
            str(FIXTURES / "tools.jsonl"),
            "--queries",
            str(FIXTURES / "queries.jsonl"),
            "--run-dir",
            str(run_dir),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_cpu_fixture_benchmark_is_frozen_bound_and_reproducible(tmp_path: Path) -> None:
    first_dir = tmp_path / "run-one"
    second_dir = tmp_path / "run-two"
    first_result = run_benchmark(first_dir)
    second_result = run_benchmark(second_dir)
    assert first_result["ok"] is True
    assert second_result["ok"] is True

    expected_files = {
        "manifest.json",
        "predictions.jsonl",
        "features.parquet",
        "metrics.json",
        "errors.csv",
        "policy.json",
        "policy-empirical.json",
        "policy-conservative.json",
        "reliability-card.md",
    }
    assert expected_files <= {path.name for path in first_dir.iterdir()}
    assert len(list((first_dir / "figures").glob("*.png"))) == 4

    metrics = load_json(first_dir / "metrics.json")
    lifecycle = metrics["lifecycle"]
    assert lifecycle["calibration_only_fit"] is True
    assert lifecycle["configuration_frozen"] is True
    assert lifecycle["test_evaluations"] == 1
    assert set(lifecycle["fit_query_ids"]).isdisjoint(lifecycle["test_query_ids"])
    assert len(lifecycle["fit_query_ids"]) == 6
    assert len(lifecycle["test_query_ids"]) == 6

    manifest = load_json(first_dir / "manifest.json")
    policy = load_json(first_dir / "policy.json")
    assert manifest["status"] == "complete"
    assert policy["minimum_accepted"] == 3
    assert policy["calibration_sample_size"] == 6
    assert policy["policy_id"].startswith("non-production-fixture-")
    assert policy["available"] is False
    assert policy["unavailable_reason"] == "no_qualifying_threshold"
    for field in (
        "router_id",
        "router_version",
        "model_version",
        "tool_corpus_hash",
        "feature_version",
        "score_metadata",
    ):
        assert policy[field] == manifest[field]

    first_hashes = manifest["artifact_checksums"]
    second_hashes = load_json(second_dir / "manifest.json")["artifact_checksums"]
    assert first_hashes == second_hashes

    predictions = {
        row["query_id"]: RouterPrediction.model_validate(row)
        for row in load_jsonl(first_dir / "predictions.jsonl")
    }
    feature_rows = pd.read_parquet(first_dir / "features.parquet").to_dict("records")
    features = {
        row["query_id"]: ConfidenceFeatures.model_validate(
            {key: value for key, value in row.items() if key != "is_correct"}
        )
        for row in feature_rows
    }
    queries = {case.query_id: case for case in load_queries(FIXTURES / "queries.jsonl")}
    artifact = PolicyArtifact.model_validate(policy)
    test_id = lifecycle["test_query_ids"][0]
    with pytest.raises(PolicyUnavailableError, match="no_qualifying_threshold"):
        apply_policy(
            artifact,
            queries[test_id].to_inference_query(),
            predictions[test_id],
            features[test_id],
        )
