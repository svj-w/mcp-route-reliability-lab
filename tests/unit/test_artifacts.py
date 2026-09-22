from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mcp_route_reliability.artifacts import RunManifest, RunWriter
from mcp_route_reliability.models import ScoreMetadata


def manifest(run_id: str) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        source_commit="abc123",
        dataset_version="v0.1",
        dataset_hash="sha256:dataset",
        tool_corpus_version="v0.1",
        tool_corpus_hash="sha256:tools",
        router_id="bm25",
        router_version="1.0.0",
        model_version="rank-bm25:0.2.2",
        score_metadata=ScoreMetadata(
            score_type="similarity",
            higher_is_better=True,
            comparable_across_queries=False,
        ),
        feature_version="1.0",
        prediction_importer_version="1.0",
        random_seed=20260921,
        python_version="3.11",
        dependency_versions={"pydantic": "2.x"},
        started_at="2026-09-22T00:00:00Z",
        commands=("mrrl evaluate --config config.yaml",),
    )


def test_incomplete_temporary_run_is_never_exposed_as_final(tmp_path: Path) -> None:
    final = tmp_path / "run-incomplete"
    with RunWriter(tmp_path, "run-incomplete") as writer:
        writer.write_manifest(manifest("run-incomplete"))
        writer.write_json("metrics.json", {"accuracy": 1.0})
        with pytest.raises(ValueError, match="missing required artifacts"):
            writer.finalize()
        assert not final.exists()
    assert not final.exists()
    assert not list(tmp_path.glob(".tmp-run-incomplete-*"))


def test_complete_run_is_atomic_and_checksum_bound(tmp_path: Path) -> None:
    with RunWriter(tmp_path, "run-complete") as writer:
        assert not (tmp_path / "run-complete").exists()
        writer.write_manifest(manifest("run-complete"))
        writer.write_jsonl("predictions.jsonl", [{"query_id": "q1"}])
        writer.write_features_parquet([{"query_id": "q1", "top1_score": 0.8, "is_correct": True}])
        writer.write_json("metrics.json", {"accuracy": 1.0})
        writer.write_csv("errors.csv", [{"query_id": "q1", "decision": "accept"}])
        writer.write_json("policy.json", {"policy_id": "selected"})
        writer.write_json("policy-empirical.json", {"policy_id": "empirical"})
        writer.write_json("policy-conservative.json", {"policy_id": "conservative"})
        writer.write_text("reliability-card.md", "# Reliability Card\n")
        for name in (
            "reliability-diagram.png",
            "risk-coverage.png",
            "method-comparison.png",
            "robustness.png",
        ):
            writer.write_bytes(f"figures/{name}", b"png")
        final = writer.finalize()

    assert final == tmp_path / "run-complete"
    expected = {
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
    assert expected <= {path.name for path in final.iterdir()}
    assert len(list((final / "figures").glob("*.png"))) == 4
    saved_manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["status"] == "complete"
    assert set(saved_manifest["artifact_checksums"]) == {
        path.relative_to(final).as_posix()
        for path in final.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    frame = pd.read_parquet(final / "features.parquet")
    assert list(frame.columns) == ["query_id", "top1_score", "is_correct"]


def test_feature_parquet_rejects_raw_gold_columns(tmp_path: Path) -> None:
    with (
        RunWriter(tmp_path, "run-gold-leak") as writer,
        pytest.raises(ValueError, match="gold-label columns"),
    ):
        writer.write_features_parquet([{"query_id": "q1", "relevant_tool_ids": ["tool.a"]}])
