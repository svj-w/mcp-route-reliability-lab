from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mcp_route_reliability.cli import BenchmarkConfig, app
from mcp_route_reliability.datasets import load_queries
from mcp_route_reliability.hashing import canonical_sha256

ROOT = Path(__file__).resolve().parents[2]


def test_ai_benchmark_configs_are_explicitly_experimental() -> None:
    for name in ("v1-ai-bm25.yaml", "v1-ai-embedding.yaml"):
        raw = json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))
        config = BenchmarkConfig.model_validate(raw)
        assert config.profile == "ai-experimental"
        assert config.seed == 20260924


def test_frozen_reports_bind_same_query_set_and_test_once() -> None:
    queries = load_queries(ROOT / "data" / "v1-ai-experimental" / "queries.jsonl")
    expected = canonical_sha256(queries)
    hashes: set[str] = set()
    for router in ("bm25", "embedding"):
        report = ROOT / "reports" / "v1-ai-experimental" / router
        manifest = json.loads((report / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((report / "metrics.json").read_text(encoding="utf-8"))
        card = (report / "reliability-card.md").read_text(encoding="utf-8")
        assert manifest["dataset_version"] == "ai-experimental"
        assert manifest["dataset_hash"] == expected
        assert manifest["python_version"] == "3.11"
        assert metrics["lifecycle"]["test_evaluations"] == 1
        assert metrics["ranking"]["sample_size"] == 190
        assert "not independent human review" in card
        hashes.add(manifest["tool_corpus_hash"])
    assert hashes == {"sha256:8983296fa7cc6343c94a11fe15307e87f3bf5c6f03bcd036e971b9e9e18de069"}


def test_live_v1_benchmark_reports_all_90_pairs_without_fabricated_decisions(
    tmp_path: Path,
) -> None:
    run = tmp_path / "v1-bm25"
    result = CliRunner().invoke(
        app,
        [
            "benchmark", "--config", str(ROOT / "configs/v1-ai-bm25.yaml"),
            "--tools", str(ROOT / "data/v1-ai-experimental/tools.json"),
            "--queries", str(ROOT / "data/v1-ai-experimental/queries.jsonl"),
            "--run-dir", str(run), "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    robustness = metrics["robustness"]
    assert metrics["ranking"]["sample_size"] == 190
    assert robustness["status"] == "computed"
    assert robustness["decision_status"] == "policy_unavailable"
    assert robustness["overall"]["pair_count"] == 90
    assert robustness["overall"]["decision_flip_rate"] is None
    assert set(robustness["by_perturbation"]) == {"paraphrase", "cross_language", "noise"}
    assert all(row["pair_count"] == 30 for row in robustness["by_perturbation"].values())
    assert len(robustness["paired_deltas"]) == 90
    assert all(row["decision_flipped"] is None for row in robustness["paired_deltas"])
    predictions = {
        row["query_id"]: row
        for row in map(json.loads, (run / "predictions.jsonl").read_text().splitlines())
    }
    variants = [
        row for row in load_queries(ROOT / "data/v1-ai-experimental/queries.jsonl")
        if row.split == "test" and row.perturbation_type is not None
    ]
    expected_flips = sum(
        predictions[row.query_id]["candidates"][0]["tool_id"]
        != predictions[row.base_query_id]["candidates"][0]["tool_id"]
        for row in variants
    )
    assert robustness["overall"]["top1_flip_rate"] == pytest.approx(expected_flips / 90)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert re.fullmatch(r"[0-9a-f]{40}(?::dirty)?", manifest["source_commit"])
    assert "configs/v1-ai-bm25.yaml" in manifest["commands"][0]
    card = (run / "reliability-card.md").read_text(encoding="utf-8")
    assert "| overall | 90 |" in card
    assert "No perturbation variants" not in card
