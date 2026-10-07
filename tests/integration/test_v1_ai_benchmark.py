from __future__ import annotations

import json
from pathlib import Path

from mcp_route_reliability.cli import BenchmarkConfig
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
