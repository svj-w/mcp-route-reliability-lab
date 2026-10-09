from __future__ import annotations

import hashlib
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
        assert manifest["status"] == "complete"
        assert manifest["source_commit"] == "6319120606b19f34802d0c9cad4dbf65866921cd"
        assert len(manifest["artifact_checksums"]) == 12
        for relative, expected_checksum in manifest["artifact_checksums"].items():
            actual = "sha256:" + hashlib.sha256((report / relative).read_bytes()).hexdigest()
            assert actual == expected_checksum, relative
        command = manifest["commands"][0]
        assert f"--config configs/v1-ai-{router}.yaml" in command
        assert "--tools data/v1-ai-experimental/tools.json" in command
        assert "--queries data/v1-ai-experimental/queries.jsonl" in command
        for name in ("policy.json", "policy-empirical.json", "policy-conservative.json"):
            policy = json.loads((report / name).read_text(encoding="utf-8"))
            assert policy["source_commit"] == manifest["source_commit"]
            assert policy["available"] is False
            assert policy["threshold"] is None
            assert policy["target_risk"] == 0.05
        robustness = metrics["robustness"]
        assert robustness["status"] == "computed"
        assert robustness["decision_status"] == "policy_unavailable"
        assert set(robustness["by_perturbation"]) == {"paraphrase", "cross_language", "noise"}
        predictions = {
            row["query_id"]: row
            for row in map(json.loads, (report / "predictions.jsonl").read_text().splitlines())
        }
        by_id = {row.query_id: row for row in queries if row.split == "test"}
        variants = {key: row for key, row in by_id.items() if row.perturbation_type is not None}
        assert len(predictions) == 390
        deltas = robustness["paired_deltas"]
        assert len(deltas) == len(variants) == 90
        assert {row["variant_query_id"] for row in deltas} == set(variants)
        for delta in deltas:
            variant = variants[delta["variant_query_id"]]
            base = by_id[variant.base_query_id]
            top1 = predictions[variant.query_id]["candidates"][0]["tool_id"]
            base_top1 = predictions[base.query_id]["candidates"][0]["tool_id"]
            assert delta["base_query_id"] == base.query_id
            assert delta["perturbation_type"] == variant.perturbation_type
            assert delta["decision_flipped"] is None
            assert delta["top1_flipped"] == (top1 != base_top1)
            correct = not variant.should_abstain and top1 in variant.relevant_tool_ids
            base_correct = not base.should_abstain and base_top1 in base.relevant_tool_ids
            assert delta["accuracy_shift"] == int(correct) - int(base_correct)
        slices = {"overall": robustness["overall"], **robustness["by_perturbation"]}
        for condition, aggregate in slices.items():
            selected = deltas if condition == "overall" else [
                row for row in deltas if row["perturbation_type"] == condition
            ]
            assert aggregate["pair_count"] == len(selected) == (
                90 if condition == "overall" else 30
            )
            assert aggregate["decision_flip_rate"] is None
            for metric, field in (
                ("top1_flip_rate", "top1_flipped"),
                ("mean_accuracy_shift", "accuracy_shift"),
                ("mean_confidence_shift", "confidence_shift"),
            ):
                assert aggregate[metric] == pytest.approx(
                    sum(row[field] for row in selected) / len(selected)
                )
            expected_row = (
                f"| {condition} | {len(selected)} | {aggregate['mean_accuracy_shift']:.4f} | "
                f"{aggregate['top1_flip_rate']:.4f} | not_available | "
                f"{aggregate['mean_confidence_shift']:.4f} |"
            )
            assert expected_row in card
        assert "No perturbation variants" not in card
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
