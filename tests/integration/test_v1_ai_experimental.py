from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from mcp_route_reliability.ai_dataset import build_ai_dataset
from mcp_route_reliability.datasets import load_queries
from mcp_route_reliability.validation import get_validation_profile

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "data" / "v1-candidate"
INPUTS = ROOT / "data" / "v1-ai-inputs"


def test_derived_snapshot_is_valid_and_not_human_reviewed(tmp_path: Path) -> None:
    output = tmp_path / "out"
    summary = build_ai_dataset(CANDIDATE, INPUTS, output)
    queries = load_queries(output / "queries.jsonl")
    assert summary.base_query_count == 300
    assert summary.variant_count == 90
    assert len({row.query_id for row in queries}) == 390
    card = (output / "DATASET_CARD.md").read_text(encoding="utf-8")
    assert "AI experimental" in card
    assert "not human-reviewed" in card
    assert (output / "raw" / "completed-secondary.csv").read_bytes() == (
        INPUTS / "raw" / "completed-secondary.csv"
    ).read_bytes()
    with pytest.raises(FileExistsError):
        build_ai_dataset(CANDIDATE, INPUTS, output)


def test_derived_snapshot_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_ai_dataset(CANDIDATE, INPUTS, first)
    build_ai_dataset(CANDIDATE, INPUTS, second)
    for name in ("tools.json", "queries.jsonl", "split-manifest.csv", "source-manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert (first / "annotations" / "final.jsonl").read_bytes() == (
        second / "annotations" / "final.jsonl"
    ).read_bytes()


def test_ai_profile_does_not_hardcode_disputed_case_counts() -> None:
    profile = get_validation_profile("ai-experimental")
    assert profile.case_type_base_counts is None
    assert profile.variant_count == 90


def test_rejects_tampered_raw_csv_before_publishing(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    shutil.copytree(INPUTS, inputs)
    source = inputs / "raw" / "completed-secondary.csv"
    source.write_bytes(source.read_bytes() + b"tampered")
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="SHA-256"):
        build_ai_dataset(CANDIDATE, inputs, output)
    assert not output.exists()


def test_rejects_missing_changed_label_decision(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    shutil.copytree(INPUTS, inputs)
    decisions = inputs / "decisions.jsonl"
    lines = [
        line for line in decisions.read_text(encoding="utf-8").splitlines()
        if '"query_id":"v01_q_039"' not in line
    ]
    decisions.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="requires final_gold decision"):
        build_ai_dataset(CANDIDATE, inputs, output)
    assert not output.exists()
