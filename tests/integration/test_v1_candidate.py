from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from mcp_route_reliability.datasets import load_annotations, load_queries, load_tools
from mcp_route_reliability.validation import get_validation_profile, validate_dataset

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "data" / "v1-candidate"
LABEL_FIELDS = (
    "relevant_tool_ids",
    "should_abstain",
    "abstain_reason",
    "label_rationale",
    "candidate_conflict_note",
    "annotator_id",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_ci_mypy_targets_the_runtime_python_matrix() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "python -m mypy --python-version ${{ matrix.python-version }} src" in workflow


def test_candidate_rebuild_matches_frozen_snapshot_with_lf_csv(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = scripts / "build_v1_candidate.py"
    shutil.copy2(ROOT / "scripts" / script.name, script)
    inputs = tmp_path / "data"
    inputs.mkdir()
    for name in ("tools.json", "queries.jsonl", "sources.csv"):
        shutil.copy2(ROOT / "data" / name, inputs / name)
    subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, check=True,
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    rebuilt = inputs / "v1-candidate"
    csv_paths = sorted(rebuilt.rglob("*.csv"))
    assert len(csv_paths) == 4
    bom_names = {"primary-recheck-assignment.csv", "secondary-assignment.csv"}
    for path in csv_paths:
        payload = path.read_bytes()
        assert b"\r\n" not in payload, f"CSV uses CRLF: {path.name}"
        assert payload.endswith(b"\n")
        assert payload.startswith(b"\xef\xbb\xbf") == (path.name in bom_names)
    frozen_paths = {
        path.relative_to(CANDIDATE) for path in CANDIDATE.rglob("*") if path.is_file()
    }
    rebuilt_paths = {
        path.relative_to(rebuilt) for path in rebuilt.rglob("*") if path.is_file()
    }
    assert rebuilt_paths == frozen_paths
    for relative in frozen_paths:
        expected = (CANDIDATE / relative).read_bytes().replace(b"\r\n", b"\n")
        assert (rebuilt / relative).read_bytes() == expected, str(relative)


def test_v1_candidate_is_complete_but_not_misrepresented_as_reviewed() -> None:
    tools = load_tools(CANDIDATE / "tools.json")
    queries = load_queries(CANDIDATE / "queries.jsonl")
    summary = json.loads((CANDIDATE / "candidate-summary.json").read_text(encoding="utf-8"))
    validation = validate_dataset(tools, queries, get_validation_profile("production"))

    assert validation.tool_count == 40
    assert validation.base_query_count == 300
    assert validation.variant_count == 90
    assert summary["tool_corpus_hash"] == validation.tool_corpus_hash
    assert summary["secondary_labels_present"] is False

    sources = read_csv(CANDIDATE / "sources.csv")
    assert len(sources) == 40
    assert {row["tool_id"] for row in sources} == {tool.tool_id for tool in tools}

    primary = load_annotations(CANDIDATE / "annotations" / "primary.jsonl")
    assert len(primary) == 390
    assert {annotation.query_id for annotation in primary} == {
        query.query_id for query in queries
    }
    assert {annotation.tool_corpus_hash for annotation in primary} == {
        validation.tool_corpus_hash
    }

    secondary = read_csv(CANDIDATE / "annotations" / "secondary-assignment.csv")
    manifest = read_csv(CANDIDATE / "annotations" / "secondary-sample-manifest.csv")
    assert len(secondary) == len(manifest) == 100
    assert {row["query_id"] for row in secondary} == {row["query_id"] for row in manifest}
    assert all(not row[field] for row in secondary for field in LABEL_FIELDS)

    recheck = read_csv(CANDIDATE / "annotations" / "primary-recheck-assignment.csv")
    expected_recheck_ids = {
        query.query_id
        for query in queries
        if query.query_id == query.base_query_id
        and query.case_type in {"clarify", "hard_negative", "no_match"}
    }
    assert len(recheck) == 180
    assert {row["query_id"] for row in recheck} == expected_recheck_ids
    assert all(not row[field] for row in recheck for field in LABEL_FIELDS)

    pending_outputs = (
        "secondary.jsonl",
        "agreement.json",
        "adjudicated.jsonl",
        "disagreements.csv",
    )
    assert all(not (CANDIDATE / "annotations" / name).exists() for name in pending_outputs)
