from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mcp_route_reliability.cli import app

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
RUNNER = CliRunner()


def test_help_lists_public_workflow_commands() -> None:
    result = RUNNER.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "validate",
        "route",
        "import-predictions",
        "evaluate",
        "fit-policy",
        "apply-policy",
        "report",
        "benchmark",
    ):
        assert command in result.stdout


def test_validate_emits_machine_readable_success() -> None:
    result = RUNNER.invoke(
        app,
        [
            "validate",
            "--tools",
            str(FIXTURES / "tools.jsonl"),
            "--queries",
            str(FIXTURES / "queries.jsonl"),
            "--profile",
            "fixture",
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["summary"]["tool_count"] == 10
    assert payload["summary"]["base_query_count"] == 20


def test_missing_file_has_nonzero_exit_and_json_error(tmp_path: Path) -> None:
    result = RUNNER.invoke(
        app,
        [
            "validate",
            "--tools",
            str(tmp_path / "missing.json"),
            "--queries",
            str(FIXTURES / "queries.jsonl"),
            "--profile",
            "fixture",
            "--json",
        ],
    )
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error_type"] == "DatasetLoadError"
    assert "missing.json" in payload["message"]


def test_validation_failure_has_nonzero_exit_and_json_error() -> None:
    result = RUNNER.invoke(
        app,
        [
            "validate",
            "--tools",
            str(FIXTURES / "tools.jsonl"),
            "--queries",
            str(FIXTURES / "queries.jsonl"),
            "--profile",
            "v0.1",
            "--json",
        ],
    )
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error_type"] == "DatasetValidationError"


def test_route_rejects_invalid_config_with_json_error(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("not-json-compatible-yaml", encoding="utf-8")
    result = RUNNER.invoke(
        app,
        [
            "route",
            "--config",
            str(invalid),
            "--tools",
            str(FIXTURES / "tools.jsonl"),
            "--queries",
            str(FIXTURES / "queries.jsonl"),
            "--output",
            str(tmp_path / "predictions.jsonl"),
            "--json",
        ],
    )
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error_type"] == "ConfigError"
