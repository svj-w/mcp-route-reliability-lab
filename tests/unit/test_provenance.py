from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mcp_route_reliability.provenance import reproduction_command, resolve_source_commit


@pytest.mark.skipif(shutil.which("git") is None, reason="Git unavailable")
def test_source_commit_distinguishes_clean_dirty_and_absent_repository(tmp_path: Path) -> None:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    assert resolve_source_commit(tmp_path) == "not_available"
    git("init", "--initial-branch=main")
    git("config", "user.name", "Test contributor")
    git("config", "user.email", "test@example.invalid")
    (tmp_path / "code.py").write_text("value = 1\n", encoding="utf-8")
    git("add", "code.py")
    git("commit", "-m", "test source")
    commit = git("rev-parse", "HEAD")
    assert resolve_source_commit(tmp_path) == commit
    (tmp_path / "code.py").write_text("value = 2\n", encoding="utf-8")
    assert resolve_source_commit(tmp_path) == f"{commit}:dirty"


def test_reproduction_command_preserves_actual_paths_with_spaces(tmp_path: Path) -> None:
    command = reproduction_command(
        config_path=tmp_path / "config folder" / "v1-ai-bm25.yaml",
        tools_path=tmp_path / "data" / "tools.json",
        queries_path=tmp_path / "data" / "queries.jsonl",
        profile="ai-experimental", router="bm25", project_root=tmp_path,
    )
    args = shlex.split(command)
    assert args[args.index("--config") + 1] == "config folder/v1-ai-bm25.yaml"
    assert args[args.index("--tools") + 1] == "data/tools.json"
    assert args[args.index("--queries") + 1] == "data/queries.jsonl"
    assert args[args.index("--run-dir") + 1] == "runs/reproduce-ai-experimental-bm25"


def test_posix_reproduction_command_quotes_apostrophes_and_metacharacters(tmp_path: Path) -> None:
    config = tmp_path / "config O'Brien $cash `literal" / "bm25.yaml"
    command = reproduction_command(
        config_path=config, tools_path=tmp_path / "tools.json",
        queries_path=tmp_path / "queries.jsonl", profile="ai-experimental",
        router="bm25", project_root=tmp_path, shell="posix",
    )
    args = shlex.split(command)
    assert args[args.index("--config") + 1] == config.relative_to(tmp_path).as_posix()


@pytest.mark.skipif(os.name != "nt", reason="PowerShell-native argument round-trip")
def test_powershell_reproduction_command_preserves_literal_native_arguments(tmp_path: Path) -> None:
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell unavailable")
    config = tmp_path / "config O'Brien $cash `literal" / "bm25.yaml"
    command = reproduction_command(
        config_path=config, tools_path=tmp_path / "tools.json",
        queries_path=tmp_path / "queries.jsonl", profile="ai-experimental",
        router="bm25", project_root=tmp_path,
    )
    probe = tmp_path / "argv_probe.py"
    probe.write_text("import json, sys\nprint(json.dumps(sys.argv[1:]))\n", encoding="utf-8")

    def quote(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    script = (
        f"& {quote(sys.executable)} {quote(str(probe))} "
        + command.removeprefix("mrrl benchmark ")
    )
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
        check=True, capture_output=True, text=True, timeout=30,
    )
    args = json.loads(result.stdout)
    assert args[args.index("--config") + 1] == config.relative_to(tmp_path).as_posix()
    assert args[args.index("--tools") + 1] == "tools.json"
