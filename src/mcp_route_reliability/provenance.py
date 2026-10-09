"""Local source identity and concrete, shell-quoted reproduction commands."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Literal


def resolve_source_commit(root: Path) -> str:
    """Identify this exact repository; never label uncommitted code as a clean SHA."""
    root = root.resolve()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()

    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
            return "not_available"
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain=v1", "--untracked-files=normal"))
    except (OSError, subprocess.SubprocessError):
        return "not_available"
    return f"{commit}:dirty" if dirty else commit


def reproduction_command(
    *,
    config_path: Path,
    tools_path: Path,
    queries_path: Path,
    profile: str,
    router: str,
    project_root: Path,
    shell: Literal["posix", "powershell"] | None = None,
) -> str:
    """Use the actual bound inputs, with a fresh run destination independent of this run."""
    def portable(path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(project_root.resolve()).as_posix()
        except ValueError:
            return resolved.as_posix()

    arguments = [
        "mrrl", "benchmark",
        "--config", portable(config_path),
        "--tools", portable(tools_path),
        "--queries", portable(queries_path),
        "--run-dir", f"runs/reproduce-{profile}-{router}",
    ]
    selected_shell = shell or ("powershell" if os.name == "nt" else "posix")
    if selected_shell == "posix":
        return shlex.join(arguments)

    def powershell_quote(value: str) -> str:
        if re.fullmatch(r"[A-Za-z0-9_./-]+", value):
            return value
        return "'" + value.replace("'", "''") + "'"

    return " ".join(powershell_quote(argument) for argument in arguments)
