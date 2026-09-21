"""Canonical serialization and content hashes used by all artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel

from mcp_route_reliability.models import ToolRecord


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON-compatible value without platform-dependent whitespace."""

    text = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return text.encode("utf-8")


def canonical_sha256(value: object) -> str:
    digest = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return f"sha256:{digest}"


def tool_corpus_sha256(tools: Sequence[ToolRecord]) -> str:
    """Hash a corpus by tool identity rather than caller-provided list order."""

    ordered = sorted(tools, key=lambda tool: tool.tool_id)
    return canonical_sha256(ordered)
