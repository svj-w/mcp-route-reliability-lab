"""Strict JSON and JSONL loaders for versioned dataset records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from mcp_route_reliability.models import QueryCase, ToolRecord

ModelT = TypeVar("ModelT", bound=BaseModel)


class DatasetLoadError(ValueError):
    """A dataset file could not be parsed into its declared record contract."""


def _validate_record(raw: object, model: type[ModelT], path: Path, location: str) -> ModelT:
    try:
        return model.model_validate(raw)
    except ValidationError as error:
        raise DatasetLoadError(f"{path} {location}: {error}") from error


def _load_jsonl(path: Path, model: type[ModelT]) -> list[ModelT]:
    records: list[ModelT] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as error:
        raise DatasetLoadError(f"cannot read {path}: {error}") from error

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise DatasetLoadError(f"{path} line {line_number}: {error.msg}") from error
        records.append(_validate_record(raw, model, path, f"line {line_number}"))
    return records


def _load_json_array(path: Path, model: type[ModelT]) -> list[ModelT]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as error:
        raise DatasetLoadError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise DatasetLoadError(f"{path} line {error.lineno}: {error.msg}") from error

    if not isinstance(raw, list):
        raise DatasetLoadError(f"{path}: expected a top-level JSON array")
    return [
        _validate_record(item, model, path, f"item {index}")
        for index, item in enumerate(raw, start=1)
    ]


def _load_records(path: Path, model: type[ModelT]) -> list[ModelT]:
    if path.suffix.casefold() in {".jsonl", ".ndjson"}:
        return _load_jsonl(path, model)
    if path.suffix.casefold() == ".json":
        return _load_json_array(path, model)
    raise DatasetLoadError(f"{path}: expected .json, .jsonl, or .ndjson")


def load_tools(path: Path) -> list[ToolRecord]:
    return _load_records(path, ToolRecord)


def load_queries(path: Path) -> list[QueryCase]:
    return _load_records(path, QueryCase)
