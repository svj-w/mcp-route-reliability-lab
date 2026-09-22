"""Strict JSON and JSONL loaders for versioned dataset records."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from mcp_route_reliability.models import (
    Annotation,
    DisagreementRecord,
    QueryCase,
    Split,
    StrictModel,
    ToolRecord,
)

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


def load_annotations(path: Path) -> list[Annotation]:
    return _load_records(path, Annotation)

class AgreementStatistic(StrictModel):
    value: float | None
    available: bool
    reason: str | None
    sample_size: int


class SetAgreementResult(StrictModel):
    mean: float
    per_sample: tuple[float, ...]
    sample_size: int


def assign_grouped_splits(
    cases: Sequence[QueryCase],
    seed: int,
    targets: Mapping[Split, int],
    tool_domains: Mapping[str, str] | None = None,
) -> dict[str, Split]:
    """Assign whole base-query groups while approximately preserving strata."""

    split_order: tuple[Split, ...] = ("development", "calibration", "test")
    if set(targets) != set(split_order):
        raise ValueError("split targets must contain development, calibration, and test")
    if any(targets[split] < 0 for split in split_order):
        raise ValueError("split target counts cannot be negative")

    groups: dict[str, list[QueryCase]] = defaultdict(list)
    for case in cases:
        groups[case.base_query_id].append(case)
    if sum(targets.values()) != len(groups):
        raise ValueError(
            f"split target total {sum(targets.values())} does not match {len(groups)} groups"
        )

    representatives: dict[str, QueryCase] = {}
    for base_query_id, group in groups.items():
        bases = [case for case in group if case.query_id == base_query_id]
        if len(bases) != 1:
            raise ValueError(
                f"base_query_id {base_query_id!r} requires exactly one base record"
            )
        representatives[base_query_id] = bases[0]

    by_stratum: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for base_query_id, case in representatives.items():
        domains = sorted(
            {
                tool_domains[tool_id]
                for tool_id in case.relevant_tool_ids
                if tool_domains is not None and tool_id in tool_domains
            }
        )
        domain = domains[0] if domains else "__none__"
        key = (case.case_type, case.language, case.difficulty, domain)
        by_stratum[key].append(base_query_id)

    rng = random.Random(seed)
    remaining = {split: targets[split] for split in split_order}
    assignments: dict[str, Split] = {}
    for stratum in sorted(by_stratum):
        group_ids = sorted(by_stratum[stratum])
        rng.shuffle(group_ids)
        for base_query_id in group_ids:
            candidates = [split for split in split_order if remaining[split] > 0]
            if not candidates:
                raise RuntimeError("split assignment exhausted capacity unexpectedly")
            selected = max(
                candidates,
                key=lambda split: (
                    remaining[split] / targets[split] if targets[split] else -1.0,
                    remaining[split],
                    -split_order.index(split),
                ),
            )
            assignments[base_query_id] = selected
            remaining[selected] -= 1

    if any(remaining.values()):
        raise RuntimeError(f"split assignment left unused capacity: {remaining}")
    return assignments


def cohen_kappa(first: Sequence[str], second: Sequence[str]) -> AgreementStatistic:
    if len(first) != len(second):
        raise ValueError("Kappa inputs must have the same length")
    if not first:
        raise ValueError("Kappa inputs must be nonempty")

    sample_size = len(first)
    observed = sum(left == right for left, right in zip(first, second, strict=True)) / sample_size
    first_counts = Counter(first)
    second_counts = Counter(second)
    labels = set(first_counts) | set(second_counts)
    expected = sum(
        (first_counts[label] / sample_size) * (second_counts[label] / sample_size)
        for label in labels
    )
    denominator = 1.0 - expected
    if abs(denominator) <= 1e-15:
        return AgreementStatistic(
            value=None,
            available=False,
            reason="degenerate_labels",
            sample_size=sample_size,
        )
    return AgreementStatistic(
        value=(observed - expected) / denominator,
        available=True,
        reason=None,
        sample_size=sample_size,
    )


def mean_set_jaccard(
    first: Sequence[frozenset[str]], second: Sequence[frozenset[str]]
) -> SetAgreementResult:
    if len(first) != len(second):
        raise ValueError("Jaccard inputs must have the same length")
    if not first:
        raise ValueError("Jaccard inputs must be nonempty")

    values: list[float] = []
    for left, right in zip(first, second, strict=True):
        union = left | right
        values.append(1.0 if not union else len(left & right) / len(union))
    return SetAgreementResult(
        mean=sum(values) / len(values),
        per_sample=tuple(values),
        sample_size=len(values),
    )


def _annotation_map(values: Sequence[Annotation], label: str) -> dict[str, Annotation]:
    result: dict[str, Annotation] = {}
    for annotation in values:
        if annotation.query_id in result:
            raise ValueError(f"{label} contains duplicate query_id {annotation.query_id!r}")
        result[annotation.query_id] = annotation
    return result


def build_disagreement_records(
    primary: Sequence[Annotation],
    secondary: Sequence[Annotation],
    adjudicated: Sequence[Annotation],
) -> list[DisagreementRecord]:
    primary_by_id = _annotation_map(primary, "primary annotations")
    secondary_by_id = _annotation_map(secondary, "secondary annotations")
    adjudicated_by_id = _annotation_map(adjudicated, "adjudicated annotations")
    if not (set(primary_by_id) == set(secondary_by_id) == set(adjudicated_by_id)):
        raise ValueError("annotation query ID sets must match exactly")

    rows: list[DisagreementRecord] = []
    for query_id in sorted(primary_by_id):
        first = primary_by_id[query_id]
        second = secondary_by_id[query_id]
        final = adjudicated_by_id[query_id]
        if first.annotator_id == second.annotator_id:
            raise ValueError(f"{query_id}: primary and secondary annotators must be independent")
        if len({first.tool_corpus_hash, second.tool_corpus_hash, final.tool_corpus_hash}) != 1:
            raise ValueError(f"{query_id}: annotation corpus hashes must match")
        first_label = (
            frozenset(first.relevant_tool_ids),
            first.should_abstain,
            first.abstain_reason,
        )
        second_label = (
            frozenset(second.relevant_tool_ids),
            second.should_abstain,
            second.abstain_reason,
        )
        if first_label != second_label:
            rows.append(
                DisagreementRecord(
                    query_id=query_id,
                    primary=first,
                    secondary=second,
                    adjudicated=final,
                )
            )
    return rows
