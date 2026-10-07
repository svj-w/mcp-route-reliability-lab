"""Cross-record validation for tool corpora, queries, and split profiles."""

from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

from mcp_route_reliability.hashing import tool_corpus_sha256
from mcp_route_reliability.models import (
    CaseType,
    Language,
    PerturbationType,
    QueryCase,
    Split,
    StrictModel,
    ToolRecord,
)

CountKey = TypeVar("CountKey", bound=str)


class DatasetValidationError(ValueError):
    """One or more cross-record dataset invariants failed."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("dataset validation failed:\n- " + "\n- ".join(self.errors))


@dataclass(frozen=True)
class ValidationProfile:
    name: str
    minimum_tools: int
    maximum_tools: int
    split_base_counts: Mapping[Split, int]
    variant_count: int
    allowed_case_types: frozenset[CaseType]
    language_base_counts: Mapping[Language, int] | None = None
    case_type_base_counts: Mapping[CaseType, int] | None = None
    perturbation_counts: Mapping[PerturbationType, int] | None = None


class DatasetSummary(StrictModel):
    schema_version: str = "1.0"
    profile_name: str
    tool_count: int
    total_query_records: int
    base_query_count: int
    variant_count: int
    tool_corpus_hash: str
    split_base_counts: dict[str, int]
    language_base_counts: dict[str, int]
    case_type_base_counts: dict[str, int]
    perturbation_counts: dict[str, int]


ALL_CASE_TYPES: frozenset[CaseType] = frozenset(
    {"clear", "hard_negative", "multi_valid", "clarify", "no_match"}
)

FIXTURE_PROFILE = ValidationProfile(
    name="fixture",
    minimum_tools=10,
    maximum_tools=10,
    split_base_counts={"development": 8, "calibration": 6, "test": 6},
    variant_count=0,
    allowed_case_types=ALL_CASE_TYPES,
    language_base_counts={"zh": 10, "en": 10},
    case_type_base_counts={
        "clear": 4,
        "hard_negative": 4,
        "multi_valid": 4,
        "clarify": 4,
        "no_match": 4,
    },
)

V0_1_PROFILE = ValidationProfile(
    name="v0.1",
    minimum_tools=20,
    maximum_tools=20,
    split_base_counts={"development": 40, "calibration": 30, "test": 30},
    variant_count=0,
    allowed_case_types=frozenset({"clear", "hard_negative", "clarify", "no_match"}),
    language_base_counts={"zh": 50, "en": 50},
)

PRODUCTION_PROFILE = ValidationProfile(
    name="production",
    minimum_tools=40,
    maximum_tools=60,
    split_base_counts={"development": 100, "calibration": 100, "test": 100},
    variant_count=90,
    allowed_case_types=ALL_CASE_TYPES,
    language_base_counts={"zh": 150, "en": 150},
    case_type_base_counts={
        "clear": 100,
        "hard_negative": 60,
        "multi_valid": 20,
        "clarify": 60,
        "no_match": 60,
    },
    perturbation_counts={"paraphrase": 30, "cross_language": 30, "noise": 30},
)

AI_EXPERIMENTAL_PROFILE = ValidationProfile(
    name="ai-experimental",
    minimum_tools=40,
    maximum_tools=40,
    split_base_counts={"development": 100, "calibration": 100, "test": 100},
    variant_count=90,
    allowed_case_types=ALL_CASE_TYPES,
    language_base_counts={"zh": 150, "en": 150},
    case_type_base_counts=None,
    perturbation_counts={"paraphrase": 30, "cross_language": 30, "noise": 30},
)

VALIDATION_PROFILES: dict[str, ValidationProfile] = {
    profile.name: profile
    for profile in (FIXTURE_PROFILE, V0_1_PROFILE, PRODUCTION_PROFILE, AI_EXPERIMENTAL_PROFILE)
}


def get_validation_profile(name: str) -> ValidationProfile:
    try:
        return VALIDATION_PROFILES[name]
    except KeyError as error:
        available = ", ".join(sorted(VALIDATION_PROFILES))
        message = f"unknown validation profile {name!r}; choose one of: {available}"
        raise ValueError(message) from error


def normalize_query_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def _append_count_errors(
    errors: list[str],
    label: str,
    actual: Mapping[CountKey, int],
    expected: Mapping[CountKey, int],
) -> None:
    normalized_actual = {key: actual.get(key, 0) for key in expected}
    if normalized_actual != dict(expected):
        errors.append(f"{label} counts {normalized_actual} do not match {dict(expected)}")


def _validate_local_semantics(case: QueryCase, errors: list[str]) -> None:
    if case.case_type == "no_match":
        if case.relevant_tool_ids:
            errors.append(f"{case.query_id}: no_match cannot contain relevant tools")
        if not case.should_abstain or case.abstain_reason != "no_match":
            errors.append(f"{case.query_id}: no_match must abstain with reason no_match")
    elif case.case_type == "clarify":
        if not case.should_abstain:
            errors.append(f"{case.query_id}: clarify must abstain")
        if case.abstain_reason not in {"ambiguous", "insufficient_information"}:
            errors.append(f"{case.query_id}: clarify has an invalid abstention reason")
    else:
        if case.should_abstain or case.abstain_reason is not None:
            errors.append(f"{case.query_id}: routeable cases cannot carry abstention labels")
        if not case.relevant_tool_ids:
            errors.append(f"{case.query_id}: routeable cases require a relevant tool")
    if case.case_type == "multi_valid" and len(case.relevant_tool_ids) < 2:
        errors.append(f"{case.query_id}: multi_valid requires at least two relevant tools")


def validate_dataset(
    tools: Sequence[ToolRecord],
    queries: Sequence[QueryCase],
    profile: ValidationProfile,
) -> DatasetSummary:
    errors: list[str] = []
    tool_ids = [tool.tool_id for tool in tools]
    duplicate_tool_ids = sorted(
        tool_id for tool_id, count in Counter(tool_ids).items() if count > 1
    )
    if duplicate_tool_ids:
        errors.append(f"duplicate tool_id values: {duplicate_tool_ids}")

    query_ids = [case.query_id for case in queries]
    duplicate_query_ids = sorted(
        query_id for query_id, count in Counter(query_ids).items() if count > 1
    )
    if duplicate_query_ids:
        errors.append(f"duplicate query_id values: {duplicate_query_ids}")

    normalized_texts: dict[str, list[str]] = defaultdict(list)
    for case in queries:
        normalized_texts[normalize_query_text(case.text)].append(case.query_id)
    for query_group in normalized_texts.values():
        if len(query_group) > 1:
            errors.append(f"duplicate normalized query text: {sorted(query_group)}")

    corpus_hash = tool_corpus_sha256(tools)
    known_tools = set(tool_ids)
    base_rows: list[QueryCase] = []
    variant_rows: list[QueryCase] = []
    splits_by_base: dict[str, set[Split]] = defaultdict(set)

    for case in queries:
        _validate_local_semantics(case, errors)
        unknown = sorted(set(case.relevant_tool_ids) - known_tools)
        if unknown:
            errors.append(f"{case.query_id}: unknown relevant tool IDs: {unknown}")
        if case.tool_corpus_hash != corpus_hash:
            errors.append(
                f"{case.query_id}: tool_corpus_hash {case.tool_corpus_hash!r} "
                f"does not match {corpus_hash!r}"
            )
        splits_by_base[case.base_query_id].add(case.split)

        if case.perturbation_type is None:
            base_rows.append(case)
            if case.query_id != case.base_query_id:
                errors.append(f"{case.query_id}: base record must use query_id as base_query_id")
        else:
            variant_rows.append(case)
            if case.query_id == case.base_query_id:
                errors.append(f"{case.query_id}: variant query_id must differ from base_query_id")

    for base_query_id, splits in splits_by_base.items():
        if len(splits) > 1:
            errors.append(f"base_query_id {base_query_id!r} crosses splits: {sorted(splits)}")

    base_ids = {case.query_id for case in base_rows}
    for variant in variant_rows:
        if variant.base_query_id not in base_ids:
            errors.append(
                f"{variant.query_id}: variant references missing base {variant.base_query_id!r}"
            )

    if not profile.minimum_tools <= len(tools) <= profile.maximum_tools:
        errors.append(
            f"tool count {len(tools)} is outside "
            f"[{profile.minimum_tools}, {profile.maximum_tools}]"
        )

    unexpected_case_types = sorted(
        {case.case_type for case in base_rows} - profile.allowed_case_types
    )
    if unexpected_case_types:
        errors.append(f"case types not allowed by {profile.name}: {unexpected_case_types}")

    split_counts = Counter(case.split for case in base_rows)
    language_counts = Counter(case.language for case in base_rows)
    case_type_counts = Counter(case.case_type for case in base_rows)
    perturbation_counts = Counter(
        case.perturbation_type for case in variant_rows if case.perturbation_type is not None
    )

    _append_count_errors(errors, "split", split_counts, profile.split_base_counts)
    if len(variant_rows) != profile.variant_count:
        errors.append(
            f"variant count {len(variant_rows)} does not match {profile.variant_count}"
        )
    if profile.language_base_counts is not None:
        _append_count_errors(errors, "language", language_counts, profile.language_base_counts)
    if profile.case_type_base_counts is not None:
        _append_count_errors(errors, "case type", case_type_counts, profile.case_type_base_counts)
    if profile.perturbation_counts is not None:
        _append_count_errors(
            errors, "perturbation", perturbation_counts, profile.perturbation_counts
        )

    if errors:
        raise DatasetValidationError(errors)

    return DatasetSummary(
        profile_name=profile.name,
        tool_count=len(tools),
        total_query_records=len(queries),
        base_query_count=len(base_rows),
        variant_count=len(variant_rows),
        tool_corpus_hash=corpus_hash,
        split_base_counts=dict(sorted(split_counts.items())),
        language_base_counts=dict(sorted(language_counts.items())),
        case_type_base_counts=dict(sorted(case_type_counts.items())),
        perturbation_counts=dict(sorted(perturbation_counts.items())),
    )
