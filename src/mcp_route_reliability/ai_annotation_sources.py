"""Read immutable AI annotation CSVs without changing their source text or labels."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from mcp_route_reliability.annotation_workflow import ASSIGNMENT_FIELDS
from mcp_route_reliability.hashing import tool_corpus_sha256
from mcp_route_reliability.models import Annotation, Language, ToolRecord


@dataclass(frozen=True)
class CompletedRow:
    query_id: str
    text: str
    language: Language
    annotation: Annotation


def load_completed_rows(
    path: Path,
    expected_ids: Collection[str],
    tools: Sequence[ToolRecord],
    expected_sha256: str,
) -> dict[str, CompletedRow]:
    """Validate a completed assignment and retain its exact wording and provenance."""

    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(f"{path}: SHA-256 mismatch: {actual_sha256}")

    corpus_hash = tool_corpus_sha256(tools)
    valid_tool_ids = {tool.tool_id for tool in tools}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(ASSIGNMENT_FIELDS):
            raise ValueError(f"{path}: unexpected CSV columns")
        source_rows = list(reader)

    result: dict[str, CompletedRow] = {}
    for line_number, source in enumerate(source_rows, start=2):
        if None in source or any(value is None for value in source.values()):
            raise ValueError(f"{path} line {line_number}: malformed CSV row")

        query_id = source["query_id"].strip()
        if not query_id:
            raise ValueError(f"{path} line {line_number}: empty query_id")
        if query_id in result:
            raise ValueError(f"{path}: duplicate query_id {query_id}")
        if source["tool_corpus_hash"].strip() != corpus_hash:
            raise ValueError(f"{path}: {query_id} has the wrong Tool Corpus hash")

        raw_tool_ids = source["relevant_tool_ids"].strip()
        tool_ids = tuple(part.strip() for part in raw_tool_ids.split("|")) if raw_tool_ids else ()
        if any(not part for part in tool_ids) or len(tool_ids) != len(set(tool_ids)):
            raise ValueError(f"{path}: {query_id} has malformed Tool IDs")
        unknown = sorted(set(tool_ids) - valid_tool_ids)
        if unknown:
            raise ValueError(f"{path}: {query_id} has unknown Tool IDs {unknown}")

        bool_value = source["should_abstain"].strip().lower()
        if bool_value not in {"true", "false"}:
            raise ValueError(f"{path}: {query_id} has invalid should_abstain")
        should_abstain = bool_value == "true"
        abstain_reason = source["abstain_reason"].strip() or None
        if should_abstain and abstain_reason != "no_match" and not tool_ids:
            raise ValueError(f"{path}: {query_id} needs a relevant Tool for clarification")
        if not should_abstain and not tool_ids:
            raise ValueError(f"{path}: {query_id} cannot route with an empty Tool set")

        language = source["language"].strip()
        if language not in {"zh", "en"}:
            raise ValueError(f"{path}: {query_id} has invalid language")
        text = source["text"]
        if not text.strip():
            raise ValueError(f"{path}: {query_id} has empty text")
        try:
            annotation = Annotation.model_validate(
                {
                    "annotation_id": f"{path.stem}-{query_id}",
                    "annotator_id": source["annotator_id"].strip(),
                    "query_id": query_id,
                    "relevant_tool_ids": tool_ids,
                    "should_abstain": should_abstain,
                    "abstain_reason": abstain_reason,
                    "label_rationale": source["label_rationale"].strip(),
                    "candidate_conflict_note": source["candidate_conflict_note"].strip() or None,
                    "tool_corpus_hash": corpus_hash,
                }
            )
        except ValidationError as error:
            raise ValueError(f"{path}: invalid annotation for {query_id}: {error}") from error
        result[query_id] = CompletedRow(
            query_id=query_id,
            text=text,
            language=cast(Language, language),
            annotation=annotation,
        )

    expected = set(expected_ids)
    if len(expected) != len(expected_ids):
        raise ValueError("expected query IDs contain duplicates")
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise ValueError(f"{path}: query IDs differ from assignment; missing={missing}, extra={extra}")
    return result
