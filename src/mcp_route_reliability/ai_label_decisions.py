"""Explicit, auditable decisions for the AI experimental dataset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from mcp_route_reliability.ai_annotation_sources import CompletedRow
from mcp_route_reliability.hashing import tool_corpus_sha256
from mcp_route_reliability.models import (
    AbstainReason,
    Annotation,
    CaseType,
    QueryCase,
    StrictModel,
    ToolRecord,
)


class Decision(StrictModel):
    """One post-hoc decision, separate from the raw annotation that motivated it."""

    query_id: str = Field(min_length=1)
    target: Literal["final_gold", "secondary_posthoc"]
    text: str = Field(min_length=1)
    case_type: CaseType
    relevant_tool_ids: tuple[str, ...]
    should_abstain: bool
    abstain_reason: AbstainReason | None
    rationale: str = Field(min_length=1)
    candidate_conflict_note: str | None = None
    reviewer_type: Literal["ai_posthoc", "ai_source_import"]
    source_role: Literal["primary_recheck", "secondary_raw", "candidate_primary"]
    source_query_id: str = Field(min_length=1)


@dataclass(frozen=True)
class ResolvedBase:
    query: QueryCase
    final_annotation: Annotation
    provenance: str


def load_decisions(path: Path) -> tuple[Decision, ...]:
    """Parse a decision log and forbid two decisions for one target and query."""

    decisions: list[Decision] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            decision = Decision.model_validate_json(line)
        except ValidationError as error:
            raise ValueError(f"{path} line {line_number}: invalid decision: {error}") from error
        if decision.source_query_id != decision.query_id:
            raise ValueError(f"{path}: {decision.query_id} source_query_id must match query_id")
        decisions.append(decision)
    keys = [(decision.query_id, decision.target) for decision in decisions]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate decision target")
    return tuple(decisions)


def _label_key(annotation: Annotation) -> tuple[frozenset[str], bool, str | None]:
    return (
        frozenset(annotation.relevant_tool_ids),
        annotation.should_abstain,
        annotation.abstain_reason,
    )


def _query_from_decision(base: QueryCase, decision: Decision, corpus_hash: str) -> QueryCase:
    payload = base.model_dump(mode="json")
    payload.update(
        {
            "text": decision.text,
            "case_type": decision.case_type,
            "relevant_tool_ids": decision.relevant_tool_ids,
            "should_abstain": decision.should_abstain,
            "abstain_reason": decision.abstain_reason,
            "label_rationale": decision.rationale,
            "candidate_conflict_note": decision.candidate_conflict_note,
            "tool_corpus_hash": corpus_hash,
        }
    )
    return QueryCase.model_validate(payload)


def _annotation_from_query(query: QueryCase, annotator_id: str) -> Annotation:
    return Annotation(
        annotation_id=f"final-{query.query_id}",
        annotator_id=annotator_id,
        query_id=query.query_id,
        relevant_tool_ids=query.relevant_tool_ids,
        should_abstain=query.should_abstain,
        abstain_reason=query.abstain_reason,
        label_rationale=query.label_rationale,
        candidate_conflict_note=query.candidate_conflict_note,
        tool_corpus_hash=query.tool_corpus_hash,
    )


def resolve_bases(
    candidate_bases: Sequence[QueryCase],
    candidate_primary: Sequence[Annotation],
    primary_rows: Mapping[str, CompletedRow],
    decisions: Sequence[Decision],
    tools: Sequence[ToolRecord],
) -> tuple[ResolvedBase, ...]:
    """Apply the primary AI pass, requiring a decision for every source change."""

    corpus_hash = tool_corpus_sha256(tools)
    known_tools = {tool.tool_id for tool in tools}
    primary_by_id = {annotation.query_id: annotation for annotation in candidate_primary}
    if len(primary_by_id) != len(candidate_primary):
        raise ValueError("candidate primary annotations contain duplicate query IDs")
    base_ids = {base.query_id for base in candidate_bases}
    if len(base_ids) != len(candidate_bases):
        raise ValueError("candidate bases contain duplicate query IDs")
    if not base_ids <= set(primary_by_id):
        raise ValueError("candidate primary annotations omit base IDs")
    if not set(primary_rows) <= base_ids:
        raise ValueError("AI primary recheck contains unknown base IDs")

    gold_decisions = {
        decision.query_id: decision for decision in decisions if decision.target == "final_gold"
    }
    if not set(gold_decisions) <= base_ids:
        raise ValueError("final_gold decision contains unknown base ID")

    resolved: list[ResolvedBase] = []
    for base in candidate_bases:
        original = primary_by_id[base.query_id]
        if original.tool_corpus_hash != corpus_hash or base.tool_corpus_hash != corpus_hash:
            raise ValueError(f"{base.query_id}: candidate Tool Corpus hash mismatch")
        recheck = primary_rows.get(base.query_id)
        source_text = recheck.text if recheck else base.text
        source_annotation = recheck.annotation if recheck else original
        if recheck and recheck.language != base.language:
            raise ValueError(f"{base.query_id}: primary recheck changed language")
        if source_annotation.tool_corpus_hash != corpus_hash:
            raise ValueError(f"{base.query_id}: AI Tool Corpus hash mismatch")

        changed = source_text != base.text or _label_key(source_annotation) != _label_key(original)
        decision = gold_decisions.get(base.query_id)
        if changed and decision is None:
            raise ValueError(f"{base.query_id}: changed AI source requires final_gold decision")
        if decision is not None:
            if decision.text != source_text:
                raise ValueError(f"{base.query_id}: final_gold text differs from AI source")
            if not set(decision.relevant_tool_ids) <= known_tools:
                raise ValueError(f"{base.query_id}: decision contains unknown Tool IDs")
            query = _query_from_decision(base, decision, corpus_hash)
            provenance = "ai_posthoc_adjudicated"
            annotator_id = "codex_ai_posthoc_2026_10_07"
        elif recheck is not None:
            payload = base.model_dump(mode="json")
            payload.update(
                {
                    "text": source_text,
                    "relevant_tool_ids": source_annotation.relevant_tool_ids,
                    "should_abstain": source_annotation.should_abstain,
                    "abstain_reason": source_annotation.abstain_reason,
                    "label_rationale": source_annotation.label_rationale,
                    "candidate_conflict_note": source_annotation.candidate_conflict_note,
                }
            )
            query = QueryCase.model_validate(payload)
            provenance = "ai_primary_recheck"
            annotator_id = source_annotation.annotator_id
        else:
            query = base
            provenance = "generator_provisional"
            annotator_id = original.annotator_id
        resolved.append(
            ResolvedBase(
                query=query,
                final_annotation=_annotation_from_query(query, annotator_id),
                provenance=provenance,
            )
        )
    return tuple(resolved)


def build_secondary_posthoc(
    secondary_rows: Mapping[str, CompletedRow],
    final_bases: Mapping[str, ResolvedBase],
    decisions: Sequence[Decision],
    tools: Sequence[ToolRecord],
) -> dict[str, Annotation]:
    """Materialize post-hoc revisions only where raw secondary text is stale."""

    corpus_hash = tool_corpus_sha256(tools)
    known_tools = {tool.tool_id for tool in tools}
    revision_decisions = {
        decision.query_id: decision
        for decision in decisions
        if decision.target == "secondary_posthoc"
    }
    changed_ids = {
        query_id
        for query_id, row in secondary_rows.items()
        if query_id in final_bases and row.text != final_bases[query_id].query.text
    }
    if set(revision_decisions) != changed_ids:
        missing = sorted(changed_ids - set(revision_decisions))
        extra = sorted(set(revision_decisions) - changed_ids)
        raise ValueError(f"secondary post-hoc IDs mismatch: missing={missing}, extra={extra}")

    revisions: dict[str, Annotation] = {}
    for query_id in sorted(changed_ids):
        decision = revision_decisions[query_id]
        query = final_bases[query_id].query
        if decision.text != query.text or decision.source_role != "secondary_raw":
            raise ValueError(f"{query_id}: secondary post-hoc text or provenance mismatch")
        if not set(decision.relevant_tool_ids) <= known_tools:
            raise ValueError(f"{query_id}: secondary revision contains unknown Tool IDs")
        _query_from_decision(query, decision, corpus_hash)
        revisions[query_id] = Annotation(
            annotation_id=f"secondary-posthoc-{query_id}",
            annotator_id="codex_ai_posthoc_2026_10_07",
            query_id=query_id,
            relevant_tool_ids=decision.relevant_tool_ids,
            should_abstain=decision.should_abstain,
            abstain_reason=decision.abstain_reason,
            label_rationale=decision.rationale,
            candidate_conflict_note=decision.candidate_conflict_note,
            tool_corpus_hash=corpus_hash,
        )
    return revisions
