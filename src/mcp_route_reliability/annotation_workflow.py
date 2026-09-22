"""Auditable import and adjudication workflow for independent annotations."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import Field

from mcp_route_reliability.datasets import (
    AgreementStatistic,
    build_disagreement_records,
    cohen_kappa,
    mean_set_jaccard,
)
from mcp_route_reliability.hashing import tool_corpus_sha256
from mcp_route_reliability.models import Annotation, QueryCase, StrictModel, ToolRecord

ASSIGNMENT_FIELDS = (
    "query_id",
    "text",
    "language",
    "relevant_tool_ids",
    "should_abstain",
    "abstain_reason",
    "label_rationale",
    "candidate_conflict_note",
    "annotator_id",
    "tool_corpus_hash",
)

ADJUDICATION_FIELDS = (
    "query_id",
    "text",
    "language",
    "primary_relevant_tool_ids",
    "primary_should_abstain",
    "primary_abstain_reason",
    "primary_label_rationale",
    "primary_candidate_conflict_note",
    "primary_annotator_id",
    "secondary_relevant_tool_ids",
    "secondary_should_abstain",
    "secondary_abstain_reason",
    "secondary_label_rationale",
    "secondary_candidate_conflict_note",
    "secondary_annotator_id",
    "adjudicated_relevant_tool_ids",
    "adjudicated_should_abstain",
    "adjudicated_abstain_reason",
    "adjudicated_label_rationale",
    "adjudicated_candidate_conflict_note",
    "adjudicator_id",
    "tool_corpus_hash",
)

DISAGREEMENT_FIELDS = ADJUDICATION_FIELDS
RECHECK_CHANGE_FIELDS = (
    "query_id",
    "text",
    "language",
    "original_relevant_tool_ids",
    "original_should_abstain",
    "original_abstain_reason",
    "original_label_rationale",
    "original_candidate_conflict_note",
    "original_annotator_id",
    "recheck_relevant_tool_ids",
    "recheck_should_abstain",
    "recheck_abstain_reason",
    "recheck_label_rationale",
    "recheck_candidate_conflict_note",
    "recheck_annotator_id",
    "tool_corpus_hash",
)
RecordT = TypeVar("RecordT")


class AnnotationWorkflowError(ValueError):
    """Human-review input is incomplete, inconsistent, or has been tampered with."""


class AgreementReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    sample_size: int = Field(ge=1)
    tool_corpus_hash: str = Field(min_length=1)
    primary_annotator_ids: tuple[str, ...]
    secondary_annotator_id: str = Field(min_length=1)
    route_abstain_kappa: AgreementStatistic
    relevant_tool_set_jaccard_mean: float = Field(ge=0, le=1)
    relevant_tool_set_jaccard_by_query: dict[str, float]
    exact_label_agreement_count: int = Field(ge=0)
    disagreement_count: int = Field(ge=0)
    raw_labels_frozen: Literal[True] = True


class PreparedReview(StrictModel):
    primary: tuple[Annotation, ...]
    secondary: tuple[Annotation, ...]
    disagreement_query_ids: tuple[str, ...]
    report: AgreementReport


class SelfCheckReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    sample_size: int = Field(ge=1)
    tool_corpus_hash: str = Field(min_length=1)
    original_annotator_ids: tuple[str, ...]
    recheck_annotator_id: str = Field(min_length=1)
    exact_label_agreement_count: int = Field(ge=0)
    changed_label_count: int = Field(ge=0)
    changed_query_ids: tuple[str, ...]
    raw_recheck_frozen: Literal[True] = True


class PreparedSelfCheck(StrictModel):
    original: tuple[Annotation, ...]
    recheck: tuple[Annotation, ...]
    report: SelfCheckReport


def _read_csv(path: Path, expected_fields: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(expected_fields):
                raise AnnotationWorkflowError(
                    f"{path}: columns must exactly match {list(expected_fields)!r}"
                )
            rows: list[dict[str, str]] = []
            for line_number, raw in enumerate(reader, start=2):
                if None in raw or any(value is None for value in raw.values()):
                    raise AnnotationWorkflowError(f"{path} line {line_number}: malformed CSV row")
                rows.append({key: value for key, value in raw.items() if key is not None})
    except OSError as error:
        raise AnnotationWorkflowError(f"cannot read {path}: {error}") from error
    return rows


def load_manifest_query_ids(path: Path) -> tuple[str, ...]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "query_id" not in reader.fieldnames:
                raise AnnotationWorkflowError(f"{path}: manifest requires query_id")
            query_ids = [str(row.get("query_id", "")).strip() for row in reader]
    except OSError as error:
        raise AnnotationWorkflowError(f"cannot read {path}: {error}") from error
    if any(not query_id for query_id in query_ids):
        raise AnnotationWorkflowError(f"{path}: manifest contains an empty query_id")
    if len(set(query_ids)) != len(query_ids):
        raise AnnotationWorkflowError(f"{path}: manifest contains duplicate query IDs")
    if not query_ids:
        raise AnnotationWorkflowError(f"{path}: manifest is empty")
    return tuple(query_ids)


def _unique_map(
    values: Sequence[RecordT], id_getter: str, label: str
) -> dict[str, RecordT]:
    result: dict[str, RecordT] = {}
    for value in values:
        identifier = getattr(value, id_getter)
        if not isinstance(identifier, str):
            raise TypeError(f"{label} identifier must be a string")
        if identifier in result:
            raise AnnotationWorkflowError(f"{label} contains duplicate ID {identifier!r}")
        result[identifier] = value
    return result


def _parse_bool(value: str, *, path: Path, query_id: str, field: str) -> bool:
    normalized = value.strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise AnnotationWorkflowError(
        f"{path}: {query_id} field {field} must be exactly true or false"
    )


def _parse_tool_ids(value: str, *, path: Path, query_id: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    parts = tuple(part.strip() for part in value.split("|"))
    if any(not part for part in parts):
        raise AnnotationWorkflowError(f"{path}: {query_id} contains an empty Tool ID")
    if len(set(parts)) != len(parts):
        raise AnnotationWorkflowError(f"{path}: {query_id} contains duplicate Tool IDs")
    return parts


def _annotation_from_row(
    row: Mapping[str, str],
    *,
    path: Path,
    prefix: str,
    annotation_id_prefix: str,
    annotator_field: str,
    valid_tool_ids: frozenset[str],
    corpus_hash: str,
) -> Annotation:
    query_id = row["query_id"].strip()
    annotator_id = row[annotator_field].strip()
    rationale = row[f"{prefix}label_rationale"].strip()
    if not annotator_id:
        raise AnnotationWorkflowError(f"{path}: {query_id} has an empty {annotator_field}")
    if not rationale:
        raise AnnotationWorkflowError(f"{path}: {query_id} has an empty label_rationale")
    if row["tool_corpus_hash"].strip() != corpus_hash:
        raise AnnotationWorkflowError(f"{path}: {query_id} has the wrong Tool Corpus hash")

    relevant = _parse_tool_ids(
        row[f"{prefix}relevant_tool_ids"], path=path, query_id=query_id
    )
    unknown = sorted(set(relevant) - valid_tool_ids)
    if unknown:
        raise AnnotationWorkflowError(f"{path}: {query_id} has unknown Tool IDs {unknown!r}")
    should_abstain = _parse_bool(
        row[f"{prefix}should_abstain"],
        path=path,
        query_id=query_id,
        field=f"{prefix}should_abstain",
    )
    reason_value = row[f"{prefix}abstain_reason"].strip() or None
    if not should_abstain and not relevant:
        raise AnnotationWorkflowError(f"{path}: {query_id} routes with an empty Tool set")
    if should_abstain and reason_value != "no_match" and not relevant:
        raise AnnotationWorkflowError(
            f"{path}: {query_id} non-no-match abstention requires a relevant Tool"
        )

    try:
        return Annotation.model_validate(
            {
                "annotation_id": f"{annotation_id_prefix}-{query_id}",
                "annotator_id": annotator_id,
                "query_id": query_id,
                "relevant_tool_ids": relevant,
                "should_abstain": should_abstain,
                "abstain_reason": reason_value,
                "label_rationale": rationale,
                "candidate_conflict_note": (
                    row[f"{prefix}candidate_conflict_note"].strip() or None
                ),
                "tool_corpus_hash": corpus_hash,
            }
        )
    except ValueError as error:
        raise AnnotationWorkflowError(
            f"{path}: invalid annotation for {query_id}: {error}"
        ) from error


def _same_label(first: Annotation, second: Annotation) -> bool:
    return (
        frozenset(first.relevant_tool_ids) == frozenset(second.relevant_tool_ids)
        and first.should_abstain == second.should_abstain
        and first.abstain_reason == second.abstain_reason
    )


def import_secondary_annotations(
    path: Path,
    *,
    expected_query_ids: Sequence[str],
    primary: Sequence[Annotation],
    queries: Sequence[QueryCase],
    tools: Sequence[ToolRecord],
) -> PreparedReview:
    """Validate a completed blinded assignment and calculate pre-adjudication agreement."""

    rows = _read_csv(path, ASSIGNMENT_FIELDS)
    expected = set(expected_query_ids)
    if len(expected) != len(expected_query_ids):
        raise AnnotationWorkflowError("expected query IDs contain duplicates")
    row_ids = [row["query_id"].strip() for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise AnnotationWorkflowError(f"{path}: completed assignment contains duplicate query IDs")
    if set(row_ids) != expected:
        missing = sorted(expected - set(row_ids))
        extra = sorted(set(row_ids) - expected)
        raise AnnotationWorkflowError(
            f"{path}: assignment IDs differ from manifest; missing={missing!r}, extra={extra!r}"
        )

    primary_by_id = _unique_map(primary, "query_id", "primary annotations")
    query_by_id = _unique_map(queries, "query_id", "queries")
    if not expected <= set(primary_by_id):
        raise AnnotationWorkflowError("primary annotations do not cover the manifest sample")
    if not expected <= set(query_by_id):
        raise AnnotationWorkflowError("queries do not cover the manifest sample")

    corpus_hash = tool_corpus_sha256(tools)
    valid_tool_ids = frozenset(tool.tool_id for tool in tools)
    secondary: list[Annotation] = []
    for row in rows:
        query_id = row["query_id"].strip()
        query = query_by_id[query_id]
        if row["text"] != query.text or row["language"].strip() != query.language:
            raise AnnotationWorkflowError(f"{path}: {query_id} text or language was modified")
        first = primary_by_id[query_id]
        if first.tool_corpus_hash != corpus_hash or query.tool_corpus_hash != corpus_hash:
            raise AnnotationWorkflowError(f"{query_id}: source records use the wrong corpus hash")
        parsed = _annotation_from_row(
            row,
            path=path,
            prefix="",
            annotation_id_prefix="secondary",
            annotator_field="annotator_id",
            valid_tool_ids=valid_tool_ids,
            corpus_hash=corpus_hash,
        )
        if parsed.annotator_id == first.annotator_id:
            raise AnnotationWorkflowError(
                f"{path}: {query_id} secondary annotator must be independent"
            )
        secondary.append(parsed)

    annotator_ids = {annotation.annotator_id for annotation in secondary}
    if len(annotator_ids) != 1:
        raise AnnotationWorkflowError(
            f"{path}: use one stable secondary annotator_id, found {sorted(annotator_ids)!r}"
        )

    ordered_ids = tuple(sorted(expected))
    ordered_primary = tuple(primary_by_id[query_id] for query_id in ordered_ids)
    secondary_by_id = _unique_map(secondary, "query_id", "secondary annotations")
    ordered_secondary = tuple(secondary_by_id[query_id] for query_id in ordered_ids)
    route_labels_primary = [
        "abstain" if annotation.should_abstain else "route" for annotation in ordered_primary
    ]
    route_labels_secondary = [
        "abstain" if annotation.should_abstain else "route" for annotation in ordered_secondary
    ]
    jaccard = mean_set_jaccard(
        [frozenset(annotation.relevant_tool_ids) for annotation in ordered_primary],
        [frozenset(annotation.relevant_tool_ids) for annotation in ordered_secondary],
    )
    disagreement_ids = tuple(
        query_id
        for query_id, first, second in zip(
            ordered_ids, ordered_primary, ordered_secondary, strict=True
        )
        if not _same_label(first, second)
    )
    report = AgreementReport(
        sample_size=len(ordered_ids),
        tool_corpus_hash=corpus_hash,
        primary_annotator_ids=tuple(
            sorted({annotation.annotator_id for annotation in ordered_primary})
        ),
        secondary_annotator_id=next(iter(annotator_ids)),
        route_abstain_kappa=cohen_kappa(route_labels_primary, route_labels_secondary),
        relevant_tool_set_jaccard_mean=jaccard.mean,
        relevant_tool_set_jaccard_by_query=dict(zip(ordered_ids, jaccard.per_sample, strict=True)),
        exact_label_agreement_count=len(ordered_ids) - len(disagreement_ids),
        disagreement_count=len(disagreement_ids),
    )
    return PreparedReview(
        primary=ordered_primary,
        secondary=ordered_secondary,
        disagreement_query_ids=disagreement_ids,
        report=report,
    )


def import_primary_recheck(
    path: Path,
    *,
    expected_query_ids: Sequence[str],
    primary: Sequence[Annotation],
    queries: Sequence[QueryCase],
    tools: Sequence[ToolRecord],
) -> PreparedSelfCheck:
    """Validate a blinded second pass over high-risk primary labels."""

    rows = _read_csv(path, ASSIGNMENT_FIELDS)
    expected = set(expected_query_ids)
    if len(expected) != len(expected_query_ids):
        raise AnnotationWorkflowError("expected recheck query IDs contain duplicates")
    row_ids = [row["query_id"].strip() for row in rows]
    if len(set(row_ids)) != len(row_ids) or set(row_ids) != expected:
        raise AnnotationWorkflowError(
            f"{path}: recheck IDs must exactly match {sorted(expected)!r}"
        )

    primary_by_id = _unique_map(primary, "query_id", "primary annotations")
    query_by_id = _unique_map(queries, "query_id", "queries")
    if not expected <= set(primary_by_id) or not expected <= set(query_by_id):
        raise AnnotationWorkflowError("primary records do not cover the recheck assignment")
    corpus_hash = tool_corpus_sha256(tools)
    valid_tool_ids = frozenset(tool.tool_id for tool in tools)
    recheck: list[Annotation] = []
    for row in rows:
        query_id = row["query_id"].strip()
        query = query_by_id[query_id]
        original = primary_by_id[query_id]
        if row["text"] != query.text or row["language"].strip() != query.language:
            raise AnnotationWorkflowError(f"{path}: {query_id} text or language was modified")
        if original.tool_corpus_hash != corpus_hash or query.tool_corpus_hash != corpus_hash:
            raise AnnotationWorkflowError(f"{query_id}: source records use the wrong corpus hash")
        recheck.append(
            _annotation_from_row(
                row,
                path=path,
                prefix="",
                annotation_id_prefix="primary-recheck",
                annotator_field="annotator_id",
                valid_tool_ids=valid_tool_ids,
                corpus_hash=corpus_hash,
            )
        )
    recheck_ids = {annotation.annotator_id for annotation in recheck}
    if len(recheck_ids) != 1:
        raise AnnotationWorkflowError(
            f"{path}: use one stable recheck annotator_id, found {sorted(recheck_ids)!r}"
        )

    ordered_ids = tuple(sorted(expected))
    original_rows = tuple(primary_by_id[query_id] for query_id in ordered_ids)
    recheck_by_id = _unique_map(recheck, "query_id", "recheck annotations")
    recheck_rows = tuple(recheck_by_id[query_id] for query_id in ordered_ids)
    changed_ids = tuple(
        query_id
        for query_id, original, checked in zip(
            ordered_ids, original_rows, recheck_rows, strict=True
        )
        if not _same_label(original, checked)
    )
    report = SelfCheckReport(
        sample_size=len(ordered_ids),
        tool_corpus_hash=corpus_hash,
        original_annotator_ids=tuple(
            sorted({annotation.annotator_id for annotation in original_rows})
        ),
        recheck_annotator_id=next(iter(recheck_ids)),
        exact_label_agreement_count=len(ordered_ids) - len(changed_ids),
        changed_label_count=len(changed_ids),
        changed_query_ids=changed_ids,
    )
    return PreparedSelfCheck(original=original_rows, recheck=recheck_rows, report=report)


def _annotation_columns(annotation: Annotation, prefix: str) -> dict[str, str]:
    annotator_field = "adjudicator_id" if prefix == "adjudicated_" else f"{prefix}annotator_id"
    return {
        f"{prefix}relevant_tool_ids": "|".join(sorted(annotation.relevant_tool_ids)),
        f"{prefix}should_abstain": str(annotation.should_abstain).lower(),
        f"{prefix}abstain_reason": annotation.abstain_reason or "",
        f"{prefix}label_rationale": annotation.label_rationale,
        f"{prefix}candidate_conflict_note": annotation.candidate_conflict_note or "",
        annotator_field: annotation.annotator_id,
    }


def _ensure_new(paths: Sequence[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite frozen review artifacts: {existing!r}")


def _write_jsonl(path: Path, values: Sequence[StrictModel]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(value.model_dump_json() + "\n")


def write_prepared_review(
    review: PreparedReview, *, output_dir: Path, queries: Sequence[QueryCase]
) -> tuple[Path, Path, Path]:
    """Freeze secondary labels, agreement, and a human adjudication assignment."""

    output_dir.mkdir(parents=True, exist_ok=True)
    secondary_path = output_dir / "secondary.jsonl"
    report_path = output_dir / "agreement.json"
    assignment_path = output_dir / "adjudication-assignment.csv"
    _ensure_new((secondary_path, report_path, assignment_path))

    query_by_id = _unique_map(queries, "query_id", "queries")
    primary_by_id = _unique_map(review.primary, "query_id", "primary annotations")
    secondary_by_id = _unique_map(review.secondary, "query_id", "secondary annotations")
    _write_jsonl(secondary_path, review.secondary)
    report_path.write_text(
        review.report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with assignment_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ADJUDICATION_FIELDS))
        writer.writeheader()
        for query_id in review.disagreement_query_ids:
            query = query_by_id[query_id]
            first = primary_by_id[query_id]
            second = secondary_by_id[query_id]
            writer.writerow(
                {
                    "query_id": query_id,
                    "text": query.text,
                    "language": query.language,
                    **_annotation_columns(first, "primary_"),
                    **_annotation_columns(second, "secondary_"),
                    "adjudicated_relevant_tool_ids": "",
                    "adjudicated_should_abstain": "",
                    "adjudicated_abstain_reason": "",
                    "adjudicated_label_rationale": "",
                    "adjudicated_candidate_conflict_note": "",
                    "adjudicator_id": "",
                    "tool_corpus_hash": review.report.tool_corpus_hash,
                }
            )
    return secondary_path, report_path, assignment_path


def write_primary_recheck(
    review: PreparedSelfCheck, *, output_dir: Path, queries: Sequence[QueryCase]
) -> tuple[Path, Path, Path]:
    """Freeze the self-check pass and export every proposed label change."""

    output_dir.mkdir(parents=True, exist_ok=True)
    recheck_path = output_dir / "primary-recheck.jsonl"
    report_path = output_dir / "primary-recheck-report.json"
    changes_path = output_dir / "primary-recheck-changes.csv"
    _ensure_new((recheck_path, report_path, changes_path))

    query_by_id = _unique_map(queries, "query_id", "queries")
    original_by_id = _unique_map(review.original, "query_id", "original annotations")
    recheck_by_id = _unique_map(review.recheck, "query_id", "recheck annotations")
    _write_jsonl(recheck_path, review.recheck)
    report_path.write_text(
        review.report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with changes_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RECHECK_CHANGE_FIELDS))
        writer.writeheader()
        for query_id in review.report.changed_query_ids:
            query = query_by_id[query_id]
            writer.writerow(
                {
                    "query_id": query_id,
                    "text": query.text,
                    "language": query.language,
                    **_annotation_columns(original_by_id[query_id], "original_"),
                    **_annotation_columns(recheck_by_id[query_id], "recheck_"),
                    "tool_corpus_hash": review.report.tool_corpus_hash,
                }
            )
    return recheck_path, report_path, changes_path


def finalize_adjudication(
    path: Path,
    *,
    primary: Sequence[Annotation],
    secondary: Sequence[Annotation],
    queries: Sequence[QueryCase],
    tools: Sequence[ToolRecord],
) -> tuple[Annotation, ...]:
    """Validate one human adjudication for every raw-label disagreement."""

    rows = _read_csv(path, ADJUDICATION_FIELDS)
    primary_by_id = _unique_map(primary, "query_id", "primary annotations")
    secondary_by_id = _unique_map(secondary, "query_id", "secondary annotations")
    query_by_id = _unique_map(queries, "query_id", "queries")
    if set(primary_by_id) != set(secondary_by_id):
        raise AnnotationWorkflowError("primary and secondary sample IDs must match exactly")
    expected = {
        query_id
        for query_id in primary_by_id
        if not _same_label(primary_by_id[query_id], secondary_by_id[query_id])
    }
    row_ids = [row["query_id"].strip() for row in rows]
    if len(set(row_ids)) != len(row_ids) or set(row_ids) != expected:
        raise AnnotationWorkflowError(
            f"{path}: adjudication IDs must exactly match disagreements {sorted(expected)!r}"
        )

    corpus_hash = tool_corpus_sha256(tools)
    valid_tool_ids = frozenset(tool.tool_id for tool in tools)
    adjudicated: list[Annotation] = []
    for row in rows:
        query_id = row["query_id"].strip()
        first = primary_by_id[query_id]
        second = secondary_by_id[query_id]
        query = query_by_id.get(query_id)
        if query is None or row["text"] != query.text or row["language"].strip() != query.language:
            raise AnnotationWorkflowError(f"{path}: {query_id} text or language was modified")
        expected_raw = {
            **_annotation_columns(first, "primary_"),
            **_annotation_columns(second, "secondary_"),
        }
        for field, expected_value in expected_raw.items():
            if row[field] != expected_value:
                raise AnnotationWorkflowError(f"{path}: {query_id} modified frozen field {field}")
        parsed = _annotation_from_row(
            row,
            path=path,
            prefix="adjudicated_",
            annotation_id_prefix="adjudicated",
            annotator_field="adjudicator_id",
            valid_tool_ids=valid_tool_ids,
            corpus_hash=corpus_hash,
        )
        adjudicated.append(parsed)
    if len({annotation.annotator_id for annotation in adjudicated}) > 1:
        raise AnnotationWorkflowError(f"{path}: use one stable adjudicator_id")
    return tuple(sorted(adjudicated, key=lambda annotation: annotation.query_id))


def write_final_adjudication(
    adjudicated: Sequence[Annotation],
    *,
    output_dir: Path,
    primary: Sequence[Annotation],
    secondary: Sequence[Annotation],
    queries: Sequence[QueryCase],
) -> tuple[Path, Path]:
    """Freeze adjudications and export flattened, auditable disagreement rows."""

    output_dir.mkdir(parents=True, exist_ok=True)
    adjudicated_path = output_dir / "adjudicated.jsonl"
    disagreements_path = output_dir / "disagreements.csv"
    _ensure_new((adjudicated_path, disagreements_path))
    adjudicated_ids = {annotation.query_id for annotation in adjudicated}
    primary_subset = [item for item in primary if item.query_id in adjudicated_ids]
    secondary_subset = [item for item in secondary if item.query_id in adjudicated_ids]
    records = build_disagreement_records(primary_subset, secondary_subset, adjudicated)
    if len(records) != len(adjudicated):
        raise AnnotationWorkflowError("adjudication set contains a non-disagreement row")

    query_by_id = _unique_map(queries, "query_id", "queries")
    _write_jsonl(adjudicated_path, adjudicated)
    with disagreements_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DISAGREEMENT_FIELDS))
        writer.writeheader()
        for record in records:
            query = query_by_id[record.query_id]
            writer.writerow(
                {
                    "query_id": record.query_id,
                    "text": query.text,
                    "language": query.language,
                    **_annotation_columns(record.primary, "primary_"),
                    **_annotation_columns(record.secondary, "secondary_"),
                    **_annotation_columns(record.adjudicated, "adjudicated_"),
                    "tool_corpus_hash": record.adjudicated.tool_corpus_hash,
                }
            )
    return adjudicated_path, disagreements_path
