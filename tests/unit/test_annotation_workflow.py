from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mcp_route_reliability.annotation_workflow import (
    ADJUDICATION_FIELDS,
    ASSIGNMENT_FIELDS,
    AnnotationWorkflowError,
    finalize_adjudication,
    import_primary_recheck,
    import_secondary_annotations,
    write_final_adjudication,
    write_prepared_review,
    write_primary_recheck,
)
from mcp_route_reliability.models import Annotation, QueryCase, ToolRecord


def annotation_from_query(query: QueryCase, annotator_id: str) -> Annotation:
    return Annotation(
        annotation_id=f"{annotator_id}-{query.query_id}",
        annotator_id=annotator_id,
        query_id=query.query_id,
        relevant_tool_ids=query.relevant_tool_ids,
        should_abstain=query.should_abstain,
        abstain_reason=query.abstain_reason,
        label_rationale=query.label_rationale,
        candidate_conflict_note=query.candidate_conflict_note,
        tool_corpus_hash=query.tool_corpus_hash,
    )


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def secondary_rows(
    queries: list[QueryCase], annotator_id: str = "human-secondary"
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for query in queries:
        relevant = query.relevant_tool_ids
        should_abstain = query.should_abstain
        reason = query.abstain_reason
        rationale = f"Independent review for {query.query_id}."
        if query.query_id == "q_001":
            relevant = ()
            should_abstain = True
            reason = "no_match"
            rationale = "No available Tool supports the requested capability."
        if query.query_id == "q_002":
            relevant = tuple(reversed(relevant))
        rows.append(
            {
                "query_id": query.query_id,
                "text": query.text,
                "language": query.language,
                "relevant_tool_ids": "|".join(relevant),
                "should_abstain": str(should_abstain).lower(),
                "abstain_reason": reason or "",
                "label_rationale": rationale,
                "candidate_conflict_note": "",
                "annotator_id": annotator_id,
                "tool_corpus_hash": query.tool_corpus_hash,
            }
        )
    return rows


def prepare_fixture(
    tmp_path: Path,
    tiny_tools: list[ToolRecord],
    tiny_queries: list[QueryCase],
) -> tuple[Path, list[QueryCase], list[Annotation]]:
    queries = tiny_queries[:4]
    primary = [annotation_from_query(query, "human-primary") for query in queries]
    completed = tmp_path / "completed-secondary.csv"
    write_csv(completed, ASSIGNMENT_FIELDS, secondary_rows(queries))
    return completed, queries, primary


def test_prepare_and_finalize_preserve_raw_labels(
    tmp_path: Path,
    tiny_tools: list[ToolRecord],
    tiny_queries: list[QueryCase],
) -> None:
    completed, queries, primary = prepare_fixture(tmp_path, tiny_tools, tiny_queries)
    review = import_secondary_annotations(
        completed,
        expected_query_ids=[query.query_id for query in queries],
        primary=primary,
        queries=queries,
        tools=tiny_tools,
    )

    assert review.disagreement_query_ids == ("q_001",)
    assert review.report.sample_size == 4
    assert review.report.exact_label_agreement_count == 3
    assert review.report.disagreement_count == 1
    assert review.report.route_abstain_kappa.value == pytest.approx(0.5)
    assert review.report.relevant_tool_set_jaccard_by_query["q_002"] == 1.0

    output = tmp_path / "review"
    secondary_path, report_path, assignment_path = write_prepared_review(
        review, output_dir=output, queries=queries
    )
    assert len(secondary_path.read_text(encoding="utf-8").splitlines()) == 4
    assert json.loads(report_path.read_text(encoding="utf-8"))["raw_labels_frozen"] is True
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_prepared_review(review, output_dir=output, queries=queries)

    with assignment_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    rows[0].update(
        {
            "adjudicated_relevant_tool_ids": queries[1].relevant_tool_ids[0],
            "adjudicated_should_abstain": "false",
            "adjudicated_abstain_reason": "",
            "adjudicated_label_rationale": "The repository operation is supported.",
            "adjudicated_candidate_conflict_note": "Reviewed the neighboring capability.",
            "adjudicator_id": "human-adjudicator",
        }
    )
    completed_adjudication = tmp_path / "completed-adjudication.csv"
    write_csv(completed_adjudication, ADJUDICATION_FIELDS, rows)

    adjudicated = finalize_adjudication(
        completed_adjudication,
        primary=review.primary,
        secondary=review.secondary,
        queries=queries,
        tools=tiny_tools,
    )
    assert len(adjudicated) == 1
    assert adjudicated[0].query_id == "q_001"
    assert adjudicated[0].annotator_id == "human-adjudicator"

    final_dir = tmp_path / "final"
    adjudicated_path, disagreements_path = write_final_adjudication(
        adjudicated,
        output_dir=final_dir,
        primary=review.primary,
        secondary=review.secondary,
        queries=queries,
    )
    assert "human-adjudicator" in adjudicated_path.read_text(encoding="utf-8")
    with disagreements_path.open(encoding="utf-8-sig", newline="") as handle:
        disagreement_rows = list(csv.DictReader(handle))
    assert disagreement_rows[0]["primary_should_abstain"] == "false"
    assert disagreement_rows[0]["secondary_should_abstain"] == "true"
    assert disagreement_rows[0]["adjudicated_should_abstain"] == "false"


def test_secondary_import_rejects_modified_query_text(
    tmp_path: Path,
    tiny_tools: list[ToolRecord],
    tiny_queries: list[QueryCase],
) -> None:
    completed, queries, primary = prepare_fixture(tmp_path, tiny_tools, tiny_queries)
    rows = secondary_rows(queries)
    rows[0]["text"] = "tampered"
    write_csv(completed, ASSIGNMENT_FIELDS, rows)

    with pytest.raises(AnnotationWorkflowError, match="text or language was modified"):
        import_secondary_annotations(
            completed,
            expected_query_ids=[query.query_id for query in queries],
            primary=primary,
            queries=queries,
            tools=tiny_tools,
        )


def test_primary_recheck_exports_only_changed_labels(
    tmp_path: Path,
    tiny_tools: list[ToolRecord],
    tiny_queries: list[QueryCase],
) -> None:
    completed, queries, primary = prepare_fixture(tmp_path, tiny_tools, tiny_queries)
    write_csv(
        completed,
        ASSIGNMENT_FIELDS,
        secondary_rows(queries, annotator_id="human-primary-pass-2"),
    )

    review = import_primary_recheck(
        completed,
        expected_query_ids=[query.query_id for query in queries],
        primary=primary,
        queries=queries,
        tools=tiny_tools,
    )

    assert review.report.changed_query_ids == ("q_001",)
    assert review.report.changed_label_count == 1
    assert review.report.exact_label_agreement_count == 3
    _, report_path, changes_path = write_primary_recheck(
        review, output_dir=tmp_path / "recheck", queries=queries
    )
    assert json.loads(report_path.read_text(encoding="utf-8"))["raw_recheck_frozen"] is True
    with changes_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["original_should_abstain"] == "false"
    assert rows[0]["recheck_should_abstain"] == "true"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("should_abstain", "yes", "exactly true or false"),
        ("relevant_tool_ids", "unknown.tool", "unknown Tool IDs"),
        ("annotator_id", "human-primary", "must be independent"),
    ],
)
def test_secondary_import_rejects_invalid_labels(
    tmp_path: Path,
    tiny_tools: list[ToolRecord],
    tiny_queries: list[QueryCase],
    field: str,
    value: str,
    message: str,
) -> None:
    completed, queries, primary = prepare_fixture(tmp_path, tiny_tools, tiny_queries)
    rows = secondary_rows(queries)
    rows[0][field] = value
    write_csv(completed, ASSIGNMENT_FIELDS, rows)

    with pytest.raises(AnnotationWorkflowError, match=message):
        import_secondary_annotations(
            completed,
            expected_query_ids=[query.query_id for query in queries],
            primary=primary,
            queries=queries,
            tools=tiny_tools,
        )
