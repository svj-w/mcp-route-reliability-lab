"""Freeze independent V1 labels, calculate agreement, and finalize adjudication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_route_reliability.annotation_workflow import (  # noqa: E402
    finalize_adjudication,
    import_primary_recheck,
    import_secondary_annotations,
    load_manifest_query_ids,
    write_final_adjudication,
    write_prepared_review,
    write_primary_recheck,
)
from mcp_route_reliability.datasets import (  # noqa: E402
    load_annotations,
    load_queries,
    load_tools,
)

DEFAULT_CANDIDATE = ROOT / "data" / "v1-candidate"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="import a completed blinded assignment and freeze raw labels"
    )
    prepare.add_argument("--completed", type=Path, required=True)
    prepare.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE)

    recheck = subparsers.add_parser(
        "recheck", help="freeze the completed high-risk primary self-check"
    )
    recheck.add_argument("--completed", type=Path, required=True)
    recheck.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE)

    finalize = subparsers.add_parser(
        "finalize", help="import completed human adjudications and freeze disagreements"
    )
    finalize.add_argument("--completed", type=Path, required=True)
    finalize.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE)
    return parser


def _prepare(completed: Path, candidate_dir: Path) -> None:
    annotations_dir = candidate_dir / "annotations"
    tools = load_tools(candidate_dir / "tools.json")
    queries = load_queries(candidate_dir / "queries.jsonl")
    primary = load_annotations(annotations_dir / "primary.jsonl")
    expected_ids = load_manifest_query_ids(annotations_dir / "secondary-sample-manifest.csv")
    if len(expected_ids) != 100:
        raise ValueError(f"V1 secondary review requires 100 queries, found {len(expected_ids)}")
    review = import_secondary_annotations(
        completed,
        expected_query_ids=expected_ids,
        primary=primary,
        queries=queries,
        tools=tools,
    )
    paths = write_prepared_review(review, output_dir=annotations_dir, queries=queries)
    print(
        json.dumps(
            {
                "status": "secondary_labels_frozen",
                "outputs": [str(path) for path in paths],
                "agreement": review.report.model_dump(mode="json"),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def _finalize(completed: Path, candidate_dir: Path) -> None:
    annotations_dir = candidate_dir / "annotations"
    tools = load_tools(candidate_dir / "tools.json")
    queries = load_queries(candidate_dir / "queries.jsonl")
    primary_all = load_annotations(annotations_dir / "primary.jsonl")
    secondary = load_annotations(annotations_dir / "secondary.jsonl")
    if len(secondary) != 100:
        raise ValueError(f"V1 secondary review requires 100 labels, found {len(secondary)}")
    sample_ids = {annotation.query_id for annotation in secondary}
    primary = [annotation for annotation in primary_all if annotation.query_id in sample_ids]
    adjudicated = finalize_adjudication(
        completed,
        primary=primary,
        secondary=secondary,
        queries=queries,
        tools=tools,
    )
    paths = write_final_adjudication(
        adjudicated,
        output_dir=annotations_dir,
        primary=primary,
        secondary=secondary,
        queries=queries,
    )
    print(
        json.dumps(
            {
                "status": "adjudication_frozen",
                "adjudicated_count": len(adjudicated),
                "outputs": [str(path) for path in paths],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def _recheck(completed: Path, candidate_dir: Path) -> None:
    annotations_dir = candidate_dir / "annotations"
    tools = load_tools(candidate_dir / "tools.json")
    queries = load_queries(candidate_dir / "queries.jsonl")
    primary = load_annotations(annotations_dir / "primary.jsonl")
    expected_ids = [
        query.query_id
        for query in queries
        if query.query_id == query.base_query_id
        and query.case_type in {"clarify", "hard_negative", "no_match"}
    ]
    if len(expected_ids) != 180:
        raise ValueError(f"V1 primary recheck requires 180 base queries, found {len(expected_ids)}")
    review = import_primary_recheck(
        completed,
        expected_query_ids=expected_ids,
        primary=primary,
        queries=queries,
        tools=tools,
    )
    paths = write_primary_recheck(review, output_dir=annotations_dir, queries=queries)
    print(
        json.dumps(
            {
                "status": "primary_recheck_frozen",
                "outputs": [str(path) for path in paths],
                "report": review.report.model_dump(mode="json"),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        _prepare(args.completed, args.candidate_dir)
    elif args.command == "recheck":
        _recheck(args.completed, args.candidate_dir)
    else:
        _finalize(args.completed, args.candidate_dir)


if __name__ == "__main__":
    main()
