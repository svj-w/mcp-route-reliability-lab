"""Build a deterministic, provenance-preserving V1 AI experimental snapshot."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from mcp_route_reliability.ai_annotation_sources import load_completed_rows
from mcp_route_reliability.ai_label_decisions import (
    ResolvedBase,
    build_secondary_posthoc,
    load_decisions,
    resolve_bases,
)
from mcp_route_reliability.annotation_workflow import load_manifest_query_ids
from mcp_route_reliability.datasets import load_annotations, load_queries, load_tools
from mcp_route_reliability.intent_families import (
    assign_family_splits,
    intent_family_key,
    synchronize_variants,
)
from mcp_route_reliability.models import Annotation, QueryCase
from mcp_route_reliability.validation import (
    DatasetSummary,
    get_validation_profile,
    validate_dataset,
)

PRIMARY_SHA256 = "4497cdb9e0e53ab5d351e93fd5af7990cebc900e95216d6342d67e8b8d339377"
SECONDARY_SHA256 = "9aefbd47bd2cec803da3aedd50a65a9c75bb67370c88e8b7f66fcd2fc24a7990"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(rows: list[QueryCase] | list[Annotation]) -> str:
    return "".join(row.model_dump_json() + "\n" for row in rows)


def _with_split(resolved: ResolvedBase, split: str) -> ResolvedBase:
    query = QueryCase.model_validate({**resolved.query.model_dump(mode="json"), "split": split})
    return ResolvedBase(query, resolved.final_annotation, resolved.provenance)


def build_ai_dataset(candidate_dir: Path, input_dir: Path, output_dir: Path) -> DatasetSummary:
    """Validate all sources and decisions before atomically creating a new snapshot."""

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_dir}")
    tools = load_tools(candidate_dir / "tools.json")
    candidates = load_queries(candidate_dir / "queries.jsonl")
    bases = [row for row in candidates if row.perturbation_type is None]
    variants = [row for row in candidates if row.perturbation_type is not None]
    primary = load_annotations(candidate_dir / "annotations" / "primary.jsonl")
    recheck = load_completed_rows(
        input_dir / "raw" / "completed-primary-recheck.csv",
        load_manifest_query_ids(candidate_dir / "annotations" / "primary-recheck-assignment.csv"),
        tools,
        PRIMARY_SHA256,
    )
    secondary = load_completed_rows(
        input_dir / "raw" / "completed-secondary.csv",
        load_manifest_query_ids(candidate_dir / "annotations" / "secondary-sample-manifest.csv"),
        tools,
        SECONDARY_SHA256,
    )
    decisions = load_decisions(input_dir / "decisions.jsonl")
    resolved = resolve_bases(bases, primary, recheck, decisions, tools)
    pinned = {row.base_query_id for row in variants}
    splits = assign_family_splits([row.query for row in resolved], pinned)
    split_resolved = [_with_split(row, splits[row.query.query_id]) for row in resolved]
    by_id = {row.query.query_id: row for row in split_resolved}
    posthoc = build_secondary_posthoc(secondary, by_id, decisions, tools)
    final_bases = {query_id: row.query for query_id, row in by_id.items()}
    revised_variants = list(synchronize_variants(variants, final_bases, splits))
    queries = [row.query for row in split_resolved] + revised_variants
    summary = validate_dataset(tools, queries, get_validation_profile("ai-experimental"))

    final_annotations = [row.final_annotation for row in split_resolved]
    for variant in revised_variants:
        final_annotations.append(Annotation(
            annotation_id=f"final-{variant.query_id}",
            annotator_id=by_id[variant.base_query_id].final_annotation.annotator_id,
            query_id=variant.query_id,
            relevant_tool_ids=variant.relevant_tool_ids,
            should_abstain=variant.should_abstain,
            abstain_reason=variant.abstain_reason,
            label_rationale=variant.label_rationale,
            candidate_conflict_note=variant.candidate_conflict_note,
            tool_corpus_hash=variant.tool_corpus_hash,
        ))
    if len(final_annotations) != 390 or len(posthoc) != 15:
        raise ValueError("final annotations or post-hoc revisions have unexpected counts")

    provenance = Counter(row.provenance for row in split_resolved)
    decision_counts = Counter(row.target for row in decisions)
    source_manifest = {
        "schema_version": "1.0",
        "release": "v1-ai-experimental",
        "review_status": "AI annotations and AI post-hoc review; not human-reviewed",
        "candidate_queries_sha256": _sha256(candidate_dir / "queries.jsonl"),
        "candidate_primary_sha256": _sha256(candidate_dir / "annotations" / "primary.jsonl"),
        "primary_raw_sha256": PRIMARY_SHA256,
        "secondary_raw_sha256": SECONDARY_SHA256,
        "decision_log_sha256": _sha256(input_dir / "decisions.jsonl"),
        "tool_corpus_hash": summary.tool_corpus_hash,
        "base_count": summary.base_query_count,
        "variant_count": summary.variant_count,
        "primary_recheck_count": len(recheck),
        "not_primary_rechecked_count": len(bases) - len(recheck),
        "secondary_raw_count": len(secondary),
        "secondary_posthoc_count": len(posthoc),
        "decision_counts": dict(sorted(decision_counts.items())),
        "provenance_counts": dict(sorted(provenance.items())),
        "case_type_base_counts": summary.case_type_base_counts,
        "split_base_counts": summary.split_base_counts,
    }
    card = (
        "# V1 AI experimental Dataset Card\n\n"
        "This is an AI experimental, synthetic tool-routing benchmark. It is not human-reviewed, "
        "not an independent inter-annotator study, and not a production safety guarantee.\n\n"
        "- 40 Tool Corpus records; 300 base queries; 90 selected variants.\n"
        "- 180 bases have an AI primary recheck; 120 do not.\n"
        "- 100 raw AI secondary rows are retained unchanged. Fifteen require separately logged "
        "post-hoc relabeling because their raw text differs from the revised primary text.\n"
        "- All decisions are in `decisions.jsonl`; source hashes and provenance counts are in "
        "`source-manifest.json`. Generator-provisional labels are identified there.\n"
        "- Numbered scenario templates are grouped into one split. This prevents measured "
        "cross-split template leakage under the documented normalization, but cannot prove "
        "that every semantic paraphrase is isolated.\n"
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-stage-", dir=output_dir.parent))
    try:
        (stage / "annotations").mkdir()
        (stage / "raw").mkdir()
        shutil.copyfile(candidate_dir / "tools.json", stage / "tools.json")
        shutil.copyfile(
            input_dir / "raw" / "completed-primary-recheck.csv",
            stage / "raw" / "completed-primary-recheck.csv",
        )
        shutil.copyfile(
            input_dir / "raw" / "completed-secondary.csv",
            stage / "raw" / "completed-secondary.csv",
        )
        shutil.copyfile(input_dir / "decisions.jsonl", stage / "decisions.jsonl")
        (stage / "queries.jsonl").write_text(_jsonl(queries), encoding="utf-8")
        (stage / "annotations" / "final.jsonl").write_text(
            _jsonl(final_annotations), encoding="utf-8"
        )
        (stage / "annotations" / "secondary-posthoc.jsonl").write_text(
            _jsonl([posthoc[key] for key in sorted(posthoc)]), encoding="utf-8"
        )
        with (stage / "split-manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("query_id", "split", "intent_family_key", "provenance"))
            for row in sorted(split_resolved, key=lambda item: item.query.query_id):
                writer.writerow((
                    row.query.query_id,
                    row.query.split,
                    intent_family_key(row.query.text),
                    row.provenance,
                ))
        (stage / "source-manifest.json").write_text(
            json.dumps(source_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (stage / "DATASET_CARD.md").write_text(card, encoding="utf-8")
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing dataset: {output_dir}")
        os.rename(stage, output_dir)
    finally:
        if stage.exists():
            stage.resolve().relative_to(output_dir.parent.resolve())
            shutil.rmtree(stage)
    return summary
