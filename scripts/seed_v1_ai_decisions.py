"""Seed the reviewed AI decision log from frozen inputs, without overwriting it."""

from __future__ import annotations

from pathlib import Path

from mcp_route_reliability.ai_annotation_sources import load_completed_rows
from mcp_route_reliability.ai_label_decisions import Decision
from mcp_route_reliability.annotation_workflow import load_manifest_query_ids
from mcp_route_reliability.datasets import load_annotations, load_queries, load_tools

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "data" / "v1-candidate"
INPUTS = ROOT / "data" / "v1-ai-inputs"
PRIMARY_SHA256 = "4497cdb9e0e53ab5d351e93fd5af7990cebc900e95216d6342d67e8b8d339377"
SECONDARY_SHA256 = "9aefbd47bd2cec803da3aedd50a65a9c75bb67370c88e8b7f66fcd2fc24a7990"

EXACT_NODE_IDS = {"v1_q_101", "v1_q_127", "v1_q_142", "v1_q_205"}
MISSING_REPO_IDS = {"v1_q_195", "v1_q_197", "v1_q_213", "v1_q_238"}
SECONDARY_GROUPS = {
    ("git.diff_staged", "git.diff_unstaged"): {"v01_q_001", "v01_q_004", "v01_q_041"},
    ("filesystem.directory_tree", "filesystem.list_directory"): {
        "v01_q_009", "v01_q_012", "v01_q_028", "v01_q_065"
    },
    ("memory.open_nodes", "memory.search_nodes"): {
        "v01_q_017", "v01_q_020", "v01_q_057", "v01_q_060"
    },
    ("time.convert_time", "time.get_current_time"): {
        "v01_q_033", "v01_q_036", "v01_q_073", "v01_q_076"
    },
}
SECONDARY_RATIONALES = {
    "git.diff_staged": (
        "The revised request does not specify staged versus unstaged changes; "
        "both diff tools remain plausible until clarified."
    ),
    "filesystem.directory_tree": (
        "The revised request does not distinguish a recursive tree from a direct "
        "directory listing; ask which view is wanted."
    ),
    "memory.open_nodes": (
        "The revised request leaves exact-name opening versus broader node search "
        "unresolved; ask for the intended lookup."
    ),
    "time.convert_time": (
        "The revised request does not distinguish converting a stated time from "
        "obtaining the current time; ask for clarification."
    ),
}


def main() -> None:
    output = INPUTS / "decisions.jsonl"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite reviewed decision log: {output}")
    tools = load_tools(CANDIDATE / "tools.json")
    queries = load_queries(CANDIDATE / "queries.jsonl")
    bases = {row.query_id: row for row in queries if row.perturbation_type is None}
    primary = {
        row.query_id: row
        for row in load_annotations(CANDIDATE / "annotations" / "primary.jsonl")
    }
    recheck = load_completed_rows(
        INPUTS / "raw" / "completed-primary-recheck.csv",
        load_manifest_query_ids(CANDIDATE / "annotations" / "primary-recheck-assignment.csv"),
        tools,
        PRIMARY_SHA256,
    )
    secondary = load_completed_rows(
        INPUTS / "raw" / "completed-secondary.csv",
        load_manifest_query_ids(CANDIDATE / "annotations" / "secondary-sample-manifest.csv"),
        tools,
        SECONDARY_SHA256,
    )
    decisions: list[Decision] = []
    for query_id, source in sorted(recheck.items()):
        base = bases[query_id]
        prior = primary[query_id]
        annotation = source.annotation
        changed = source.text != base.text or (
            set(annotation.relevant_tool_ids) != set(prior.relevant_tool_ids)
            or annotation.should_abstain != prior.should_abstain
            or annotation.abstain_reason != prior.abstain_reason
        )
        if not changed:
            continue
        case_type = (
            "clarify" if annotation.should_abstain and annotation.abstain_reason != "no_match"
            else "no_match" if annotation.abstain_reason == "no_match"
            else base.case_type
        )
        rationale = annotation.label_rationale
        conflict_note = annotation.candidate_conflict_note
        if query_id == "v01_q_039":
            rationale = (
                "filesystem.search_files requires a search root (path), but the request says "
                "the location is unknown and provides no root; ask for a path before routing."
            )
            conflict_note = "The candidate route ignored the required path input."
        decisions.append(Decision(
            query_id=query_id, target="final_gold", text=source.text,
            case_type=case_type, relevant_tool_ids=annotation.relevant_tool_ids,
            should_abstain=annotation.should_abstain,
            abstain_reason=annotation.abstain_reason,
            rationale=rationale, candidate_conflict_note=conflict_note,
            reviewer_type="ai_source_import", source_role="primary_recheck",
            source_query_id=query_id,
        ))
    for query_id in sorted(EXACT_NODE_IDS | MISSING_REPO_IDS):
        base = bases[query_id]
        if query_id in EXACT_NODE_IDS:
            case_type, tool_ids, abstain, reason = (
                "hard_negative", ("memory.open_nodes",), False, None
            )
            rationale = (
                "The request supplies an exact node name. memory.open_nodes accepts exact names; "
                "memory.search_nodes performs broader search and is not equivalent."
            )
        else:
            case_type, tool_ids, abstain, reason = (
                "clarify", ("git.show",), True, "insufficient_information"
            )
            rationale = (
                "git.show can inspect the named commit, but its schema requires repo_path; "
                "the request does not identify the repository, so clarification is needed."
            )
        decisions.append(Decision(
            query_id=query_id, target="final_gold", text=base.text,
            case_type=case_type, relevant_tool_ids=tool_ids,
            should_abstain=abstain, abstain_reason=reason,
            rationale=rationale,
            candidate_conflict_note=(
                "Post-hoc Tool-schema correction to the provisional multi-valid label."
            ),
            reviewer_type="ai_posthoc", source_role="secondary_raw",
            source_query_id=query_id,
        ))
    for tool_ids, query_ids in SECONDARY_GROUPS.items():
        for query_id in sorted(query_ids):
            source = recheck[query_id]
            if secondary[query_id].text == source.text:
                raise ValueError(f"{query_id}: secondary text was expected to be stale")
            decisions.append(Decision(
                query_id=query_id, target="secondary_posthoc", text=source.text,
                case_type="clarify", relevant_tool_ids=tool_ids,
                should_abstain=True, abstain_reason="ambiguous",
                rationale=f"{query_id}: {SECONDARY_RATIONALES[tool_ids[0]]}",
                candidate_conflict_note=(
                    "Post-hoc relabeling against revised primary text; "
                    "not an independent blind annotation."
                ),
                reviewer_type="ai_posthoc", source_role="secondary_raw",
                source_query_id=query_id,
            ))
    keys = [(row.query_id, row.target) for row in decisions]
    if len(keys) != len(set(keys)) or len(decisions) != 52:
        raise ValueError(f"Unexpected decision coverage: {len(decisions)}")
    output.write_text(
        "".join(
            row.model_dump_json() + "\n"
            for row in sorted(decisions, key=lambda d: (d.query_id, d.target))
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(decisions)} decisions to {output}")


if __name__ == "__main__":
    main()
