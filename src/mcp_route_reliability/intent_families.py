"""Keep scenario-template families within one split and synchronize variants."""

# ruff: noqa: RUF001 -- Chinese query text intentionally uses Chinese punctuation.

from __future__ import annotations

import random
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence

from mcp_route_reliability.models import QueryCase, Split
from mcp_route_reliability.validation import normalize_query_text

SCENARIO_SUFFIX = re.compile(
    r"\s*(?:\((?:v1\s*)?(?:scenario|场景)\s*\d+\)|"
    r"\[(?:v1\s*)?scenario\s*\d+\])$",
    flags=re.IGNORECASE,
)


def intent_family_key(text: str) -> str:
    """Strip numbered generator suffixes before comparing normalized intent text."""

    normalized = unicodedata.normalize("NFKC", text)
    return normalize_query_text(SCENARIO_SUFFIX.sub("", normalized))


def _choose_families(
    groups: Mapping[str, tuple[str, ...]], target: int, seed: int
) -> set[str]:
    if target < 0:
        raise ValueError(f"intent-family capacity is negative: {target}")
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    ordered = sorted(groups)
    random.Random(seed).shuffle(ordered)
    for key in ordered:
        size = len(groups[key])
        for count, selected in sorted(reachable.items(), reverse=True):
            new_count = count + size
            if new_count <= target and new_count not in reachable:
                reachable[new_count] = (*selected, key)
    if target not in reachable:
        raise ValueError(f"intent families cannot fill split capacity {target}")
    return set(reachable[target])


def assign_family_splits(
    bases: Sequence[QueryCase], pinned_test_ids: set[str], seed: int = 20260924
) -> dict[str, Split]:
    """Pin selected-variant families to Test, then fill all splits by whole families."""

    ids = {row.query_id for row in bases}
    if len(ids) != len(bases) or len(bases) != 300:
        raise ValueError("family allocation requires 300 unique base queries")
    if not pinned_test_ids <= ids:
        raise ValueError("pinned Test IDs contain an unknown base")
    groups_mutable: dict[str, list[str]] = defaultdict(list)
    for row in bases:
        groups_mutable[intent_family_key(row.text)].append(row.query_id)
    groups = {key: tuple(sorted(values)) for key, values in groups_mutable.items()}
    pinned_families = {
        key for key, values in groups.items() if pinned_test_ids.intersection(values)
    }
    pinned_count = sum(len(groups[key]) for key in pinned_families)
    if pinned_count > 100:
        raise ValueError(f"pinned Test families exceed 100 bases: {pinned_count}")
    available = {key: values for key, values in groups.items() if key not in pinned_families}
    test_families = pinned_families | _choose_families(available, 100 - pinned_count, seed)
    remaining = {key: values for key, values in groups.items() if key not in test_families}
    calibration_families = _choose_families(remaining, 100, seed + 1)
    result: dict[str, Split] = {}
    for key, values in groups.items():
        split: Split = (
            "test" if key in test_families else "calibration"
            if key in calibration_families else "development"
        )
        result.update({query_id: split for query_id in values})
    expected_splits: tuple[Split, ...] = ("development", "calibration", "test")
    if {split: list(result.values()).count(split) for split in expected_splits} != {
        "development": 100, "calibration": 100, "test": 100,
    }:
        raise ValueError("intent-family allocation did not achieve exact split sizes")
    return result


_REWRITTEN_183 = {
    "paraphrase": (
        "请查看 /project，不过我还没决定要列出当前目录，还是递归找出匹配 *.md 的路径 "
        "[variant v1_q_183]"
    ),
    "cross_language": (
        "In /project, should I list the current directory or recursively find paths "
        "matching *.md? [variant v1_q_183]"
    ),
    "noise": "那个……看看 /project 吧，是列当前层，还是递归找 *.md 路径来着？ "
    "[variant v1_q_183]",
}


def synchronize_variants(
    candidate_variants: Sequence[QueryCase],
    final_bases: Mapping[str, QueryCase],
    split_by_id: Mapping[str, Split],
) -> tuple[QueryCase, ...]:
    """Inherit each final base's gold labels and split without changing variant IDs."""

    revised: list[QueryCase] = []
    for variant in candidate_variants:
        if variant.perturbation_type is None:
            raise ValueError(f"{variant.query_id}: expected a variant")
        base = final_bases[variant.base_query_id]
        split = split_by_id[base.query_id]
        text = variant.text
        if base.query_id == "v1_q_183":
            text = _REWRITTEN_183[variant.perturbation_type]
        payload = variant.model_dump(mode="json")
        payload.update({
            "text": text,
            "case_type": base.case_type,
            "difficulty": base.difficulty,
            "relevant_tool_ids": base.relevant_tool_ids,
            "should_abstain": base.should_abstain,
            "abstain_reason": base.abstain_reason,
            "split": split,
            "label_rationale": base.label_rationale,
            "candidate_conflict_note": base.candidate_conflict_note,
            "tool_corpus_hash": base.tool_corpus_hash,
        })
        revised.append(QueryCase.model_validate(payload))
    return tuple(revised)
