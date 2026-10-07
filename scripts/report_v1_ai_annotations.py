"""Write provenance-aware annotation diagnostics for the frozen AI experiment."""

# ruff: noqa: RUF001 -- Chinese report copy intentionally uses Chinese punctuation.

from __future__ import annotations

import json
from pathlib import Path

from mcp_route_reliability.ai_annotation_sources import load_completed_rows
from mcp_route_reliability.ai_dataset import PRIMARY_SHA256, SECONDARY_SHA256
from mcp_route_reliability.annotation_workflow import load_manifest_query_ids
from mcp_route_reliability.datasets import (
    cohen_kappa,
    load_annotations,
    load_queries,
    load_tools,
    mean_set_jaccard,
)
from mcp_route_reliability.models import Annotation

ROOT = Path(__file__).resolve().parents[1]


def _category(row: Annotation) -> str:
    return row.abstain_reason if row.should_abstain and row.abstain_reason else "route"


def _comparison(left: list[Annotation], right: list[Annotation]) -> dict[str, object]:
    if len(left) != len(right) or not left:
        raise ValueError("annotation comparison needs nonempty equal-sized populations")
    kappa = cohen_kappa([_category(row) for row in left], [_category(row) for row in right])
    jaccard = mean_set_jaccard(
        [frozenset(row.relevant_tool_ids) for row in left],
        [frozenset(row.relevant_tool_ids) for row in right],
    )
    exact = sum(
        _category(a) == _category(b)
        and frozenset(a.relevant_tool_ids) == frozenset(b.relevant_tool_ids)
        for a, b in zip(left, right, strict=True)
    )
    return {
        "n": len(left),
        "exact": exact,
        "exact_rate": exact / len(left),
        "jaccard": jaccard.mean,
        "kappa": kappa.value if kappa.available else None,
        "kappa_reason": kappa.reason,
    }


def main() -> None:
    candidate = ROOT / "data" / "v1-candidate"
    inputs = ROOT / "data" / "v1-ai-inputs"
    output = ROOT / "data" / "v1-ai-experimental"
    tools = load_tools(candidate / "tools.json")
    raw_primary = load_completed_rows(
        inputs / "raw" / "completed-primary-recheck.csv",
        load_manifest_query_ids(candidate / "annotations" / "primary-recheck-assignment.csv"),
        tools, PRIMARY_SHA256,
    )
    raw_secondary = load_completed_rows(
        inputs / "raw" / "completed-secondary.csv",
        load_manifest_query_ids(candidate / "annotations" / "secondary-sample-manifest.csv"),
        tools, SECONDARY_SHA256,
    )
    provisional = {
        row.query_id: row
        for row in load_annotations(candidate / "annotations" / "primary.jsonl")
    }
    final = {row.query_id: row for row in load_annotations(output / "annotations" / "final.jsonl")}
    revisions = {
        row.query_id: row
        for row in load_annotations(output / "annotations" / "secondary-posthoc.jsonl")
    }
    final_text = {
        row.query_id: row.text
        for row in load_queries(output / "queries.jsonl")
        if row.perturbation_type is None
    }
    recheck_ids = sorted(
        query_id for query_id, row in raw_secondary.items()
        if query_id in raw_primary and row.text == raw_primary[query_id].text
    )
    generator_ids = sorted(
        query_id for query_id in raw_secondary
        if query_id not in raw_primary
    )
    changed_ids = sorted(
        query_id for query_id, row in raw_secondary.items()
        if row.text != final_text[query_id]
    )
    if (len(recheck_ids), len(generator_ids), len(changed_ids)) != (45, 40, 15):
        raise ValueError("unexpected same-text comparison populations")
    recheck_result = _comparison(
        [raw_primary[key].annotation for key in recheck_ids],
        [raw_secondary[key].annotation for key in recheck_ids],
    )
    generator_result = _comparison(
        [provisional[key] for key in generator_ids],
        [raw_secondary[key].annotation for key in generator_ids],
    )
    posthoc_result = _comparison(
        [final[key] for key in sorted(raw_secondary)],
        [revisions.get(key, raw_secondary[key].annotation) for key in sorted(raw_secondary)],
    )
    manifest = json.loads((output / "source-manifest.json").read_text(encoding="utf-8"))

    def fmt(result: dict[str, object]) -> str:
        kappa = (
            f"{result['kappa']:.3f}" if result["kappa"] is not None
            else f"不可用（{result['kappa_reason']}）"
        )
        return (
            f"{result['n']} | {result['exact']}/{result['n']} "
            f"({result['exact_rate']:.1%}) | {result['jaccard']:.3f} | {kappa}"
        )

    report = (
        "# V1 AI experimental：标注来源与一致性诊断\n\n"
        "本报告只描述 AI 标注源之间的差异；不是独立人工复核，也不是正式的标注员一致性研究。"
        "原始 CSV 均未修改。\n\n"
        "| 可比较样本及来源 | n | 完全一致 | 工具集合 Jaccard | 弃权类别 Cohen κ |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| 相同文本：AI 主复核 vs AI 次标注 | {fmt(recheck_result)} |\n"
        f"| 相同文本：生成器暂定主标注 vs AI 次标注 | {fmt(generator_result)} |\n\n"
        "上述两组共 85 条，但主标注来源不同，故不合并称为独立标注一致性。"
        "若 κ 不可用，是因为类别常量导致分母为零，并非完美一致。\n\n"
        "| 事后描述性对照 | n | 完全一致 | 工具集合 Jaccard | 弃权类别 Cohen κ |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| 最终 gold vs 次标注（含 15 条事后修订） | {fmt(posthoc_result)} |\n\n"
        "15 条旧文本次标注保存在 `raw/completed-secondary.csv`；对应新文本修订保存在 "
        "`annotations/secondary-posthoc.jsonl`。这 15 条不属于原始同文本比较。\n\n"
        f"- 主复核覆盖：{manifest['primary_recheck_count']} 条；未覆盖："
        f"{manifest['not_primary_rechecked_count']} 条。\n"
        f"- 最终标签来源：{manifest['provenance_counts']}。其中 "
        f"{manifest['provenance_counts']['generator_provisional']} 条仍为生成器暂定标签。\n"
        f"- 最终基础样本类型：{manifest['case_type_base_counts']}。\n"
        f"- 15 条事后修订 ID：{', '.join(changed_ids)}。\n"
        "- `v01_q_039` 和 8 条原 `multi_valid` 争议样本的修正见 `decisions.jsonl`。\n"
        "- 编号模板按归一化意图族分组；这只能消除已识别模板的跨划分泄漏，"
        "不能证明所有语义近重复都已消除。\n"
    )
    (output / "ANNOTATION_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "same_text_ai_recheck": recheck_result,
        "same_text_generator": generator_result,
        "posthoc_descriptive": posthoc_result,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
