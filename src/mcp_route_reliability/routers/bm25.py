"""Deterministic lexical baseline with mixed Chinese/English tokenization."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from importlib.metadata import version

from rank_bm25 import BM25Okapi

from mcp_route_reliability.hashing import tool_corpus_sha256
from mcp_route_reliability.models import (
    InferenceQuery,
    RankedCandidate,
    RouterPrediction,
    ScoreMetadata,
    ToolRecord,
)

_TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)*|[\u3400-\u4dbf\u4e00-\u9fff]+"
)


def tokenize(text: str) -> list[str]:
    """Tokenize Latin words and emit CJK unigrams plus adjacent bigrams."""

    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(text.casefold()):
        value = match.group(0)
        if "\u3400" <= value[0] <= "\u9fff":
            characters = list(value)
            tokens.extend(characters)
            tokens.extend(
                characters[index] + characters[index + 1]
                for index in range(len(characters) - 1)
            )
        else:
            tokens.append(value)
    return tokens


def _parameter_names(value: object) -> list[str]:
    names: list[str] = []
    if isinstance(value, Mapping):
        properties = value.get("properties")
        if isinstance(properties, Mapping):
            names.extend(str(name) for name in properties)
        for nested in value.values():
            names.extend(_parameter_names(nested))
    elif isinstance(value, list):
        for nested in value:
            names.extend(_parameter_names(nested))
    return names


def _tool_document(tool: ToolRecord) -> str:
    parameter_text = " ".join(_parameter_names(tool.input_schema))
    return f"{tool.name} {tool.description} {parameter_text}".strip()


class BM25Router:
    """Fit BM25 on the supplied immutable corpus and rank a single query."""

    router_id = "bm25"
    router_version = "1.0.0"

    def rank(
        self,
        query: InferenceQuery,
        tools: Sequence[ToolRecord],
        top_k: int,
        query_set_hash: str,
    ) -> RouterPrediction:
        if not tools:
            raise ValueError("BM25 routing requires at least one tool")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not query_set_hash:
            raise ValueError("query_set_hash must not be empty")
        if len({tool.tool_id for tool in tools}) != len(tools):
            raise ValueError("tool IDs must be unique")

        corpus_tokens = [tokenize(_tool_document(tool)) for tool in tools]
        bm25 = BM25Okapi(corpus_tokens)
        raw_scores = bm25.get_scores(tokenize(query.text))
        scored = [
            (tool.tool_id, float(score))
            for tool, score in zip(tools, raw_scores, strict=True)
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))

        candidates = tuple(
            RankedCandidate(tool_id=tool_id, score=score, rank=rank)
            for rank, (tool_id, score) in enumerate(scored[: min(top_k, len(scored))], start=1)
        )
        return RouterPrediction(
            query_id=query.query_id,
            router_id=self.router_id,
            router_version=self.router_version,
            model_version=f"rank-bm25:{version('rank-bm25')}",
            tool_corpus_hash=tool_corpus_sha256(tools),
            query_set_hash=query_set_hash,
            score_metadata=ScoreMetadata(
                score_type="similarity",
                higher_is_better=True,
                comparable_across_queries=False,
            ),
            candidates=candidates,
            latency_ms=None,
            error=None,
        )
