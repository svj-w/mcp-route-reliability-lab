"""Injectable multilingual embedding router with deterministic cosine ranking."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from mcp_route_reliability.hashing import tool_corpus_sha256
from mcp_route_reliability.models import (
    InferenceQuery,
    RankedCandidate,
    RouterPrediction,
    ScoreMetadata,
    ToolRecord,
)


class EmbeddingProvider(Protocol):
    @property
    def version(self) -> str: ...

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]: ...


class _SentenceEncoder(Protocol):
    def encode(
        self,
        sentences: Sequence[str],
        *,
        convert_to_numpy: bool,
        show_progress_bar: bool,
        normalize_embeddings: bool,
    ) -> object: ...


class SentenceTransformerProvider:
    """Load sentence-transformers only when embeddings are actually requested."""

    def __init__(
        self,
        model_name: str,
        *,
        revision: str = "main",
        device: str = "cpu",
    ) -> None:
        if device != "cpu":
            raise ValueError("V1.0 SentenceTransformerProvider requires device='cpu'")
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self._model: _SentenceEncoder | None = None

    @property
    def version(self) -> str:
        try:
            package_version = version("sentence-transformers")
        except PackageNotFoundError:
            package_version = "not-installed"
        return (
            f"sentence-transformers:{package_version}|model:{self.model_name}"
            f"|revision:{self.revision}|device:{self.device}"
        )

    def _load_model(self) -> _SentenceEncoder:
        if self._model is None:
            try:
                module = importlib.import_module("sentence_transformers")
            except ImportError as error:
                raise RuntimeError(
                    "sentence-transformers is optional; install the 'embedding' extra"
                ) from error
            factory = cast(Callable[..., _SentenceEncoder], module.SentenceTransformer)
            self._model = factory(
                self.model_name,
                revision=self.revision,
                device=self.device,
            )
        return self._model

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]:
        values = self._load_model().encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return np.asarray(values, dtype=np.float64)


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


def _tool_text(tool: ToolRecord) -> str:
    parameters = " ".join(_parameter_names(tool.input_schema))
    return f"{tool.name}\n{tool.description}\n{parameters}".strip()


class EmbeddingRouter:
    router_id = "embedding"
    router_version = "1.0.0"

    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider

    def rank(
        self,
        query: InferenceQuery,
        tools: Sequence[ToolRecord],
        top_k: int,
        query_set_hash: str,
    ) -> RouterPrediction:
        if not tools:
            raise ValueError("embedding routing requires at least one tool")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not query_set_hash:
            raise ValueError("query_set_hash must not be empty")
        if len({tool.tool_id for tool in tools}) != len(tools):
            raise ValueError("tool IDs must be unique")

        texts = [query.text, *(_tool_text(tool) for tool in tools)]
        embeddings = np.asarray(self.provider.encode(texts), dtype=np.float64)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional matrix")
        if embeddings.shape[0] != len(texts):
            raise ValueError(
                f"embedding row count {embeddings.shape[0]} does not match {len(texts)} texts"
            )
        if embeddings.shape[1] < 1:
            raise ValueError("embedding dimension must be at least 1")
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings must contain only finite values")

        query_vector = embeddings[0]
        tool_vectors = embeddings[1:]
        query_norm = float(np.linalg.norm(query_vector))
        tool_norms = np.linalg.norm(tool_vectors, axis=1)
        denominators = tool_norms * query_norm
        similarities = np.zeros(len(tools), dtype=np.float64)
        nonzero = denominators > 0
        similarities[nonzero] = (tool_vectors[nonzero] @ query_vector) / denominators[nonzero]

        scored = [
            (tool.tool_id, float(score)) for tool, score in zip(tools, similarities, strict=True)
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
            model_version=self.provider.version,
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
