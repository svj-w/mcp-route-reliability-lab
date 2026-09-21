from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

import mcp_route_reliability.routers.embedding as embedding_module
from mcp_route_reliability.models import InferenceQuery, ToolRecord
from mcp_route_reliability.routers.embedding import EmbeddingRouter


class FakeProvider:
    def __init__(self, vectors: NDArray[np.float64]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[str, ...]] = []

    @property
    def version(self) -> str:
        return "fake:1"

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]:
        self.calls.append(tuple(texts))
        return self.vectors.copy()


def make_tool(tool_id: str, description: str) -> ToolRecord:
    return ToolRecord(
        tool_id=tool_id,
        name=tool_id.rsplit(".", maxsplit=1)[-1],
        description=description,
        input_schema={"type": "object", "properties": {}},
        server="test",
        domain="test",
        source_url="https://example.invalid/embedding-test",
        license="CC0-1.0",
    )


def make_query() -> InferenceQuery:
    return InferenceQuery(query_id="q_001", text="find source code", language="en")


def test_cosine_similarity_orders_candidates() -> None:
    tools = [make_tool("tool.b", "weather"), make_tool("tool.a", "source code")]
    provider = FakeProvider(
        np.asarray(
            [
                [1.0, 0.0],  # query
                [0.0, 1.0],  # tool.b
                [0.8, 0.2],  # tool.a
            ],
            dtype=np.float64,
        )
    )

    prediction = EmbeddingRouter(provider).rank(
        make_query(), tools, top_k=2, query_set_hash="sha256:queries"
    )

    assert [candidate.tool_id for candidate in prediction.candidates] == ["tool.a", "tool.b"]
    assert prediction.candidates[0].score == pytest.approx(0.9701425)
    assert prediction.candidates[1].score == 0.0
    assert prediction.model_version == "fake:1"
    assert len(provider.calls) == 1


def test_zero_vectors_have_zero_similarity() -> None:
    tools = [make_tool("tool.zero", "nothing")]
    provider = FakeProvider(np.zeros((2, 3), dtype=np.float64))

    prediction = EmbeddingRouter(provider).rank(
        make_query(), tools, top_k=1, query_set_hash="sha256:queries"
    )

    assert prediction.candidates[0].score == 0.0


def test_tied_similarities_use_tool_id() -> None:
    tools = [make_tool("tool.z", "z"), make_tool("tool.a", "a")]
    provider = FakeProvider(np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float64))

    prediction = EmbeddingRouter(provider).rank(
        make_query(), tools, top_k=5, query_set_hash="sha256:queries"
    )

    assert [candidate.tool_id for candidate in prediction.candidates] == ["tool.a", "tool.z"]


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        (np.asarray([1.0, 2.0]), "two-dimensional"),
        (np.zeros((2, 2)), "row count"),
        (np.zeros((3, 0)), "dimension"),
        (np.asarray([[1.0], [np.nan], [0.0]]), "finite"),
    ],
)
def test_invalid_embedding_shapes_are_rejected(vectors: NDArray[np.float64], message: str) -> None:
    tools = [make_tool("tool.a", "a"), make_tool("tool.b", "b")]
    with pytest.raises(ValueError, match=message):
        EmbeddingRouter(FakeProvider(vectors)).rank(
            make_query(), tools, top_k=2, query_set_hash="sha256:queries"
        )


def test_fake_provider_path_never_imports_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(name: str) -> object:
        raise AssertionError(f"unexpected optional import: {name}")

    monkeypatch.setattr(embedding_module.importlib, "import_module", fail_import)
    provider = FakeProvider(np.asarray([[1.0], [1.0]], dtype=np.float64))

    prediction = EmbeddingRouter(provider).rank(
        make_query(),
        [make_tool("tool.a", "a")],
        top_k=1,
        query_set_hash="sha256:queries",
    )

    assert prediction.candidates[0].tool_id == "tool.a"


def test_invalid_router_inputs_are_rejected() -> None:
    provider = FakeProvider(np.asarray([[1.0]], dtype=np.float64))
    router = EmbeddingRouter(provider)
    with pytest.raises(ValueError, match="at least one tool"):
        router.rank(make_query(), [], top_k=1, query_set_hash="sha256:queries")
    with pytest.raises(ValueError, match="top_k"):
        router.rank(
            make_query(),
            [make_tool("tool.a", "a")],
            top_k=0,
            query_set_hash="sha256:queries",
        )
