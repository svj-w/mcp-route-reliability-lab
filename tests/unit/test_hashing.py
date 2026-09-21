from __future__ import annotations

from conftest import make_tiny_tools
from mcp_route_reliability.hashing import canonical_sha256, tool_corpus_sha256


def test_dictionary_key_order_does_not_change_hash() -> None:
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})


def test_list_order_changes_hash() -> None:
    assert canonical_sha256(["a", "b"]) != canonical_sha256(["b", "a"])


def test_tool_corpus_hash_is_independent_of_input_order() -> None:
    tools = make_tiny_tools(3)
    assert tool_corpus_sha256(tools) == tool_corpus_sha256(list(reversed(tools)))


def test_hash_has_explicit_algorithm_prefix() -> None:
    value = canonical_sha256({"hello": "世界"})
    assert value.startswith("sha256:")
    assert len(value) == len("sha256:") + 64
