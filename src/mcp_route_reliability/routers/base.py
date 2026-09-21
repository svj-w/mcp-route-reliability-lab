"""Protocol implemented by built-in and adapter-backed routers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from mcp_route_reliability.models import InferenceQuery, RouterPrediction, ToolRecord


class Router(Protocol):
    def rank(
        self,
        query: InferenceQuery,
        tools: Sequence[ToolRecord],
        top_k: int,
        query_set_hash: str,
    ) -> RouterPrediction: ...
