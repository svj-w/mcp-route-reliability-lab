"""Built-in router baselines and the shared router protocol."""

from mcp_route_reliability.routers.base import Router
from mcp_route_reliability.routers.bm25 import BM25Router
from mcp_route_reliability.routers.embedding import EmbeddingRouter, SentenceTransformerProvider

__all__ = ["BM25Router", "EmbeddingRouter", "Router", "SentenceTransformerProvider"]
