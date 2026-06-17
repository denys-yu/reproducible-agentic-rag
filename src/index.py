"""ChromaDB persistent index builder and deterministic retrieval.

Builds a `PersistentClient` collection in cosine space from the chunked corpus and exposes
retrieval that breaks score ties deterministically by `(score DESC, doc_id ASC)` — never relying
on vector-store insertion order.
"""

from __future__ import annotations

from typing import Any

from src.config import Config


def build_index(chunks: list[dict[str, Any]], config: Config) -> Any:
    """Build/persist the ChromaDB cosine collection from chunks; return the vector store handle."""
    raise NotImplementedError


def get_vectorstore(config: Config) -> Any:
    """Open the persisted ChromaDB collection for querying."""
    raise NotImplementedError


def retrieve(vectorstore: Any, query: str, config: Config) -> list[tuple[str, float]]:
    """Return the top-k `(doc_id, score)` hits, sorted by `(score DESC, doc_id ASC)`."""
    raise NotImplementedError
