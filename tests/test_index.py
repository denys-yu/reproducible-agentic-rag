"""Offline tests for src.index retrieval determinism and tie-breaking.

A tiny deterministic fake embedder over an in-memory (ephemeral) Chroma collection exercises the
real build + query path without touching the OpenAI API. The real API build stays behind the CLI.
"""

from __future__ import annotations

import chromadb
import pytest

from src.config import Config
from src.data import Chunk
from src.index import build_index, retrieve

# Fixed 2-D embeddings. Two q1 docs share the SAME vector as the query (exact similarity tie),
# so ordering between them must be decided by doc_id ASC. "other" is orthogonal (lower similarity).
# The q2 doc is highly similar but must be excluded by the per-question `where` filter.
_VECTORS: dict[str, list[float]] = {
    "QUERY": [1.0, 0.0],
    "dup-b": [1.0, 0.0],
    "dup-a": [1.0, 0.0],
    "other": [0.0, 1.0],
    "q2doc": [1.0, 0.0],
}


class FakeEmbedder:
    """Deterministic lookup embedder; raises on unknown text to catch fixture mistakes."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(_VECTORS[text]) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return list(_VECTORS[text])


def _chunk(question_id: str, doc_id: str, text: str, *, is_gold: bool = False) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        question_id=question_id,
        title=f"title-{doc_id}",
        text=text,
        is_gold=is_gold,
        para_index=0,
        chunk_index=0,
    )


_FIXTURE: list[Chunk] = [
    # Two tied q1 docs whose doc_ids deliberately sort id_a < id_b while text differs.
    _chunk("q1", "id_b", "dup-b", is_gold=True),
    _chunk("q1", "id_a", "dup-a"),
    _chunk("q1", "id_c", "other"),
    # A q2 doc with maximal similarity to the probe — must never leak into q1 results.
    _chunk("q2", "id_z", "q2doc"),
]


@pytest.fixture
def collection():
    config = Config()
    client = chromadb.EphemeralClient()
    return build_index(config, chunks=_FIXTURE, embedder=FakeEmbedder(), client=client)


def test_retrieval_is_scoped_to_question(collection):
    hits = retrieve("QUERY", "q1", k=10, collection=collection, embedder=FakeEmbedder())
    assert {h["question_id"] for h in hits} == {"q1"}
    assert "id_z" not in {h["doc_id"] for h in hits}  # the q2 doc is excluded


def test_tie_breaks_by_doc_id_ascending(collection):
    hits = retrieve("QUERY", "q1", k=10, collection=collection, embedder=FakeEmbedder())
    # id_a and id_b are exact similarity ties -> ordered by doc_id ASC; id_c (orthogonal) last.
    assert [h["doc_id"] for h in hits] == ["id_a", "id_b", "id_c"]


def test_retrieval_ordering_is_deterministic(collection):
    fake = FakeEmbedder()
    first = [h["doc_id"] for h in retrieve("QUERY", "q1", k=10, collection=collection, embedder=fake)]
    second = [h["doc_id"] for h in retrieve("QUERY", "q1", k=10, collection=collection, embedder=fake)]
    assert first == second


def test_similarity_formula_is_one_minus_distance(collection):
    hits = retrieve("QUERY", "q1", k=10, collection=collection, embedder=FakeEmbedder())
    by_id = {h["doc_id"]: h for h in hits}
    assert by_id["id_a"]["similarity"] == pytest.approx(1.0, abs=1e-6)  # identical vector
    assert by_id["id_c"]["similarity"] == pytest.approx(0.0, abs=1e-6)  # orthogonal vector


def test_k_limits_results(collection):
    hits = retrieve("QUERY", "q1", k=1, collection=collection, embedder=FakeEmbedder())
    assert [h["doc_id"] for h in hits] == ["id_a"]


def test_unknown_question_returns_empty(collection):
    assert retrieve("QUERY", "missing", k=5, collection=collection, embedder=FakeEmbedder()) == []
