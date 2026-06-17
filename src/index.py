"""ChromaDB persistent index builder and deterministic per-question retrieval.

Builds one cosine-space `PersistentClient` collection over the sampled corpus, then retrieves
within a single question's paragraphs with explicit, reproducible tie-breaking.

Design choices that protect reproducibility:
- **Composite record id** `f"{question_id}:{doc_id}"`. HotpotQA distractor reuses the same
  paragraph text across questions, so the same normalized chunk yields the same `doc_id`. Keying a
  global collection by `doc_id` alone would silently dedupe those, corrupting per-question
  retrieval. The composite id keeps one record per (question, chunk).
- **Per-question scoping.** Retrieval always passes `where={"question_id": ...}`; it never crosses
  question boundaries.
- **Explicit tie-breaking.** We fetch *all* of a question's candidates, convert Chroma's cosine
  distance to similarity (`similarity = 1 - distance`), and sort by `(similarity DESC, doc_id ASC)`
  ourselves — never trusting Chroma's internal ordering for ties.
- **Idempotent build.** The collection is dropped and recreated from the (deduped) input, so a
  rebuild over the same input yields the same contents.

Embeddings go through an injectable `Embedder`; the default hits the pinned OpenAI model, and tests
inject a deterministic fake so retrieval logic is exercised without the API.
"""

from __future__ import annotations

import argparse
from itertools import batched
from typing import Any, Protocol, TypedDict

from src.config import Config
from src.data import Chunk, load_sampled_questions, to_chunks

COLLECTION_NAME = "hotpotqa_distractor"
_EMBED_BATCH = 256  # inputs per embedding request / Chroma add


class Embedder(Protocol):
    """Minimal embedding interface (documents for indexing, single query for retrieval)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class RetrievedChunk(TypedDict):
    """One retrieval hit, scoped to its question and carrying its cosine similarity."""

    doc_id: str
    question_id: str
    title: str
    text: str
    is_gold: bool
    similarity: float


class OpenAIEmbedder:
    """Default embedder: the pinned `text-embedding-3-small` snapshot at fixed `dimensions`."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            key = (
                self._config.openai_api_key.get_secret_value()
                if self._config.openai_api_key is not None
                else None
            )
            self._client = OpenAI(api_key=key) if key else OpenAI()
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._get_client().embeddings.create(
            model=self._config.embedding_model,
            input=list(texts),
            dimensions=self._config.embedding_dimensions,
        )
        # Re-sort by index defensively so output order always matches input order.
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_client(config: Config) -> Any:
    """Open the persistent ChromaDB client rooted at `config.chroma_dir`."""
    import chromadb

    return chromadb.PersistentClient(path=str(config.chroma_dir))


def composite_id(chunk: Chunk) -> str:
    """Return the per-question record id `f"{question_id}:{doc_id}"`."""
    return f"{chunk['question_id']}:{chunk['doc_id']}"


def _record_metadata(chunk: Chunk) -> dict[str, Any]:
    return {
        "doc_id": chunk["doc_id"],
        "question_id": chunk["question_id"],
        "title": chunk["title"],
        "is_gold": chunk["is_gold"],
        "para_index": chunk["para_index"],
        "chunk_index": chunk["chunk_index"],
    }


def _dedupe_by_record_id(chunks: list[Chunk]) -> list[Chunk]:
    """Drop chunks sharing a composite id (identical text within a question), keeping the first."""
    seen: set[str] = set()
    unique: list[Chunk] = []
    for chunk in chunks:
        rid = composite_id(chunk)
        if rid in seen:
            continue
        seen.add(rid)
        unique.append(chunk)
    return unique


def iter_corpus_chunks(config: Config) -> list[Chunk]:
    """Load the sampled questions and flatten them into the full chunk corpus."""
    chunks: list[Chunk] = []
    for question in load_sampled_questions(config):
        chunks.extend(to_chunks(question, config))
    return chunks


def _drop_collection_if_present(client: Any, name: str) -> None:
    existing = {c if isinstance(c, str) else c.name for c in client.list_collections()}
    if name in existing:
        client.delete_collection(name)


def build_index(
    config: Config,
    *,
    chunks: list[Chunk] | None = None,
    embedder: Embedder | None = None,
    client: Any | None = None,
) -> Any:
    """Build and persist the cosine collection from chunks; return the Chroma collection.

    Idempotent: the collection is dropped and recreated from the deduped input. Embeddings are
    computed via `embedder` (default: pinned OpenAI). `chunks`/`embedder`/`client` are injectable
    so the build path can run offline in tests.
    """
    if chunks is None:
        chunks = iter_corpus_chunks(config)
    if embedder is None:
        embedder = OpenAIEmbedder(config)
    if client is None:
        client = get_client(config)

    records = _dedupe_by_record_id(chunks)

    _drop_collection_if_present(client, COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    for batch in batched(records, _EMBED_BATCH):
        texts = [chunk["text"] for chunk in batch]
        collection.add(
            ids=[composite_id(chunk) for chunk in batch],
            documents=texts,
            metadatas=[_record_metadata(chunk) for chunk in batch],
            embeddings=embedder.embed_documents(texts),
        )
    return collection


def open_collection(config: Config, *, client: Any | None = None) -> Any:
    """Open the persisted collection for querying (raises if it has not been built yet)."""
    if client is None:
        client = get_client(config)
    return client.get_collection(COLLECTION_NAME)


def retrieve(
    query: str,
    question_id: str,
    k: int,
    *,
    collection: Any,
    embedder: Embedder,
) -> list[RetrievedChunk]:
    """Return the top-k chunks for one question, deterministically tie-broken.

    Scoped to `question_id` via a `where` filter. All of the question's candidates are scored, then
    sorted by `(similarity DESC, doc_id ASC)` — Chroma's internal ordering is never trusted for
    ties. Cosine distance is converted to similarity as `similarity = 1 - distance`.
    """
    candidate_ids = collection.get(where={"question_id": question_id}, include=[])["ids"]
    if not candidate_ids:
        return []

    result = collection.query(
        query_embeddings=[embedder.embed_query(query)],
        n_results=len(candidate_ids),
        where={"question_id": question_id},
        include=["distances", "metadatas", "documents"],
    )
    metadatas = result["metadatas"][0]
    documents = result["documents"][0]
    distances = result["distances"][0]

    hits: list[RetrievedChunk] = [
        RetrievedChunk(
            doc_id=meta["doc_id"],
            question_id=meta["question_id"],
            title=meta["title"],
            text=document,
            is_gold=bool(meta["is_gold"]),
            similarity=1.0 - float(distance),  # Chroma cosine distance = 1 - cosine similarity
        )
        for meta, document, distance in zip(metadatas, documents, distances, strict=True)
    ]
    hits.sort(key=lambda hit: (-hit["similarity"], hit["doc_id"]))
    return hits[:k]


def _run_build(config: Config) -> None:
    chunks = iter_corpus_chunks(config)
    collection = build_index(config, chunks=chunks)
    n_questions = len({chunk["question_id"] for chunk in chunks})
    print(
        f"Built collection '{COLLECTION_NAME}' at {config.chroma_dir}: "
        f"{n_questions} questions, {len(chunks)} chunks "
        f"(deduped collection count={collection.count()})"
    )


def _run_smoke_test(config: Config, question_id: str) -> None:
    embedder = OpenAIEmbedder(config)
    collection = open_collection(config)
    probe = "reproducibility determinism probe"  # fixed query; we assert ordering, not relevance

    first = retrieve(probe, question_id, config.top_k, collection=collection, embedder=embedder)
    second = retrieve(probe, question_id, config.top_k, collection=collection, embedder=embedder)
    if not first:
        raise SystemExit(f"No chunks indexed for question_id={question_id!r}. Did you --build?")

    ids_first = [hit["doc_id"] for hit in first]
    ids_second = [hit["doc_id"] for hit in second]
    assert ids_first == ids_second, f"Non-deterministic ordering: {ids_first} != {ids_second}"

    print(f"Smoke test OK for question_id={question_id!r}: identical ordering across two retrievals.")
    for rank, hit in enumerate(first):
        print(
            f"  #{rank}  sim={hit['similarity']:.6f}  gold={hit['is_gold']}  "
            f"doc_id={hit['doc_id'][:12]}...  title={hit['title']!r}"
        )


def main(argv: list[str] | None = None) -> None:
    """CLI: build/persist the index, or run a per-question retrieval determinism check."""
    parser = argparse.ArgumentParser(
        prog="python -m src.index",
        description="Build the persistent vector index or run a retrieval determinism smoke test.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true", help="build and persist the index")
    group.add_argument(
        "--smoke-test",
        metavar="QUESTION_ID",
        help="retrieve twice for QUESTION_ID and assert identical orderings",
    )
    args = parser.parse_args(argv)

    config = Config()
    if args.build:
        _run_build(config)
    else:
        _run_smoke_test(config, args.smoke_test)


if __name__ == "__main__":
    main()
